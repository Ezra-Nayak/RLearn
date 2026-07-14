import os
import glob
import time
from collections import deque
import random
import numpy as np

import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn
import torch.nn.functional as F  # <--- ADD THIS
from torch.distributions import Categorical
import cv2

# --- HYPERPARAMETERS & CONFIG ---
LR_ACTOR = 8e-6         # EXTREME SCALPEL: Half the LR to stop KL Overload thrashing
LR_CRITIC = 1e-4        # HIGHER LR: Allow Critic to quickly map the new GAE scale
TARGET_KL = 0.025       # RELAXED BRAKE: Allow the network a bit more room to breathe before aborting
ENTROPY_COEF = 0.002    # ALMOST ZERO: Stop the agent from exploring stupid moves, rely on mastery
GAMMA = 0.998           # EAGLE EYE: Expands the Critic's planning horizon to ~500 steps (50 rows)
GAE_LAMBDA = 0.95
EPS_CLIP = 0.1          # Tighter clipping for precise weight adjustments
K_EPOCHS = 4
HIDDEN_DIM = 512
NUM_ENVS = 48
ROLLOUT_STEPS = 128
MINIBATCH_SIZE = 64
CHECKPOINT_INTERVAL = 10000

# Targeted Model to Fine-Tune (The Grandmaster baseline)
TARGET_MODEL = "checkpoints/ppo_redemption_latest.pth"

# --- WARMUP CONFIG ---
# Crucial: Allow Critic to see the new +0.25 rewards before the Actor changes behavior
WARMUP_UPDATES = 25     # Extended warmup to guarantee Critic stability

VAE_CHECKPOINT = "checkpoints/sim_vae_best.pth"

# DEVICE MAPPING
# SpatialVQVAE -> DirectML GPU (for speed)
# ActorCritic -> CPU (for DirectML backwards-pass stability)
def setup_device():
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device()
    except ImportError:
        pass
    return torch.device("cpu")

VAE_DEVICE = setup_device()
PPO_DEVICE = torch.device("cpu")


# =====================================================================
# 1. SIMULATED ENVIRONMENT & WRAPPER
# =====================================================================

from crossy_gym_env import CrossyGymEnv


class FrameStackWrapper(gym.ObservationWrapper):
    """
    Gymnasium Observation Wrapper that stacks the last 4 grayscale images.
    """
    def __init__(self, env, stack_size=4):
        super(FrameStackWrapper, self).__init__(env)
        self.stack_size = stack_size
        self.frame_buffer = deque(maxlen=stack_size)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0.0, high=1.0, shape=(stack_size, 160, 160), dtype=np.float32),
            "scalars": env.observation_space["scalars"]
        })

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frame_buffer.clear()
        initial_frame = obs["image"]
        for _ in range(self.stack_size):
            self.frame_buffer.append(initial_frame)
        obs["image"] = np.array(self.frame_buffer, dtype=np.float32)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frame_buffer.append(obs["image"])
        obs["image"] = np.array(self.frame_buffer, dtype=np.float32)
        return obs, reward, terminated, truncated, info


# =====================================================================
# 2. VECTOR ROLLOUT MEMORY
# =====================================================================

class VectorRolloutBuffer:
    def __init__(self, num_envs, rollout_steps, ppo_device):
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self.device = ppo_device
        self.reset()

    def reset(self):
        self.states_latent = []
        self.states_scalar = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.terminals = []
        self.values = []
        self.action_masks = []

    def store(self, latents, scalars, actions, logprobs, rewards, terminals, values, masks):
        self.states_latent.append(latents.clone().detach().to(self.device))
        self.states_scalar.append(scalars.clone().detach().to(self.device))
        self.actions.append(actions.clone().detach().to(self.device))
        self.logprobs.append(logprobs.clone().detach().to(self.device))
        self.rewards.append(torch.tensor(rewards, dtype=torch.float32, device=self.device))
        self.terminals.append(torch.tensor(terminals, dtype=torch.float32, device=self.device))
        self.values.append(values.clone().detach().squeeze(-1).to(self.device))
        self.action_masks.append(masks.clone().detach().to(self.device))

    def compute_gae(self, next_values, gamma, gae_lambda):
        states_latent = torch.stack(self.states_latent)
        states_scalar = torch.stack(self.states_scalar)
        actions = torch.stack(self.actions)
        logprobs = torch.stack(self.logprobs)
        rewards = torch.stack(self.rewards)
        terminals = torch.stack(self.terminals)
        values = torch.stack(self.values)
        action_masks = torch.stack(self.action_masks)

        advantages = torch.zeros(self.rollout_steps, self.num_envs, device=self.device)
        last_gae_lam = 0.0

        for step in reversed(range(self.rollout_steps)):
            if step == self.rollout_steps - 1:
                next_non_terminal = 1.0 - terminals[step]
                next_val = next_values.squeeze(-1)
            else:
                next_non_terminal = 1.0 - terminals[step]
                next_val = values[step + 1]

            delta = rewards[step] + gamma * next_val * next_non_terminal - values[step]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam

        returns = advantages + values
        return (
            states_latent.view(-1, *states_latent.shape[2:]),
            states_scalar.view(-1, *states_scalar.shape[2:]),
            actions.view(-1),
            logprobs.view(-1),
            advantages.view(-1),
            returns.view(-1),
            action_masks.view(-1, *action_masks.shape[2:])
        )


# =====================================================================
# 3. POLICY ARCHITECTURE
# =====================================================================

def layer_init(layer, std=1.414, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# =====================================================================
# TRIED AND TESTED VAE ARCHITECTURE (ORIGINAL PARITY)
# =====================================================================

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs):
        # inputs shape: [Batch, Channels, Height, Width]
        flat_inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)

        # Spherical VQ (Cosine Similarity)
        flat_norm = F.normalize(flat_inputs, p=2, dim=1)
        weight_norm = F.normalize(self.embedding.weight, p=2, dim=1)

        # Distance is (1 - cosine_similarity)
        distances = 1.0 - torch.matmul(flat_norm, weight_norm.t())
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # Dead-code revival (only active during .train() mode)
        if self.training:
            usage_map = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
            usage_map.scatter_(1, encoding_indices, 1)
            usage_count = torch.sum(usage_map, dim=0)
            dead_codes = torch.nonzero(usage_count == 0).squeeze(-1)

            if dead_codes.numel() > 0:
                rand_indices = torch.randint(0, flat_inputs.shape[0], (dead_codes.numel(),), device=inputs.device)
                self.embedding.weight.data[dead_codes] = flat_inputs[rand_indices].detach()
                # Recompute distances after revival
                weight_norm = F.normalize(self.embedding.weight, p=2, dim=1)
                distances = 1.0 - torch.matmul(flat_norm, weight_norm.t())
                encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize the latents
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape[0], inputs.shape[2],
                                                                        inputs.shape[3], self.embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # Loss calculation
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        # Perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity


class SpatialVQVAE(nn.Module):
    def __init__(self):
        super(SpatialVQVAE, self).__init__()
        self.num_embeddings = 512
        self.embedding_dim = 64

        # ENCODER
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=4, stride=2, padding=1),  # 80x80
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 40x40
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 20x20
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 20x20
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.vq_c = VectorQuantizer(self.num_embeddings, self.embedding_dim)
        self.vq_t = VectorQuantizer(self.num_embeddings, self.embedding_dim)

        # DECODER
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.embedding_dim, 128, kernel_size=3, stride=1, padding=1),  # 20x20
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 40x40
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 80x80
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 160x160
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.encoder(x)
        z_c, z_t = torch.split(h, self.embedding_dim, dim=1)

        quantized_c, vq_loss_c, perplexity_c = self.vq_c(z_c)
        quantized_t, vq_loss_t, perplexity_t = self.vq_t(z_t)

        recon_static = self.decoder(quantized_c)
        pred_next = self.decoder(quantized_t)

        return recon_static, pred_next, vq_loss_c, vq_loss_t, perplexity_c, perplexity_t, quantized_c, quantized_t

class ActorCritic(nn.Module):
    def __init__(self, action_dim=4):
        super(ActorCritic, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(130, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        cnn_out_dim = 800
        scalar_dim = 2

        self.actor = nn.Sequential(
            layer_init(nn.Linear(cnn_out_dim + scalar_dim, HIDDEN_DIM)),
            nn.Tanh(),
            nn.Dropout(0.1),
            layer_init(nn.Linear(HIDDEN_DIM, HIDDEN_DIM)),
            nn.Tanh(),
            nn.Dropout(0.1),
            layer_init(nn.Linear(HIDDEN_DIM, action_dim), std=0.01)
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(cnn_out_dim + scalar_dim, HIDDEN_DIM)),
            nn.Tanh(),
            layer_init(nn.Linear(HIDDEN_DIM, HIDDEN_DIM)),
            nn.Tanh(),
            layer_init(nn.Linear(HIDDEN_DIM, 1), std=1.0)
        )

    def _get_features(self, latents, scalars):
        B, C, H, W = latents.shape
        y_coords = torch.linspace(-1, 1, H, device=latents.device).view(1, 1, H, 1).expand(B, 1, H, W)
        x_coords = torch.linspace(-1, 1, W, device=latents.device).view(1, 1, 1, W).expand(B, 1, H, W)
        latents_coord = torch.cat([latents, y_coords, x_coords], dim=1)
        cnn_out = self.cnn(latents_coord)
        return torch.cat([cnn_out, scalars], dim=1)

    def act(self, latents, scalars, action_mask):
        features = self._get_features(latents, scalars)
        action_logits = self.actor(features) + action_mask
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_value = self.critic(features)
        return action, action_logprob, state_value

    def evaluate(self, latents, scalars, action, action_mask):
        features = self._get_features(latents, scalars)
        action_logits = self.actor(features) + action_mask
        dist = Categorical(logits=action_logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(features)
        return action_logprobs, state_values, dist_entropy

# =====================================================================
# 5. VECTOR TRAINING PROCESS PIPELINE
# =====================================================================

def make_env():
    def _init():
        raw_env = CrossyGymEnv()
        return FrameStackWrapper(raw_env, stack_size=4)
    return _init

def main():
    os.makedirs("checkpoints", exist_ok=True)

    # Metrics
    total_timesteps = 0
    best_score = 0
    rolling_scores = deque(maxlen=100)
    rolling_rewards = deque(maxlen=100)
    avg_fps = 0.0
    start_time = time.time()

    # 1. Load Pretrained Spatial VQ-VAE Eyes
    print(f"[EYES] Loading frozen Spatial VQ-VAE onto {VAE_DEVICE}...")
    vae = SpatialVQVAE().to(VAE_DEVICE)
    vae_checkpoints = glob.glob(VAE_CHECKPOINT)
    if not vae_checkpoints:
        raise FileNotFoundError(f"[ERROR] VAE Checkpoint not found at: {VAE_CHECKPOINT}. Pretrain the VAE first.")
    
    vae.load_state_dict(torch.load(vae_checkpoints[0], map_location=VAE_DEVICE))
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    print("[EYES] Eyes loaded and frozen.")

    # 2. Setup Vectorized Environments
    print(f"[ENV] Spawning {NUM_ENVS} parallel environments...")
    envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(NUM_ENVS)])
    
    policy = ActorCritic(action_dim=4).to(PPO_DEVICE)
    policy_old = ActorCritic(action_dim=4).to(PPO_DEVICE)
    policy_old.load_state_dict(policy.state_dict())

    optimizer = torch.optim.Adam([
        {'params': policy.actor.parameters(), 'lr': LR_ACTOR},
        {'params': policy.cnn.parameters(), 'lr': LR_ACTOR},
        {'params': policy.critic.parameters(), 'lr': LR_CRITIC}
    ])
    mse_loss_fn = nn.MSELoss()
    buffer = VectorRolloutBuffer(NUM_ENVS, ROLLOUT_STEPS, PPO_DEVICE)

    # 3. Checkpoint Resuming logic
    start_update = 1
    step_checkpoints = glob.glob("checkpoints/ppo_sim_*_step.pth")
    if step_checkpoints:
        latest_cp = max(step_checkpoints, key=lambda x: int(x.split('_')[-2]))
        print(f"[RESUME] Loading active checkpoint: {latest_cp}")
        checkpoint = torch.load(latest_cp, map_location=PPO_DEVICE)

        policy.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        total_timesteps = checkpoint['total_timesteps']
        best_score = checkpoint['best_score']
        policy_old.load_state_dict(policy.state_dict())
        start_update = (total_timesteps // (NUM_ENVS * ROLLOUT_STEPS)) + 1
        print(f"[RESUME] Resuming at global step: {total_timesteps:,}")
    else:
        # Load the Target Model culmination checkpoint
        if os.path.exists(TARGET_MODEL):
            print(f"[BOOTSTRAP] Found pre-trained Grandmaster model at {TARGET_MODEL}. Loading parameters...")
            checkpoint = torch.load(TARGET_MODEL, map_location=PPO_DEVICE)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                policy.load_state_dict(checkpoint['model_state_dict'])
            else:
                policy.load_state_dict(checkpoint)
            policy_old.load_state_dict(policy.state_dict())
            print("[BOOTSTRAP] Warm start parameters loaded successfully (Actor & Critic synchronized).")
        else:
            print(f"[INIT] Target model {TARGET_MODEL} not found. Initiating fresh parameters.")

    obs, info = envs.reset()
    episode_rewards = np.zeros(NUM_ENVS)
    last_checkpoint_step = (total_timesteps // CHECKPOINT_INTERVAL) * CHECKPOINT_INTERVAL

    try:
        for update in range(start_update, 1000000):
            update_start = time.time()

            # --- VALUE FUNCTION WARMUP LOGIC ---
            # If we are in the warmup phase, freeze the Actor to protect
            # pre-trained weights while the Critic calibrates.
            is_warmup = update < (start_update + WARMUP_UPDATES)
            if is_warmup:
                for param in policy.actor.parameters():
                    param.requires_grad = False
                # Optional: Ensure CNN stays frozen if it's part of the pre-trained features
                for param in policy.cnn.parameters():
                    param.requires_grad = False
            else:
                for param in policy.actor.parameters():
                    param.requires_grad = True
                for param in policy.cnn.parameters():
                    param.requires_grad = True

            for _ in range(ROLLOUT_STEPS):
                total_timesteps += NUM_ENVS
                
                img_batch = torch.tensor(obs["image"], dtype=torch.float32, device=VAE_DEVICE)
                scalars_batch = torch.tensor(obs["scalars"], dtype=torch.float32, device=PPO_DEVICE)
                masks_batch = torch.tensor(info["action_mask"], dtype=torch.float32, device=PPO_DEVICE)
                
                with torch.no_grad():
                    _, _, _, _, _, _, quant_c, quant_t = vae(img_batch)
                    latents_batch = torch.cat([quant_c, quant_t], dim=1).to(PPO_DEVICE)
                    actions, logprobs, values = policy_old.act(latents_batch, scalars_batch, masks_batch)
                
                next_obs, rewards, terminated, truncated, next_info = envs.step(actions.cpu().numpy())
                
                episode_rewards += rewards
                dones = terminated | truncated
                
                for i in range(NUM_ENVS):
                    if dones[i]:
                        rolling_scores.append(info["score"][i])
                        rolling_rewards.append(episode_rewards[i])
                        if info["score"][i] > best_score:
                            best_score = info["score"][i]
                            print(f" >>> [NEW BEST] Score: {best_score}")
                            checkpoint_best = {
                                'model_state_dict': policy.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'total_timesteps': total_timesteps,
                                'best_score': best_score
                            }
                            torch.save(checkpoint_best, "checkpoints/ppo_sim_best.pth")
                        episode_rewards[i] = 0.0

                buffer.store(latents_batch, scalars_batch, actions, logprobs, rewards, dones, values, masks_batch)
                obs, info = next_obs, next_info
                
                if total_timesteps - last_checkpoint_step >= CHECKPOINT_INTERVAL:
                    target_checkpoint_step = (total_timesteps // CHECKPOINT_INTERVAL) * CHECKPOINT_INTERVAL
                    new_checkpoint_path = f"checkpoints/ppo_sim_{target_checkpoint_step}_step.pth"
                    for old_file in glob.glob("checkpoints/ppo_sim_*_step.pth"):
                        try: os.remove(old_file)
                        except: pass
                    checkpoint_state = {
                        'model_state_dict': policy.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'total_timesteps': total_timesteps,
                        'best_score': best_score
                    }
                    torch.save(checkpoint_state, new_checkpoint_path)
                    print(f"[*] Step Checkpoint: {new_checkpoint_path}")
                    last_checkpoint_step = target_checkpoint_step

            img_batch = torch.tensor(obs["image"], dtype=torch.float32, device=VAE_DEVICE)
            with torch.no_grad():
                _, _, _, _, _, _, quant_c, quant_t = vae(img_batch)
                next_lat_batch = torch.cat([quant_c, quant_t], dim=1).to(PPO_DEVICE)
                next_sca_batch = torch.tensor(obs["scalars"], dtype=torch.float32, device=PPO_DEVICE)
                next_features = policy_old._get_features(next_lat_batch, next_sca_batch)
                next_values = policy_old.critic(next_features)

            (fl, fs, fa, flog, fadv, fret, fm) = buffer.compute_gae(next_values, GAMMA, GAE_LAMBDA)
            fadv = (fadv - fadv.mean()) / (fadv.std() + 1e-8)

            batch_indices = np.arange(NUM_ENVS * ROLLOUT_STEPS)
            kl_break = False

            for epoch in range(K_EPOCHS):
                np.random.shuffle(batch_indices)
                epoch_kls = []

                for start in range(0, len(batch_indices), MINIBATCH_SIZE):
                    mb_idx = batch_indices[start: start + MINIBATCH_SIZE]
                    lps, svs, dent = policy.evaluate(fl[mb_idx], fs[mb_idx], fa[mb_idx], fm[mb_idx])

                    ratios = torch.exp(lps - flog[mb_idx])
                    surr1 = ratios * fadv[mb_idx]
                    surr2 = torch.clamp(ratios, 1.0 - EPS_CLIP, 1.0 + EPS_CLIP) * fadv[mb_idx]

                    loss_pi = -torch.min(surr1, surr2).mean()
                    loss_v = 0.5 * mse_loss_fn(svs.squeeze(-1), fret[mb_idx])

                    # Static Low Entropy: We want to preserve pathing, just micro-explore
                    loss_ent = ENTROPY_COEF * dent.mean()

                    loss = loss_pi + loss_v - loss_ent

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                    optimizer.step()

                    with torch.no_grad():
                        approx_kl = (flog[mb_idx] - lps).mean().item()
                        epoch_kls.append(approx_kl)

                avg_epoch_kl = np.mean(epoch_kls)
                # KL Divergence check: Only check if Actor is actually being updated (not Warmup)
                if not is_warmup and avg_epoch_kl > TARGET_KL:
                    print(
                        f"      [!] KL Overload ({avg_epoch_kl:.4f} > {TARGET_KL}). Early stopping epoch {epoch + 1}/{K_EPOCHS} to protect policy.")
                    kl_break = True
                    break

            policy_old.load_state_dict(policy.state_dict())
            buffer.reset()

            it_fps = (NUM_ENVS * ROLLOUT_STEPS) / (time.time() - update_start)
            avg_fps = it_fps if avg_fps == 0.0 else (0.9 * avg_fps) + (0.1 * it_fps)
            avg_score = sum(rolling_scores) / len(rolling_scores) if rolling_scores else 0

            warmup_status = "[WARMUP]" if is_warmup else ""
            print(
                f"Update: {update} {warmup_status} | Steps: {total_timesteps:,} | FPS: {avg_fps:.1f} | Avg Score: {avg_score:.2f} | Best: {best_score}")

    except KeyboardInterrupt:
        print("\n[!] Shutdown: Saving current state...")
        checkpoint_final = {
            'model_state_dict': policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'total_timesteps': total_timesteps,
            'best_score': best_score
        }
        torch.save(checkpoint_final, f"checkpoints/ppo_sim_{total_timesteps}_step.pth")
    finally:
        envs.close()

if __name__ == "__main__":
    main()