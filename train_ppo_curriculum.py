# --- train_ppo_curriculum.py ---
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

# Import components from your existing files
from crossy_gym_env import CrossyGymEnv
from train_ppo_sim import (
    FrameStackWrapper, VectorRolloutBuffer, SpatialVQVAE, ActorCritic,
    setup_device, layer_init
)

# --- HYPERPARAMETERS ---
LR_ACTOR = 1.5e-5  # Slightly higher to learn new spatial concepts
LR_CRITIC = 1e-4
TARGET_KL = 0.020  # Slightly looser to allow for unlearning bad habits
ENTROPY_COEF = 0.025  # INCREASED: Forces the agent to try sideways moves when stuck
GAMMA = 0.99
GAE_LAMBDA = 0.95
EPS_CLIP = 0.15
K_EPOCHS = 4
NUM_ENVS = 48
ROLLOUT_STEPS = 128
MINIBATCH_SIZE = 64
CHECKPOINT_INTERVAL = 10000

# Targeted Model to Fine-Tune
TARGET_MODEL = "checkpoints/ppo_sim_3100000_step.pth"

VAE_DEVICE = setup_device()
PPO_DEVICE = torch.device("cpu")


# =====================================================================
# 1. CURRICULUM ENVIRONMENT WRAPPER
# =====================================================================

class CurriculumCrossyEnv(CrossyGymEnv):
    """
    A modified environment that forces the agent to experience dense grass
    and jailbreaks frequently to train lateral spatial planning.
    """

    def _generate_chunk(self):
        mode = random.random()
        start_z = self.highest_generated_z + 1
        end_z = start_z + self.chunk_size - 1

        if mode < 0.40:
            # 40% Normal Generation (To maintain traffic dodging skills)
            super()._generate_chunk()
            return

        elif mode < 0.70:
            # 30% Dense Grass Maze Curriculum
            attempts = 0
            while attempts < 100:
                local_terrain = {}
                local_obstacles = {}
                for z in range(start_z, end_z + 1):
                    local_terrain[z] = 'G'
                    if z < start_z + 2: continue  # Safe buffer
                    for x in range(self.GRID_MIN_X, self.GRID_MAX_X + 1):
                        # 28% obstacle density (almost double normal)
                        if random.random() < 0.28:
                            local_obstacles[(x, z)] = True

                if self._verify_connectivity(start_z, end_z, local_obstacles, local_terrain):
                    self.terrain_map.update(local_terrain)
                    self.obstacle_map.update(local_obstacles)
                    self.highest_generated_z = end_z
                    return
                attempts += 1

        else:
            # 30% Jailbreak Trap Curriculum
            local_terrain = {}
            local_obstacles = {}

            for z in range(start_z, end_z + 1):
                local_terrain[z] = 'G'

            # Build the immediate trap
            trap_z = start_z + 1
            local_obstacles[(0, trap_z)] = True  # Block center

            # Block one side completely to force movement to the other
            if random.random() < 0.5:
                local_obstacles[(-1, trap_z - 1)] = True
                local_obstacles[(-1, trap_z)] = True
            else:
                local_obstacles[(1, trap_z - 1)] = True
                local_obstacles[(1, trap_z)] = True

            # Populate the rest normally
            for z in range(trap_z + 2, end_z + 1):
                for x in range(self.GRID_MIN_X, self.GRID_MAX_X + 1):
                    if random.random() < 0.15:
                        local_obstacles[(x, z)] = True

            # For Jailbreak, we don't strictly verify connectivity because we want to
            # teach it localized escapes. If it's a dead end later, that's fine.
            self.terrain_map.update(local_terrain)
            self.obstacle_map.update(local_obstacles)
            self.highest_generated_z = end_z

    def step(self, action):
        # Override camera speed to be a flat, forgiving 1.0 during spatial learning.
        # This gives the agent TIME to figure out the maze without the camera killing it instantly.
        obs, rew, done, trunc, info = super().step(action)
        self.camera_speed = 1.0
        return obs, rew, done, trunc, info


def make_curriculum_env():
    def _init():
        raw_env = CurriculumCrossyEnv()
        return FrameStackWrapper(raw_env, stack_size=4)

    return _init


# =====================================================================
# 2. MAIN TRAINING LOOP
# =====================================================================

def main():
    print("[SYSTEM] Booting Targeted Curriculum Training (Obstacles & Jailbreaks)...")

    vae = SpatialVQVAE().to(VAE_DEVICE)
    vae.load_state_dict(torch.load("checkpoints/sim_vae_best.pth", map_location=VAE_DEVICE, weights_only=False))
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    print(f"[ENV] Spawning {NUM_ENVS} Curriculum Environments...")
    envs = gym.vector.AsyncVectorEnv([make_curriculum_env() for _ in range(NUM_ENVS)])

    policy = ActorCritic(action_dim=4).to(PPO_DEVICE)
    policy_old = ActorCritic(action_dim=4).to(PPO_DEVICE)

    print(f"[MODEL] Loading Baseline Agent: {TARGET_MODEL}")
    checkpoint = torch.load(TARGET_MODEL, map_location=PPO_DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['model_state_dict'])
    else:
        policy.load_state_dict(checkpoint)
    policy_old.load_state_dict(policy.state_dict())

    optimizer = torch.optim.Adam([
        {'params': policy.actor.parameters(), 'lr': LR_ACTOR},
        {'params': policy.cnn.parameters(), 'lr': LR_ACTOR},
        {'params': policy.critic.parameters(), 'lr': LR_CRITIC}
    ])

    mse_loss_fn = nn.MSELoss()
    buffer = VectorRolloutBuffer(NUM_ENVS, ROLLOUT_STEPS, PPO_DEVICE)

    total_timesteps = 0
    best_score = 0
    rolling_scores = deque(maxlen=100)
    avg_fps = 0.0
    start_time = time.time()
    last_checkpoint_step = 0

    obs, info = envs.reset()

    try:
        for update in range(1, 1000000):
            update_start = time.time()

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
                dones = terminated | truncated

                for i in range(NUM_ENVS):
                    if dones[i]:
                        rolling_scores.append(info["score"][i])
                        if info["score"][i] > best_score:
                            best_score = info["score"][i]

                buffer.store(latents_batch, scalars_batch, actions, logprobs, rewards, dones, values, masks_batch)
                obs, info = next_obs, next_info

                if total_timesteps - last_checkpoint_step >= CHECKPOINT_INTERVAL:
                    target_checkpoint_step = (total_timesteps // CHECKPOINT_INTERVAL) * CHECKPOINT_INTERVAL
                    new_checkpoint_path = f"checkpoints/ppo_curriculum_{target_checkpoint_step}_step.pth"

                    for old_file in glob.glob("checkpoints/ppo_curriculum_*_step.pth"):
                        try:
                            os.remove(old_file)
                        except:
                            pass

                    torch.save(policy.state_dict(), new_checkpoint_path)
                    print(f"[*] Curriculum Checkpoint: {new_checkpoint_path}")
                    last_checkpoint_step = target_checkpoint_step

            # Compute GAE
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

                    # Entropy bonus encourages lateral exploration in rock traps
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
                if avg_epoch_kl > TARGET_KL:
                    print(f"      [!] KL Break ({avg_epoch_kl:.4f}). Protected policy.")
                    kl_break = True
                    break

            policy_old.load_state_dict(policy.state_dict())
            buffer.reset()

            it_fps = (NUM_ENVS * ROLLOUT_STEPS) / (time.time() - update_start)
            avg_fps = it_fps if avg_fps == 0.0 else (0.9 * avg_fps) + (0.1 * it_fps)
            avg_score = sum(rolling_scores) / len(rolling_scores) if rolling_scores else 0

            print(
                f"Update: {update} | Steps: {total_timesteps:,} | FPS: {avg_fps:.1f} | Avg Score: {avg_score:.2f} | Best: {best_score}")

    except KeyboardInterrupt:
        print("\n[!] Shutdown: Saving current state...")
        torch.save(policy.state_dict(), f"checkpoints/ppo_curriculum_{total_timesteps}_step.pth")
    finally:
        envs.close()


if __name__ == "__main__":
    main()