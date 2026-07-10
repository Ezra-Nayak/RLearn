# --- train_bc.py ---
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import concurrent.futures

from crossy_gym_env import CrossyGymEnv
from train_ppo_sim import FrameStackWrapper, SpatialVQVAE, ActorCritic, setup_device
from play_oracle import plan_best_action

# --- CONFIG ---
CHECKPOINT_DIR = "checkpoints"
TARGET_STEPS = 15000
BATCH_SIZE = 128
EPOCHS = 25  # Increased training epochs from 15 to 25
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5


class CrossyExpertDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.FloatTensor(s['latent']),
            torch.FloatTensor(s['scalars']),
            torch.FloatTensor(s['mask']),
            torch.tensor(s['action'], dtype=torch.long),
            torch.tensor(s['return'], dtype=torch.float32)
        )


def worker_collect_trajectory(worker_id):
    """
    Independent worker process that collects a single raw oracle trajectory.
    """
    raw_env = CrossyGymEnv()
    env = FrameStackWrapper(raw_env, stack_size=4)
    sim_env_for_planner = CrossyGymEnv()

    obs, info = env.reset()
    ep_data = []

    while raw_env.player_z < 100:
        action = plan_best_action(raw_env, lookahead_steps=12, sim_env=sim_env_for_planner)

        ep_data.append({
            'image': obs["image"],  # Shape: (4, 160, 160)
            'scalars': obs["scalars"],
            'mask': info["action_mask"],
            'action': action
        })

        obs, reward, terminated, truncated, info = env.step(action)
        ep_data[-1]['reward'] = reward

        if terminated or truncated:
            break

    env.close()

    # Calculate episodic returns locally
    g = 0.0
    for step in reversed(ep_data):
        g = step['reward'] + 0.99 * g
        step['return'] = g

    return ep_data


def gather_expert_samples(vae, vae_device, target_steps):
    print(f"[ORACLE] Spawning parallel workers to harvest {target_steps:,} transitions...")
    start_time = time.time()

    raw_samples = []
    num_workers = max(1, os.cpu_count() - 2)

    # PHASE 1: Parallel CPU Search
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        steps_submitted = 0

        # Over-provision slightly since exact episode length varies
        while steps_submitted < target_steps + 2000:
            futures.append(executor.submit(worker_collect_trajectory, len(futures)))
            steps_submitted += 100

        for future in concurrent.futures.as_completed(futures):
            ep_data = future.result()
            raw_samples.extend(ep_data)

            elapsed = time.time() - start_time
            fps = len(raw_samples) / elapsed
            print(f"\r[ORACLE] CPU Collection: {len(raw_samples):,}/{target_steps:,} raw steps | {fps:.1f} steps/sec",
                  end="")

            if len(raw_samples) >= target_steps:
                break

    raw_samples = raw_samples[:target_steps]
    print(f"\n[ORACLE] CPU Harvesting complete in {(time.time() - start_time) / 60:.2f} minutes.")

    # PHASE 2: Batched GPU Latent Encoding
    print(f"[EYES] Forward passing {len(raw_samples):,} visual frames through VAE on {vae_device}...")
    final_samples = []
    batch_size = 128

    for i in range(0, len(raw_samples), batch_size):
        batch = raw_samples[i:i + batch_size]

        # Push image stack to device: (B, 4, 160, 160)
        img_batch = torch.tensor(np.stack([s['image'] for s in batch]), dtype=torch.float32, device=vae_device)

        with torch.no_grad():
            _, _, _, _, _, _, quant_c, quant_t = vae(img_batch)
            latents_batch = torch.cat([quant_c, quant_t], dim=1).cpu().numpy()

        for j, s in enumerate(batch):
            final_samples.append({
                'latent': latents_batch[j],
                'scalars': s['scalars'],
                'mask': s['mask'],
                'action': s['action'],
                'return': s['return']
            })

        print(f"\r[EYES] Encoded {len(final_samples):,}/{target_steps:,} latents...", end="")

    print("\n[SYSTEM] GPU Encoding complete. Validating and normalizing dataset...")

    # Normalize the Returns to fix MSE Loss scaling (Crucial step to prevent CNN destruction)
    returns_arr = np.array([s['return'] for s in final_samples])
    mean_ret = returns_arr.mean()
    std_ret = returns_arr.std() + 1e-8

    for s in final_samples:
        s['return'] = (s['return'] - mean_ret) / std_ret

    return final_samples


def train_behavioral_cloning():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    vae_device = setup_device()

    vae_path = "checkpoints/sim_vae_best.pth"
    if not os.path.exists(vae_path):
        raise FileNotFoundError(f"[ERROR] VAE checkpoint missing from {vae_path}. Pretrain the VAE first.")

    print(f"[EYES] Loading pretrained VAE onto {vae_device}...")
    vae = SpatialVQVAE().to(vae_device)
    vae.load_state_dict(torch.load(vae_path, map_location=vae_device))
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    # Gather via parallel pool
    samples = gather_expert_samples(vae, vae_device, TARGET_STEPS)
    full_dataset = CrossyExpertDataset(samples)

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cpu")
    print(f"[MODEL] Initializing Joint Actor-Critic Network on {device}...")
    model = ActorCritic(action_dim=4).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    actor_criterion = nn.CrossEntropyLoss()
    critic_criterion = nn.MSELoss()

    best_val_loss = float('inf')

    print(f"[TRAIN] Initiating Behavioral Cloning for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for latents, scalars, masks, actions, returns in train_loader:
            latents = latents.to(device)
            scalars = scalars.to(device)
            masks = masks.to(device)
            actions = actions.to(device)
            returns = returns.to(device)

            optimizer.zero_grad()

            features = model._get_features(latents, scalars)
            action_logits = model.actor(features)

            # CRITICAL SAFETY: Detach CNN features before Critic so massive value variance
            # does not backpropagate and ruin the BC representation learned by the Actor.
            state_values = model.critic(features.detach()).squeeze(-1)

            # FORCE 1D SHAPES TO PREVENT SILENT BROADCASTING
            state_values = state_values.reshape(-1)
            returns = returns.reshape(-1)

            # One-time first batch shape/normalization diagnostic verification
            if epoch == 1 and total_train == 0:
                print(f"\n[DEBUG] Epoch 1 First Batch Diagnostics:")
                print(
                    f"  * state_values: shape={state_values.shape}, mean={state_values.mean().item():.4f}, std={state_values.std().item():.4f}")
                print(
                    f"  * returns:      shape={returns.shape}, mean={returns.mean().item():.4f}, std={returns.std().item():.4f}")
                print(f"  * sample returns: {returns[:5].tolist()}")

            masked_logits = action_logits + masks

            loss_actor = actor_criterion(masked_logits, actions)
            loss_critic = critic_criterion(state_values, returns)
            loss = loss_actor + 0.5 * loss_critic

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(masked_logits, 1)
            total_train += actions.size(0)
            correct_train += (predicted == actions).sum().item()

        train_acc = 100.0 * correct_train / total_train
        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for latents, scalars, masks, actions, returns in val_loader:
                latents, scalars, masks, actions, returns = (
                    latents.to(device), scalars.to(device), masks.to(device), actions.to(device), returns.to(device)
                )
                features = model._get_features(latents, scalars)
                masked_logits = model.actor(features) + masks
                state_values = model.critic(features).squeeze(-1)

                # FORCE 1D SHAPES TO PREVENT SILENT BROADCASTING
                state_values = state_values.reshape(-1)
                returns = returns.reshape(-1)

                loss_actor = actor_criterion(masked_logits, actions)
                loss_critic = critic_criterion(state_values, returns)
                loss = loss_actor + 0.5 * loss_critic
                val_loss += loss.item()

                _, predicted = torch.max(masked_logits, 1)
                total_val += actions.size(0)
                correct_val += (predicted == actions).sum().item()

        val_acc = 100.0 * correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.1f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(CHECKPOINT_DIR, "ppo_sim_bc.pth")
            torch.save(model.state_dict(), save_path)

    print("\n[SUCCESS] Joint Policy-Value Behavioral Cloning Complete.")
    print("Execute train_ppo_sim.py to run standard vectorized PPO with these pre-trained checkpoints.")


if __name__ == "__main__":
    train_behavioral_cloning()