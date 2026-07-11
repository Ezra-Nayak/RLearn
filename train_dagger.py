# --- train_dagger.py ---
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from crossy_gym_env import CrossyGymEnv
from train_ppo_sim import FrameStackWrapper, SpatialVQVAE, ActorCritic, setup_device
from play_oracle import plan_best_action, sim_env_step_physics_only

# --- CONFIG ---
CHECKPOINT_DIR = "checkpoints"
INPUT_CHECKPOINT = "checkpoints/ppo_sim_9430000_step.pth"
OUTPUT_CHECKPOINT = "checkpoints/ppo_sim_dagger_aligned.pth"
TARGET_SAMPLES = 8000
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 3e-5  # Low learning rate to prevent destroying existing PPO progress
WEIGHT_DECAY = 1e-5


class CrossyDaggerDataset(Dataset):
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
            torch.FloatTensor(s['fatal_mask']),
            torch.tensor(s['action'], dtype=torch.long),
            torch.tensor(s['return'], dtype=torch.float32)
        )


def get_env_state_dict(e):
    return {
        'player_x': e.player_x,
        'player_z': e.player_z,
        'camera_z': e.camera_z,
        'camera_speed': e.camera_speed,
        'highest_generated_z': e.highest_generated_z,
        'terrain_map': e.terrain_map.copy(),
        'obstacle_map': e.obstacle_map.copy(),
        'road_parameters': e.road_parameters.copy(),
        'active_cars': {z: list(cars) for z, cars in e.active_cars.items()}
    }


def restore_env_state(e, s):
    e.player_x = s['player_x']
    e.player_z = s['player_z']
    e.camera_z = s['camera_z']
    e.camera_speed = s['camera_speed']
    e.highest_generated_z = s['highest_generated_z']
    e.terrain_map = s['terrain_map'].copy()
    e.obstacle_map = s['obstacle_map'].copy()
    e.road_parameters = s['road_parameters'].copy()
    e.active_cars = {z: list(cars) for z, cars in s['active_cars'].items()}


def collect_dagger_samples(vae, policy, vae_device, ppo_device, target_samples):
    print(f"[DAGGER] Collecting {target_samples} targeted correction samples...")
    raw_env = CrossyGymEnv()
    env = FrameStackWrapper(raw_env, stack_size=4)
    sim_env_for_planner = CrossyGymEnv()

    samples = []
    error_count = 0
    total_steps = 0

    obs, info = env.reset()
    trajectory_buffer = []

    while len(samples) < target_samples:
        img_tensor = torch.tensor(obs["image"], dtype=torch.float32, device=vae_device).unsqueeze(0)
        scalars_tensor = torch.tensor(obs["scalars"], dtype=torch.float32, device=ppo_device).unsqueeze(0)
        masks_tensor = torch.tensor(info["action_mask"], dtype=torch.float32, device=ppo_device).unsqueeze(0)

        # Get PPO model's current spatial representations
        with torch.no_grad():
            _, _, _, _, _, _, quant_c, quant_t = vae(img_tensor)
            latents_tensor = torch.cat([quant_c, quant_t], dim=1).to(ppo_device)

            features = policy._get_features(latents_tensor, scalars_tensor)
            action_logits = policy.actor(features) + masks_tensor
            ppo_action = action_logits.argmax(dim=-1).item()

        # Query Oracle expert for correct survival action
        oracle_action = plan_best_action(raw_env, lookahead_steps=12, sim_env=sim_env_for_planner)

        # Calculate immediate fatal moves
        fatal_mask = np.zeros(4, dtype=np.float32)
        start_state = get_env_state_dict(raw_env)
        for a in [0, 1, 2, 3]:
            if a == 1 and raw_env.player_x <= raw_env.GRID_MIN_X:
                fatal_mask[a] = 1.0
                continue
            elif a == 2 and raw_env.player_x >= raw_env.GRID_MAX_X:
                fatal_mask[a] = 1.0
                continue

            rand_state = random.getstate()
            restore_env_state(sim_env_for_planner, start_state)
            done = sim_env_step_physics_only(sim_env_for_planner, a)
            random.setstate(rand_state)
            if done:
                fatal_mask[a] = 1.0

        is_error = (ppo_action != oracle_action)
        if is_error:
            error_count += 1

        # Keep 100% of error states, but keep only 20% of correct traversals to prevent bias
        if is_error or (random.random() < 0.20):
            trajectory_buffer.append({
                'latent': latents_tensor.squeeze(0).cpu().numpy(),
                'scalars': obs["scalars"],
                'mask': info["action_mask"],
                'fatal_mask': fatal_mask,
                'action': oracle_action,
                'reward': 0.0,  # Computed at terminal
                'return': 0.0
            })

        # Execute PPO's action to let it experience its own suboptimal pathing
        obs, reward, terminated, truncated, info = env.step(ppo_action)
        total_steps += 1

        if len(trajectory_buffer) > 0:
            trajectory_buffer[-1]['reward'] = reward

        if terminated or truncated:
            # Backcalculate rewards for the collected steps in this trajectory
            g = 0.0
            for step in reversed(trajectory_buffer):
                g = step['reward'] + 0.99 * g
                step['return'] = g
                samples.append(step)

            trajectory_buffer = []
            obs, info = env.reset()

            print(
                f"\r[DAGGER] Progress: {len(samples)}/{target_samples} samples | captured {error_count} model deviations...",
                end="")

    env.close()
    print(f"\n[DAGGER] Collection complete. Disagreement rate: {100.0 * error_count / total_steps:.1f}%")

    # Normalize returns for value function stability
    returns_arr = np.array([s['return'] for s in samples])
    mean_ret = returns_arr.mean()
    std_ret = returns_arr.std() + 1e-8
    for s in samples:
        s['return'] = (s['return'] - mean_ret) / std_ret

    return samples[:target_samples]


def train_dagger():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    vae_device = setup_device()

    vae_path = "checkpoints/sim_vae_best.pth"
    if not os.path.exists(vae_path):
        raise FileNotFoundError("[ERROR] Pretrained VAE checkpoint missing.")

    print(f"[EYES] Loading pretrained VAE onto {vae_device}...")
    vae = SpatialVQVAE().to(vae_device)
    vae.load_state_dict(torch.load(vae_path, map_location=vae_device))
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    device = torch.device("cpu")
    print(f"[MODEL] Loading PPO Checkpoint for alignment fine-tuning...")
    model = ActorCritic(action_dim=4).to(device)

    checkpoint = torch.load(INPUT_CHECKPOINT, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Harvest targeted training pairs
    samples = collect_dagger_samples(vae, model, vae_device, device, TARGET_SAMPLES)
    dataset = CrossyDaggerDataset(samples)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    actor_criterion = nn.CrossEntropyLoss()
    critic_criterion = nn.MSELoss()

    print(f"[TRAIN] Aligning model weights via DAgger for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0

        for latents, scalars, masks, fatal_masks, actions, returns in train_loader:
            latents, scalars, masks, fatal_masks, actions, returns = (
                latents.to(device), scalars.to(device), masks.to(device),
                fatal_masks.to(device), actions.to(device), returns.to(device)
            )

            optimizer.zero_grad()
            features = model._get_features(latents, scalars)
            action_logits = model.actor(features)
            state_values = model.critic(features.detach()).squeeze(-1)

            state_values = state_values.reshape(-1)
            returns = returns.reshape(-1)

            masked_logits = action_logits + masks

            loss_actor = actor_criterion(masked_logits, actions)
            loss_critic = critic_criterion(state_values, returns)

            action_probs = torch.softmax(action_logits, dim=-1)
            fatal_penalty = torch.mean(torch.sum(action_probs * fatal_masks, dim=-1))

            loss = loss_actor + 0.5 * loss_critic + 2.0 * fatal_penalty

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
            for latents, scalars, masks, fatal_masks, actions, returns in val_loader:
                latents, scalars, masks, fatal_masks, actions, returns = (
                    latents.to(device), scalars.to(device), masks.to(device),
                    fatal_masks.to(device), actions.to(device), returns.to(device)
                )
                features = model._get_features(latents, scalars)
                action_logits = model.actor(features)
                masked_logits = action_logits + masks
                state_values = model.critic(features).squeeze(-1)

                state_values = state_values.reshape(-1)
                returns = returns.reshape(-1)

                loss_actor = actor_criterion(masked_logits, actions)
                loss_critic = critic_criterion(state_values, returns)

                action_probs = torch.softmax(action_logits, dim=-1)
                fatal_penalty = torch.mean(torch.sum(action_probs * fatal_masks, dim=-1))

                loss = loss_actor + 0.5 * loss_critic + 2.0 * fatal_penalty
                val_loss += loss.item()

                _, predicted = torch.max(masked_logits, 1)
                total_val += actions.size(0)
                correct_val += (predicted == actions).sum().item()

        val_acc = 100.0 * correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.1f}%")

    # Save aligned weights
    torch.save(model.state_dict(), OUTPUT_CHECKPOINT)
    print(f"[SUCCESS] Aligned model weights saved to: '{OUTPUT_CHECKPOINT}'")


if __name__ == "__main__":
    train_dagger()