# --- train_redemption.py ---
import os
import time
import copy
import random
import numpy as np
import concurrent.futures

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from crossy_gym_env import CrossyGymEnv
from train_ppo_sim import FrameStackWrapper, SpatialVQVAE, ActorCritic, setup_device
from play_oracle import plan_best_action

# --- CONFIGURATION ---
DATA_DIR = r"D:\python\RLearn\sim_data"
INPUT_CHECKPOINT = "checkpoints/ppo_redemption_latest.pth"
REDEMPTION_CHECKPOINT = "checkpoints/ppo_redemption_patience.pth"

REDEMPTIONS_PER_LOOP = 2000  # How many steps of "alternate timeline survival" to gather
BC_MIX_RATIO = 10000  # How many baseline BC samples to mix in to prevent forgetting
BATCH_SIZE = 128
EPOCHS_PER_LOOP = 3
LEARNING_RATE = 2e-5  # Micro LR for gentle, continuous correction


# =====================================================================
# PATIENCE CURRICULUM ENVIRONMENT
# =====================================================================

class PatienceCurriculumEnv(CrossyGymEnv):
    """
    Overrides generation to force Patience Checkpoints (Traffic Jams).
    Mixes normal levels with forced idle scenarios so the Oracle can
    demonstrate waiting, which is then appended to the BC dataset.
    """
    def _generate_chunk(self):
        mode = random.random()
        start_z = self.highest_generated_z + 1
        end_z = start_z + self.chunk_size - 1

        if mode < 0.50:
            super()._generate_chunk()
            return
        else:
            local_terrain = {}
            local_obstacles = {}
            local_road_params = {}

            for z in range(start_z, start_z + 2):
                local_terrain[z] = 'G'

            choke_z = start_z + 2
            local_terrain[choke_z] = 'G'
            safe_col = random.randint(-1, 1)
            for x in range(self.GRID_MIN_X, self.GRID_MAX_X + 1):
                if x != safe_col:
                    local_obstacles[(x, choke_z)] = True

            road_z = start_z + 3
            local_terrain[road_z] = 'R'
            local_road_params[road_z] = {'speed': random.uniform(2.5, 4.0), 'direction': random.choice([-1, 1])}

            self.terrain_map.update(local_terrain)
            self.obstacle_map.update(local_obstacles)
            self.road_parameters.update(local_road_params)

            self.active_cars[road_z] = []
            last_x = -7.0
            for _ in range(5):
                spawn_x = last_x + random.uniform(2.5, 3.5)
                if spawn_x < 7.0:
                    self.active_cars[road_z].append(spawn_x)
                    last_x = spawn_x

            for z in range(road_z + 1, end_z + 1):
                local_terrain[z] = 'G'
                for x in range(self.GRID_MIN_X, self.GRID_MAX_X + 1):
                    if random.random() < 0.1:
                        local_obstacles[(x, z)] = True

            self.highest_generated_z = end_z

    def step(self, action):
        obs, rew, done, trunc, info = super().step(action)
        self.camera_speed = 0.5  # Forgiving camera to guarantee Idling is physically viable
        return obs, rew, done, trunc, info

# =====================================================================
# STATE MANAGEMENT UTILITIES
# =====================================================================

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


# =====================================================================
# DATASET & WORKER LOGIC
# =====================================================================

class RedemptionDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.FloatTensor(s['latent']),
            torch.FloatTensor(s['scalars']),
            torch.FloatTensor(s['mask']),
            torch.tensor(s['action'], dtype=torch.long),
            torch.tensor(s['return'], dtype=torch.float32)
        )


def worker_gather_redemption(worker_id):
    """
    Worker plays as the agent until death. Upon death, it rewinds 12 steps,
    spawns the Oracle to navigate the trap successfully, and returns the
    alternate surviving trajectory.
    """
    vae = SpatialVQVAE()
    vae.load_state_dict(torch.load("checkpoints/sim_vae_best.pth", map_location="cpu", weights_only=False))
    vae.eval()

    policy = ActorCritic(action_dim=4)
    checkpoint_to_load = REDEMPTION_CHECKPOINT if os.path.exists(REDEMPTION_CHECKPOINT) else INPUT_CHECKPOINT

    checkpoint = torch.load(checkpoint_to_load, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['model_state_dict'])
    else:
        policy.load_state_dict(checkpoint)
    policy.eval()

    # Inject the modified Curriculum Environment here to force the traffic jam traps
    raw_env = PatienceCurriculumEnv()
    env = FrameStackWrapper(raw_env, stack_size=4)
    planner_env = PatienceCurriculumEnv()

    obs, info = env.reset()
    state_history = []

    # 1. Agent plays until it makes a fatal mistake
    done = False
    while not done:
        # Save exact physical state and motion blur
        state_history.append({
            'env_state': get_env_state_dict(raw_env),
            'frame_stack': copy.deepcopy(env.frame_buffer),
            'scalars': obs["scalars"],
            'mask': info["action_mask"]
        })

        img_tensor = torch.tensor(obs["image"], dtype=torch.float32).unsqueeze(0)
        sca_tensor = torch.tensor(obs["scalars"], dtype=torch.float32).unsqueeze(0)
        msk_tensor = torch.tensor(info["action_mask"], dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            _, _, _, _, _, _, qc, qt = vae(img_tensor)
            lats = torch.cat([qc, qt], dim=1)
            feats = policy._get_features(lats, sca_tensor)
            logits = policy.actor(feats) + msk_tensor
            agent_action = logits.argmax(dim=-1).item()

        obs, reward, terminated, truncated, info = env.step(agent_action)
        done = terminated or truncated

        # Stop early if the agent is surviving perfectly to force a reset and find edge cases
        if raw_env.player_z > 80:
            return []

            # 2. Death Detected. Time Travel 12 steps backward (or to start).
    rewind_steps = min(12, len(state_history) - 1)
    if rewind_steps < 2: return []  # Too short to learn anything

    history_point = state_history[-rewind_steps]
    restore_env_state(raw_env, history_point['env_state'])
    env.frame_buffer = copy.deepcopy(history_point['frame_stack'])

    redemption_traj = []

    # 3. Oracle plays forward to demonstrate survival
    for _ in range(rewind_steps + 4):  # Play slightly past the death point
        obs_img = np.array(env.frame_buffer, dtype=np.float32)
        norm_x = raw_env.player_x / 5.0
        threat = max(0.0, min(1.0, 1.0 - ((raw_env.player_z - raw_env.camera_z) / 6.0)))
        scalars = np.array([norm_x, threat], dtype=np.float32)
        mask = raw_env._get_action_mask()

        oracle_action = plan_best_action(raw_env, lookahead_steps=12, sim_env=planner_env)

        redemption_traj.append({
            'image': obs_img,
            'scalars': scalars,
            'mask': mask,
            'action': oracle_action,
            'reward': 1.0,  # Positive reward for surviving this step
            'return': 0.0
        })

        obs, reward, term, trunc, info = env.step(oracle_action)
        if term or trunc: break  # Even the Oracle failed (impossible trap), discard

    env.close()

    # Backcalculate returns
    g = 0.0
    for step in reversed(redemption_traj):
        g = step['reward'] + 0.99 * g
        step['return'] = g

    return redemption_traj


# =====================================================================
# ENDLESS REDEMPTION LOOP
# =====================================================================

def main():
    print("=====================================================")
    print("      ENDLESS HINDSIGHT REDEMPTION TRAINING          ")
    print("=====================================================")

    vae_device = setup_device()
    train_device = torch.device("cpu")

    vae = SpatialVQVAE().to(vae_device)
    vae.load_state_dict(torch.load("checkpoints/sim_vae_best.pth", map_location=vae_device, weights_only=False))
    vae.eval()
    for p in vae.parameters(): p.requires_grad = False

    policy = ActorCritic(action_dim=4).to(train_device)
    if not os.path.exists(REDEMPTION_CHECKPOINT):
        print(f"[INIT] Saving starting anchor point to {REDEMPTION_CHECKPOINT}")
        checkpoint = torch.load(INPUT_CHECKPOINT, map_location=train_device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            policy.load_state_dict(checkpoint['model_state_dict'])
        else:
            policy.load_state_dict(checkpoint)
        torch.save(policy.state_dict(), REDEMPTION_CHECKPOINT)

    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    actor_criterion = nn.CrossEntropyLoss()
    critic_criterion = nn.MSELoss()

    # Load BC Foundation (To prevent catastrophic forgetting of normal traffic)
    bc_data_path = os.path.join(DATA_DIR, "bc_dataset.pt")
    if not os.path.exists(bc_data_path):
        raise FileNotFoundError(f"[FATAL] Cannot find {bc_data_path}. Foundation required.")

    print("[SYSTEM] Loading permanent BC Foundation Dataset...")
    full_bc_samples = torch.load(bc_data_path, map_location='cpu')

    loop_count = 1
    num_workers = max(1, os.cpu_count() - 2)

    while True:
        print(f"\n--- REDEMPTION CYCLE {loop_count} ---")
        start_time = time.time()

        raw_redemptions = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            active_futures = {executor.submit(worker_gather_redemption, i) for i in range(num_workers * 2)}

            while len(raw_redemptions) < REDEMPTIONS_PER_LOOP:
                done, active_futures = concurrent.futures.wait(active_futures,
                                                               return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    try:
                        traj = future.result()
                        raw_redemptions.extend(traj)
                    except Exception as e:
                        print(f" Worker Error: {e}")
                    active_futures.add(executor.submit(worker_gather_redemption, 0))

                print(f"\r[GATHER] Collected {len(raw_redemptions)}/{REDEMPTIONS_PER_LOOP} Oracle Corrections...",
                      end="")

            for f in active_futures: f.cancel()

        raw_redemptions = raw_redemptions[:REDEMPTIONS_PER_LOOP]
        print(f"\n[ENCODE] Pushing {len(raw_redemptions)} alternate-timeline frames through VAE GPU...")

        encoded_redemptions = []
        batch_size = 128
        for i in range(0, len(raw_redemptions), batch_size):
            batch = raw_redemptions[i:i + batch_size]
            img_batch = torch.tensor(np.stack([s['image'] for s in batch]), dtype=torch.float32, device=vae_device)
            with torch.no_grad():
                _, _, _, _, _, _, qc, qt = vae(img_batch)
                lats = torch.cat([qc, qt], dim=1).cpu().numpy()

            for j, s in enumerate(batch):
                encoded_redemptions.append({
                    'latent': lats[j],
                    'scalars': s['scalars'],
                    'mask': s['mask'],
                    'action': s['action'],
                    'return': s['return']
                })

        # Sub-sample BC Foundation to mix
        random.shuffle(full_bc_samples)
        bc_mix = full_bc_samples[:BC_MIX_RATIO]

        # Merge and normalize
        combined_dataset = bc_mix + encoded_redemptions
        random.shuffle(combined_dataset)

        returns_arr = np.array([s['return'] for s in combined_dataset])
        mean_ret, std_ret = returns_arr.mean(), returns_arr.std() + 1e-8
        for s in combined_dataset: s['return'] = (s['return'] - mean_ret) / std_ret

        loader = DataLoader(RedemptionDataset(combined_dataset), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        print(
            f"[TRAIN] Fine-tuning on {len(combined_dataset)} samples ({BC_MIX_RATIO} Base + {len(encoded_redemptions)} Redemptions)")

        policy.load_state_dict(torch.load(REDEMPTION_CHECKPOINT, map_location=train_device, weights_only=False))
        policy.train()

        for epoch in range(1, EPOCHS_PER_LOOP + 1):
            total_loss = 0.0
            correct = 0
            total = 0
            for lats, scas, msks, acts, rets in loader:
                lats, scas, msks, acts, rets = lats.to(train_device), scas.to(train_device), msks.to(
                    train_device), acts.to(train_device), rets.to(train_device)

                optimizer.zero_grad()
                feats = policy._get_features(lats, scas)
                logits = policy.actor(feats) + msks
                vals = policy.critic(feats.detach()).squeeze(-1)

                loss_actor = actor_criterion(logits, acts)
                loss_critic = critic_criterion(vals, rets)
                loss = loss_actor + 0.5 * loss_critic

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, preds = torch.max(logits, 1)
                total += acts.size(0)
                correct += (preds == acts).sum().item()

            print(
                f"  Epoch {epoch}/{EPOCHS_PER_LOOP} | Loss: {total_loss / len(loader):.4f} | Accuracy: {100. * correct / total:.1f}%")

        torch.save(policy.state_dict(), REDEMPTION_CHECKPOINT)
        print(f"[SUCCESS] Cycle {loop_count} Complete in {time.time() - start_time:.1f}s. Checkpoint updated.")
        loop_count += 1


if __name__ == "__main__":
    main()