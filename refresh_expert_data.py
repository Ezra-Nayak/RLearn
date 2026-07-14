# --- refresh_expert_data.py ---
import os
import time
import numpy as np
import torch
import concurrent.futures

from crossy_gym_env import CrossyGymEnv
from train_ppo_sim import FrameStackWrapper, SpatialVQVAE, setup_device
from play_oracle import plan_best_action

# --- CONFIGURATION ---
DATA_DIR = r"D:\python\RLearn\sim_data"
TARGET_STEPS = 20000  # Generate a large, fresh pool of expert knowledge
VAE_CHECKPOINT = "checkpoints/sim_vae_best.pth"
BATCH_SIZE = 128


def worker_collect_expert(worker_id):
    """ Runs a single expert episode using the Oracle """
    raw_env = CrossyGymEnv()
    env = FrameStackWrapper(raw_env, stack_size=4)
    planner_env = CrossyGymEnv()

    obs, info = env.reset()
    traj = []

    # Run until death or a reasonable progress cap to ensure diversity
    done = False
    while not done and raw_env.player_z < 150:
        # Save pre-transition state
        current_obs_img = obs["image"].copy()
        current_scalars = obs["scalars"].copy()
        current_mask = info["action_mask"].copy()

        # Oracle Decision
        action = plan_best_action(raw_env, lookahead_steps=12, sim_env=planner_env)

        obs, reward, terminated, truncated, info = env.step(action)

        traj.append({
            'image': current_obs_img,
            'scalars': current_scalars,
            'mask': current_mask,
            'action': action,
            'reward': reward,
            'return': 0.0
        })
        done = terminated or truncated

    env.close()

    # Calculate episodic returns
    g = 0.0
    for step in reversed(traj):
        g = step['reward'] + 0.99 * g
        step['return'] = g

    return traj


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    save_path = os.path.join(DATA_DIR, "bc_dataset.pt")

    print("=====================================================")
    print("       EXPERT DATA HARVESTER (DETERMINISTIC)         ")
    print("=====================================================")

    vae_device = setup_device()
    vae = SpatialVQVAE().to(vae_device)
    vae.load_state_dict(torch.load(VAE_CHECKPOINT, map_location=vae_device, weights_only=False))
    vae.eval()

    raw_samples = []
    num_workers = max(1, os.cpu_count() - 2)
    start_time = time.time()

    # PHASE 1: Parallel CPU Collection
    print(f"[STEP 1] Spawning {num_workers} Oracle workers...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker_collect_expert, i) for i in range(num_workers * 3)]

        while len(raw_samples) < TARGET_STEPS:
            done, futures = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                raw_samples.extend(f.result())
                # Add a new task to keep the pool full
                futures.add(executor.submit(worker_collect_expert, len(futures)))

            print(f"\r[GATHER] Raw steps collected: {len(raw_samples)}/{TARGET_STEPS}...", end="")

    raw_samples = raw_samples[:TARGET_STEPS]
    cpu_time = time.time() - start_time
    print(f"\n[SYSTEM] CPU Collection Complete in {cpu_time:.1f}s.")

    # PHASE 2: Batched GPU Encoding
    print(f"[STEP 2] Encoding {len(raw_samples)} frames into latents on {vae_device}...")
    final_encoded_samples = []

    for i in range(0, len(raw_samples), BATCH_SIZE):
        batch = raw_samples[i:i + BATCH_SIZE]

        # Prepare batch for GPU
        img_stack = np.stack([s['image'] for s in batch])
        img_tensor = torch.tensor(img_stack, dtype=torch.float32, device=vae_device)

        with torch.no_grad():
            _, _, _, _, _, _, qc, qt = vae(img_tensor)
            # Concatenate Context and Trend latents into 128-channel map
            latents = torch.cat([qc, qt], dim=1).cpu().numpy()

        for j, s in enumerate(batch):
            final_encoded_samples.append({
                'latent': latents[j],
                'scalars': s['scalars'],
                'mask': s['mask'],
                'action': s['action'],
                'return': s['return']
            })

        if (i // BATCH_SIZE) % 10 == 0:
            print(f"\r[ENCODE] Progress: {len(final_encoded_samples)}/{TARGET_STEPS}...", end="")

    # PHASE 3: Save to SSD
    print(f"\n[STEP 3] Saving dataset to {save_path}...")
    torch.save(final_encoded_samples, save_path)

    total_time = time.time() - start_time
    print(f"=====================================================")
    print(f" SUCCESS: {TARGET_STEPS} Expert Steps Harvested.")
    print(f" Total Time: {total_time:.1f}s | Final Size: {os.path.getsize(save_path) / 1e6:.2f} MB")
    print("=====================================================")


if __name__ == "__main__":
    main()