import os
import time
import numpy as np
import gymnasium as gym
import random
from collections import deque
from crossy_gym_env import CrossyGymEnv
from gym_wrappers import FrameStackWrapper
from play_oracle import plan_best_action

# Configuration
DATA_DIR = "sim_data"
TARGET_SAMPLES = 8000
SAVE_INTERVAL = 1000


def main():
    print("[SYSTEM] Starting Simulated Data Collection using Oracle Autopilot...")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    raw_env = CrossyGymEnv()
    env = FrameStackWrapper(raw_env, stack_size=4)

    data_cache = []
    total_sequences_saved = 0

    obs, info = env.reset()
    spamming_idle = False

    # Store a rolling historical buffer to align future targets [t+1]
    # We require [t-3, t-2, t-1, t] as inputs and [t+1] as the target frame.
    frame_history = deque(maxlen=5)
    frame_history.append(raw_env._render_canvas())

    try:
        while total_sequences_saved + len(data_cache) < TARGET_SAMPLES:
            if spamming_idle:
                action = 3  # Spam Idle
            else:
                # Check for idle-spam triggers (Score >= 50 AND on Grass block)
                current_z = raw_env.player_z
                current_terrain = raw_env.terrain_map.get(current_z, 'G')
                if current_z >= 50 and current_terrain == 'G':
                    if random.random() < 0.10:
                        spamming_idle = True
                        print(f"[ORACLE] Score {current_z} reached on Grass. Commencing idle spam (10% trigger)...")

                if spamming_idle:
                    action = 3
                else:
                    action = plan_best_action(raw_env, lookahead_steps=12)

            # Record current raw frame state before applying step
            frame_history.append(raw_env._render_canvas())

            obs, reward, terminated, truncated, info = env.step(action)

            # Align sequences when the history buffer contains 5 consecutive frames:
            # - Input Stack: frames at indices 0, 1, 2, 3 (corresponding to t-3 to t)
            # - Target Frame: frame at index 4 (corresponding to t+1)
            if len(frame_history) == 5:
                input_stack = np.array([frame_history[0], frame_history[1], frame_history[2], frame_history[3]],
                                       dtype=np.float32)
                target_frame = frame_history[4]

                data_cache.append((input_stack, target_frame))

            # Force reset if agent reaches score 100 to balance data
            score_limit_reached = False
            if raw_env.player_z >= 100:
                print(f"[ORACLE] Score 100 reached. Resetting run for simulation balance.")
                score_limit_reached = True

            if terminated or truncated or score_limit_reached:
                obs, info = env.reset()
                frame_history.clear()
                frame_history.append(raw_env._render_canvas())
                spamming_idle = False

            # Periodically write data cache to disk
            if len(data_cache) >= SAVE_INTERVAL:
                timestamp = int(time.time())
                save_path = os.path.join(DATA_DIR, f"sim_chunk_{timestamp}.npy")
                np.save(save_path, np.array(data_cache, dtype=object))

                total_sequences_saved += len(data_cache)
                print(
                    f"[SAVE] Saved {len(data_cache)} samples to {save_path}. Progress: {total_sequences_saved}/{TARGET_SAMPLES}")
                data_cache = []

    except KeyboardInterrupt:
        print("\n[SYSTEM] Collection interrupted manually.")
    finally:
        if len(data_cache) > 0:
            timestamp = int(time.time())
            save_path = os.path.join(DATA_DIR, f"sim_chunk_{timestamp}.npy")
            np.save(save_path, np.array(data_cache, dtype=object))
            total_sequences_saved += len(data_cache)
            print(f"[SAVE] Final save of {len(data_cache)} samples to {save_path}.")

        env.close()
        print(f"[SUCCESS] Data collection complete. Total samples collected: {total_sequences_saved}")


if __name__ == "__main__":
    main()