# --- visualize_scenarios.py ---
import os
import time
import cv2
import numpy as np
import torch

from crossy_gym_env import CrossyGymEnv
from train_ppo_sim import FrameStackWrapper, SpatialVQVAE, ActorCritic, setup_device

# --- CONFIGURATION ---
MODEL_PATH = "../checkpoints/ppo_redemption_latest.pth"
VAE_PATH = "../checkpoints/sim_vae_best.pth"
WIN_CONDITION_Z = 15
MAX_STEPS = 150


def patch_environment_scenario(raw_env, scenario_type, rng):
    """
    Temporarily overrides the environment's chunk generator to build
    highly specific, deterministic sandboxes for cognitive testing.
    """

    def custom_generate_chunk():
        if getattr(raw_env, '_scenario_generated', False):
            return

        raw_env.terrain_map.clear()
        raw_env.obstacle_map.clear()
        raw_env.road_parameters.clear()
        raw_env.active_cars.clear()

        chunk_end = 40
        raw_env.highest_generated_z = chunk_end

        if scenario_type == "Grass Navigation":
            while True:
                local_obs = {}
                for z in range(chunk_end + 1):
                    raw_env.terrain_map[z] = 'G'
                    if z < 2: continue
                    for x in range(raw_env.GRID_MIN_X, raw_env.GRID_MAX_X + 1):
                        if rng.rand() < 0.25:
                            local_obs[(x, z)] = True
                if raw_env._verify_connectivity(0, WIN_CONDITION_Z, local_obs, raw_env.terrain_map):
                    raw_env.obstacle_map.update(local_obs)
                    break

        elif scenario_type == "Simple Roads":
            for z in range(chunk_end + 1):
                is_road = z in [3, 4, 8, 9, 13, 14]
                raw_env.terrain_map[z] = 'R' if is_road else 'G'
                if is_road:
                    raw_env.road_parameters[z] = {'speed': rng.uniform(1.5, 2.0), 'direction': rng.choice([-1, 1])}
                    raw_env.active_cars[z] = [rng.uniform(-4.0, 4.0)]

        elif scenario_type == "Complex Traffic":
            for z in range(chunk_end + 1):
                is_road = 3 <= z <= 12
                raw_env.terrain_map[z] = 'R' if is_road else 'G'
                if is_road:
                    direction = 1 if z % 2 == 0 else -1
                    raw_env.road_parameters[z] = {'speed': rng.uniform(2.5, 4.5), 'direction': direction}
                    raw_env.active_cars[z] = [rng.uniform(-6.0, -1.0), rng.uniform(1.0, 6.0)]

        elif scenario_type == "Obstacle Jailbreak":
            for z in range(chunk_end + 1):
                raw_env.terrain_map[z] = 'G'

            raw_env.obstacle_map[(0, 1)] = True
            raw_env.obstacle_map[(0, -1)] = True

            if rng.rand() < 0.5:
                raw_env.obstacle_map[(-1, 0)] = True
                raw_env.obstacle_map[(-1, 1)] = True
            else:
                raw_env.obstacle_map[(1, 0)] = True
                raw_env.obstacle_map[(1, 1)] = True

            for z in range(3, chunk_end + 1):
                for x in range(raw_env.GRID_MIN_X, raw_env.GRID_MAX_X + 1):
                    if rng.rand() < 0.15:
                        raw_env.obstacle_map[(x, z)] = True

        raw_env._scenario_generated = True

    raw_env._generate_chunk = custom_generate_chunk


def main():
    print("[SYSTEM] Booting Interactive Visual Sandbox...")

    vae_device = setup_device()
    ppo_device = torch.device("cpu")

    raw_env = CrossyGymEnv()

    # FORCE CONSTANT CAMERA SPEED
    original_step = raw_env.step

    def patched_step(action):
        obs, rew, done, trunc, info = original_step(action)
        raw_env.camera_speed = 1.0
        return obs, rew, done, trunc, info

    raw_env.step = patched_step

    env = FrameStackWrapper(raw_env, stack_size=4)

    # Load Models
    vae = SpatialVQVAE().to(vae_device)
    vae.load_state_dict(torch.load(VAE_PATH, map_location=vae_device, weights_only=False))
    vae.eval()

    policy = ActorCritic(action_dim=4).to(ppo_device)
    checkpoint = torch.load(MODEL_PATH, map_location=ppo_device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['model_state_dict'])
    else:
        policy.load_state_dict(checkpoint)
    policy.eval()

    scenarios = {
        '1': "Grass Navigation",
        '2': "Simple Roads",
        '3': "Complex Traffic",
        '4': "Obstacle Jailbreak"
    }

    current_scenario = "Grass Navigation"
    seed = 0

    print("\n--- CONTROLS ---")
    print("[1] Grass Navigation")
    print("[2] Simple Roads")
    print("[3] Complex Traffic")
    print("[4] Obstacle Jailbreak")
    print("[Q/ESC] Quit")
    print("----------------\n")

    try:
        while True:
            # Setup new episode
            rng = np.random.RandomState(seed)
            raw_env._scenario_generated = False
            patch_environment_scenario(raw_env, current_scenario, rng)

            obs, info = env.reset(seed=int(seed))
            done = False
            steps = 0
            outcome = ""

            while not done:
                # 1. Capture user input for switching scenarios in real-time
                key = cv2.waitKey(40) & 0xFF  # ~25 FPS visual playback
                if key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                    current_scenario = scenarios[chr(key)]
                    seed += 1
                    break  # Break inner loop to restart with new scenario
                elif key == ord('q') or key == 27:
                    return

                # 2. Agent Decision
                img_batch = torch.tensor(obs["image"], dtype=torch.float32, device=vae_device).unsqueeze(0)
                scalars_batch = torch.tensor(obs["scalars"], dtype=torch.float32, device=ppo_device).unsqueeze(0)
                masks_batch = torch.tensor(info["action_mask"], dtype=torch.float32, device=ppo_device).unsqueeze(0)

                with torch.no_grad():
                    _, _, _, _, _, _, quant_c, quant_t = vae(img_batch)
                    latents_batch = torch.cat([quant_c, quant_t], dim=1).to(ppo_device)
                    features = policy._get_features(latents_batch, scalars_batch)
                    action_logits = policy.actor(features) + masks_batch
                    best_action = torch.argmax(action_logits, dim=-1).item()

                obs, reward, terminated, truncated, info = env.step(best_action)
                steps += 1

                # 3. Check Conditions
                if raw_env.player_z >= WIN_CONDITION_Z:
                    outcome = "PASS!"
                    done = True
                elif terminated or truncated:
                    if raw_env.player_z <= raw_env.camera_z:
                        outcome = "DEAD: CAMERA (Too slow/stuck)"
                    else:
                        outcome = "DEAD: CAR (Hit traffic)"
                    done = True
                elif steps >= MAX_STEPS:
                    outcome = "TIMEOUT"
                    done = True

                # 4. Render HUD
                raw_frame = obs["image"][3]  # Most recent grayscale frame
                canvas = cv2.cvtColor(np.uint8(raw_frame * 255), cv2.COLOR_GRAY2BGR)
                canvas = cv2.resize(canvas, (640, 640), interpolation=cv2.INTER_NEAREST)

                # Draw Overlay
                cv2.rectangle(canvas, (0, 0), (640, 60), (20, 20, 20), -1)
                cv2.putText(canvas, f"SCENARIO: {current_scenario.upper()} (Keys 1-4)", (15, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(canvas, f"Seed: {seed} | Step: {steps}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200, 200, 200), 1)

                if done:
                    color = (0, 255, 0) if outcome == "PASS!" else (0, 0, 255)
                    cv2.putText(canvas, outcome, (150, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

                cv2.imshow("Cognitive Scenario Visualizer", canvas)

            # If the episode naturally finished (done=True), pause for a moment to let user read the outcome
            if done:
                # Wait 1.5 seconds, but allow user to interrupt with 1-4 or Q
                start_wait = time.time()
                while time.time() - start_wait < 1.5:
                    key = cv2.waitKey(10) & 0xFF
                    if key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                        current_scenario = scenarios[chr(key)]
                        break
                    elif key == ord('q') or key == 27:
                        return

                # Increment seed for variety
                seed += 1

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print("[SYSTEM] Visualizer shut down safely.")


if __name__ == "__main__":
    main()