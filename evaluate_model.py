# --- evaluate_scenarios.py ---
import os
import time
import numpy as np
import torch

from crossy_gym_env import CrossyGymEnv
from train_ppo_sim import FrameStackWrapper, SpatialVQVAE, ActorCritic, setup_device

# --- CONFIGURATION ---
MODEL_PATH = "checkpoints/ppo_redemption_latest.pth"
VAE_PATH = "checkpoints/sim_vae_best.pth"
TESTS_PER_SCENARIO = 250
WIN_CONDITION_Z = 15
MAX_STEPS = 150


def patch_environment_scenario(raw_env, scenario_type, rng):
    """
    Temporarily overrides the environment's chunk generator to build
    highly specific, deterministic sandboxes for cognitive testing.
    """

    def custom_generate_chunk():
        # Prevent the env from generating more chunks once the sandbox is built
        if getattr(raw_env, '_scenario_generated', False):
            return

        raw_env.terrain_map.clear()
        raw_env.obstacle_map.clear()
        raw_env.road_parameters.clear()
        raw_env.active_cars.clear()

        chunk_end = 40
        raw_env.highest_generated_z = chunk_end

        # -------------------------------------------------------------
        # SCENARIO 1: GRASS OBSTACLE NAVIGATION
        # -------------------------------------------------------------
        if scenario_type == "Grass Navigation":
            while True:
                local_obs = {}
                for z in range(chunk_end + 1):
                    raw_env.terrain_map[z] = 'G'
                    if z < 2: continue  # Clear start zone
                    for x in range(raw_env.GRID_MIN_X, raw_env.GRID_MAX_X + 1):
                        if rng.rand() < 0.25:  # 25% density
                            local_obs[(x, z)] = True

                # Ensure it's mathematically possible to pass
                if raw_env._verify_connectivity(0, WIN_CONDITION_Z, local_obs, raw_env.terrain_map):
                    raw_env.obstacle_map.update(local_obs)
                    break

        # -------------------------------------------------------------
        # SCENARIO 2: SIMPLE ROAD CROSSINGS
        # -------------------------------------------------------------
        elif scenario_type == "Simple Roads":
            for z in range(chunk_end + 1):
                is_road = z in [3, 4, 8, 9, 13, 14]
                raw_env.terrain_map[z] = 'R' if is_road else 'G'
                if is_road:
                    # Slow speeds, single cars
                    raw_env.road_parameters[z] = {'speed': rng.uniform(1.5, 2.0), 'direction': rng.choice([-1, 1])}
                    raw_env.active_cars[z] = [rng.uniform(-4.0, 4.0)]

        # -------------------------------------------------------------
        # SCENARIO 3: COMPLEX TRAFFIC MATRIX
        # -------------------------------------------------------------
        elif scenario_type == "Complex Traffic":
            for z in range(chunk_end + 1):
                is_road = 3 <= z <= 12  # Massive 10-lane highway
                raw_env.terrain_map[z] = 'R' if is_road else 'G'
                if is_road:
                    # Fast speeds, opposing directions, multiple cars
                    direction = 1 if z % 2 == 0 else -1
                    raw_env.road_parameters[z] = {'speed': rng.uniform(2.5, 4.5), 'direction': direction}
                    raw_env.active_cars[z] = [rng.uniform(-6.0, -1.0), rng.uniform(1.0, 6.0)]

        # -------------------------------------------------------------
        # SCENARIO 4: THE OBSTACLE JAILBREAK
        # -------------------------------------------------------------
        elif scenario_type == "Obstacle Jailbreak":
            for z in range(chunk_end + 1):
                raw_env.terrain_map[z] = 'G'

            # The Trap: Block directly ahead
            raw_env.obstacle_map[(0, 1)] = True
            raw_env.obstacle_map[(0, -1)] = True  # Prevent backing up

            # Randomly block Left or Right, forcing a specific lateral escape
            if rng.rand() < 0.5:
                # Block Left side completely
                raw_env.obstacle_map[(-1, 0)] = True
                raw_env.obstacle_map[(-1, 1)] = True
            else:
                # Block Right side completely
                raw_env.obstacle_map[(1, 0)] = True
                raw_env.obstacle_map[(1, 1)] = True

            # Add light obstacles afterwards so the agent doesn't just run straight
            for z in range(3, chunk_end + 1):
                for x in range(raw_env.GRID_MIN_X, raw_env.GRID_MAX_X + 1):
                    if rng.rand() < 0.15:
                        raw_env.obstacle_map[(x, z)] = True

        raw_env._scenario_generated = True

    # Apply the patch
    raw_env._generate_chunk = custom_generate_chunk


def main():
    print("=====================================================")
    print("    CROSSY ROAD - COGNITIVE SCENARIO BENCHMARK       ")
    print("=====================================================")

    vae_device = setup_device()
    ppo_device = torch.device("cpu")

    raw_env = CrossyGymEnv()

    # FORCE CONSTANT CAMERA SPEED (Lock to 1.0)
    original_step = raw_env.step

    def patched_step(action):
        obs, rew, done, trunc, info = original_step(action)
        raw_env.camera_speed = 1.0
        return obs, rew, done, trunc, info

    raw_env.step = patched_step

    env = FrameStackWrapper(raw_env, stack_size=4)

    print(f"[SYSTEM] Loading VAE...")
    vae = SpatialVQVAE().to(vae_device)
    vae.load_state_dict(torch.load(VAE_PATH, map_location=vae_device, weights_only=False))
    vae.eval()

    print(f"[SYSTEM] Loading Policy from {MODEL_PATH}...\n")
    policy = ActorCritic(action_dim=4).to(ppo_device)
    checkpoint = torch.load(MODEL_PATH, map_location=ppo_device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['model_state_dict'])
    else:
        policy.load_state_dict(checkpoint)
    policy.eval()

    scenarios = ["Grass Navigation", "Simple Roads", "Complex Traffic", "Obstacle Jailbreak"]
    report_data = {}

    start_time = time.time()

    for sc_idx, scenario in enumerate(scenarios):
        print(f"--- Testing Scenario: {scenario} ---")

        passes = 0
        death_car = 0
        death_eagle = 0
        death_timeout = 0
        total_steps_on_pass = 0

        for i in range(TESTS_PER_SCENARIO):
            seed = i + (sc_idx * 1000)
            rng = np.random.RandomState(seed)

            # Inject Sandbox Map
            raw_env._scenario_generated = False
            patch_environment_scenario(raw_env, scenario, rng)

            obs, info = env.reset(seed=int(seed))
            done = False
            steps = 0

            while not done:
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

                # Check custom Win/Loss conditions
                if raw_env.player_z >= WIN_CONDITION_Z:
                    passes += 1
                    total_steps_on_pass += steps
                    done = True
                elif terminated or truncated:
                    if raw_env.player_z <= raw_env.camera_z:
                        death_eagle += 1
                    else:
                        death_car += 1
                    done = True
                elif steps >= MAX_STEPS:
                    death_timeout += 1
                    done = True

            if (i + 1) % 50 == 0:
                print(f"  [{i + 1}/{TESTS_PER_SCENARIO}] Pass Rate: {(passes / (i + 1)) * 100:.1f}%")

        avg_steps = (total_steps_on_pass / passes) if passes > 0 else 0
        report_data[scenario] = {
            "pass_rate": (passes / TESTS_PER_SCENARIO) * 100.0,
            "passes": passes,
            "car_deaths": death_car,
            "eagle_deaths": death_eagle,
            "timeouts": death_timeout,
            "avg_steps": avg_steps
        }

    # ==========================================
    # FINAL REPORT GENERATION
    # ==========================================
    total_time = time.time() - start_time
    total_tests = len(scenarios) * TESTS_PER_SCENARIO
    total_passes = sum(d["passes"] for d in report_data.values())
    overall_pass_rate = (total_passes / total_tests) * 100.0

    print("\n=====================================================")
    print("              FINAL COGNITIVE REPORT                 ")
    print("=====================================================")
    print(f"Model          : {os.path.basename(MODEL_PATH)}")
    print(f"Total Tests    : {total_tests} ({TESTS_PER_SCENARIO} per scenario)")
    print(f"Overall Grade  : {overall_pass_rate:.1f}% PASS RATE")
    print(f"Execution Time : {total_time:.1f} seconds")
    print("=====================================================\n")

    for scenario, data in report_data.items():
        print(f"[{scenario.upper()}]")
        print(f"  Pass Rate      : {data['pass_rate']:.1f}% ({data['passes']}/{TESTS_PER_SCENARIO})")
        if data['passes'] < TESTS_PER_SCENARIO:
            print(
                f"  Failure Causes : {data['car_deaths']} Car Hits | {data['eagle_deaths']} Camera Deaths | {data['timeouts']} Timeouts")
        print(f"  Avg Speed      : {data['avg_steps']:.1f} steps to clear")
        print("-" * 53)


if __name__ == "__main__":
    main()