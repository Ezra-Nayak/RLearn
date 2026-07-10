# --- play_oracle.py ---
import cv2
import numpy as np
import random
import time
from crossy_gym_env import CrossyGymEnv


def sim_env_step_physics_only(sim_env, action):
    """
    Lightweight physics update for planning lookup.
    Bypasses visual rendering loops to maximize execution speed.
    """
    prev_z = sim_env.player_z
    prev_x = sim_env.player_x

    target_x = sim_env.player_x
    target_z = sim_env.player_z

    if action == 0:  # Up
        target_z += 1
    elif action == 1:  # Left
        target_x = max(sim_env.GRID_MIN_X, sim_env.player_x - 1)
    elif action == 2:  # Right
        target_x = min(sim_env.GRID_MAX_X, sim_env.player_x + 1)

    obstacle_hit = sim_env.obstacle_map.get((target_x, target_z), False)

    if obstacle_hit:
        target_x = sim_env.player_x
        target_z = sim_env.player_z
    else:
        sim_env.player_x = target_x
        sim_env.player_z = target_z

    sim_env._update_physics()

    # Camera updates
    sim_env.camera_z += sim_env.camera_speed * sim_env.DT
    sim_env.camera_speed = min(2.5, 1.0 + (sim_env.player_z * 0.005))

    target_camera_z = sim_env.player_z - 3.0
    if sim_env.camera_z < target_camera_z:
        sim_env.camera_z += (target_camera_z - sim_env.camera_z) * 1.5 * sim_env.DT

    max_dist = 8.0
    if sim_env.player_z - sim_env.camera_z > max_dist:
        sim_env.camera_z = sim_env.player_z - max_dist

    if sim_env.highest_generated_z - sim_env.player_z < 30:
        sim_env._generate_chunk()

    done = False
    if sim_env.player_z <= sim_env.camera_z:
        done = True
    elif sim_env._check_collisions():
        done = True

    return done


def plan_best_action(env, lookahead_steps=12):
    """
    Performs real-time time-expanded state-space BFS to find optimal actions.
    Uses cloned simulator transitions to verify paths up to lookahead_steps ahead.
    """
    # Capture current random state to maintain prediction parity
    rand_state = random.getstate()

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

    sim_env = CrossyGymEnv()
    start_state = get_env_state_dict(env)

    # Dictionary representing active search wave: (player_x, player_z) -> (state_dict, action_path, random_state)
    current_level = {(env.player_x, env.player_z): (start_state, [], random.getstate())}

    best_survival_path = []
    max_survival_depth = 0

    best_path = []
    best_final_z = env.player_z

    for depth in range(1, lookahead_steps + 1):
        next_level = {}
        for (px, pz), (state_dict, path, r_state) in current_level.items():
            for action in [0, 1, 2, 3]:  # Up, Left, Right, Idle
                # Reconstruct simulator state for branching paths
                random.setstate(r_state)
                restore_env_state(sim_env, state_dict)

                done = sim_env_step_physics_only(sim_env, action)

                if not done:
                    nxt_px, nxt_pz = sim_env.player_x, sim_env.player_z
                    if (nxt_px, nxt_pz) not in next_level:
                        next_level[(nxt_px, nxt_pz)] = (
                            get_env_state_dict(sim_env),
                            path + [action],
                            random.getstate()
                        )

                        # Track furthest surviving progress
                        if nxt_pz > best_final_z:
                            best_final_z = nxt_pz
                            best_path = path + [action]
                        elif nxt_pz == best_final_z:
                            # Prefer shorter/faster paths to reach same progress
                            if not best_path or len(path) + 1 < len(best_path):
                                best_path = path + [action]

                        # Track survival path in case of complete blockages
                        if depth > max_survival_depth:
                            max_survival_depth = depth
                            best_survival_path = path + [action]

        if not next_level:
            # All planning horizons are fatal, fallback to maximal survival strategy
            break
        current_level = next_level

    # Restore the original system random sequence
    random.setstate(rand_state)

    if best_path:
        return best_path[0]
    elif best_survival_path:
        return best_survival_path[0]
    return 3  # Default fallback to Idle


def main():
    print("--- [ORACLE AUTOPLAY] Simulated Perfect Pathing ---")
    print("Controls:")
    print("  'Spacebar' -> Pause / Resume")
    print("  'r'        -> Force Reset")
    print("  'q'        -> Quit\n")

    env = CrossyGymEnv()
    obs, info = env.reset()

    steps = 0
    total_reward = 0.0
    action_map = {0: "Up", 1: "Left", 2: "Right", 3: "Idle"}

    paused = False
    last_action = "None"
    tactile_bump = False

    while True:
        start_time = time.time()

        # Compute optimal decision using oracle planning
        if not paused:
            action = plan_best_action(env, lookahead_steps=12)
            last_action = action_map[action]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            tactile_bump = info.get("tactile_bump", False)
        else:
            action = -1

        # Get visual frame from environment
        rgb_frame = env.render()
        display = cv2.resize(rgb_frame, (640, 640), interpolation=cv2.INTER_NEAREST)

        # Draw Diagnostic Overlays
        cv2.rectangle(display, (0, 0), (640, 140), (15, 15, 15), -1)
        cv2.putText(display, f"Step: {steps} | Score: {info['score']}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        status_text = "PAUSED" if paused else f"Oracle Action: {last_action}"
        cv2.putText(display, status_text, (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, f"Cumulative Reward: {total_reward:+.2f}", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1, cv2.LINE_AA)

        if tactile_bump:
            cv2.rectangle(display, (0, 0), (640, 640), (0, 0, 255), 3)

        cv2.imshow("Oracle Autoplay", display)

        # Pause handling / reset controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            paused = not paused
        elif key == ord('r'):
            obs, info = env.reset()
            steps = 0
            total_reward = 0.0
            last_action = "RESET"
            tactile_bump = False
        elif key == ord('q'):
            print("[INFO] Exiting oracle autopilot.")
            break

        if not paused and (terminated or truncated):
            # Flash game-over highlight
            death_overlay = display.copy()
            cv2.rectangle(death_overlay, (0, 0), (640, 640), (0, 0, 255), -1)
            cv2.addWeighted(display, 0.4, death_overlay, 0.6, 0, dst=display)
            cv2.putText(display, f"GAME OVER (FINAL SCORE: {info['score']})", (130, 340),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Oracle Autoplay", display)
            cv2.waitKey(2000)

            # Re-initialize
            obs, info = env.reset()
            steps = 0
            total_reward = 0.0
            last_action = "RESET"
            tactile_bump = False

        # Cap visualization rate for visual comfort
        elapsed = time.time() - start_time
        sleep_duration = max(0.0, 0.05 - elapsed)
        time.sleep(sleep_duration)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()