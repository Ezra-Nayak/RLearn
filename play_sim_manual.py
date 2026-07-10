# --- play_sim_manual.py ---
import cv2
import numpy as np
import time
from crossy_gym_env import CrossyGymEnv


def main():
    print("--- [MANUAL PLAY] Interactive Simulator ---")
    print("Controls (Keep the 'Interactive Simulator' window in focus):")
    print("  'w' -> Step Forward (Up)")
    print("  'a' -> Step Left")
    print("  'd' -> Step Right")
    print("  's' -> Idle (No Move)")
    print("  'q' -> Quit\n")

    env = CrossyGymEnv()
    obs, info = env.reset()

    steps = 0
    total_reward = 0.0
    action_map = {0: "Up", 1: "Left", 2: "Right", 3: "Idle"}
    last_action = "None"
    tactile_bump = False

    while True:
        # Get standard RGB canvas render from env (160x160x3)
        rgb_frame = env.render()

        # Scale up by 4x for comfortable viewing
        display = cv2.resize(rgb_frame, (640, 640), interpolation=cv2.INTER_NEAREST)

        # Draw Diagnostic Overlay Panel
        cv2.rectangle(display, (0, 0), (640, 140), (15, 15, 15), -1)
        cv2.putText(display, f"Step: {steps} | Score: {info['score']}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, f"Last Move: {last_action}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, f"Cumulative Reward: {total_reward:+.2f}", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1, cv2.LINE_AA)

        # Warn if the player bumped into an obstacle
        if tactile_bump:
            cv2.rectangle(display, (0, 0), (640, 640), (0, 0, 255), 3)
            cv2.putText(display, "BUMP DETECTED!", (350, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Interactive Simulator", display)

        # Wait indefinitely for keyboard input
        key = cv2.waitKey(0) & 0xFF

        action = -1
        if key == ord('w'):
            action = 0
        elif key == ord('a'):
            action = 1
        elif key == ord('d'):
            action = 2
        elif key == ord('s'):
            action = 3
        elif key == ord('q'):
            print("[INFO] Exiting interactive simulator.")
            break

        if action != -1:
            last_action = action_map[action]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            tactile_bump = info.get("tactile_bump", False)

            if terminated or truncated:
                # Flash red screen on collision
                death_overlay = display.copy()
                cv2.rectangle(death_overlay, (0, 0), (640, 640), (0, 0, 255), -1)
                cv2.addWeighted(display, 0.4, death_overlay, 0.6, 0, dst=display)
                cv2.putText(display, "GAME OVER (RESETTING...)", (130, 340),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("Interactive Simulator", display)
                cv2.waitKey(2000)

                # Reset
                obs, info = env.reset()
                steps = 0
                total_reward = 0.0
                last_action = "RESET"
                tactile_bump = False

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()