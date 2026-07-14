# --- diagnose_agent.py ---
import os
import sys
import glob
import time
import cv2
import torch
import torch.nn.functional as F
import numpy as np

# Parity Imports from train_ppo_sim
from train_ppo_sim import (
    CrossyGymEnv,
    FrameStackWrapper,
    SpatialVQVAE,
    ActorCritic,
    setup_device,
    VAE_DEVICE,
    PPO_DEVICE
)

# Setup Paths
CHECKPOINT_PATH = "../checkpoints/ppo_redemption_latest.pth"
VAE_CHECKPOINT = "../checkpoints/sim_vae_best.pth"


def compute_saliency_map(policy, latents, scalars, action_mask):
    """ Backpropagation Saliency Heatmap """
    latents.requires_grad_()
    features = policy._get_features(latents, scalars)
    action_logits = policy.actor(features)

    if action_mask is not None:
        action_logits = action_logits + action_mask

    best_action = action_logits.argmax(dim=-1)
    best_logit = action_logits[0, best_action]

    policy.zero_grad()
    best_logit.backward(retain_graph=True)

    # Gradients out of the latent layer
    saliency = latents.grad.abs().squeeze(0).max(dim=0)[0].cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    return saliency, action_logits.detach(), best_action.item()


def draw_scorecard(canvas, stats):
    """ Overlays widescreen-centered performance results """
    h, w, _ = canvas.shape
    overlay = canvas.copy()
    card_w, card_h = int(w * 0.5), int(h * 0.6)
    x0, y0 = (w - card_w) // 2, (h - card_h) // 2

    cv2.rectangle(overlay, (x0, y0), (x0 + card_w, y0 + card_h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.90, canvas, 0.10, 0, dst=canvas)
    cv2.rectangle(canvas, (x0, y0), (x0 + card_w, y0 + card_h), (0, 255, 255), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "RUN EVALUATION SCORECARD", (x0 + 40, y0 + 50), font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, (x0 + 40, y0 + 65), (x0 + card_w - 40, y0 + 65), (50, 50, 50), 1)

    y_pos = y0 + 110
    metrics = [
        ("Final Z-Score (Progress):", f"{stats['score']}"),
        ("Steps Survived:", f"{stats['steps']}"),
        ("Avg Critic Confidence:", f"{stats['avg_val']:.4f}"),
        ("Cumulative Reward:", f"{stats['total_rew']:.2f}"),
        ("Decision Entropy:", f"{stats['entropy']:.4f}"),
    ]

    for label, val in metrics:
        cv2.putText(canvas, label, (x0 + 40, y_pos), font, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, val, (x0 + card_w - 180, y_pos), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y_pos += 40

    cv2.line(canvas, (x0 + 40, y_pos - 15), (x0 + card_w - 40, y_pos - 15), (50, 50, 50), 1)
    cv2.putText(canvas, "Action Distribution:", (x0 + 40, y_pos + 10), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    y_pos += 45

    actions = ["Up", "Left", "Right", "Idle"]
    for i, act in enumerate(actions):
        pct = stats['action_pcts'][i]
        cv2.putText(canvas, f"{act}: {pct:.1f}%", (x0 + 50, y_pos), font, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
        bar_len = int(pct * 1.5)
        cv2.rectangle(canvas, (x0 + 180, y_pos - 10), (x0 + 180 + bar_len, y_pos + 2), (0, 255, 0), -1)
        y_pos += 25

    cv2.putText(canvas, "[PRESS ANY KEY TO RESET RUN]", (x0 + 40, y0 + card_h - 30), font, 0.45, (0, 255, 0), 1, cv2.LINE_AA)


def main():
    print("[SYSTEM] Booting Sci-Fi Widescreen Telemetry Scanner...")

    # Load Checks
    if not os.path.exists(VAE_CHECKPOINT):
        print(f"[ERROR] Missing VAE checkpoint: '{VAE_CHECKPOINT}'")
        sys.exit(1)

    selected_checkpoint = CHECKPOINT_PATH
    if not os.path.exists(selected_checkpoint):
        fallback_checkpoints = glob.glob("checkpoints/ppo_sim_*_step.pth")
        if fallback_checkpoints:
            selected_checkpoint = max(fallback_checkpoints, key=os.path.getmtime)
            print(f"[WARN] Requested checkpoint missing. Opening newest fallback: '{selected_checkpoint}'")
        else:
            print(f"[ERROR] No valid policy checkpoints available.")
            sys.exit(1)

    # Initialize environment
    raw_env = CrossyGymEnv()
    env = FrameStackWrapper(raw_env, stack_size=4)

    # Load Neural networks
    vae = SpatialVQVAE().to(VAE_DEVICE)
    vae.load_state_dict(torch.load(VAE_CHECKPOINT, map_location=VAE_DEVICE, weights_only=False))
    vae.eval()

    policy = ActorCritic(action_dim=4).to(PPO_DEVICE)
    checkpoint = torch.load(selected_checkpoint, map_location=PPO_DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['model_state_dict'])
    else:
        policy.load_state_dict(checkpoint)
    policy.eval()

    # Dynamic Grid Calibration variables
    cell_size = 16.0
    offset = 16.0

    # Rolling Plotter & Stats
    critic_history = []
    run_steps = 0
    run_score = 0
    run_values = []
    run_actions_taken = []
    run_entropies = []
    run_rewards = []

    obs, info = env.reset()

    try:
        while True:
            # Wrap inputs
            img_batch = torch.tensor(obs["image"], dtype=torch.float32, device=VAE_DEVICE).unsqueeze(0)
            scalars_batch = torch.tensor(obs["scalars"], dtype=torch.float32, device=PPO_DEVICE).unsqueeze(0)
            masks_batch = torch.tensor(info["action_mask"], dtype=torch.float32, device=PPO_DEVICE).unsqueeze(0)

            # VAE Feed Forward
            with torch.no_grad():
                recon_static, _, _, _, _, _, quant_c, quant_t = vae(img_batch)
                latents_batch = torch.cat([quant_c, quant_t], dim=1).to(PPO_DEVICE)

            # Saliency mapping
            saliency, logits, action_idx = compute_saliency_map(policy, latents_batch, scalars_batch, masks_batch)

            with torch.no_grad():
                features = policy._get_features(latents_batch, scalars_batch)
                value = policy.critic(features).item()

            probs = F.softmax(logits.squeeze(0), dim=0).cpu().numpy()
            entropy = -np.sum(probs * np.log(probs + 1e-8))

            # Record Telemetry
            run_steps += 1
            run_values.append(value)
            run_actions_taken.append(action_idx)
            run_entropies.append(entropy)
            critic_history.append(value)
            if len(critic_history) > 100:
                critic_history.pop(0)

            # Get player coordinates to recalibrate grid mapping dynamically
            raw_grayscale_frame = obs["image"][3]
            player_pixels = np.argwhere(np.abs(raw_grayscale_frame - 0.9) < 0.05)
            if len(player_pixels) > 0:
                centroid_y, centroid_x = player_pixels.mean(axis=0)
                if abs(raw_env.player_x) > 0.01:
                    cell_size = (centroid_x - 80.0) / raw_env.player_x
                    cell_size = max(12.0, min(20.0, cell_size))
                offset = 160.0 - centroid_y - cell_size * (raw_env.player_z - raw_env.camera_z)
                offset = max(0.0, min(50.0, offset))

            # ----------------- CANVAS RENDERING (1920x1080) -----------------
            canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)

            # Create game visual baseline
            color_frame = cv2.cvtColor(np.uint8(raw_grayscale_frame * 255), cv2.COLOR_GRAY2BGR)
            game_board = cv2.resize(color_frame, (1080, 1080), interpolation=cv2.INTER_NEAREST)

            # Convert logical agent positions to pixel locations
            px = cell_size * raw_env.player_x + 80.0
            py = 160.0 - (cell_size * (raw_env.player_z - raw_env.camera_z) + offset)
            player_canvas_x = int(px * 6.75)
            player_canvas_y = int(py * 6.75)

            # 1. CRITIC ENERGY RETICLE (removed)

            # 2. CRITICAL THREAT RED ZONES (80%+ Saliency Focus)
            flat_indices = np.argsort(saliency.flatten())[-5:]
            for s_idx in flat_indices:
                sy, sx = np.unravel_index(s_idx, saliency.shape)
                sal_val = saliency[sy, sx]

                if sal_val > 0.80:
                    pixel_x, pixel_y = int(sx * 8 + 4), int(sy * 8 + 4)
                    target_canvas_x, target_canvas_y = int(pixel_x * 6.75), int(pixel_y * 6.75)

                    # Draw a simple pulsing red threat box
                    pulse = int(5 * np.sin(time.time() * 10))
                    cv2.rectangle(game_board, (target_canvas_x - 30 - pulse, target_canvas_y - 30 - pulse),
                                  (target_canvas_x + 30 + pulse, target_canvas_y + 30 + pulse), (0, 0, 255), 2,
                                  cv2.LINE_AA)
                    cv2.putText(game_board, "THREAT", (target_canvas_x - 25, target_canvas_y - 40 - pulse),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)


            # 3. TRANS-TEMPORAL GHOST ORACLE (REMOVED)
            # 4. ACTION PROBABILITY VECTORS (REMOVED)

            # 5. SCI-FI CORNER HUD DECORATIONS
            hud_color = (50, 50, 50)
            cv2.line(game_board, (20, 20), (120, 20), hud_color, 2)
            cv2.line(game_board, (20, 20), (20, 120), hud_color, 2)
            cv2.line(game_board, (1060, 20), (960, 20), hud_color, 2)
            cv2.line(game_board, (1060, 20), (1060, 120), hud_color, 2)
            cv2.line(game_board, (20, 1060), (120, 1060), hud_color, 2)
            cv2.line(game_board, (20, 1060), (20, 960), hud_color, 2)
            cv2.line(game_board, (1060, 1060), (960, 1060), hud_color, 2)
            cv2.line(game_board, (1060, 1060), (1060, 960), hud_color, 2)

            # Stitch game board to Center of Widescreen Canvas
            canvas[:, 420:1500] = game_board

            # ----------------- LEFT PANEL: BIOMETRICS & HUD -----------------
            left_panel = np.zeros((1080, 420, 3), dtype=np.uint8)
            cv2.putText(left_panel, "AGENT COGNITIVE HUD", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.line(left_panel, (30, 80), (390, 80), (30, 80, 80), 1)

            # Glowing Threat Index Bar
            threat_index = max(0.0, min(1.0, 1.0 - (value + 1.0) / 2.0))
            bar_x0, bar_y0 = 30, 160
            bar_w, bar_h = 40, 220
            cv2.rectangle(left_panel, (bar_x0, bar_y0), (bar_x0 + bar_w, bar_y0 + bar_h), (15, 15, 15), -1)
            cv2.rectangle(left_panel, (bar_x0, bar_y0), (bar_x0 + bar_w, bar_y0 + bar_h), (50, 50, 50), 1)

            fill_h = int(threat_index * bar_h)
            threat_color = (0, int((1.0 - threat_index) * 255), int(threat_index * 255))
            cv2.rectangle(left_panel, (bar_x0 + 2, bar_y0 + bar_h - fill_h), (bar_x0 + bar_w - 2, bar_y0 + bar_h - 2), threat_color, -1)
            cv2.putText(left_panel, "THREAT", (bar_x0, bar_y0 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(left_panel, "INDEX", (bar_x0, bar_y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(left_panel, f"{threat_index * 100.0:.0f}%", (bar_x0, bar_y0 + bar_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, threat_color, 1, cv2.LINE_AA)

            # Performance telemetry readouts
            cv2.putText(left_panel, "SYSTEM DATA READOUTS", (120, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA)
            cv2.putText(left_panel, f"Steps Lived:  {run_steps}", (120, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(left_panel, f"Z-Position:   {raw_env.player_z}", (120, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(left_panel, f"Target Value: {value:+.4f}", (120, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(left_panel, f"State Entropy: {entropy:.4f}", (120, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            # Critic History plotter
            plot_x0, plot_y0 = 30, 480
            plot_w, plot_h = 360, 220
            cv2.rectangle(left_panel, (plot_x0, plot_y0), (plot_x0 + plot_w, plot_y0 + plot_h), (10, 10, 10), -1)
            cv2.rectangle(left_panel, (plot_x0, plot_y0), (plot_x0 + plot_w, plot_y0 + plot_h), (40, 40, 40), 1)
            cv2.putText(left_panel, "NEURAL VALUATION TIMELINE", (plot_x0, plot_y0 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            if len(critic_history) > 1:
                pts = []
                for idx, val in enumerate(critic_history):
                    pt_x = plot_x0 + int(idx * (plot_w / 100.0))
                    pt_y = int(plot_y0 + plot_h / 2.0 - (val / 1.5) * (plot_h / 2.0))
                    pt_y = max(plot_y0, min(plot_y0 + plot_h, pt_y))
                    pts.append((pt_x, pt_y))

                for idx in range(len(pts) - 1):
                    cv2.line(left_panel, pts[idx], pts[idx + 1], (0, 255, 0), 2, cv2.LINE_AA)

            canvas[:, 0:420] = left_panel

            # ----------------- RIGHT PANEL: VAE EYE & TELEMETRY -----------------
            right_panel = np.zeros((1080, 420, 3), dtype=np.uint8)
            cv2.putText(right_panel, "TACTICAL DIAGNOSTICS", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.line(right_panel, (30, 80), (390, 80), (30, 80, 80), 1)

            # VAE Feed reconstruction viewport
            recon_static_np = recon_static.squeeze(0).squeeze(0).cpu().numpy()
            recon_colored = cv2.cvtColor(np.uint8(recon_static_np * 255), cv2.COLOR_GRAY2BGR)
            recon_rescaled = cv2.resize(recon_colored, (256, 256), interpolation=cv2.INTER_CUBIC)

            right_panel[120:376, 82:338] = recon_rescaled
            cv2.rectangle(right_panel, (82, 120), (338, 376), (0, 255, 255), 1)
            cv2.putText(right_panel, "LATENT VISUAL EYE (VAE)", (82, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            # Real-time action selection sliders
            action_labels = ["Up", "Left", "Right", "Idle"]
            bar_offset_y = 480
            cv2.putText(right_panel, "DECISION CONFIDENCE INDEX", (30, bar_offset_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            for i, label in enumerate(action_labels):
                prob = probs[i]
                y_pos = bar_offset_y + i * 45
                color = (0, 255, 0) if i == action_idx else (120, 120, 120)

                cv2.putText(right_panel, f"{label}:", (30, y_pos + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                cv2.rectangle(right_panel, (100, y_pos), (380, y_pos + 15), (15, 15, 15), -1)
                cv2.rectangle(right_panel, (100, y_pos), (380, y_pos + 15), (40, 40, 40), 1)

                prob_len = int(prob * 280)
                cv2.rectangle(right_panel, (100, y_pos), (100 + prob_len, y_pos + 15), color, -1)
                cv2.putText(right_panel, f"{prob * 100:.1f}%", (310, y_pos - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

            canvas[:, 1500:1920] = right_panel

            # Draw outer border separating HUD panels
            cv2.line(canvas, (420, 0), (420, 1080), (30, 50, 50), 2)
            cv2.line(canvas, (1500, 0), (1500, 1080), (30, 50, 50), 2)

            # ----------------- DISPLAY INTERACTIVE LOOPS -----------------
            cv2.imshow("WIDESCREEN PPO AGENT SCANNER HUD", canvas)

            # Step clock control
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                print("[INFO] Scanner interrupted.")
                break

            # Execute active frame updates
            obs, reward, terminated, truncated, info = env.step(action_idx)
            run_rewards.append(reward)
            run_score = info.get("score", run_score)

            # Terminal trigger scorecard visualization
            if terminated or truncated:
                print(f"[EVENT] Trajectory complete. Rendering metrics overlay...")

                actions_arr = np.array(run_actions_taken)
                action_counts = [np.sum(actions_arr == i) for i in range(4)]
                action_pcts = [100.0 * count / len(run_actions_taken) for count in action_counts]

                run_stats = {
                    'score': run_score,
                    'steps': run_steps,
                    'avg_val': np.mean(run_values),
                    'total_rew': np.sum(run_rewards),
                    'entropy': np.mean(run_entropies),
                    'action_pcts': action_pcts
                }

                # Overlay metrics card on game view
                draw_scorecard(canvas, run_stats)
                cv2.imshow("WIDESCREEN PPO AGENT SCANNER HUD", canvas)

                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    break

                # Reset
                obs, info = env.reset()
                run_steps = 0
                run_score = 0
                run_values = []
                run_actions_taken = []
                run_entropies = []
                run_rewards = []

    except KeyboardInterrupt:
        print("\n[INFO] Running sequence halted.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()