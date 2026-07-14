# --- verify_vae_sim.py ---
import os
import glob
import cv2
import torch
import numpy as np
from train_ppo_sim import CrossyGymEnv, FrameStackWrapper, SpatialVQVAE, setup_device, VAE_DEVICE, VAE_CHECKPOINT


def main():
    print("--- [DIAGNOSTIC] VAE Visual Verification ---")
    if not os.path.exists(VAE_CHECKPOINT):
        print(f"[ERROR] VAE Checkpoint missing at '{VAE_CHECKPOINT}'. Cannot run verification.")
        return

    # Load frozen VAE
    device = setup_device()
    vae = SpatialVQVAE().to(device)
    vae.load_state_dict(torch.load(VAE_CHECKPOINT, map_location=device, weights_only=False))
    vae.eval()
    print(f"[SUCCESS] Loaded VAE onto {device}")

    raw_env = CrossyGymEnv()
    env = FrameStackWrapper(raw_env, stack_size=4)
    obs, info = env.reset()

    print("\nControls:")
    print("  [Spacebar] -> Pause/Resume Autoplay")
    print("  'q'        -> Exit Diagnostic\n")

    autoplay = True
    delay = 150  # Delay in milliseconds

    while True:
        # Prepare inputs for the network
        image_batch = torch.tensor(obs["image"], dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            # forward output shapes: (1, 1, 160, 160)
            recon, pred, _, _, _, _, _, _ = vae(image_batch)

        # Extract frames
        img_input = obs["image"][3]  # Current input frame (t)
        img_recon = recon[0, 0].cpu().numpy()
        img_pred = pred[0, 0].cpu().numpy()

        # Calculate metrics
        mae = np.mean(np.abs(img_input - img_recon))
        in_variance = np.var(img_input)
        recon_variance = np.var(img_recon)
        variance_ratio = (recon_variance / (in_variance + 1e-8)) * 100.0

        # Construct visual panels
        vis_input = cv2.cvtColor(np.uint8(img_input * 255), cv2.COLOR_GRAY2BGR)
        vis_recon = cv2.cvtColor(np.uint8(img_recon * 255), cv2.COLOR_GRAY2BGR)
        vis_pred = cv2.cvtColor(np.uint8(img_pred * 255), cv2.COLOR_GRAY2BGR)

        # Absolute Error Heatmap
        error_diff = np.abs(img_input - img_recon)
        error_heatmap = cv2.applyColorMap(np.uint8(error_diff * 255), cv2.COLORMAP_JET)

        # Text descriptors
        cv2.putText(vis_input, "1. Input (t)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(vis_recon, "2. Reconstruction", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 100), 1)
        cv2.putText(vis_pred, "3. Prediction (t+1)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
        cv2.putText(error_heatmap, "4. L1 Error Heatmap", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Assemble Panels
        top_row = np.hstack([vis_input, vis_recon])
        bottom_row = np.hstack([vis_pred, error_heatmap])
        dashboard = np.vstack([top_row, bottom_row])

        # Double canvas scale for display
        display = cv2.resize(dashboard, (640, 640), interpolation=cv2.INTER_NEAREST)

        # Telemetry Bar
        cv2.rectangle(display, (0, 0), (640, 45), (10, 10, 10), -1)
        cv2.putText(display, f"MAE: {mae:.4f} | Variance Preserved: {variance_ratio:.1f}%", (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("VAE Visual Diagnostic", display)

        # Loop timing and key evaluation
        key = cv2.waitKey(delay if autoplay else 0) & 0xFF

        if key == ord(' '):
            autoplay = not autoplay
            print(f"[INFO] Autoplay: {'RESUMED' if autoplay else 'PAUSED'}")
        elif key == ord('q'):
            print("[INFO] Exiting VAE diagnostic.")
            break

        if autoplay:
            # Step randomly
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()

    cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    main()