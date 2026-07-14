# --- debug_canvas_black_bar.py ---
import os
import glob
import numpy as np
import matplotlib.pyplot as plt


def analyze_dataset(data_dir="sim_data"):
    print("[DEBUG] Scanning directory:", data_dir)
    files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
    if not files:
        print(f"[ERROR] No dataset files found in {data_dir}. Please run data collection first.")
        return

    print(f"[DEBUG] Found {len(files)} data chunks. Loading up to 3 chunks for analysis...")

    frames = []
    for f in files[:3]:
        try:
            chunk = np.load(f, allow_pickle=True)
            for seq in chunk:
                # Each sequence in dataset is (input_stack, target_frame)
                # input_stack shape is (4, 160, 160)
                input_stack, target_frame = seq
                frames.append(input_stack[3])  # Frame at t
                frames.append(target_frame)  # Frame at t+1
        except Exception as e:
            print(f"[ERROR] Failed to read chunk {f}: {e}")

    if not frames:
        print("[ERROR] No valid frames loaded.")
        return

    frames = np.array(frames, dtype=np.float32)
    num_frames, height, width = frames.shape
    print(f"[DEBUG] Successfully loaded {num_frames} frames of shape ({height}x{width}).")

    # Calculate row-wise statistics across all loaded frames and columns
    row_means = np.mean(frames, axis=(0, 2))
    row_stds = np.std(frames, axis=(0, 2))

    # Detect constant (dead) rows with zero variance
    constant_rows = []
    for r in range(height):
        if row_stds[r] < 1e-5:  # Extremely low standard deviation
            constant_rows.append((r, row_means[r]))

    print("\n" + "=" * 60)
    print("                 DIAGNOSTIC REPORT: VISUAL DEAD ZONES")
    print("=" * 60)
    print(f"Total rows analyzed: {height}")

    groups = []
    if constant_rows:
        print(f"Detected {len(constant_rows)} constant/dead rows:")
        start_r, val = constant_rows[0]
        prev_r = start_r
        for r, v in constant_rows[1:]:
            if r == prev_r + 1:
                prev_r = r
            else:
                groups.append((start_r, prev_r, val))
                start_r = r
                prev_r = r
                val = v
        groups.append((start_r, prev_r, val))

        for start, end, val in groups:
            num_dead = end - start + 1
            percent = (num_dead / height) * 100
            print(f"  * Rows {start:03d} to {end:03d} ({num_dead} rows, {percent:.2f}% of screen) are dead.")
            print(f"    - Constant Value: {val:.4f} (0.0000 = Pitch Black)")
    else:
        print("  * No dead rows detected! Every vertical row has active pixel variations.")

    # Render assessment
    print("\n" + "-" * 60)
    print("                 GRID LAYOUT ASSESSMENT")
    print("-" * 60)

    has_top_dead_zone = any(start == 0 and end == 29 for start, end, _ in groups)
    if has_top_dead_zone:
        print("[CONFIRMED] The top 30 rows are completely unused, dead black pixels.")
        print("Impact: The VAE is wasting 18.75% of its latent capacity reconstructing zeros.")
        print("\nRecommendations to recover VAE capacity:")
        print("  Adjust `crossy_gym_env.py` to utilize 100% of the active vertical pixels:")
        print("    1. Increase the vertical viewport tracking window to 16 rows:")
        print("       Change: z_top = z_bottom + 15  (instead of +12)")
        print("    2. Remove the row_start offset so rendering begins at row 0:")
        print("       Change: row_start = v_row * 10 (instead of 30 + v_row * 10)")
        print("    3. Change loop bounds to iterate 16 rows:")
        print("       Change: for v_row in range(16): (instead of range(13))")
    else:
        print("[OK] No vertical rendering misalignment detected.")

    print("=" * 60)

    # Generate diagnostic plot
    plt.figure(figsize=(10, 6))
    plt.plot(row_means, label='Row Mean Intensity', color='blue', lw=2)
    plt.fill_between(range(height), row_means - row_stds, row_means + row_stds,
                     alpha=0.2, color='blue', label='Row Std Dev (Variance)')
    plt.axvline(x=29, color='red', linestyle='--', label='HUD Boundary (Row 29)')
    plt.title('Dataset Vertical Row Intensity Distribution')
    plt.xlabel('Row Index (0 = Top, 159 = Bottom)')
    plt.ylabel('Normalized Intensity (0.0=Black, 1.0=White)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    report_img = "debug_black_bar_report.png"
    plt.savefig(report_img, dpi=150)
    print(f"\n[SUCCESS] Visual report plot saved to: {report_img}")


if __name__ == "__main__":
    analyze_dataset()