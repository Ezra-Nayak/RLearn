# --- inspect_dataset.py ---
import os
import glob
import numpy as np
import cv2


def format_panel(img_array, label, is_diff=False):
    """ Scales the 160x160 array to 250x250 BGR and adds text """
    img_uint8 = (img_array * 255).astype(np.uint8)

    if is_diff:
        # Apply a heatmap to the diff so it's easier to see changes
        panel = cv2.applyColorMap(img_uint8, cv2.COLORMAP_JET)
        # If it's pure black (no difference), keep it black instead of dark blue
        panel[img_uint8 == 0] = (0, 0, 0)
    else:
        panel = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)

    panel = cv2.resize(panel, (250, 250), interpolation=cv2.INTER_NEAREST)

    # Add shadow text for readability
    cv2.putText(panel, label, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(panel, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

    return panel


def main():
    print("--- [DATASET INSPECTOR] ---")
    data_dir = "../sim_data"

    files = glob.glob(os.path.join(data_dir, "*.npy"))
    if not files:
        print(f"[ERROR] No .npy files found in '{data_dir}/'")
        return

    print(f"Loading {len(files)} chunk files...")

    dataset = []
    for f in files:
        dataset.extend(np.load(f, allow_pickle=True))

    print(f"[SUCCESS] Loaded {len(dataset)} sequence samples.")
    print("\nControls:")
    print("  'd' -> Next Sample")
    print("  'a' -> Previous Sample")
    print("  'q' -> Quit")

    idx = 0
    while True:
        inputs, target = dataset[idx]

        # Inputs shape is (4, 160, 160). We want to look at T-1 and T+0
        t_minus_1 = inputs[2]
        t_0 = inputs[3]
        t_plus_1 = target

        # Calculate mathematical absolute differences
        diff_prev_current = np.abs(t_minus_1 - t_0)
        diff_current_target = np.abs(t_0 - t_plus_1)

        # Build Top Row (The actual frames)
        p_t_minus_1 = format_panel(t_minus_1, "T-1 (Input)")
        p_t_0 = format_panel(t_0, "T+0 (Last Input)")
        p_t_plus_1 = format_panel(t_plus_1, "T+1 (TARGET)")
        top_row = np.hstack([p_t_minus_1, p_t_0, p_t_plus_1])

        # Build Bottom Row (The differences)
        blank = np.zeros_like(p_t_minus_1)
        p_diff_prev = format_panel(diff_prev_current, "Diff: T-1 vs T+0", is_diff=True)
        p_diff_targ = format_panel(diff_current_target, "Diff: T+0 vs TARGET", is_diff=True)
        bottom_row = np.hstack([blank, p_diff_prev, p_diff_targ])

        # Combine
        grid = np.vstack([top_row, bottom_row])

        # Add footer instructions
        footer = np.zeros((40, grid.shape[1], 3), dtype=np.uint8)
        cv2.putText(footer, f"Sample {idx + 1} / {len(dataset)} | 'a' = Prev | 'd' = Next | 'q' = Quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        ui = np.vstack([grid, footer])

        cv2.imshow("Dataset Inspector", ui)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            idx = (idx + 1) % len(dataset)
        elif key == ord('a'):
            idx = (idx - 1) % len(dataset)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()