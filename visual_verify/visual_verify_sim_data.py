import numpy as np
import cv2
import os
import glob
import random


def main():
    data_dir = "../sim_data"
    files = glob.glob(os.path.join(data_dir, "*.npy"))

    if not files:
        print(f"[ERROR] No data found in {data_dir}. Run data collection first.")
        return

    print(f"[INFO] Found {len(files)} chunks. Loading a random chunk for verification...")

    # Load a random file to see different parts of the dataset
    target_file = random.choice(files)
    data = np.load(target_file, allow_pickle=True)

    print(f"[INFO] Loaded {target_file}")
    print(f"[INFO] Samples in chunk: {len(data)}")
    print("[CONTROLS] Press any key for NEXT sample | Press 'q' to QUIT")

    for i, (input_stack, target_frame) in enumerate(data):
        # input_stack shape: (4, 160, 160)
        # target_frame shape: (160, 160)

        # 1. Convert normalized floats (0-1) to uint8 (0-255) for display
        stack_imgs = [(f * 255).astype(np.uint8) for f in input_stack]
        target_img = (target_frame * 255).astype(np.uint8)

        # 2. Create labels
        def add_label(img, text):
            # Convert to BGR so we can use colored text
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.putText(img_bgr, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            return img_bgr

        # Label the inputs t-3 down to t
        labeled_inputs = [add_label(stack_imgs[j], f"t-{3 - j}") for j in range(4)]
        # Label the future target t+1
        labeled_target = add_label(target_img, "TARGET (t+1)")

        # 3. Create a layout
        # Top Row: [t-3] [t-2] [t-1] [t]
        top_row = np.hstack(labeled_inputs)

        # Bottom Row: [Target] [Empty] [Empty] [Empty]
        empty = np.zeros((160, 160, 3), dtype=np.uint8)
        bottom_row = np.hstack([labeled_target, empty, empty, empty])

        # Full Dashboard
        dashboard = np.vstack([top_row, bottom_row])

        # 4. Display
        cv2.imshow("Dataset Verification", dashboard)

        key = cv2.waitKey(0)
        if key & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print("[INFO] Verification closed.")


if __name__ == "__main__":
    main()