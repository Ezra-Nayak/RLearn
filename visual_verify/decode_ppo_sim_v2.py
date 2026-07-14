import os
import glob
import cv2
import torch
import torch.nn.functional as F
import numpy as np

# Parity Imports
from train_ppo_sim import CrossyGymEnv, FrameStackWrapper, SpatialVQVAE, ActorCritic, setup_device, VAE_DEVICE, \
    PPO_DEVICE

# Setup Paths & Hardcoded Toggles
CHECKPOINT_PATH = "../checkpoints/ppo_sim_best.pth"
VAE_CHECKPOINT = "../checkpoints/sim_vae_best.pth"

# Hardcoded Param: Toggle to save frames to file
SAVE_PNGS = False
DECODE_LOG_DIR = "decode_sim_network"


def print_instructions():
    print("=" * 70)
    print(" 🧠 CYBERPUNK PPO DECODER - SCANNER BOOTING... 🧠 ")
    print("=" * 70)
    print("CONTROLS:")
    print("  [SPACE] or [ANY KEY] : Step one frame forward through the rollout.")
    print("  [Q] or [ESC]         : Shut down the scanner.")
    print(f"  SAVE_PNGS            : {SAVE_PNGS} (Saves to '{DECODE_LOG_DIR}')\n")
    print("DASHBOARD LAYOUT:")
    print("  [TOP-LEFT]  - Original Grayscale Viewport [t] (Upscaled sharp)")
    print("  [BSM-LEFT]  - Decision Probabilities & Critic Valuation (Telemetry)")
    print("  [RIGHT]     - Cyber-Synaptic flow highlighting the PREDICTED NEXT MOVE")
    print("=" * 70)


# --- COLOR PALETTE & LATENT EXTRACTION (from vision_verifier.py) ---
_hsv_colors = np.zeros((1, 512, 3), dtype=np.uint8)
for i in range(512):
    _hsv_colors[0, i] = [int((i * 137.5) % 180), 160, 60]
_bgr_colors = cv2.cvtColor(_hsv_colors, cv2.COLOR_HSV2BGR)[0]
COLOR_PALETTE = [tuple(int(c) for c in color) for color in _bgr_colors]


def get_discrete_indices(quant_tensor, embedding_weight):
    """ Extract discrete codebook indices by Euclidean distance to codebook embeddings """
    B, C, H, W = quant_tensor.shape
    flat_quant = quant_tensor.permute(0, 2, 3, 1).reshape(-1, C).cpu()
    emb_weight = embedding_weight.detach().cpu()
    distances = torch.cdist(flat_quant, emb_weight)
    indices = torch.argmin(distances, dim=1)
    return indices.view(H, W).numpy()


class SimulatedBrainScanner:
    def __init__(self, policy):
        self.policy = policy
        self.activations = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()

            return hook

        # Hooks matching train_ppo_sim.py Actor heads
        self.hooks.append(self.policy.actor[1].register_forward_hook(get_activation('actor_tanh1')))
        self.hooks.append(self.policy.actor[4].register_forward_hook(get_activation('actor_tanh2')))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


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

    saliency = latents.grad.abs().squeeze(0).max(dim=0)[0].cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency, action_logits.detach(), best_action.item(), features.detach().cpu().numpy()[0]


def extract_reduced_weights(policy):
    """ Downsamples actor connection matrices for visual graph rendering """
    w1 = policy.actor[0].weight.detach().cpu().numpy()  # L0 -> L1
    w2 = policy.actor[3].weight.detach().cpu().numpy()  # L1 -> L2
    w3 = policy.actor[6].weight.detach().cpu().numpy()  # L2 -> L3

    def reduce_w(w, in_nodes, out_nodes):
        out_chunk = max(1, w.shape[0] // out_nodes)
        in_chunk = max(1, w.shape[1] // in_nodes)
        w_red = np.zeros((out_nodes, in_nodes))
        for o in range(out_nodes):
            for i in range(in_nodes):
                block = w[o * out_chunk:(o + 1) * out_chunk, i * in_chunk:(i + 1) * in_chunk]
                idx = np.unravel_index(np.argmax(np.abs(block)), block.shape)
                w_red[o, i] = block[idx]
        return w_red

    # Compress network into an 8 -> 12 -> 12 -> 4 visualization model
    W01 = reduce_w(w1, 8, 12)
    W12 = reduce_w(w2, 12, 12)
    W23 = reduce_w(w3, 12, 4)
    return [W01, W12, W23]


def get_reduced_acts(features, acts1, acts2, logits):
    """ Downsamples active layer signals for node visualization """

    def reduce_a(a, num_nodes):
        chunk = max(1, len(a) // num_nodes)
        res = [np.mean(np.abs(a[i * chunk:(i + 1) * chunk])) for i in range(num_nodes)]
        res = np.array(res)
        if np.max(res) > 0:
            res = res / np.max(res)
        return res

    a0 = reduce_a(features, 8)
    a1 = reduce_a(acts1, 12)
    a2 = reduce_a(acts2, 12)
    a3 = F.softmax(torch.tensor(logits), dim=0).numpy()
    return [a0, a1, a2, a3]


def render_cyberpunk_dashboard(frame, recon_frame, indices_c, nn_acts, nn_weights, logits, action_idx, saliency):
    """
    Renders the beautiful Bloom/Cyberpunk Synaptic Dashboard.
    Displays Raw Input & VAE Recon on top, and 20x20 Latent Codebook Matrix below.
    """
    H, W = 900, 1600  # 16:9 Cinematic Aspect Ratio
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    bloom = np.zeros((H, W, 3), dtype=np.uint8)  # Additive glow layer

    # ==========================================
    # 1. LEFT PANEL - TOP: RAW MATRIX & VAE RECON WITH SALIENCY
    # ==========================================
    size_box = 256

    # Left: Raw Matrix (Input)
    raw_crisp = cv2.resize(frame, (size_box, size_box), interpolation=cv2.INTER_NEAREST)
    raw_bgr = cv2.cvtColor((raw_crisp * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    canvas[0:size_box, 0:size_box] = raw_bgr

    # Right: VAE Reconstruction
    recon_crisp = cv2.resize(recon_frame, (size_box, size_box), interpolation=cv2.INTER_NEAREST)
    recon_bgr = cv2.cvtColor((recon_crisp * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Saliency Heatmap Overlay (JET Colormap)
    saliency_resized = cv2.resize(saliency, (size_box, size_box), interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap((saliency_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Opacity Masking: High saliency gets more heatmap coloring, low saliency preserves the clear VAE recon
    saliency_mask = saliency_resized[..., np.newaxis]
    blended_recon = (recon_bgr * (1.0 - saliency_mask * 0.7) + heatmap * (saliency_mask * 0.7)).astype(np.uint8)

    canvas[0:size_box, size_box:512] = blended_recon

    # Overlay labels
    cv2.putText(canvas, "RAW INPUT", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
    cv2.putText(canvas, "RAW INPUT", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    cv2.putText(canvas, "VAE RECON + SALIENCY", (size_box + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
    cv2.putText(canvas, "VAE RECON + SALIENCY", (size_box + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # ==========================================
    # 2. LEFT PANEL - MIDDLE: COMPACT ACTION PROBABILITIES
    # ==========================================
    cv2.line(canvas, (0, size_box), (512, size_box), (50, 40, 60), 2)

    probs = F.softmax(logits.squeeze(0), dim=-1).cpu().numpy()
    action_labels = ["UP", "LEFT", "RIGHT", "IDLE"]

    # Compact horizontal probability indicators
    start_y = 270
    for i, (prob, label) in enumerate(zip(probs, action_labels)):
        y_pos = start_y + i * 22
        cv2.putText(canvas, f"{label:5s}:", (15, y_pos + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        cv2.putText(canvas, f"{prob * 100:5.1f}%", (75, y_pos + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        bar_len = int(prob * 100)
        color = (255, 255, 0) if i == action_idx else (80, 70, 90)
        cv2.rectangle(canvas, (130, y_pos + 2), (130 + bar_len, y_pos + 10), color, -1)
        cv2.rectangle(canvas, (130, y_pos + 2), (130 + 100, y_pos + 10), (40, 30, 50), 1)

    # ==========================================
    # 3. LEFT PANEL - BOTTOM: 20x20 LATENT CODEBOOK MATRIX
    # ==========================================
    matrix_y_start = 370
    cv2.line(canvas, (0, matrix_y_start - 10), (512, matrix_y_start - 10), (50, 40, 60), 2)
    cv2.putText(canvas, "CONTEXT CODEBOOK INDICES (20x20)", (15, matrix_y_start + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

    # Grid Cell Dimensions
    cell_w = 24
    cell_h = 24
    grid_start_x = 16  # Center 20 * 24 = 480px inside 512px width
    grid_start_y = matrix_y_start + 25

    grid_h, grid_w = indices_c.shape  # 20x20
    for r in range(grid_h):
        for c in range(grid_w):
            val = int(indices_c[r, c])
            val_str = str(val)

            cx = grid_start_x + c * cell_w
            cy = grid_start_y + r * cell_h

            # Background color from HSV palette
            bg_color = COLOR_PALETTE[val % 512]

            # Draw cell box & subtle dark border
            cv2.rectangle(canvas, (cx, cy), (cx + cell_w, cy + cell_h), bg_color, -1)
            cv2.rectangle(canvas, (cx, cy), (cx + cell_w, cy + cell_h), (30, 30, 30), 1)

            # Center white text integer value
            text_size = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_SIMPLEX, 0.26, 1)[0]
            tx = cx + (cell_w - text_size[0]) // 2
            ty = cy + (cell_h + text_size[1]) // 2
            cv2.putText(canvas, val_str, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.26, (255, 255, 255), 1, cv2.LINE_AA)

    # ==========================================
    # 3. RIGHT PANEL: SYNAPTIC FLOW GRAPH
    # ==========================================
    # Draw faint Cyber Grid background starting from X=512
    for i in range(0, H, 40):
        cv2.line(canvas, (512, i), (W, i), (25, 20, 35), 1)
    for i in range(512, W, 40):
        cv2.line(canvas, (i, 0), (i, H), (25, 20, 35), 1)

    # Centered layering parameters
    x_margin, y_margin, x_step = 612, 100, 280
    nodes_config = [len(a) for a in nn_acts]
    num_layers = len(nodes_config)

    # Map Node Coordinates
    node_pos = []
    for i, size in enumerate(nodes_config):
        layer_x = x_margin + i * x_step
        y_step = (H - 2 * y_margin) / max(1, size)
        y_start = y_margin + (H - 2 * y_margin - y_step * size) / 2
        layer_pos = [(layer_x, int(y_start + j * y_step + y_step / 2)) for j in range(size)]
        node_pos.append(layer_pos)

    # --- ACTIVE FLOW BACKTRACING ---
    # Trace backwards from the NEXT Action decision tree
    active_edges = set()
    active_nodes = {3: [action_idx], 2: [], 1: [], 0: []}

    # L3 <- L2 (Top 3 contributors)
    contrib_2 = np.abs(nn_acts[2] * nn_weights[2][action_idx])
    top_l2 = np.argsort(contrib_2)[-3:]
    active_nodes[2].extend(top_l2)
    for n in top_l2: active_edges.add((2, n, 3, action_idx))

    # L2 <- L1 (Top 2 per active L2)
    for node_l2 in top_l2:
        contrib_1 = np.abs(nn_acts[1] * nn_weights[1][node_l2])
        top_l1 = np.argsort(contrib_1)[-2:]
        active_nodes[1].extend(top_l1)
        for n in top_l1: active_edges.add((1, n, 2, node_l2))

    # L1 <- L0 (Top 1 per active L1)
    for node_l1 in set(active_nodes[1]):
        contrib_0 = np.abs(nn_acts[0] * nn_weights[0][node_l1])
        top_l0 = np.argsort(contrib_0)[-1:]
        active_nodes[0].extend(top_l0)
        for n in top_l0: active_edges.add((0, n, 1, node_l1))

    # --- DRAW WIRES (DROOPING BEZIER) ---
    for l in range(num_layers - 1):
        for i in range(nodes_config[l]):
            for j in range(nodes_config[l + 1]):
                p0, p3 = node_pos[l][i], node_pos[l + 1][j]
                weight = nn_weights[l][j, i]
                is_active = (l, i, l + 1, j) in active_edges

                # Gravity sag droop equation
                dx = p3[0] - p0[0]
                droop = abs(dx) * 0.08  # Taut wires, sagging slightly
                p1 = (int(p0[0] + dx * 0.33), int(p0[1] + droop))
                p2 = (int(p0[0] + dx * 0.66), int(p3[1] + droop))

                # Interpolate Bezier points
                t = np.linspace(0, 1, 25)
                x = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0]
                y = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1]
                curve = np.vstack((x, y)).astype(np.int32).T.reshape((-1, 1, 2))

                if is_active:
                    color = (255, 255, 0) if weight > 0 else (0, 140, 255)  # Cyan positive, Orange negative
                    cv2.polylines(canvas, [curve], False, color, 2, lineType=cv2.LINE_AA)
                    cv2.polylines(bloom, [curve], False, color, 6, lineType=cv2.LINE_AA)
                else:
                    color = (40, 30, 50)  # Thin, dark dormant wires
                    cv2.polylines(canvas, [curve], False, color, 1, lineType=cv2.LINE_AA)

    # --- DRAW NODES ---
    for l in range(num_layers):
        for i, pos in enumerate(node_pos[l]):
            is_active_node = i in active_nodes[l]

            if is_active_node:
                cv2.circle(canvas, pos, 6, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(bloom, pos, 12, (255, 255, 0), -1, cv2.LINE_AA)  # Cyan Bloom
            else:
                cv2.circle(canvas, pos, 4, (30, 30, 30), -1, cv2.LINE_AA)
                cv2.circle(canvas, pos, 4, (80, 80, 80), 1, cv2.LINE_AA)

            # Node Labels on Terminal Layer
            if l == num_layers - 1:
                label = action_labels[i]
                if i == action_idx:
                    cv2.putText(canvas, label, (pos[0] + 15, pos[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2)
                    cv2.putText(bloom, label, (pos[0] + 15, pos[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0),
                                4)
                else:
                    cv2.putText(canvas, label, (pos[0] + 15, pos[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80),
                                1)

    # ==========================================
    # 4. MERGE DYNAMICS & DRAW LABELS
    # ==========================================
    bloom_blurred = cv2.GaussianBlur(bloom, (25, 25), 0)
    final_canvas = cv2.add(canvas, bloom_blurred)

    cv2.putText(final_canvas, "Neural Network [t+1 PREDICT]", (542, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)

    return final_canvas


def main():
    print_instructions()
    if SAVE_PNGS:
        os.makedirs(DECODE_LOG_DIR, exist_ok=True)

    frame_count = 0

    # Initialize environment wrappers
    raw_env = CrossyGymEnv()
    env = FrameStackWrapper(raw_env, stack_size=4)

    # Load pre-trained vision compression
    print(f"[EYES] Loading frozen Spatial VQ-VAE...")
    vae = SpatialVQVAE().to(VAE_DEVICE)
    vae.load_state_dict(torch.load(VAE_CHECKPOINT, map_location=VAE_DEVICE, weights_only=False))
    vae.eval()

    # Load target weights
    print(f"[BRAIN] Loading model weights: '{CHECKPOINT_PATH}'")
    policy = ActorCritic(action_dim=4).to(PPO_DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=PPO_DEVICE, weights_only=False)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['model_state_dict'])
    else:
        policy.load_state_dict(checkpoint)
    policy.eval()

    scanner = SimulatedBrainScanner(policy)
    reduced_weights = extract_reduced_weights(policy)

    obs, info = env.reset()

    try:
        while True:
            # Prepare dimensions for model forwarding
            img_batch = torch.tensor(obs["image"], dtype=torch.float32, device=VAE_DEVICE).unsqueeze(0)
            scalars_batch = torch.tensor(obs["scalars"], dtype=torch.float32, device=PPO_DEVICE).unsqueeze(0)
            masks_batch = torch.tensor(info["action_mask"], dtype=torch.float32, device=PPO_DEVICE).unsqueeze(0)

            # Compute latent compression mapping, VAE Reconstruction, & Codebook Indices
            with torch.no_grad():
                recon_static, _, _, _, _, _, quant_c, quant_t = vae(img_batch)
                latents_batch = torch.cat([quant_c, quant_t], dim=1).to(PPO_DEVICE)
                recon_frame = recon_static.squeeze(0).squeeze(0).cpu().numpy()
                indices_c = get_discrete_indices(quant_c, vae.vq_c.embedding.weight)

            # Saliency pass (extracting action_idx and raw_features)
            saliency, logits, action_idx, raw_features = compute_saliency_map(
                policy, latents_batch, scalars_batch, masks_batch
            )

            # Critic State evaluation
            with torch.no_grad():
                features = policy._get_features(latents_batch, scalars_batch)
                value = policy.critic(features).item()

            # Active activations sampling
            acts1 = scanner.activations['actor_tanh1'].squeeze(0).cpu().numpy()
            acts2 = scanner.activations['actor_tanh2'].squeeze(0).cpu().numpy()
            reduced_acts = get_reduced_acts(raw_features, acts1, acts2, logits.squeeze(0).cpu().numpy())

            # Render Dashboard
            raw_grayscale_frame = obs["image"][3]  # Current viewport frame
            dashboard = render_cyberpunk_dashboard(
                raw_grayscale_frame, recon_frame, indices_c, reduced_acts, reduced_weights, logits, action_idx, saliency
            )

            cv2.imshow("PPO Brain fMRI Scanner (Cinematic)", dashboard)

            if SAVE_PNGS:
                log_frame_path = os.path.join(DECODE_LOG_DIR, f"scan_frame_{frame_count:06d}.png")
                cv2.imwrite(log_frame_path, dashboard)

            frame_count += 1

            # Keypress blocking
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q') or key == 27:  # Quit on 'q' or Escape
                break

            # Advance rollout state
            obs, reward, terminated, truncated, info = env.step(action_idx)
            if terminated or truncated:
                obs, info = env.reset()

    except KeyboardInterrupt:
        pass
    finally:
        scanner.remove_hooks()
        cv2.destroyAllWindows()
        print("[SHUTDOWN] Scanner finalized successfully.")


if __name__ == "__main__":
    main()