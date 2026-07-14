# --- decode_ppo_sim.py ---
import os
import glob
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import time

# Parity Imports
from train_ppo_sim import CrossyGymEnv, FrameStackWrapper, SpatialVQVAE, ActorCritic, setup_device, VAE_DEVICE, \
    PPO_DEVICE

# Setup Paths
CHECKPOINT_PATH = "../checkpoints/ppo_sim_best.pth"
VAE_CHECKPOINT = "../checkpoints/sim_vae_best.pth"
DECODE_LOG_DIR = "logs/decode_sim"


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

    # Calculate absolute max gradients across channels
    saliency = latents.grad.abs().squeeze(0).max(dim=0)[0].cpu().numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    return saliency, action_logits.detach(), best_action.item(), features.detach().cpu().numpy()[0]


def extract_reduced_weights(policy):
    """ Downsamples actor connection matrices for visual graph rendering """
    w1 = policy.actor[0].weight.detach().cpu().numpy()  # (512, 802)
    w2 = policy.actor[3].weight.detach().cpu().numpy()  # (512, 512)
    w3 = policy.actor[6].weight.detach().cpu().numpy()  # (4, 512)

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


def get_bezier_curve(p0, p2, resolution=15):
    p1 = ((p0[0] + p2[0]) // 2, p0[1])
    p3 = ((p0[0] + p2[0]) // 2, p2[1])
    t = np.linspace(0, 1, resolution)
    x = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t ** 2 * p3[0] + t ** 3 * p2[0]
    y = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t ** 2 * p3[1] + t ** 3 * p2[1]
    pts = np.vstack((x, y)).astype(np.int32).T
    return pts.reshape((-1, 1, 2))


def draw_nn_graph(canvas, acts, weights_reduced, best_action):
    h, w, _ = canvas.shape
    nodes_config = [len(a) for a in acts]
    num_layers = len(nodes_config)
    action_labels = ["Up", "Left", "Right", "Idle"]

    node_pos = []
    x_margin = int(w * 0.15)
    y_margin = int(h * 0.15)
    x_step = (w - 2 * x_margin) // (num_layers - 1)

    for i, size in enumerate(nodes_config):
        layer_x = x_margin + i * x_step
        y_step = (h - 2 * y_margin) / size
        layer_pos = [(layer_x, int(y_margin + j * y_step + y_step / 2)) for j in range(size)]
        node_pos.append(layer_pos)

    edges = []
    for l in range(num_layers - 1):
        w_mat = weights_reduced[l]
        act_in = acts[l]
        for i in range(nodes_config[l]):
            for j in range(nodes_config[l + 1]):
                signal = act_in[i] * w_mat[j, i]
                edges.append((signal, node_pos[l][i], node_pos[l + 1][j]))

    edges.sort(key=lambda x: abs(x[0]))
    max_sig = max(1e-8, max(abs(e[0]) for e in edges))

    for signal, p0, p2 in edges:
        norm_sig = signal / max_sig
        if abs(norm_sig) < 0.15: continue

        thickness = max(1, int(abs(norm_sig) * 2))
        color = (255, 255, 0) if norm_sig > 0 else (255, 0, 255)
        intensity = int(abs(norm_sig) * 255)
        dimmed_color = (color[0] * intensity // 255, color[1] * intensity // 255, color[2] * intensity // 255)

        curve = get_bezier_curve(p0, p2)
        cv2.polylines(canvas, [curve], False, dimmed_color, thickness, lineType=cv2.LINE_AA)

    for l in range(num_layers):
        for i, pos in enumerate(node_pos[l]):
            act_val = acts[l][i]
            intensity = int(act_val * 255)
            node_color = (intensity, intensity, intensity)

            if l == num_layers - 1:
                if best_action == i:
                    node_color = (0, 255, 0)
                    cv2.putText(canvas, action_labels[i], (pos[0] + 10, pos[1] + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                else:
                    cv2.putText(canvas, action_labels[i], (pos[0] + 10, pos[1] + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

            cv2.circle(canvas, pos, 4, node_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, pos, 4, (200, 200, 200), 1, lineType=cv2.LINE_AA)


def render_dashboard(frame, saliency, nn_acts, nn_weights, logits, value, action_idx):
    PANEL_SIZE = 320

    # 1. Base Grayscale Input -> Scaled RGB Canvas
    frame_resized = cv2.resize(frame, (PANEL_SIZE, PANEL_SIZE))

    # 2. Resized attention overlay
    saliency_resized = cv2.resize(saliency, (PANEL_SIZE, PANEL_SIZE))
    heatmap = cv2.applyColorMap(np.uint8(255 * saliency_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame_resized, 0.5, heatmap, 0.5, 0)

    # 3. Graph display panel
    graph_panel = np.zeros((PANEL_SIZE, PANEL_SIZE, 3), dtype=np.uint8)
    draw_nn_graph(graph_panel, nn_acts, nn_weights, action_idx)

    # 4. Telemetry stats panel
    telemetry = np.zeros((PANEL_SIZE, PANEL_SIZE, 3), dtype=np.uint8)
    probs = F.softmax(logits.squeeze(0), dim=-1).cpu().numpy()
    actions = ["Up", "Left", "Right", "Idle"]
    colors = [(0, 255, 0) if i == action_idx else (150, 150, 150) for i in range(4)]

    cv2.putText(telemetry, "DECISION DISTRIBUTION:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for i, (p, act) in enumerate(zip(probs, actions)):
        bar_len = int(p * 140)
        y_offset = 55 + (i * 35)
        cv2.rectangle(telemetry, (10, y_offset), (10 + bar_len, y_offset + 18), colors[i], -1)
        cv2.putText(telemetry, f"{act}: {p:.2f}", (160, y_offset + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[i], 1)

    cv2.putText(telemetry, "CRITIC STATE VALUATION:", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(telemetry, f"V_s: {value:+.4f}", (10, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    # Compile dashboard layout
    top_row = np.hstack([frame_resized, overlay])
    bottom_row = np.hstack([graph_panel, telemetry])
    dashboard = np.vstack([top_row, bottom_row])

    # Canvas annotations
    cv2.putText(dashboard, "1. Input Vision (t)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(dashboard, "2. Saliency Overlay (Attention)", (PANEL_SIZE + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1)
    cv2.putText(dashboard, "3. Synaptic Graph (Actor)", (10, PANEL_SIZE + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1)
    cv2.putText(dashboard, "4. Telemetry Diagnostics", (PANEL_SIZE + 10, PANEL_SIZE + 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1)

    return dashboard


def main():
    print("[SYSTEM] Starting Vectorized Simulator fMRI brain scans...")
    os.makedirs(DECODE_LOG_DIR, exist_ok=True)
    frame_count = 0

    # Initialize environment
    raw_env = CrossyGymEnv()
    env = FrameStackWrapper(raw_env, stack_size=4)

    # Initialize models
    print(f"[EYES] Loading frozen Spatial VQ-VAE...")
    vae = SpatialVQVAE().to(VAE_DEVICE)
    vae.load_state_dict(torch.load(VAE_CHECKPOINT, map_location=VAE_DEVICE, weights_only=False))
    vae.eval()

    print(f"[BRAIN] Loading model weights from checkpoint: '{CHECKPOINT_PATH}'")
    policy = ActorCritic(action_dim=4).to(PPO_DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=PPO_DEVICE, weights_only=False)

    # Handle dictionary payload
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        policy.load_state_dict(checkpoint['model_state_dict'])
    else:
        policy.load_state_dict(checkpoint)

    policy.eval()

    scanner = SimulatedBrainScanner(policy)
    reduced_weights = extract_reduced_weights(policy)

    obs, info = env.reset()

    print("\nControls:")
    print("  'q' -> Interrupt Scan")
    print("  [Any other key] -> Step Simulation forward\n")

    try:
        while True:
            # Package inputs
            img_batch = torch.tensor(obs["image"], dtype=torch.float32, device=VAE_DEVICE).unsqueeze(0)
            scalars_batch = torch.tensor(obs["scalars"], dtype=torch.float32, device=PPO_DEVICE).unsqueeze(0)
            masks_batch = torch.tensor(info["action_mask"], dtype=torch.float32, device=PPO_DEVICE).unsqueeze(0)

            # Extract quantized spatial representations from the VAE
            with torch.no_grad():
                _, _, _, _, _, _, quant_c, quant_t = vae(img_batch)
                latents_batch = torch.cat([quant_c, quant_t], dim=1).to(PPO_DEVICE)

            # Compute Saliency and forward values
            saliency, logits, action_idx, raw_features = compute_saliency_map(policy, latents_batch, scalars_batch,
                                                                              masks_batch)

            with torch.no_grad():
                features = policy._get_features(latents_batch, scalars_batch)
                value = policy.critic(features).item()

            # Map neural activations to the graph layers
            acts1 = scanner.activations['actor_tanh1'].squeeze(0).cpu().numpy()
            acts2 = scanner.activations['actor_tanh2'].squeeze(0).cpu().numpy()
            reduced_acts = get_reduced_acts(raw_features, acts1, acts2, logits.squeeze(0).cpu().numpy())

            # Render
            raw_grayscale_frame = obs["image"][3]
            color_frame = cv2.cvtColor(np.uint8(raw_grayscale_frame * 255), cv2.COLOR_GRAY2BGR)
            dashboard = render_dashboard(color_frame, saliency, reduced_acts, reduced_weights, logits, value,
                                         action_idx)

            cv2.imshow("PPO Brain fMRI Scanner (Simulation)", dashboard)

            # Save visual log to file
            log_frame_path = os.path.join(DECODE_LOG_DIR, f"scan_frame_{frame_count:06d}.png")
            cv2.imwrite(log_frame_path, dashboard)
            frame_count += 1

            # Wait for key press to advance manually
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                print("[INFO] Interrupting scan.")
                break

            # Execute decided action in environment
            obs, reward, terminated, truncated, info = env.step(action_idx)
            if terminated or truncated:
                print("[EVENT] Agent died. Resetting scan sequence.")
                obs, info = env.reset()

    except KeyboardInterrupt:
        print("\n[INFO] Run halted.")
    finally:
        scanner.remove_hooks()
        cv2.destroyAllWindows()
        print(f"[SUCCESS] Scans finalized. Saved {frame_count} sequential records in: '{DECODE_LOG_DIR}'")


if __name__ == "__main__":
    main()