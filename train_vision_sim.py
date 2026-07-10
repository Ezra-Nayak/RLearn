import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from torch.optim import Optimizer


# DirectML Compatibility Layer
class DMLAdam(Optimizer):
    """
    DirectML-Compatible Adam Optimizer.
    Replaces the 'lerp' operator (unsupported on Windows DirectML backends) with standard arithmetic.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(DMLAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Replaced .lerp_ with standard mathematical operators
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss


def setup_device():
    try:
        import torch_directml
        if torch_directml.is_available():
            device = torch_directml.device()
            print(f"[SYSTEM] DirectML Acceleration Engaged: {device}")
            return device
    except ImportError:
        pass
    print("[SYSTEM] DirectML not found. Falling back to CPU execution.")
    return torch.device("cpu")


DEVICE = setup_device()


class SimVisionDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "*.npy"))
        self.data = []
        for f in self.files:
            try:
                chunk = np.load(f, allow_pickle=True)
                self.data.extend(chunk)
            except Exception as e:
                print(f"[ERROR] Failed loading chunk {f}: {e}")
        print(f"[LOADER] Dataset established. Loaded {len(self.data)} sequences.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_stack, target_frame = self.data[idx]
        target_frame = np.expand_dims(target_frame, axis=0)  # Shape: (1, 160, 160)
        return torch.FloatTensor(input_stack), torch.FloatTensor(target_frame)


# Vector Quantization Codebook Representation
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs):
        flat_inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)

        # Spherical Normalization
        flat_norm = F.normalize(flat_inputs, p=2, dim=1)
        weight_norm = F.normalize(self.embedding.weight, p=2, dim=1)
        distances = 1.0 - torch.matmul(flat_norm, weight_norm.t())

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # Dead-Code Revival Check
        if self.training:
            usage_map = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
            usage_map.scatter_(1, encoding_indices, 1)
            usage_count = torch.sum(usage_map, dim=0)
            dead_codes = torch.nonzero(usage_count == 0).squeeze(-1)

            if dead_codes.numel() > 0:
                rand_indices = torch.randint(0, flat_inputs.shape[0], (dead_codes.numel(),), device=inputs.device)
                self.embedding.weight.data[dead_codes] = flat_inputs[rand_indices].detach()
                weight_norm = F.normalize(self.embedding.weight, p=2, dim=1)
                distances = 1.0 - torch.matmul(flat_norm, weight_norm.t())
                encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape[0], inputs.shape[2],
                                                                        inputs.shape[3], self.embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity


# Spatial Split-Brain VQ-VAE Architecture
class SpatialVQVAE(nn.Module):
    def __init__(self):
        super(SpatialVQVAE, self).__init__()
        self.num_embeddings = 512
        self.embedding_dim = 64

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=4, stride=2, padding=1),  # 80x80
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 40x40
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 20x20
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 20x20 Bottleneck
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        self.vq_c = VectorQuantizer(self.num_embeddings, self.embedding_dim)
        self.vq_t = VectorQuantizer(self.num_embeddings, self.embedding_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.embedding_dim, 128, kernel_size=3, stride=1, padding=1),  # 20x20
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 40x40
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 80x80
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 160x160
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.encoder(x)
        z_c, z_t = torch.split(h, self.embedding_dim, dim=1)

        quantized_c, vq_loss_c, perplexity_c = self.vq_c(z_c)
        quantized_t, vq_loss_t, perplexity_t = self.vq_t(z_t)

        recon_static = self.decoder(quantized_c)
        pred_next = self.decoder(quantized_t)

        return recon_static, pred_next, vq_loss_c, vq_loss_t, perplexity_c, perplexity_t, quantized_c, quantized_t


def sobel_loss(pred, target, device):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

    edge_x_p = F.conv2d(pred, kx, padding=1)
    edge_y_p = F.conv2d(pred, ky, padding=1)
    edge_p = torch.sqrt(edge_x_p ** 2 + edge_y_p ** 2 + 1e-6)

    edge_x_t = F.conv2d(target, kx, padding=1)
    edge_y_t = F.conv2d(target, ky, padding=1)
    edge_t = torch.sqrt(edge_x_t ** 2 + edge_y_t ** 2 + 1e-6)

    return F.l1_loss(edge_p, edge_t, reduction='none')


def verify_and_save_visuals(model, val_loader, epoch):
    model.eval()
    os.makedirs("sim_logs", exist_ok=True)
    with torch.no_grad():
        inputs, targets = next(iter(val_loader))
        inputs = inputs.to(DEVICE)
        recon, pred, _, _, _, _, _, _ = model(inputs)

        # Plot comparison: Input, Reconstruction, Prediction, Target
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        axs[0, 0].imshow(inputs[0, 3].cpu().numpy(), cmap='gray')
        axs[0, 0].set_title("Input Frame (t)")
        axs[0, 1].imshow(recon[0, 0].cpu().numpy(), cmap='gray')
        axs[0, 1].set_title("Reconstructed Frame")
        axs[1, 0].imshow(pred[0, 0].cpu().numpy(), cmap='gray')
        axs[1, 0].set_title("Predicted Frame (t+1)")
        axs[1, 1].imshow(targets[0, 0].cpu().numpy(), cmap='gray')
        axs[1, 1].set_title("Target Frame")

        for ax in axs.flat:
            ax.axis('off')

        plt.suptitle(f"Verification - Epoch {epoch}")
        plt.tight_layout()
        plt.savefig(f"sim_logs/verify_epoch_{epoch}.png")
        plt.close()


def main():
    os.makedirs("checkpoints", exist_ok=True)
    dataset = SimVisionDataset("sim_data")
    dataset_size = len(dataset)

    if dataset_size < 1000:
        print("[WARNING] Low dataset volume. Generate more data to prevent training anomalies.")

    # Split dataset into training and validation sets
    val_size = int(dataset_size * 0.1)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=min(128, val_size), shuffle=False, drop_last=False)

    model = SpatialVQVAE().to(DEVICE)
    optimizer = DMLAdam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    print(f"[SYSTEM] Starting VAE training for 50 epochs on {DEVICE}...")
    best_loss = float('inf')

    for epoch in range(1, 51):
        model.train()
        total_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            current_frame = inputs[:, 3:4, :, :]

            # Foveated Mask Generation
            # FIX 1: Raised threshold and lowered multiplier to prevent the
            # scrolling background from triggering the foveated attention.
            motion_diff = torch.abs(inputs[:, 3:4, :, :] - inputs[:, 2:3, :, :])
            dynamic_mask = (motion_diff > 0.15).float() * 2.0 + 1.0

            center_bias = torch.ones_like(current_frame)
            center_bias[:, :, 50:140, 50:110] = 1.5
            master_mask = dynamic_mask * center_bias

            recon, pred, vq_loss_c, vq_loss_t, _, _, _, _ = model(inputs)

            # Reconstruction Loss (Context Brain)
            # Safe to use Sobel here because the target is the present frame (certainty).
            l1_c = F.l1_loss(recon, current_frame, reduction='none')
            edge_c = sobel_loss(recon, current_frame, DEVICE)
            loss_context = torch.mean((l1_c + 0.5 * edge_c) * master_mask)

            # Prediction Loss (Trend Brain)
            # FIX 2: Removed Sobel Loss from the prediction path.
            # Penalizing edge differences on an uncertain scrolling background prevents gray-blob mode collapse.
            l1_t = F.l1_loss(pred, targets, reduction='none')
            loss_trend = torch.mean(l1_t * master_mask)

            loss = loss_context + (2.0 * loss_trend) + vq_loss_c + vq_loss_t

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch:02d}/50 | Train Loss: {train_loss:.4f}")

        if epoch % 5 == 0 or epoch == 1:
            verify_and_save_visuals(model, val_loader, epoch)

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), "checkpoints/sim_vae_best.pth")

    print("[SUCCESS] Training complete. Saved best model weights to checkpoints/sim_vae_best.pth")


if __name__ == "__main__":
    main()