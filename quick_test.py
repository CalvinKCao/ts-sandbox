# comprehensive_test.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import random
import math
import time

# --------------------------
# Config / seeds
# --------------------------
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

D = 128            # activation dimension
K = 128            # SAE latent size
N = 60000          # dataset size
rank = 4           # LoRA rank
inject_frac = 0.10 # fraction of samples that get the LoRA update (concept-specific)
batch = 512
epochs = 30
alpha_list = [1e-4, 1e-3, 5e-3]  # sparsity penalties to try
scale_list = [0.5, 1.0, 2.0]      # LoRA scaling
perm_tests = 50     # number of permutations for null test
topk_proj = 4       # how many SVD components to project onto

# --------------------------
# Synthetic base activations
# --------------------------
X = torch.randn(N, D) * 1.0  # gaussian activations, shape (N, D)
X = X.to(device)

# --------------------------
# LoRA creation function
# --------------------------
def make_lora_matrices(rank, D, scale_init=0.3):
    A = torch.randn(rank, D) * scale_init
    B = torch.randn(D, rank) * scale_init
    return A, B

def apply_lora(x, A, B, scale=1.0):
    # x: (..., D)
    # returns x + scale * (x @ A.T) @ B.T
    return x + scale * (x @ A.t()) @ B.t()

# --------------------------
# SAE definition
# --------------------------
class SAE(nn.Module):
    def __init__(self, D, K):
        super().__init__()
        self.enc1 = nn.Linear(D, 4*D)
        self.enc2 = nn.Linear(4*D, K)
        self.dec1 = nn.Linear(K, 4*D)
        self.dec2 = nn.Linear(4*D, D)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.enc1(x))
        z = self.relu(self.enc2(h))    # nonnegative sparse code
        h2 = self.relu(self.dec1(z))
        xhat = self.dec2(h2)
        return xhat, z
    def encode(self, x):
        with torch.no_grad():
            h = F.relu(self.enc1(x))
            z = F.relu(self.enc2(h))
        return z
    def decode(self, z):
        with torch.no_grad():
            h2 = F.relu(self.dec1(z))
            xhat = self.dec2(h2)
        return xhat

# --------------------------
# Utility metrics
# --------------------------
def cosine(a, b, eps=1e-8):
    # a, b are 1D tensors
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp(min=eps)
    return (a @ b) / denom

def svd_topk(X_np, k):
    # X_np: (n_samples, D)
    U, S, Vt = np.linalg.svd(X_np, full_matrices=False)
    return U[:, :k], S[:k], Vt[:k, :]

def fit_linear_map(Z, X):
    # Solve for W in R^{D x K} minimizing ||X - W Z||_F
    # Z: (n, K), X: (n, D)
    # W = X.T @ Z @ inv(Z.T @ Z + eps I)
    ZtZ = Z.T @ Z
    eps = 1e-6
    inv = np.linalg.inv(ZtZ + eps * np.eye(ZtZ.shape[0]))
    W = (X.T @ Z) @ inv  # (D, K)
    return W

# --------------------------
# Main experiment loop
# --------------------------
results = []

for alpha in alpha_list:
    for scale in scale_list:
        t0 = time.time()
        print("\n=== alpha=%.1e  scale=%.2f ===" % (alpha, scale))
        # new random LoRA for each run
        A, B = make_lora_matrices(rank, D, scale_init=0.3)
        A = A.to(device)
        B = B.to(device)

        # build merged dataset with mask
        mask = (torch.rand(N) < inject_frac)
        X_merged = X.clone()
        # apply LoRA only to masked indices
        if mask.any():
            X_merged[mask] = apply_lora(X[mask], A, B, scale=scale)

        # create dataloader for training on base X
        ds = TensorDataset(X)
        loader = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)

        # init SAE
        model = SAE(D, K).to(device)
        opt = optim.Adam(model.parameters(), lr=1e-3)
        mse = nn.MSELoss()

        # train SAE on base only
        for ep in range(epochs):
            model.train()
            epoch_loss = 0.0
            count = 0
            for (xb,) in loader:
                xb = xb.to(device)
                xhat, z = model(xb)
                # L1 on z: mean over batch, scale by K so values comparable across K
                l1 = z.abs().mean() * K
                loss = mse(xhat, xb) + alpha * l1
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * xb.size(0)
                count += xb.size(0)
            if (ep+1) % 10 == 0 or ep == 0:
                print(f" epoch {ep:2d} loss {epoch_loss/count:.6f}")

        model.eval()

        # encode all
        with torch.no_grad():
            Z_base = model.encode(X.to(device)).cpu().numpy()        # (N, K)
            Z_merged = model.encode(X_merged.to(device)).cpu().numpy()# (N, K)
            # compute reconstructions (if needed)
            # recon = model.decode(torch.from_numpy(Z_base).to(device)).cpu().numpy()

        # compute delta_x and delta_z per sample
        delta_X = (X_merged - X).cpu().numpy()     # (N, D)
        delta_Z = Z_merged - Z_base                # (N, K)

        # indices
        inj_idx = np.where(mask.numpy())[0]
        noninj_idx = np.where(~mask.numpy())[0]
        n_inj = len(inj_idx)
        print(" injected samples:", n_inj, " / ", N)

        if n_inj == 0:
            print("No injected samples this run (increase inject_frac). Skipping.")
            continue

        # means
        mean_delta_x_inj = delta_X[inj_idx].mean(axis=0)  # (D,)
        mean_delta_z_inj = delta_Z[inj_idx].mean(axis=0)  # (K,)

        # decode mean delta_z through decoder
        mean_delta_z_inj_t = torch.from_numpy(mean_delta_z_inj).to(device).float()
        recon_from_mean_delta_z = model.decode(mean_delta_z_inj_t).cpu().numpy()

        # primary cosine test
        cos_main = cosine(torch.from_numpy(mean_delta_x_inj), torch.from_numpy(recon_from_mean_delta_z)).item()
        print("Cosine similarity (mean delta_x vs decoder(mean delta_z)):", cos_main)

        # SVD / low-rank check of true delta_X on injected samples
        Ux, Sx, Vtx = svd_topk(delta_X[inj_idx], k=min(50, n_inj))
        svd_explained = (Sx ** 2) / np.sum(delta_X[inj_idx] ** 2)
        print("Top singular values (true Δx) [top 8]:", np.round(Sx[:8], 4))
        print("Explained energy fraction by top %d components: %.4f" % (rank, svd_explained[:rank].sum()))

        # Project recon onto top-k components of Δx
        Vtop = Vtx[:topk_proj, :]  # (k, D)
        proj_coeffs = Vtop @ recon_from_mean_delta_z  # (k,)
        proj_norm = np.linalg.norm(proj_coeffs)
        recon_norm = np.linalg.norm(recon_from_mean_delta_z)
        proj_ratio = proj_norm / (recon_norm + 1e-12)
        print(f"Projection ratio of decoder(mean Δz) onto top-{topk_proj} Δx PCs: {proj_ratio:.4f}")

        # SVD of Δz to see if delta is low-rank in code space
        Uz, Sz, Vtz = svd_topk(delta_Z[inj_idx], k=min(50, n_inj))
        print("Top singular values (Δz) [top 8]:", np.round(Sz[:8], 4))
        svd_explained_z = (Sz ** 2) / np.sum(delta_Z[inj_idx] ** 2)
        print("Explained energy fraction in Δz by top 4:", svd_explained_z[:4].sum())

        # Linear regression: fit W so that Δx ≈ W Δz (fit on injected)
        Zinj = delta_Z[inj_idx]   # (n_inj, K)
        Xinj = delta_X[inj_idx]   # (n_inj, D)
        try:
            W = fit_linear_map(Zinj, Xinj)  # (D, K)
            pred = (Zinj @ W.T)             # (n_inj, D)
            ss_res = np.sum((Xinj - pred) ** 2)
            ss_tot = np.sum((Xinj - Xinj.mean(axis=0)) ** 2)
            r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
            print(f"Linear regression R^2 (Δx ≈ W Δz) on injected samples: {r2:.4f}")
        except np.linalg.LinAlgError as e:
            print("Linear regression failed:", e)
            r2 = float('nan')

        # Null / permutation test for cosine: shuffle mask many times
        null_cos = []
        rng = np.random.RandomState(seed)
        for p in range(perm_tests):
            # pick random subset of same size as injected
            idx = rng.choice(N, size=n_inj, replace=False)
            mean_dx = delta_X[idx].mean(axis=0)
            mean_dz = delta_Z[idx].mean(axis=0)
            recon = model.decode(torch.from_numpy(mean_dz).to(device).float()).cpu().numpy()
            null_cos.append(float(cosine(torch.from_numpy(mean_dx), torch.from_numpy(recon))))
        null_cos = np.array(null_cos)
        mean_null = null_cos.mean()
        std_null = null_cos.std(ddof=1)
        zscore = (cos_main - mean_null) / (std_null + 1e-12)
        print(f"Null cos mean {mean_null:.4f}, std {std_null:.4f}, zscore of observed {zscore:.2f}")

        # add to results
        results.append({
            "alpha": alpha, "scale": scale, "cos_main": cos_main,
            "proj_ratio": proj_ratio, "r2": r2,
            "svd_frac_x_toprank": float(svd_explained[:rank].sum() if len(svd_explained)>rank else np.sum(svd_explained)),
            "svd_frac_z_top4": float(svd_explained_z[:4].sum() if len(svd_explained_z)>4 else np.sum(svd_explained_z)),
            "null_mean": mean_null, "null_std": std_null, "zscore": zscore
        })

        print("elapsed: %.1fs" % (time.time() - t0))

# --------------------------
# summary print
# --------------------------
print("\n=== SUMMARY ===")
for r in results:
    print(r)

print("\nDone.")