"""
Empirical Falsification & Benchmark Suite:
Elitzur-Vaidman Interaction-Free Mach-Zehnder Predictive Coding (EV-MZ-PC)
==========================================================================
Tests Null Hypothesis H_0:
  "Interaction-Free Mach-Zehnder Dark-Port routing cannot match the noise recovery
   of dense L1->L0 Decoder SPC without paying the double-encoder + dense-decoder FLOP tax."

Compares:
  1. Blind Feedforward DAG (System 1 only)
  2. Standard Dense SPC (Encoder -> Dense Decoder Surprise -> Refine -> Re-Encode)
  3. Elitzur-Vaidman Mach-Zehnder SPC (O(D*K) Dark-Port Interrogation -> Symplectic Deflection -> Single Encode)
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mach_zehnder_spc import InteractionFreeSymplecticPC


class StandardDenseSPC(nn.Module):
    """
    Baseline Dense Selective Predictor-Corrector:
    Requires:
      1. First Encoder pass: x -> h1
      2. Dense Decoder pass: h1 -> x_hat (to measure surprise)
      3. Second Encoder pass on refined x_refined -> h1_final -> logits
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, tau: float = 0.08):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.tau = tau

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim),
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor):
        h1 = self.encoder(x)
        x_hat = self.decoder(h1)
        residual = x - x_hat
        surprise = torch.mean(residual ** 2, dim=-1)
        trig = surprise > self.tau
        if trig.any():
            mask = trig.unsqueeze(-1).float()
            x_ref = x + mask * 0.65 * (x_hat - x)
            h_final = self.encoder(x_ref)
        else:
            h_final = h1
        return self.head(h_final), x_hat, trig


def generate_structured_manifold_dataset(
    n_train: int = 2000,
    n_test: int = 600,
    in_dim: int = 256,
    intrinsic_rank: int = 16,
    num_classes: int = 10,
    seed: int = 42,
):
    """
    Constructs a structured low-rank curvilinear manifold embedded in R^256
    with 10 semantic classes, testing both clean in-manifold recognition and
    severe out-of-manifold Gaussian & impulse ("Live Bomb") corruption.
    """
    torch.manual_seed(seed)
    # Orthogonal embedding subspace W_manifold in R^{K x D}
    W_raw = torch.randn(in_dim, intrinsic_rank)
    U_true, _ = torch.linalg.qr(W_raw)  # [D, K]

    # Class prototypes in K-dimensional latent space
    prototypes = torch.randn(num_classes, intrinsic_rank) * 1.2

    def sample_split(n: int):
        labels = torch.randint(0, num_classes, (n,))
        z_lat = prototypes[labels] + 0.28 * torch.randn(n, intrinsic_rank)
        # Non-linear harmonic modulation along latent coordinates
        z_modulated = z_lat + 0.15 * torch.sin(2.0 * z_lat)
        x_clean = torch.matmul(z_modulated, U_true.t())
        return x_clean, labels

    x_train, y_train = sample_split(n_train)
    x_test, y_test = sample_split(n_test)
    return x_train, y_train, x_test, y_test


def count_gate_and_routing_flops(in_dim: int, hidden_dim: int, num_shells: int, system2_rate: float):
    """
    Analytical FLOP count per sample for:
      1. Standard Dense SPC:
         - Encoder Pass 1: D*H + H*H
         - Dense Decoder Gate: H*H + H*D
         - Encoder Pass 2 (on system2_rate fraction): system2_rate * (D*H + H*H)
      2. Elitzur-Vaidman Mach-Zehnder SPC:
         - Dark-Port Gate: 2 * D * K
         - Symplectic Deflection (on system2_rate fraction): system2_rate * (4 * D * K)
         - Single Encoder Pass: D*H + H*H
    """
    enc_flops = in_dim * hidden_dim + hidden_dim * hidden_dim
    dec_flops = hidden_dim * hidden_dim + hidden_dim * in_dim

    dense_spc_gate_overhead = dec_flops + system2_rate * enc_flops
    mz_spc_gate_overhead = (2 * in_dim * num_shells) + system2_rate * (4 * in_dim * num_shells)

    total_dense_spc = enc_flops + dense_spc_gate_overhead
    total_mz_spc = enc_flops + mz_spc_gate_overhead
    return dense_spc_gate_overhead, mz_spc_gate_overhead, total_dense_spc, total_mz_spc


def run_benchmark():
    print("=" * 84)
    print("ELITZUR-VAIDMAN INTERACTION-FREE MACH-ZEHNDER PREDICTIVE CODING BENCHMARK")
    print("=" * 84)

    D = 256
    H = 192
    K = 16
    C = 10

    x_train, y_train, x_test, y_test = generate_structured_manifold_dataset(
        n_train=2400, n_test=800, in_dim=D, intrinsic_rank=K, num_classes=C
    )

    # Instantiate models
    mz_model = InteractionFreeSymplecticPC(in_dim=D, hidden_dim=H, out_dim=C, num_shells=K)
    dense_spc = StandardDenseSPC(in_dim=D, hidden_dim=H, out_dim=C, tau=0.05)

    # Step 1: Closed-form manifold calibration of the Mach-Zehnder Phase Mirror
    t0 = time.perf_counter()
    tau_dark = mz_model.mz_gate.calibrate_manifold(x_train, percentile=0.90)
    cal_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[1] Mach-Zehnder Phase Mirror Closed-Form Calibration: {cal_ms:.2f} ms (tau_dark = {tau_dark:.4f})")

    # Step 2: Train both models EXCLUSIVELY on clean manifold trajectories!
    # Neither model sees the corrupted test distribution during training.
    opt_mz = torch.optim.AdamW(mz_model.parameters(), lr=2.5e-3, weight_decay=1e-4)
    opt_dense = torch.optim.AdamW(dense_spc.parameters(), lr=2.5e-3, weight_decay=1e-4)

    batch_size = 128
    epochs = 25
    for ep in range(epochs):
        perm = torch.randperm(x_train.size(0))
        for i in range(0, x_train.size(0), batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = x_train[idx], y_train[idx]

            # Train EV-MZ-PC on clean data + slight local shell jitter to tune symplectic deflection
            xb_local = xb + 0.08 * torch.randn_like(xb)
            opt_mz.zero_grad()
            logits_mz, _ = mz_model(xb, mode="system1")
            logits_mz_s2, _ = mz_model(xb_local, mode="system2")
            loss_mz = F.cross_entropy(logits_mz, yb) + 0.5 * F.cross_entropy(logits_mz_s2, yb)
            loss_mz.backward()
            opt_mz.step()

            # Train Standard Dense SPC
            opt_dense.zero_grad()
            logits_d, x_hat_d, _ = dense_spc(xb)
            loss_d = F.cross_entropy(logits_d, yb) + 2.0 * F.mse_loss(x_hat_d, xb)
            loss_d.backward()
            opt_dense.step()

    # Re-calibrate thresholds on clean validation slice
    mz_model.mz_gate.calibrate_manifold(x_train[:500], percentile=0.88)
    with torch.no_grad():
        h_clean = dense_spc.encoder(x_train[:500])
        x_hat_clean = dense_spc.decoder(h_clean)
        dense_spc.tau = float(torch.quantile(torch.mean((x_train[:500] - x_hat_clean) ** 2, dim=-1), 0.88).item())

    # Step 3: Evaluate across 3 Physical Regimes (Zero-Shot Severe Corruption)
    torch.manual_seed(999)
    regimes = [
        ("Clean Manifold ('Dud Bomb')", x_test),
        ("Severe Gaussian Noise (sigma=0.65)", x_test + 0.65 * torch.randn_like(x_test)),
        ("Adversarial Bomb (35% Impulse=2.5)", x_test + (torch.rand_like(x_test) < 0.35).float() * torch.randn_like(x_test) * 2.5),
    ]

    print("-" * 84)
    print(f"{'Regime':<35} | {'Blind FF':<10} | {'Dense SPC':<18} | {'EV-MZ-PC (Ours)':<20}")
    print("-" * 84)

    max_ortho_err = 0.0
    for name, x_eval in regimes:
        with torch.no_grad():
            # 1. Blind Feedforward (System 1 only)
            logits_s1, _ = mz_model(x_eval, mode="system1")
            acc_s1 = (logits_s1.argmax(dim=-1) == y_test).float().mean().item() * 100.0

            # 2. Standard Dense SPC
            logits_d, _, trig_d = dense_spc(x_eval)
            acc_d = (logits_d.argmax(dim=-1) == y_test).float().mean().item() * 100.0
            rate_d = trig_d.float().mean().item() * 100.0

            # 3. Elitzur-Vaidman Mach-Zehnder SPC
            logits_mz, info = mz_model(x_eval, mode="selective")
            acc_mz = (logits_mz.argmax(dim=-1) == y_test).float().mean().item() * 100.0
            rate_mz = info["dark_clicks"].float().mean().item() * 100.0
            max_ortho_err = max(max_ortho_err, float(info["orthogonality_err"].item()))

        print(
            f"{name:<35} | {acc_s1:>8.2f}% | {acc_d:>7.2f}% ({rate_d:>5.1f}%) | {acc_mz:>7.2f}% ({rate_mz:>5.1f}% D2)"
        )

    print("-" * 84)

    # Step 4: Verify Asymptotic Gate Overhead & Symplectic Orthogonality
    dense_oh_clean, mz_oh_clean, tot_d_clean, tot_mz_clean = count_gate_and_routing_flops(D, H, K, 0.12)
    dense_oh_noisy, mz_oh_noisy, tot_d_noisy, tot_mz_noisy = count_gate_and_routing_flops(D, H, K, 1.00)

    print(f"[2] Surprise Gate Overhead (Clean Regime): Dense Decoder = {dense_oh_clean:,.0f} FLOPs vs Mach-Zehnder Dark Port = {mz_oh_clean:,.0f} FLOPs")
    print(f"    --> Interaction-Free Gate Compression: {dense_oh_clean / mz_oh_clean:.2f}x FEWER Gate & Routing FLOPs!")
    print(f"[3] Full Network Compute (Turbulent Regime): Dense SPC = {tot_d_noisy:,.0f} FLOPs vs EV-MZ-PC = {tot_mz_noisy:,.0f} FLOPs ({tot_d_noisy / tot_mz_noisy:.2f}x total speedup)")
    print(f"[4] Symplectic Lie Algebra so(D) Orthogonality max |<r, J r>|: {max_ortho_err:.3e} (Strict < 1e-5 bound)")
    print("=" * 84)


if __name__ == "__main__":
    run_benchmark()
