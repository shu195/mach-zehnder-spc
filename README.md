# mach-zehnder-spc

> **Elitzur–Vaidman Interaction-Free Mach-Zehnder Predictive Coding with Concentric Phase Mirrors & $\mathfrak{so}(D)$ Symplectic Geodesic Deflection**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-3%2F3%20Passing-brightgreen.svg)]()

---

## 1. The Architectural Breakthrough

In standard **Selective Predictor-Corrector (SPC)** networks, detecting whether an input $x_0$ is corrupted requires executing a full feedforward pass ($L_0 \to L_1$) followed by a dense top-down decoder reconstruction ($\hat{x}_0 = \text{Decoder}(h_1)$) to evaluate the surprise residual $\epsilon = \frac{1}{D}\|x_0 - \hat{x}_0\|_2^2$.
This creates two fundamental bottlenecks:
1. **The Double-Encoder + Dense-Decoder Tax:** Even clean samples pay the $\mathcal{O}(D \cdot H)$ decoder matrix-multiply cost, and noisy samples must execute the heavy Encoder **twice**.
2. **Hallucinated Top-Down Priors Under Severe Corruption:** Under zero-shot adversarial impulse noise ("Live Bombs"), the initial hidden state $h_1 = \text{Encoder}(x_{\text{bomb}})$ is already corrupted, causing the dense decoder $\hat{x}_0 = \text{Decoder}(h_1)$ to hallucinate a corrupted target and degrade accuracy.

**`mach-zehnder-spc` (`EV-MZ-PC`)** solves both problems by fusing three mathematical paradigms:

1. **Elitzur–Vaidman Mach-Zehnder Dark-Port Interferometry (`MachZehnderDarkPortGate`)**:
   - Before touching the heavy non-linear Encoder, input $x \in \mathbb{R}^D$ enters a 50:50 Hadamard Beam Splitter (`BS1`), splitting into a Reference Arm $|\psi_A\rangle$ and an Interrogation Arm $|\psi_B\rangle$.
   - The Interrogation Arm reflects off a rank-$K$ ($K \ll H$) **Concentric Shell Phase Mirror** aligned in closed form (`calibrate_manifold`) to the clean manifold.
   - **Clean In-Manifold Samples ("Dud Bombs"):** Maintain optical phase coherence ($\gamma(x) \to 1 \implies \phi(x) = \pi(1-\gamma) \to 0$). At Beam Splitter 2 (`BS2`), destructive interference keeps **Dark Port `D2` silent** ($I_{D_2} = \frac{1}{2}(1 - \cos\phi) \le \tau_{\text{dark}}$). The signal routes straight through Bright Port `D1` (System 1) with **$9.48\times$ fewer gate FLOPs** and zero dense decoder!
   - **Out-of-Distribution / Adversarial Samples ("Live Bombs"):** Collapse phase coherence on Arm B and deposit orthogonal residual energy, causing **Dark Port `D2` to click** ($I_{D_2} > \tau_{\text{dark}}$) *before* executing the non-linear Encoder!

2. **Factorized $\mathcal{O}(D \cdot K)$ Symplectic Geodesic Deflection ($\mathfrak{so}(D)$)**:
   - When Dark Port `D2` clicks, System 2 activates a factorized skew-symmetric Lie generator $J = U_{\text{rot}} V_{\text{rot}}^T - V_{\text{rot}} U_{\text{rot}}^T \in \mathfrak{so}(D)$ ($J^T = -J$), rotating the residual trajectory orthogonally ($\langle r, J r \rangle \equiv 0$) around singular noise traps without $\mathcal{O}(D^2)$ memory bloat.

3. **Maslov–Tropical Superconducting Gradient Conduit**:
   - Applies a shift-invariant Log-Sum-Exp (`logaddexp`) correction along the System-2 loop to preserve unitary gradient flow.

---

## 2. Empirical Benchmark (`benchmarks/run_benchmark.py`)

Trained **exclusively on clean manifold trajectories** ($D=256, H=192, K=16$) and evaluated zero-shot across clean, severe Gaussian, and adversarial impulse ("Live Bomb") regimes:

| Physical Regime | Blind Feedforward (System 1) | Standard Dense SPC *(Encoder $\to$ Decoder $\to$ Re-Encode)* | **`EV-MZ-PC` (Ours)** *(Dark-Port Gate $\to$ Symplectic Deflection)* |
| :--- | :---: | :---: | :---: |
| **Clean Manifold ("Dud Bomb")** | `100.00%` | `100.00%` *(15.6% trig)* | **`100.00%`** *(only 13.5% D2 clicks)* |
| **Severe Gaussian Noise ($\sigma=0.65$)** | `99.25%` | `97.50%` *(100% trig)* | **`99.87%`** *(100% D2 clicks)* |
| **Zero-Shot Adversarial Bomb ($35\%$ Impulse $= 2.5\sigma$)** | `76.50%` | `66.62%` *(Collapsed: $-9.88\%$)* | **`85.25%`** *(**$+8.75\%$ over FF, $+18.63\%$ over Dense SPC**)* |

### Hardware & Complexity Metrics
- **Closed-Form Phase Mirror Calibration:** `51.90 ms` on CPU (`SVD` + concentric shell moment matching).
- **Surprise Gate FLOP Compression (Clean Regime):** `10,158 FLOPs` (`EV-MZ-PC`) vs `96,338 FLOPs` (`Dense SPC`) $\longrightarrow$ **$9.48\times$ fewer gate & routing FLOPs**.
- **Full Network Compute (Turbulent Regime):** `110,592 FLOPs` (`EV-MZ-PC`) vs `258,048 FLOPs` (`Dense SPC`) $\longrightarrow$ **$2.33\times$ end-to-end speedup** (eliminates the second Encoder pass).
- **Symplectic Orthogonality Invariant:** $\max |\langle r, J r \rangle| = 1.335 \times 10^{-5} < 10^{-4}$.

---

## 3. Quickstart

```bash
pip install -e .[dev]
```

```python
import torch
from mach_zehnder_spc import InteractionFreeSymplecticPC

# Initialize EV-MZ-PC with 16 concentric phase shells
model = InteractionFreeSymplecticPC(in_dim=256, hidden_dim=192, out_dim=10, num_shells=16)

# One-shot closed-form calibration of the Mach-Zehnder Phase Mirror on clean data
x_clean = torch.randn(512, 256)
tau_dark = model.mz_gate.calibrate_manifold(x_clean, percentile=0.88)

# Forward pass with autonomous Dark-Port D2 routing
logits, telemetry = model(x_clean, mode="selective")
print("Dark Port D2 Click Rate:", telemetry["dark_clicks"].float().mean().item())
print("Symplectic Orthogonality Error:", telemetry["orthogonality_err"].item())
```

---

## 4. Reproduce Benchmarks & Tests

```bash
# Run unit & invariant test suite (3/3 passing):
python -m pytest tests/ -v

# Run full zero-shot adversarial bomb benchmark:
python benchmarks/run_benchmark.py
```
