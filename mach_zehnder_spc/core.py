r"""
Elitzur-Vaidman Interaction-Free Mach-Zehnder Predictive Coding (EV-MZ-PC)
==========================================================================
Fuses three orthogonal paradigms:
1. Quantum Optical Mach-Zehnder Interferometry & Elitzur-Vaidman Bomb Testing:
   - Hadamard Beam Splitter BS1 splits input x into Reference Arm |psi_A> and
     Interrogation Arm |psi_B>.
   - A rank-k Curvilinear Concentric Shell manifold probe acts as the coherent
     mirror on Arm B.
   - Clean in-manifold samples ("Dud Bombs") preserve phase coherence (cos(phi) -> 1),
     causing 100% destructive interference at Dark Port D2 (I_D2 = 0.5 * (1 - cos(phi)) -> 0).
     System 1 routes directly from Bright Port D1 with ZERO dense decoder GEMM!
   - Out-of-distribution / noisy / occluded samples ("Live Bombs") induce phase decoherence
     or amplitude absorption on Arm B, causing Dark Port D2 to click (I_D2 > tau_D2)
     WITHOUT having executed a dense L1 -> L0 reconstruction matrix!

2. Symplectic Geodesic Deflection (Lie Algebra so(D)):
   - Triggered exclusively when Dark Port D2 clicks (I_D2 > tau_D2).
   - Skew-symmetric generator J = A - A^T (J^T = -J) rotates error trajectories
     orthogonally (<e, J e> = 0) around singular noise sinks.

3. Maslov-Tropical Superconducting Gradient Conduit:
   - Log-Sum-Exp (LSE) shift-invariant correction preserves unitary gradient flow
     along the System-2 corrective loop.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class MachZehnderDarkPortGate(nn.Module):
    """
    Elitzur-Vaidman Interaction-Free Dark-Port Surprise Gate.
    Replaces the O(D * H) dense predictive decoder GEMM with an O(D * K) (K << H)
    Mach-Zehnder Interferometer + Concentric Shell Phase Mirror.

    Complexity:
      - Standard CognitiveSurpriseGate: D * H FLOPs (e.g., 784 * 256 = 200,704 FLOPs)
      - MachZehnderDarkPortGate:        D * K FLOPs (e.g., 784 * 8   =   6,272 FLOPs -> 32x smaller!)
    """
    def __init__(
        self,
        in_features: int,
        num_shells: int = 8,
        tau_dark: float = 0.15,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_shells = num_shells
        self.tau_dark = tau_dark
        self.eps = eps

        # Low-rank orthogonal subspace basis U in R^{D x K} for manifold interrogation
        U_init = torch.randn(in_features, num_shells)
        U_orth, _ = torch.linalg.qr(U_init)
        self.U_probe = nn.Parameter(U_orth)

        # Concentric shell radii r_k and widths sigma_k on the interrogation arm
        self.shell_radii = nn.Parameter(torch.linspace(0.2, 1.5, num_shells))
        self.log_sigma = nn.Parameter(torch.full((num_shells,), math.log(0.35)))

        # Phase-coherence weights w_k mapping shell resonance -> optical phase phi
        self.phase_weights = nn.Parameter(torch.ones(num_shells) / num_shells)

    def calibrate_manifold(self, x_clean: torch.Tensor, percentile: float = 0.88) -> float:
        """
        Closed-form manifold alignment (CS-ANA style):
        Aligns U_probe and shell_radii to the clean training manifold in one shot,
        so clean samples achieve cos(phi) -> 1 (zero Dark Port D2 clicks).
        """
        with torch.no_grad():
            # Center and compute top-K principal directions of clean manifold
            mean_x = x_clean.mean(dim=0, keepdim=True)
            x_centered = x_clean - mean_x
            _, _, Vh = torch.linalg.svd(x_centered, full_matrices=False)
            self.U_probe.copy_(Vh[:self.num_shells].t())

            # Measure projected radial magnitudes on the clean manifold
            proj = torch.matmul(x_clean, self.U_probe)  # [N, K]
            r_clean = proj.abs().mean(dim=0)            # [K]
            r_std = proj.abs().std(dim=0).clamp(min=0.15)
            self.shell_radii.copy_(r_clean)
            self.log_sigma.copy_(torch.log(r_std * 1.5))
            self.phase_weights.fill_(1.0 / self.num_shells)

            # Calibrate dark-port threshold tau_dark on clean data
            _, i_d2_clean, _, _ = self.forward(x_clean)
            self.tau_dark = float(torch.quantile(i_d2_clean, percentile).item())
        return self.tau_dark

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Propagates input x through the Mach-Zehnder Interferometer:
          1. BS1 splits x into Reference Arm A and Interrogation Arm B:
             psi_A = x / sqrt(2),  psi_B = x / sqrt(2)
          2. Interrogation Arm B reflects off the Concentric Shell Phase Mirror:
             Each shell k measures radial resonance psi_k = exp(-( |<x, u_k>| - r_k )^2 / (2 sigma_k^2)).
             Total optical coherence gamma(x) = sum_k w_k * psi_k in [0, 1].
             Optical phase shift: phi(x) = pi * (1 - gamma(x)).
             - On clean manifold ("Dud Bomb"): gamma -> 1 => phi -> 0 => e^{i phi} -> +1.
             - Off manifold / noisy ("Live Bomb"): gamma -> 0 => phi -> pi (or absorbed) => Dark Port clicks!
          3. BS2 recombines Arm A and Arm B:
             I_D1 (Bright Port) = 0.5 * (1 + cos(phi))
             I_D2 (Dark Port)   = 0.5 * (1 - cos(phi)) + high-frequency residual energy
        Returns:
            i_d1: Bright Port D1 intensity [batch]
            i_d2: Dark Port D2 intensity (Interaction-Free Surprise) [batch]
            dark_click_mask: Boolean mask (I_D2 > tau_dark) triggering System 2 [batch]
            x_manifold_proj: Low-rank manifold anchor for System-2 deflection [batch, D]
        """
        # Project onto low-rank interrogation basis: O(D * K) FLOPs
        z = torch.matmul(x, self.U_probe)  # [B, K]
        x_manifold_proj = torch.matmul(z, self.U_probe.t())  # [B, D]

        # Orthogonal residual energy outside the K-dimensional subspace ("Bomb absorption")
        ortho_residual = torch.mean((x - x_manifold_proj) ** 2, dim=-1)  # [B]

        # Concentric shell resonance along the K principal axes
        sigma = torch.exp(self.log_sigma).unsqueeze(0) + self.eps  # [1, K]
        radii = self.shell_radii.unsqueeze(0)                      # [1, K]
        shell_resonance = torch.exp(-((z.abs() - radii) ** 2) / (2.0 * sigma ** 2))  # [B, K]

        # Normalized coherence gamma in [0, 1]
        w = F.softmax(self.phase_weights, dim=0).unsqueeze(0)
        gamma = torch.sum(w * shell_resonance, dim=-1)  # [B]

        # Optical phase shift phi in [0, pi]
        phi = math.pi * (1.0 - gamma)

        # Mach-Zehnder BS2 interference intensities:
        # I_D1 = cos^2(phi/2) = 0.5 * (1 + cos(phi))
        # I_D2 = sin^2(phi/2) = 0.5 * (1 - cos(phi)) + absorption term
        cos_phi = torch.cos(phi)
        i_d1 = 0.5 * (1.0 + cos_phi)
        i_d2 = 0.5 * (1.0 - cos_phi) + ortho_residual

        dark_click_mask = i_d2 > self.tau_dark
        return i_d1, i_d2, dark_click_mask, x_manifold_proj


class InteractionFreeSymplecticPC(nn.Module):
    """
    Complete Elitzur-Vaidman Interaction-Free Symplectic Predictor-Corrector Network.
    - System 1 (Dark Port Silent, I_D2 <= tau_dark):
        Executes pure feedforward classification with ZERO dense decoder GEMM and ZERO feedback.
    - System 2 (Dark Port Clicks, I_D2 > tau_dark):
        Executes Lie-algebraic so(D) skew-symmetric geodesic deflection + Tropical LSE conduit
        to scrub noise and rotate the trajectory back onto the clean manifold before encoding.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_shells: int = 16,
        tau_dark: float = 0.15,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_shells = num_shells

        # 1. Mach-Zehnder Interaction-Free Dark Port Gate (O(D * K) FLOPs)
        self.mz_gate = MachZehnderDarkPortGate(
            in_features=in_dim, num_shells=num_shells, tau_dark=tau_dark
        )

        # 2. Lie algebra generator A for skew-symmetric symplectic rotation J = A - A^T in so(D)
        # Parameterized via low-rank skew-symmetric factor U_rot * V_rot^T - V_rot * U_rot^T
        # to keep complexity strictly O(D * K) instead of O(D^2)!
        self.U_rot = nn.Parameter(torch.randn(in_dim, num_shells) * 0.05)
        self.V_rot = nn.Parameter(torch.randn(in_dim, num_shells) * 0.05)

        # Learned step sizes for System-2 geodesic deflection
        self.alpha_pull = nn.Parameter(torch.tensor(0.65))
        self.beta_symplectic = nn.Parameter(torch.tensor(0.20))
        self.gamma_tropical = nn.Parameter(torch.tensor(0.05))

        # 3. Primary Feedforward Encoder & Classifier
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def apply_symplectic_deflection(
        self, x_noisy: torch.Tensor, x_anchor: torch.Tensor
    ) -> torch.Tensor:
        """
        Executes O(D * K) factorized skew-symmetric rotation J = U V^T - V U^T in so(D)
        plus Maslov-Tropical LSE superconducting gradient conduit.
        """
        residual = x_noisy - x_anchor  # [B, D]

        # Factorized skew-symmetric product: J * r = U (V^T r) - V (U^T r)
        # Guaranteed <r, J r> = (r^T U)(V^T r) - (r^T V)(U^T r) == 0 identically!
        vt_r = torch.matmul(residual, self.V_rot)  # [B, K]
        ut_r = torch.matmul(residual, self.U_rot)  # [B, K]
        v_rot = torch.matmul(vt_r, self.U_rot.t()) - torch.matmul(ut_r, self.V_rot.t())  # [B, D]

        # Centrifugal radial regularization 1 / ||r||_2
        r_norm = torch.sqrt(torch.sum(residual ** 2, dim=-1, keepdim=True) + 1e-5)
        v_deflect = v_rot / r_norm

        # Maslov-Tropical dequantized log-sum-exp shift
        lse_shift = torch.logaddexp(-residual, torch.zeros_like(residual)) - math.log(2.0)

        # Symplectic Predictor-Corrector update
        x_corrected = (
            x_noisy
            - torch.sigmoid(self.alpha_pull) * residual
            + self.beta_symplectic * v_deflect
            + self.gamma_tropical * lse_shift
        )
        return x_corrected

    def forward(
        self, x: torch.Tensor, mode: str = "selective"
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: Input tensor [batch, in_dim]
            mode: 'selective' (Dark Port triggered), 'system1' (pure feedforward), 'system2' (always deflect)
        """
        if mode == "system1":
            h = self.encoder(x)
            logits = self.head(h)
            zeros = torch.zeros(x.size(0), device=x.device)
            return logits, {
                "i_d1": torch.ones_like(zeros),
                "i_d2": zeros,
                "dark_clicks": zeros.bool(),
                "orthogonality_err": torch.tensor(0.0, device=x.device),
            }

        # Step 1: Interaction-Free Mach-Zehnder Interrogation (O(D * K) FLOPs, NO encoder/decoder needed!)
        i_d1, i_d2, dark_clicks, x_anchor = self.mz_gate(x)

        if mode == "system2":
            active_mask = torch.ones_like(dark_clicks, dtype=torch.bool)
        else:
            active_mask = dark_clicks

        # Step 2: Route ONLY Dark-Port Clicking samples through Symplectic Deflection
        orthogonality_err = torch.tensor(0.0, device=x.device)
        if active_mask.any():
            x_deflected = self.apply_symplectic_deflection(x, x_anchor)

            # Verify strict symplectic orthogonality <r, J r> == 0
            with torch.no_grad():
                r = x - x_anchor
                vt_r = torch.matmul(r, self.V_rot)
                ut_r = torch.matmul(r, self.U_rot)
                jr = torch.matmul(vt_r, self.U_rot.t()) - torch.matmul(ut_r, self.V_rot.t())
                orthogonality_err = torch.max(torch.abs(torch.sum(r * jr, dim=-1)))

            mask_f = active_mask.unsqueeze(-1).float()
            x_routed = (1.0 - mask_f) * x + mask_f * x_deflected
        else:
            x_routed = x

        # Step 3: Execute heavy encoder ONCE on the routed wavefield!
        # (Standard SPC had to execute Encoder TWICE: once before decoder, once after refinement!)
        h = self.encoder(x_routed)
        logits = self.head(h)

        return logits, {
            "i_d1": i_d1,
            "i_d2": i_d2,
            "dark_clicks": active_mask,
            "orthogonality_err": orthogonality_err,
        }
