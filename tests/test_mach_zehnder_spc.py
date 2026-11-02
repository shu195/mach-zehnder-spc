import torch
import pytest
from mach_zehnder_spc import MachZehnderDarkPortGate, InteractionFreeSymplecticPC


def test_dark_port_destructive_interference_on_clean_manifold():
    torch.manual_seed(42)
    D, K = 64, 8
    U_true, _ = torch.linalg.qr(torch.randn(D, K))
    z_clean = torch.randn(256, K) * 0.5
    x_clean = z_clean @ U_true.t()

    gate = MachZehnderDarkPortGate(in_features=D, num_shells=K)
    tau = gate.calibrate_manifold(x_clean, percentile=0.90)

    i_d1_clean, i_d2_clean, clicks_clean, _ = gate(x_clean)
    assert clicks_clean.float().mean().item() <= 0.12
    assert i_d1_clean.mean().item() > i_d2_clean.mean().item()

    # Inject live-bomb orthogonal noise
    x_bomb = x_clean + 0.8 * torch.randn_like(x_clean)
    _, i_d2_bomb, clicks_bomb, _ = gate(x_bomb)
    assert i_d2_bomb.mean().item() > tau
    assert clicks_bomb.float().mean().item() > 0.95


def test_symplectic_so_d_orthogonality_invariant():
    torch.manual_seed(7)
    D, H, C, K = 64, 48, 5, 8
    model = InteractionFreeSymplecticPC(in_dim=D, hidden_dim=H, out_dim=C, num_shells=K, tau_dark=0.01)
    x = torch.randn(32, D)
    logits, info = model(x, mode="system2")
    assert logits.shape == (32, C)
    # Verify <r, J r> == 0 up to floating point precision
    assert info["orthogonality_err"].item() < 1e-5


def test_system1_vs_selective_routing_modes():
    torch.manual_seed(13)
    D, H, C, K = 32, 32, 4, 6
    model = InteractionFreeSymplecticPC(in_dim=D, hidden_dim=H, out_dim=C, num_shells=K)
    x = torch.randn(16, D)

    logits_s1, info_s1 = model(x, mode="system1")
    assert info_s1["dark_clicks"].sum().item() == 0

    logits_s2, info_s2 = model(x, mode="system2")
    assert info_s2["dark_clicks"].sum().item() == 16
    assert not torch.allclose(logits_s1, logits_s2)
