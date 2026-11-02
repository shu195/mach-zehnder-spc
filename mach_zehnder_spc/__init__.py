"""
mach_zehnder_spc: Elitzur-Vaidman Interaction-Free Mach-Zehnder Predictive Coding
=================================================================================
Quantum-inspired dark-port wave interferometry + Concentric Shell phase mirrors +
Lie-algebraic so(D) symplectic geodesic deflection for zero-decoder System 1 / System 2 routing.
"""

from .core import MachZehnderDarkPortGate, InteractionFreeSymplecticPC

__version__ = "0.1.0"
__all__ = [
    "MachZehnderDarkPortGate",
    "InteractionFreeSymplecticPC",
]
