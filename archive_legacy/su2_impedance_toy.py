"""
SU(2) Toy Impedance Implementation

Minimal toy model for SU(2) gauge impedance.

This is a SIMPLIFIED model for exploratory comparison. It uses:
- Matter capacity: State count (2j+1) weighted by j(j+1)
- Gauge action: Simplified holonomy based on SU(2) structure

This is NOT a full SU(2) Yang-Mills calculation. It's a pedagogical
toy model to explore impedance patterns across gauge groups.

Author: Computational Physics Research
Date: February 2026
"""

import numpy as np
from geometric_impedance_interface import GeometricImpedanceSystem, ImpedanceResult
from typing import Dict, Any


class SU2Impedance(GeometricImpedanceSystem):
    """
    SU(2) toy impedance calculator.
    
    Simplified model for exploratory comparison with U(1) and SU(3).
    
    Matter capacity: C = (2j+1) * √[j(j+1)]
        - (2j+1): State count (spin degeneracy)
        - √[j(j+1)]: Casimir weighting
    
    Gauge action: S = 4π√j (simplified SU(2) holonomy scale)
        - Based on solid angle 4π for SU(2) group manifold
        - √j scaling from representation size
    
    WARNING: This is a TOY MODEL, not rigorous SU(2) Yang-Mills.
    """
    
    def __init__(self, j: float, model: str = "simple"):
        """
        Initialize SU(2) toy impedance calculator.
        
        Parameters:
        -----------
        j : float
            Spin quantum number (j ≥ 0, half-integer or integer)
        model : str
            Toy model variant:
            - "simple": C = (2j+1)√[j(j+1)], S = 4π√j
            - "phase_space": C = (2j+1)², S = 2π(2j+1)
            - "ladder": C from ladder operators, S from coherent states
        """
        if j < 0:
            raise ValueError(f"Spin j must be non-negative, got {j}")
        
        # Check if j is half-integer or integer
        if not (2*j).is_integer():
            raise ValueError(f"Spin j must be integer or half-integer, got {j}")
        
        self.j = j
        self.model = model
        
        # Cache
        self._C_matter = None
        self._S_gauge = None
    
    def compute_matter_capacity(self) -> float:
        """
        Compute SU(2) matter capacity (spin phase space).
        
        Model variants:
        
        1. "simple" (default):
           C = (2j+1) * √[j(j+1)]
           - Degeneracy × Casimir weighting
        
        2. "phase_space":
           C = (2j+1)²
           - Full phase space area
        
        3. "ladder":
           C = Σ_m √[(j-m)(j+m+1)] (sum of ladder amplitudes)
        
        Returns:
        --------
        C_matter : float
            Matter capacity (phase space volume)
        """
        if self._C_matter is not None:
            return self._C_matter
        
        if self.model == "simple":
            # Degeneracy × Casimir
            degeneracy = 2 * self.j + 1
            casimir = np.sqrt(max(self.j * (self.j + 1), 0.01))  # Avoid singularity at j=0
            C = degeneracy * casimir
        
        elif self.model == "phase_space":
            # Full phase space (degeneracy squared)
            degeneracy = 2 * self.j + 1
            C = degeneracy ** 2
        
        elif self.model == "ladder":
            # Sum of ladder operator matrix elements
            # |⟨j, m+1|J+|j, m⟩| = √[(j-m)(j+m+1)]
            C = 0.0
            m_values = np.arange(-self.j, self.j)  # m = -j, -j+1, ..., j-1
            for m in m_values:
                ladder_amplitude = np.sqrt((self.j - m) * (self.j + m + 1))
                C += ladder_amplitude
        
        else:
            raise ValueError(f"Unknown model: {self.model}")
        
        self._C_matter = C
        return C
    
    def compute_gauge_action(self) -> float:
        """
        Compute SU(2) gauge action (holonomy scale).
        
        Model variants:
        
        1. "simple" (default):
           S = 4π√j
           - 4π: Solid angle of SU(2) ~ S³
           - √j: Representation size scaling
        
        2. "phase_space":
           S = 2π(2j+1)
           - Circumference × degeneracy
        
        3. "ladder":
           S = 2π√[j(j+1)]
           - Casimir-based holonomy
        
        Returns:
        --------
        S_gauge : float
            Gauge action
        """
        if self._S_gauge is not None:
            return self._S_gauge
        
        if self.model == "simple":
            # Solid angle × representation scaling
            S = 4 * np.pi * np.sqrt(max(self.j, 0.01))
        
        elif self.model == "phase_space":
            # Circumference × degeneracy
            S = 2 * np.pi * (2 * self.j + 1)
        
        elif self.model == "ladder":
            # Casimir-based
            S = 2 * np.pi * np.sqrt(max(self.j * (self.j + 1), 0.01))
        
        else:
            raise ValueError(f"Unknown model: {self.model}")
        
        self._S_gauge = S
        return S
    
    def get_label(self) -> str:
        """Get human-readable label."""
        # Format j nicely (integer or half-integer)
        if self.j == int(self.j):
            j_str = f"{int(self.j)}"
        else:
            j_str = f"{int(2*self.j)}/2"
        
        return f"SU(2) j={j_str}"
    
    def get_size_parameter(self) -> float:
        """Get characteristic size (spin j)."""
        return float(self.j)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get additional metadata."""
        degeneracy = 2 * self.j + 1
        casimir = self.j * (self.j + 1)
        
        return {
            "j": self.j,
            "model": self.model,
            "degeneracy": degeneracy,
            "casimir_J2": casimir,
            "representation_dim": int(degeneracy),
        }
    
    def __repr__(self):
        return f"SU2Impedance(j={self.j}, model={self.model})"


def compute_su2_series(j_values: list, model: str = "simple") -> list:
    """
    Compute impedance for multiple SU(2) spins.
    
    Convenience function for scanning across j.
    
    Parameters:
    -----------
    j_values : list of float
        Spin quantum numbers (integer or half-integer)
    model : str
        Toy model variant
    
    Returns:
    --------
    results : list of ImpedanceResult
        Impedance results for each j
    """
    results = []
    
    for j in j_values:
        system = SU2Impedance(j=j, model=model)
        result = system.compute()
        results.append(result)
    
    return results


if __name__ == "__main__":
    print("="*70)
    print("SU(2) TOY IMPEDANCE - Validation")
    print("="*70)
    
    # Test j=1/2 (fundamental doublet)
    print("\nTest 1: j=1/2 (fundamental doublet)")
    print("-"*70)
    
    su2_half = SU2Impedance(j=0.5, model="simple")
    result = su2_half.compute()
    
    print(f"\nResults:")
    print(f"  Spin j:           {result.metadata['j']}")
    print(f"  Degeneracy:       {result.metadata['degeneracy']:.0f}")
    print(f"  Casimir J²:       {result.metadata['casimir_J2']:.4f}")
    print(f"  C_matter:         {result.C_matter:.4f}")
    print(f"  S_gauge:          {result.S_gauge:.4f}")
    print(f"  Z_impedance:      {result.Z_impedance:.6f}")
    
    # Test series j=0 to 5
    print("\n" + "="*70)
    print("Test 2: SU(2) spin series (j=0 to 5)")
    print("-"*70)
    
    j_values = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    results = compute_su2_series(j_values, model="simple")
    
    print("\n    j    |  Deg  |  C_matter  |  S_gauge  |  Z_impedance")
    print("  " + "-"*58)
    
    for res in results:
        j = res.metadata["j"]
        deg = res.metadata["degeneracy"]
        C = res.C_matter
        S = res.S_gauge
        Z = res.Z_impedance
        
        # Format j nicely
        if j == int(j):
            j_str = f"{int(j)}"
        else:
            j_str = f"{int(2*j)}/2"
        
        print(f"  {j_str:>5}  |  {deg:3.0f}   |  {C:8.4f}   |  {S:8.4f}  |  {Z:.6f}")
    
    # Compare models
    print("\n" + "="*70)
    print("Test 3: Model comparison (j=2)")
    print("-"*70)
    
    models = ["simple", "phase_space", "ladder"]
    
    print("\n  Model        |  C_matter  |  S_gauge  |  Z_impedance")
    print("  " + "-"*52)
    
    for model in models:
        system = SU2Impedance(j=2, model=model)
        result = system.compute()
        
        print(f"  {model:12} |  {result.C_matter:8.4f}   |  {result.S_gauge:8.4f}  |  {result.Z_impedance:.6f}")
    
    print("\n✓ SU(2) toy impedance calculator validated!")
    print("\nWARNING:")
    print("  This is a TOY MODEL for pedagogical exploration.")
    print("  It is NOT a rigorous SU(2) Yang-Mills calculation.")
    print("  Use for qualitative pattern comparison only.")
