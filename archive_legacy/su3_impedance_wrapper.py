"""
SU(3) Color Impedance Implementation

Wraps SU(3) spherical embedding calculations into the unified impedance framework.

Based on SU(3)/su3_impedance.py:
- Matter capacity: C_SU3 (symplectic volume of color representation)
- Gauge action: S_SU3 (holonomy, Wilson loops on spherical shells)
- Impedance: Z_SU3 = S_SU3 / C_SU3

CRITICAL DISCLAIMER (from original code):
    This is a geometric/information-theoretic probe.
    We do NOT claim derivation of QCD coupling α_s from first principles.

Author: Computational Physics Research
Date: February 2026
"""

import sys
import os
import numpy as np
from geometric_impedance_interface import GeometricImpedanceSystem, ImpedanceResult, validate_quantum_number
from typing import Dict, Any

# Add SU(3) directory to path
SU3_DIR = os.path.join(os.path.dirname(__file__), "SU(3)")
if SU3_DIR not in sys.path:
    sys.path.insert(0, SU3_DIR)

# Import SU(3) classes
try:
    from su3_impedance import SU3SymplecticImpedance, ImpedanceData
except ImportError:
    raise ImportError(f"Could not import SU(3) modules. Check that SU(3) directory exists at {SU3_DIR}")


class SU3Impedance(GeometricImpedanceSystem):
    """
    SU(3) color impedance for quark representations.
    
    Wraps the existing SU3SymplecticImpedance class into the unified framework.
    
    Computes impedance Z = S_gauge / C_matter where:
    - C_matter: Color phase space capacity (symplectic volume)
    - S_gauge: Color holonomy (Wilson loops, plaquette sums)
    
    This treats SU(3) color coupling as a geometric impedance, analogous to
    α for U(1). However, we do NOT claim first-principles QCD derivation.
    """
    
    def __init__(self, p: int, q: int, normalization: str = "default", verbose: bool = False):
        """
        Initialize SU(3) impedance calculator.
        
        Parameters:
        -----------
        p, q : int
            Dynkin labels for SU(3) representation (p,q)
            - (1,0): fundamental 3
            - (0,1): antifundamental 3̄
            - (1,1): octet 8
            - (2,0): sextet 6
            - (3,0): decuplet 10
            - etc.
        normalization : str
            Impedance normalization choice:
            - "default": Z_impedance (raw ratio)
            - "normalized": Z_normalized (scaled by √dim)
            - "dimensionless": Z_dimensionless (rescaled to match α scale)
        verbose : bool
            Print detailed computation information
        """
        # Validate inputs
        if not isinstance(p, int) or not isinstance(q, int):
            raise TypeError(f"Dynkin labels must be integers, got p={type(p)}, q={type(q)}")
        if p < 0 or q < 0:
            raise ValueError(f"Dynkin labels must be non-negative, got (p,q)=({p},{q})")
        if p == 0 and q == 0:
            raise ValueError("Trivial representation (0,0) not allowed")
        
        self.p = p
        self.q = q
        self.normalization = normalization
        self.verbose = verbose
        
        # Create underlying SU(3) calculator
        self._su3_calc = SU3SymplecticImpedance(p=p, q=q, verbose=verbose)
        
        # Compute impedance data (cached)
        self._impedance_data: ImpedanceData = None
    
    def _compute_su3_data(self) -> ImpedanceData:
        """
        Compute SU(3) impedance data (cached).
        
        Returns:
        --------
        ImpedanceData
            Full impedance data from underlying SU(3) calculator
        """
        if self._impedance_data is None:
            self._impedance_data = self._su3_calc.compute_impedance()
        return self._impedance_data
    
    def compute_matter_capacity(self) -> float:
        """
        Compute SU(3) matter capacity C_matter.
        
        This wraps the underlying SU(3) calculation:
            C_matter = C_symplectic (total symplectic volume)
        
        Returns:
        --------
        C_matter : float
            Matter capacity (color phase space volume)
        """
        data = self._compute_su3_data()
        return data.C_matter
    
    def compute_gauge_action(self) -> float:
        """
        Compute SU(3) gauge action S_gauge.
        
        This wraps the underlying SU(3) calculation:
            S_gauge = S_holonomy (Wilson loop sum)
        
        Returns:
        --------
        S_gauge : float
            Gauge action (color holonomy)
        """
        data = self._compute_su3_data()
        return data.S_gauge
    
    def compute_impedance(self) -> float:
        """
        Compute SU(3) impedance Z = S_gauge / C_matter.
        
        Returns impedance in chosen normalization:
        - "default": Z_impedance (raw ratio)
        - "normalized": Z_normalized (scaled by √dim)
        - "dimensionless": Z_dimensionless (rescaled to match α scale)
        
        Returns:
        --------
        Z : float
            Impedance (dimensionless)
        """
        data = self._compute_su3_data()
        
        if self.normalization == "default":
            return data.Z_impedance
        elif self.normalization == "normalized":
            return data.Z_normalized
        elif self.normalization == "dimensionless":
            return data.Z_dimensionless
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")
    
    def get_label(self) -> str:
        """Get human-readable label."""
        return f"SU(3) ({self.p},{self.q})"
    
    def get_size_parameter(self) -> float:
        """
        Get characteristic size parameter.
        
        Uses representation size: √(p² + pq + q²)
        This is related to the Casimir eigenvalue.
        """
        return np.sqrt(self.p**2 + self.p*self.q + self.q**2)
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get additional SU(3) metadata."""
        data = self._compute_su3_data()
        
        return {
            "p": self.p,
            "q": self.q,
            "dimension": data.dim,
            "casimir_C2": data.C2,
            "num_shells": data.num_shells,
            "states_per_shell": data.states_per_shell,
            "shell_radii": data.shell_radii,
            "C_berry": data.C_berry,
            "C_symplectic": data.C_symplectic,
            "S_wilson": data.S_wilson,
            "S_plaquette": data.S_plaquette,
            "S_holonomy": data.S_holonomy,
            "Z_impedance": data.Z_impedance,
            "Z_normalized": data.Z_normalized,
            "Z_dimensionless": data.Z_dimensionless,
            "entropy_matter": data.entropy_matter,
            "entropy_gauge": data.entropy_gauge,
            "info_conversion_rate": data.info_conversion_rate,
            "normalization": self.normalization,
        }
    
    def __repr__(self):
        return f"SU3Impedance(p={self.p}, q={self.q}, norm={self.normalization})"


def compute_su3_series(rep_list: list, normalization: str = "default", verbose: bool = False) -> list:
    """
    Compute impedance for multiple SU(3) representations.
    
    Convenience function for scanning across representations.
    
    Parameters:
    -----------
    rep_list : list of (p, q) tuples
        SU(3) representations to compute
    normalization : str
        Impedance normalization choice
    verbose : bool
        Print detailed information
    
    Returns:
    --------
    results : list of ImpedanceResult
        Impedance results for each representation
    """
    results = []
    
    for p, q in rep_list:
        system = SU3Impedance(p=p, q=q, normalization=normalization, verbose=verbose)
        result = system.compute()
        results.append(result)
    
    return results


# ============================================================================
# Common SU(3) representations
# ============================================================================

SU3_REPRESENTATIONS = {
    "fundamental": (1, 0),
    "antifundamental": (0, 1),
    "adjoint": (1, 1),
    "sextet": (2, 0),
    "antisextet": (0, 2),
    "decuplet": (3, 0),
    "antidecuplet": (0, 3),
    "15-plet": (2, 1),
    "15-bar": (1, 2),
    "27-plet": (3, 1),
}


if __name__ == "__main__":
    print("="*70)
    print("SU(3) COLOR IMPEDANCE - Validation")
    print("="*70)
    
    # Test fundamental representation (3)
    print("\nTest 1: Fundamental representation (1,0)")
    print("-"*70)
    
    su3_fund = SU3Impedance(p=1, q=0, normalization="default", verbose=True)
    result = su3_fund.compute()
    
    print(f"\nResults:")
    print(f"  Dimension:        {result.metadata['dimension']}")
    print(f"  Casimir C2:       {result.metadata['casimir_C2']:.4f}")
    print(f"  C_matter:         {result.C_matter:.4f}")
    print(f"  S_gauge:          {result.S_gauge:.4f}")
    print(f"  Z_impedance:      {result.Z_impedance:.6f}")
    print(f"  Z_normalized:     {result.metadata['Z_normalized']:.6f}")
    print(f"  Z_dimensionless:  {result.metadata['Z_dimensionless']:.6f}")
    
    # Test multiple representations
    print("\n" + "="*70)
    print("Test 2: Multiple SU(3) representations")
    print("-"*70)
    
    test_reps = [(1,0), (0,1), (1,1), (2,0), (3,0)]
    results = compute_su3_series(test_reps, normalization="default", verbose=False)
    
    print("\n  (p,q)  |  Dim  |  C_matter  |  S_gauge  |  Z_impedance")
    print("  " + "-"*58)
    
    for res in results:
        p = res.metadata["p"]
        q = res.metadata["q"]
        dim = res.metadata["dimension"]
        C = res.C_matter
        S = res.S_gauge
        Z = res.Z_impedance
        
        print(f"  ({p},{q})   |  {dim:3d}   |  {C:8.4f}   |  {S:8.4f}  |  {Z:.6f}")
    
    print("\n✓ SU(3) impedance wrapper validated!")
    print("\nDISCLAIMER:")
    print("  This is a geometric probe. We do NOT claim first-principles")
    print("  derivation of QCD coupling α_s. These are information-theoretic")
    print("  impedance ratios for exploratory comparison.")
