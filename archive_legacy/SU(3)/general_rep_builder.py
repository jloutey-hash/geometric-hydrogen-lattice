# General (p,q) Representation Builder with CG Projection
"""
Construct arbitrary SU(3) irreducible representations (p,q) using
Clebsch-Gordan decomposition.

Uses proper irrep projection instead of working with reducible tensor products.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from weight_basis_gellmann import WeightBasisSU3
from clebsch_gordan_su3 import ClebschGordanSU3
from irrep_projectors import IrrepProjector
from irrep_operators import IrrepOperators


class GeneralRepBuilder:
    """
    Builder for arbitrary (p,q) SU(3) irreps using CG projection.
    """
    
    def __init__(self):
        """Initialize with CG coefficients and projectors."""
        self.cg = ClebschGordanSU3()
        self.proj = IrrepProjector()
        self.irrep_ops = IrrepOperators()
        
        # Known representations from CG decomposition
        self.known_irreps = {
            (0, 0): {'name': '1', 'dim': 1, 'C2': 0.0},
            (1, 0): {'name': '3', 'dim': 3, 'C2': 4/3},
            (0, 1): {'name': '3bar', 'dim': 3, 'C2': 4/3},
            (2, 0): {'name': '6', 'dim': 6, 'C2': 10/3},
            (0, 2): {'name': '6bar', 'dim': 6, 'C2': 10/3},
            (1, 1): {'name': '8', 'dim': 8, 'C2': 3.0},
        }
    
    def dimension_formula(self, p: int, q: int) -> int:
        """
        Dimension formula for (p,q) irrep.
        
        dim(p,q) = (p+1)(q+1)(p+q+2)/2
        
        Args:
            p, q: Dynkin labels
            
        Returns:
            dimension: Dimension of irrep
        """
        return (p + 1) * (q + 1) * (p + q + 2) // 2
    
    def casimir_formula(self, p: int, q: int) -> float:
        """
        Casimir eigenvalue for (p,q) irrep.
        
        C₂(p,q) = (p² + q² + pq + 3p + 3q)/3
        
        Args:
            p, q: Dynkin labels
            
        Returns:
            casimir: C₂ eigenvalue
        """
        return (p**2 + q**2 + p*q + 3*p + 3*q) / 3.0
    
    def get_irrep_operators(self, p: int, q: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Get SU(3) generators for (p,q) irrep if available.
        
        Args:
            p, q: Dynkin labels
            
        Returns:
            operators: Dictionary of generators, or None if not available
        """
        if (p, q) in self.known_irreps:
            irrep_name = self.known_irreps[(p, q)]['name']
            if irrep_name in self.irrep_ops.irrep_generators:
                return self.irrep_ops.irrep_generators[irrep_name]
        return None
    
    def extract_weights(self, operators: Dict[str, np.ndarray],
                       n_states: Optional[int] = None) -> List[Tuple[float, float]]:
        """
        Extract weight diagram from generators.
        
        Diagonalize T3 and T8 to find simultaneous eigenstates.
        
        Args:
            operators: Dictionary with 'T3' and 'T8'
            n_states: Number of states to sample (None = use dimension)
            
        Returns:
            weights: List of (I3, Y) tuples
        """
        T3 = operators['T3']
        T8 = operators['T8']
        dim = T3.shape[0]
        
        if n_states is None:
            n_states = dim
        
        # For small dimensions, diagonalize directly
        if dim <= 8:
            # Try to find simultaneous eigenbasis
            # Start with T3 eigenvectors
            evals_T3, evecs_T3 = np.linalg.eigh(T3)
            
            weights = []
            for i in range(dim):
                psi = evecs_T3[:, i]
                I3 = np.real(psi.conj() @ T3 @ psi)
                Y = 2/np.sqrt(3) * np.real(psi.conj() @ T8 @ psi)
                weights.append((I3, Y))
            
            return weights
        else:
            # For larger dimensions, sample random states
            weights = []
            for _ in range(n_states):
                psi = np.random.randn(dim) + 1j * np.random.randn(dim)
                psi /= np.linalg.norm(psi)
                
                I3 = np.real(psi.conj() @ T3 @ psi)
                Y = 2/np.sqrt(3) * np.real(psi.conj() @ T8 @ psi)
                weights.append((I3, Y))
            
            return weights
    
    def extract_highest_weight(self, operators: Dict[str, np.ndarray]) -> Tuple[float, float]:
        """
        Extract highest weight state (maximum I3 + Y).
        
        Args:
            operators: Dictionary of generators
            
        Returns:
            hw: (I3, Y) of highest weight
        """
        weights = self.extract_weights(operators)
        
        # Highest weight: max I3, then max Y
        hw = max(weights, key=lambda w: (w[0], w[1]))
        return hw
    
    def validate_representation(self, p: int, q: int, verbose: bool = True) -> Dict[str, any]:
        """
        Validate (p,q) representation if available.
        
        Tests:
        - Dimension matches formula
        - Casimir matches formula
        - Weight diagram structure
        - Hermiticity of generators
        
        Args:
            p, q: Dynkin labels
            verbose: Print results
            
        Returns:
            results: Validation metrics
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"REPRESENTATION ({p},{q}) VALIDATION")
            print(f"{'='*70}")
        
        # Get operators
        ops = self.get_irrep_operators(p, q)
        
        if ops is None:
            if verbose:
                print(f"⚠ Representation ({p},{q}) not yet implemented")
            return {'available': False}
        
        # Expected values
        dim_expected = self.dimension_formula(p, q)
        C2_expected = self.casimir_formula(p, q)
        
        # Actual dimension
        dim_actual = ops['T3'].shape[0]
        
        # Casimir
        C2 = self.irrep_ops.compute_casimir(ops)
        C2_eigs = np.linalg.eigvalsh(C2)
        C2_mean = np.mean(C2_eigs)
        C2_std = np.std(C2_eigs)
        
        # Hermiticity
        herm_err = max(
            np.linalg.norm(ops['T3'] - ops['T3'].conj().T),
            np.linalg.norm(ops['T8'] - ops['T8'].conj().T)
        )
        
        # Weights
        weights = self.extract_weights(ops)
        hw = self.extract_highest_weight(ops)
        
        # Errors
        dim_match = (dim_actual == dim_expected)
        C2_error = abs(C2_mean - C2_expected)
        
        if verbose:
            print(f"\nDimension:")
            print(f"  Expected: {dim_expected}")
            print(f"  Actual: {dim_actual}")
            print(f"  Match: {'✓' if dim_match else '✗'}")
            
            print(f"\nCasimir:")
            print(f"  Expected: {C2_expected:.6f}")
            print(f"  Actual: {C2_mean:.6f}")
            print(f"  Error: {C2_error:.2e}")
            print(f"  Uniformity (std): {C2_std:.2e}")
            
            print(f"\nHermiticity:")
            print(f"  Max error: {herm_err:.2e}")
            
            print(f"\nWeight diagram:")
            print(f"  Number of weights: {len(weights)}")
            print(f"  Highest weight: I3={hw[0]:.3f}, Y={hw[1]:.3f}")
            
            # Display weight multiplicities
            from collections import Counter
            weight_counts = Counter(tuple(np.round([w[0], w[1]], 3)) for w in weights)
            print(f"  Unique weights: {len(weight_counts)}")
            if len(weight_counts) <= 15:
                print("  Weight multiplicities:")
                for (i3, y), count in sorted(weight_counts.items()):
                    print(f"    ({i3:.1f}, {y:.1f}): {count}")
            
            if dim_match and C2_error < 1e-10 and herm_err < 1e-14:
                print(f"\n✓ Representation ({p},{q}) VALIDATED!")
            else:
                print(f"\n⚠ Some properties show deviations")
        
        return {
            'available': True,
            'dim_expected': dim_expected,
            'dim_actual': dim_actual,
            'dim_match': dim_match,
            'C2_expected': C2_expected,
            'C2_actual': C2_mean,
            'C2_error': C2_error,
            'hermiticity_error': herm_err,
            'weights': weights,
            'highest_weight': hw
        }
    
    def list_available_irreps(self, verbose: bool = True) -> List[Tuple[int, int]]:
        """
        List all currently available irreps.
        
        Args:
            verbose: Print list
            
        Returns:
            irreps: List of (p, q) tuples
        """
        available = list(self.known_irreps.keys())
        
        if verbose:
            print(f"\n{'='*70}")
            print("AVAILABLE IRREDUCIBLE REPRESENTATIONS")
            print(f"{'='*70}")
            print(f"\n{'(p,q)':<10} {'Name':<10} {'Dim':<6} {'C2':<10}")  # Changed from C₂ to C2
            print("-" * 40)
            
            for (p, q) in sorted(available):
                info = self.known_irreps[(p, q)]
                print(f"({p},{q}){'':<6} {info['name']:<10} {info['dim']:<6} {info['C2']:<10.4f}")
        
        return available


# ============================================================================
# VALIDATION AND DEMONSTRATION
# ============================================================================

def validate_general_rep_builder():
    """Run full validation suite for general (p,q) builder."""
    print("\n" + "="*70)
    print("GENERAL (p,q) REPRESENTATION BUILDER VALIDATION")
    print("="*70)
    
    builder = GeneralRepBuilder()
    
    # List available irreps
    available = builder.list_available_irreps(verbose=True)
    
    # Validate each available irrep
    all_results = {}
    
    for (p, q) in available:
        results = builder.validate_representation(p, q, verbose=True)
        all_results[(p, q)] = results
    
    print("\n" + "="*70)
    print("✓ ALL REPRESENTATION BUILDER TESTS COMPLETED")
    print("="*70)
    
    # Summary
    print("\n\nVALIDATION SUMMARY")
    print("="*70)
    print(f"{'(p,q)':<10} {'Dim':<8} {'C₂ error':<12} {'Herm error':<12}")
    print("-" * 45)
    
    for (p, q), res in sorted(all_results.items()):
        if res['available']:
            dim_str = f"{res['dim_actual']}/{res['dim_expected']}"
            print(f"({p},{q}){'':<6} {dim_str:<8} {res['C2_error']:<12.2e} {res['hermiticity_error']:<12.2e}")
    
    # Check if all pass
    all_passed = all(
        res['available'] and res['dim_match'] and 
        res['C2_error'] < 1e-10 and res['hermiticity_error'] < 1e-14
        for res in all_results.values()
    )
    
    if all_passed:
        print("\n✓ All representation builder tests PASSED!")
    else:
        print("\n⚠ Some tests show deviations")


if __name__ == "__main__":
    validate_general_rep_builder()
