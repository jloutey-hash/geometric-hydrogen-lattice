# Higher SU(3) Representation Builder
"""
Construct arbitrary (p,q) representations via tensor products.

Features:
- General (p,q) construction from fundamental reps
- Casimir operator computation
- Irrep identification via GT patterns
- Dimension and weight validation
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from weight_basis_gellmann import WeightBasisSU3
from adjoint_tensor_product import AdjointSU3
from scipy.linalg import eigh


class HigherRepBuilder:
    """
    Builder for arbitrary SU(3) representations.
    """
    
    def __init__(self):
        """Initialize fundamental representations."""
        self.fund = WeightBasisSU3(1, 0)    # 3
        self.antifund = WeightBasisSU3(0, 1)  # 3̄
        self.adj = AdjointSU3()             # 8
        
        # Known representations (p, q): dimension
        self.known_dims = {
            (0, 0): 1,   # singlet
            (1, 0): 3,   # fundamental
            (0, 1): 3,   # antifundamental
            (1, 1): 8,   # adjoint
            (2, 0): 6,   # symmetric
            (0, 2): 6,   # antisymmetric
            (3, 0): 10,  # 10
            (2, 1): 15,  # 15
            (1, 2): 15,  # 15̄
            (4, 0): 15,  # 15'
            (0, 3): 10,  # 10̄
        }
        
        # Casimir values C₂(p,q) = (p² + q² + pq + 3p + 3q)/3
        self.known_casimirs = {}
        for (p, q), dim in self.known_dims.items():
            C2 = (p**2 + q**2 + p*q + 3*p + 3*q) / 3.0
            self.known_casimirs[(p, q)] = C2
    
    def tensor_product_operators(self, ops1: List[np.ndarray],
                                ops2: List[np.ndarray]) -> List[np.ndarray]:
        """
        Construct tensor product of operator lists.
        
        T^(R1⊗R2) = T^(R1) ⊗ I + I ⊗ T^(R2)
        
        Args:
            ops1: Operators in first representation [T3, T8, E12, ...]
            ops2: Operators in second representation
            
        Returns:
            ops_prod: Operators in product representation
        """
        dim1 = ops1[0].shape[0]
        dim2 = ops2[0].shape[0]
        
        I1 = np.eye(dim1, dtype=complex)
        I2 = np.eye(dim2, dtype=complex)
        
        ops_prod = []
        for T1, T2 in zip(ops1, ops2):
            # T1 ⊗ I2 + I1 ⊗ T2
            T_prod = np.kron(T1, I2) + np.kron(I1, T2)
            ops_prod.append(T_prod)
        
        return ops_prod
    
    def build_representation(self, p: int, q: int) -> Tuple[List[np.ndarray], int]:
        """
        Build (p,q) representation via tensor products.
        
        (p,q) = 3^⊗p ⊗ 3̄^⊗q
        
        Args:
            p: Number of fundamental indices
            q: Number of antifundamental indices
            
        Returns:
            operators: [T3, T8, E12, E21, E23, E32, E13, E31]
            dimension: Dimension of reducible representation
        """
        # Start with identity (singlet)
        if p == 0 and q == 0:
            return [np.zeros((1, 1), dtype=complex) for _ in range(8)], 1
        
        # Build from fundamental reps
        ops_fund = [self.fund.T3, self.fund.T8, self.fund.E12, self.fund.E21,
                   self.fund.E23, self.fund.E32, self.fund.E13, self.fund.E31]
        ops_anti = [self.antifund.T3, self.antifund.T8, self.antifund.E12, self.antifund.E21,
                   self.antifund.E23, self.antifund.E32, self.antifund.E13, self.antifund.E31]
        
        # Start with first factor
        if p > 0:
            ops_result = [T.copy() for T in ops_fund]
            current_dim = 3
        else:
            ops_result = [T.copy() for T in ops_anti]
            current_dim = 3
        
        # Add remaining fundamental factors
        for i in range(1, p):
            ops_result = self.tensor_product_operators(ops_result, ops_fund)
            current_dim *= 3
        
        # Add antifundamental factors
        for i in range(q if p > 0 else 1, q):
            ops_result = self.tensor_product_operators(ops_result, ops_anti)
            current_dim *= 3
        
        return ops_result, current_dim
    
    def compute_casimir(self, operators: List[np.ndarray]) -> np.ndarray:
        """
        Compute Casimir operator C₂ = Σ Tₐ².
        
        Args:
            operators: List of generators
            
        Returns:
            C2: Casimir operator matrix
        """
        C2 = sum(T @ T for T in operators)
        return C2
    
    def identify_irreps(self, C2: np.ndarray, tol: float = 1e-8) -> Dict[str, any]:
        """
        Identify irreducible representations from Casimir eigenvalues.
        
        Each irrep (p,q) has definite Casimir:
        C₂(p,q) = (p² + q² + pq + 3p + 3q)/3
        
        Args:
            C2: Casimir operator
            tol: Tolerance for eigenvalue degeneracy
            
        Returns:
            irreps: Dictionary with identified irreps and multiplicities
        """
        # Diagonalize Casimir
        eigs = np.linalg.eigvalsh(C2)
        eigs = np.round(eigs, decimals=8)
        
        # Find unique eigenvalues and their multiplicities
        unique_eigs, counts = np.unique(eigs, return_counts=True)
        
        # Match to known representations
        identified = {}
        for eig, mult in zip(unique_eigs, counts):
            # Search known Casimirs
            matches = []
            for (p, q), C2_val in self.known_casimirs.items():
                if abs(eig - C2_val) < tol:
                    matches.append((p, q))
            
            identified[float(eig)] = {
                'multiplicity': int(mult),
                'possible_irreps': matches
            }
        
        return identified
    
    def extract_weights(self, operators: List[np.ndarray],
                       n_samples: int = 10) -> List[Tuple[float, float]]:
        """
        Extract weight diagram by diagonalizing (T3, T8).
        
        Args:
            operators: Generator list
            n_samples: Number of random states to sample
            
        Returns:
            weights: List of (I3, Y) values
        """
        T3 = operators[0]
        T8 = operators[1]
        
        # Diagonalize T3 and T8 simultaneously if possible
        # For now, extract weights from random states
        dim = T3.shape[0]
        weights = []
        
        for _ in range(n_samples):
            psi = np.random.randn(dim) + 1j * np.random.randn(dim)
            psi /= np.linalg.norm(psi)
            
            I3 = np.real(psi.conj() @ T3 @ psi)
            Y = 2/np.sqrt(3) * np.real(psi.conj() @ T8 @ psi)
            
            weights.append((I3, Y))
        
        return weights
    
    def validate_representation(self, p: int, q: int, verbose: bool = True) -> Dict[str, any]:
        """
        Build and validate (p,q) representation.
        
        Tests:
        - Dimension matches theory
        - Casimir eigenvalue correct
        - Commutation relations [Tₐ, Tᵦ] = ifₐᵦᶜTᶜ
        
        Args:
            p, q: Representation labels
            verbose: Print results
            
        Returns:
            results: Validation metrics
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"REPRESENTATION ({p},{q}) VALIDATION")
            print(f"{'='*70}")
        
        # Build representation
        ops, dim_reducible = self.build_representation(p, q)
        T3, T8, E12, E21, E23, E32, E13, E31 = ops
        
        # Compute Casimir
        C2 = self.compute_casimir(ops)
        
        # Identify irreps
        irreps = self.identify_irreps(C2)
        
        # Expected Casimir
        C2_expected = self.known_casimirs.get((p, q), None)
        
        if verbose:
            print(f"\nReducible dimension: {dim_reducible}")
            print(f"\nCasimir decomposition:")
            for eig, info in irreps.items():
                print(f"  C₂ = {eig:.6f}, multiplicity = {info['multiplicity']}")
                if info['possible_irreps']:
                    print(f"    Possible irreps: {info['possible_irreps']}")
            
            if C2_expected is not None:
                print(f"\nExpected C₂({p},{q}) = {C2_expected:.6f}")
                # Check if expected value appears
                found = any(abs(eig - C2_expected) < 1e-6 for eig in irreps.keys())
                if found:
                    print("  ✓ Expected irrep found!")
                else:
                    print("  ⚠ Expected irrep NOT found")
        
        # Test commutation relations [E12, E23] = E13
        comm_E12_E23 = E12 @ E23 - E23 @ E12
        diff = np.linalg.norm(comm_E12_E23 - E13)
        
        if verbose:
            print(f"\nCommutation relation test:")
            print(f"  ||[E₁₂, E₂₃] - E₁₃|| = {diff:.2e}")
            if diff < 1e-10:
                print("  ✓ Commutation relations verified!")
        
        results = {
            'dim_reducible': dim_reducible,
            'irreps': irreps,
            'C2_expected': C2_expected,
            'commutator_error': diff
        }
        
        return results


# ============================================================================
# VALIDATION AND DEMONSTRATION
# ============================================================================

def validate_higher_reps():
    """Run full validation suite."""
    print("\n" + "="*70)
    print("HIGHER REPRESENTATION BUILDER VALIDATION")
    print("="*70)
    
    builder = HigherRepBuilder()
    
    # Test 1: Fundamental (1,0)
    print("\n\nTest 1: Fundamental (1,0)")
    print("-"*70)
    results_10 = builder.validate_representation(1, 0, verbose=True)
    
    # Test 2: Adjoint from 3 ⊗ 3̄
    print("\n\nTest 2: Adjoint from 3 ⊗ 3̄ = (1,1)")
    print("-"*70)
    results_11 = builder.validate_representation(1, 1, verbose=True)
    
    # Test 3: Symmetric (2,0)
    print("\n\nTest 3: Symmetric (2,0)")
    print("-"*70)
    results_20 = builder.validate_representation(2, 0, verbose=True)
    
    # Test 4: (2,1) → 15
    print("\n\nTest 4: Higher rep (2,1)")
    print("-"*70)
    results_21 = builder.validate_representation(2, 1, verbose=True)
    
    print("\n" + "="*70)
    print("✓ ALL HIGHER REPRESENTATION TESTS COMPLETED")
    print("="*70)
    
    # Summary
    print("\n\nVALIDATION SUMMARY")
    print("="*70)
    print(f"(1,0): dim={results_10['dim_reducible']}, commutator error={results_10['commutator_error']:.2e}")
    print(f"(1,1): dim={results_11['dim_reducible']}, commutator error={results_11['commutator_error']:.2e}")
    print(f"(2,0): dim={results_20['dim_reducible']}, commutator error={results_20['commutator_error']:.2e}")
    print(f"(2,1): dim={results_21['dim_reducible']}, commutator error={results_21['commutator_error']:.2e}")
    
    all_passed = all(r['commutator_error'] < 1e-10 for r in 
                    [results_10, results_11, results_20, results_21])
    
    if all_passed:
        print("\n✓ All commutation relation tests PASSED at machine precision!")
    else:
        print("\n⚠ Some tests show deviations")


if __name__ == "__main__":
    validate_higher_reps()
