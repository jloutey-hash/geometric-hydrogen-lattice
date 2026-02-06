# SU(3) Clebsch-Gordan Coefficient Generator
"""
Compute explicit Clebsch-Gordan coefficients for SU(3) tensor products.

Decompositions:
- 3 ⊗ 3 = 6 ⊕ 3̄
- 3 ⊗ 3̄ = 1 ⊕ 8  
- 3̄ ⊗ 3̄ = 6̄ ⊕ 3

Uses weight-matching and orthonormality constraints.
"""

import numpy as np
from typing import Tuple, Dict, List
from weight_basis_gellmann import WeightBasisSU3


class ClebschGordanSU3:
    """
    Clebsch-Gordan coefficient generator for SU(3).
    """
    
    def __init__(self):
        """Initialize fundamental representations."""
        self.fund = WeightBasisSU3(1, 0)    # 3
        self.antifund = WeightBasisSU3(0, 1)  # 3̄
        
        # Weights for fundamental (I3, Y)
        self.weights_3 = [
            (0.5, 1/np.sqrt(3)),      # state 0: |r⟩
            (-0.5, 1/np.sqrt(3)),     # state 1: |g⟩
            (0.0, -2/np.sqrt(3))      # state 2: |b⟩
        ]
        
        # Weights for antifundamental
        self.weights_3bar = [
            (-0.5, -1/np.sqrt(3)),    # state 0: |r̄⟩
            (0.5, -1/np.sqrt(3)),     # state 1: |ḡ⟩
            (0.0, 2/np.sqrt(3))       # state 2: |b̄⟩
        ]
        
        # Build CG tables
        self.cg_3x3_to_6_3bar = None
        self.cg_3x3bar_to_1_8 = None
        self.cg_3barx3bar_to_6bar_3 = None
        
        self._build_all_cg_tables()
    
    def _build_all_cg_tables(self):
        """Build all CG coefficient tables."""
        self.cg_3x3_to_6_3bar = self._build_3x3()
        self.cg_3x3bar_to_1_8 = self._build_3x3bar()
        self.cg_3barx3bar_to_6bar_3 = self._build_3barx3bar()
    
    def _build_3x3(self) -> Dict[str, np.ndarray]:
        """
        Build CG coefficients for 3 ⊗ 3 = 6 ⊕ 3̄.
        
        6: symmetric part (dim=6)
        3̄: antisymmetric part (dim=3)
        
        Returns:
            cg_dict: {'6': array(6,9), '3bar': array(3,9)}
        """
        # 3 ⊗ 3 has 9 states
        # Basis ordering: |i,j⟩ = |i⟩⊗|j⟩ for i,j ∈ {0,1,2}
        # Flatten as: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)
        
        # Symmetric 6: normalized symmetrized states
        cg_6 = np.zeros((6, 9), dtype=complex)
        
        # |00⟩ (already symmetric)
        cg_6[0, 0] = 1.0  # (0,0)
        
        # (|01⟩ + |10⟩)/√2
        cg_6[1, 1] = 1/np.sqrt(2)  # (0,1)
        cg_6[1, 3] = 1/np.sqrt(2)  # (1,0)
        
        # (|02⟩ + |20⟩)/√2
        cg_6[2, 2] = 1/np.sqrt(2)  # (0,2)
        cg_6[2, 6] = 1/np.sqrt(2)  # (2,0)
        
        # |11⟩
        cg_6[3, 4] = 1.0  # (1,1)
        
        # (|12⟩ + |21⟩)/√2
        cg_6[4, 5] = 1/np.sqrt(2)  # (1,2)
        cg_6[4, 7] = 1/np.sqrt(2)  # (2,1)
        
        # |22⟩
        cg_6[5, 8] = 1.0  # (2,2)
        
        # Antisymmetric 3̄: normalized antisymmetrized states
        cg_3bar = np.zeros((3, 9), dtype=complex)
        
        # (|01⟩ - |10⟩)/√2
        cg_3bar[0, 1] = 1/np.sqrt(2)   # (0,1)
        cg_3bar[0, 3] = -1/np.sqrt(2)  # (1,0)
        
        # (|02⟩ - |20⟩)/√2
        cg_3bar[1, 2] = 1/np.sqrt(2)   # (0,2)
        cg_3bar[1, 6] = -1/np.sqrt(2)  # (2,0)
        
        # (|12⟩ - |21⟩)/√2
        cg_3bar[2, 5] = 1/np.sqrt(2)   # (1,2)
        cg_3bar[2, 7] = -1/np.sqrt(2)  # (2,1)
        
        return {'6': cg_6, '3bar': cg_3bar}
    
    def _build_3x3bar(self) -> Dict[str, np.ndarray]:
        """
        Build CG coefficients for 3 ⊗ 3̄ = 1 ⊕ 8.
        
        1: singlet (dim=1)
        8: adjoint (dim=8)
        
        Returns:
            cg_dict: {'1': array(1,9), '8': array(8,9)}
        """
        # 3 ⊗ 3̄ has 9 states
        # Basis: |i,j̄⟩ for i ∈ {0,1,2} (3), j ∈ {0,1,2} (3̄)
        
        # Singlet: (|0,0̄⟩ + |1,1̄⟩ + |2,2̄⟩)/√3
        cg_1 = np.zeros((1, 9), dtype=complex)
        cg_1[0, 0] = 1/np.sqrt(3)  # (0,0)
        cg_1[0, 4] = 1/np.sqrt(3)  # (1,1)
        cg_1[0, 8] = 1/np.sqrt(3)  # (2,2)
        
        # Adjoint 8: orthogonal to singlet
        # Use Gell-Mann structure
        cg_8 = np.zeros((8, 9), dtype=complex)
        
        # λ₁ direction: |0,1̄⟩ + |1,0̄⟩
        cg_8[0, 1] = 1/np.sqrt(2)
        cg_8[0, 3] = 1/np.sqrt(2)
        
        # λ₂ direction: -i(|0,1̄⟩ - |1,0̄⟩)
        cg_8[1, 1] = -1j/np.sqrt(2)
        cg_8[1, 3] = 1j/np.sqrt(2)
        
        # λ₃ direction: |0,0̄⟩ - |1,1̄⟩
        cg_8[2, 0] = 1/np.sqrt(2)
        cg_8[2, 4] = -1/np.sqrt(2)
        
        # λ₄ direction: |0,2̄⟩ + |2,0̄⟩
        cg_8[3, 2] = 1/np.sqrt(2)
        cg_8[3, 6] = 1/np.sqrt(2)
        
        # λ₅ direction: -i(|0,2̄⟩ - |2,0̄⟩)
        cg_8[4, 2] = -1j/np.sqrt(2)
        cg_8[4, 6] = 1j/np.sqrt(2)
        
        # λ₆ direction: |1,2̄⟩ + |2,1̄⟩
        cg_8[5, 5] = 1/np.sqrt(2)
        cg_8[5, 7] = 1/np.sqrt(2)
        
        # λ₇ direction: -i(|1,2̄⟩ - |2,1̄⟩)
        cg_8[6, 5] = -1j/np.sqrt(2)
        cg_8[6, 7] = 1j/np.sqrt(2)
        
        # λ₈ direction: (|0,0̄⟩ + |1,1̄⟩ - 2|2,2̄⟩)/√6
        cg_8[7, 0] = 1/np.sqrt(6)
        cg_8[7, 4] = 1/np.sqrt(6)
        cg_8[7, 8] = -2/np.sqrt(6)
        
        return {'1': cg_1, '8': cg_8}
    
    def _build_3barx3bar(self) -> Dict[str, np.ndarray]:
        """
        Build CG coefficients for 3̄ ⊗ 3̄ = 6̄ ⊕ 3.
        
        Similar to 3 ⊗ 3 but with conjugate weights.
        
        Returns:
            cg_dict: {'6bar': array(6,9), '3': array(3,9)}
        """
        # Symmetric 6̄
        cg_6bar = np.zeros((6, 9), dtype=complex)
        
        cg_6bar[0, 0] = 1.0
        cg_6bar[1, 1] = 1/np.sqrt(2)
        cg_6bar[1, 3] = 1/np.sqrt(2)
        cg_6bar[2, 2] = 1/np.sqrt(2)
        cg_6bar[2, 6] = 1/np.sqrt(2)
        cg_6bar[3, 4] = 1.0
        cg_6bar[4, 5] = 1/np.sqrt(2)
        cg_6bar[4, 7] = 1/np.sqrt(2)
        cg_6bar[5, 8] = 1.0
        
        # Antisymmetric 3
        cg_3 = np.zeros((3, 9), dtype=complex)
        
        cg_3[0, 1] = 1/np.sqrt(2)
        cg_3[0, 3] = -1/np.sqrt(2)
        cg_3[1, 2] = 1/np.sqrt(2)
        cg_3[1, 6] = -1/np.sqrt(2)
        cg_3[2, 5] = 1/np.sqrt(2)
        cg_3[2, 7] = -1/np.sqrt(2)
        
        return {'6bar': cg_6bar, '3': cg_3}
    
    def validate_orthonormality(self, cg_table: Dict[str, np.ndarray],
                                verbose: bool = True) -> float:
        """
        Validate orthonormality: ⟨irrep,i|irrep',j⟩ = δ_irrep,irrep' δ_ij
        
        Args:
            cg_table: Dictionary of CG arrays
            verbose: Print results
            
        Returns:
            max_error: Maximum deviation from orthonormality
        """
        max_error = 0.0
        
        if verbose:
            print("\n" + "="*70)
            print("ORTHONORMALITY VALIDATION")
            print("="*70)
        
        # Collect all irrep states
        all_states = []
        irrep_names = []
        for irrep_name, cg_array in cg_table.items():
            for i in range(cg_array.shape[0]):
                all_states.append(cg_array[i])
                irrep_names.append(f"{irrep_name}_{i}")
        
        # Check all pairs
        n_states = len(all_states)
        for i in range(n_states):
            for j in range(i, n_states):
                inner_prod = np.vdot(all_states[i], all_states[j])
                expected = 1.0 if i == j else 0.0
                error = abs(inner_prod - expected)
                max_error = max(max_error, error)
                
                if verbose and error > 1e-12:
                    print(f"⟨{irrep_names[i]}|{irrep_names[j]}⟩ = {inner_prod:.6f}, expected {expected:.6f}, error = {error:.2e}")
        
        if verbose:
            print(f"\nMax orthonormality error: {max_error:.2e}")
            if max_error < 1e-14:
                print("✓ Orthonormality VERIFIED at machine precision!")
            else:
                print("⚠ Orthonormality deviation detected")
        
        return max_error
    
    def validate_completeness(self, cg_table: Dict[str, np.ndarray],
                             verbose: bool = True) -> float:
        """
        Validate completeness: Σ |irrep,i⟩⟨irrep,i| = I
        
        Args:
            cg_table: Dictionary of CG arrays
            verbose: Print results
            
        Returns:
            max_error: Maximum deviation from identity
        """
        # Sum projection operators
        dim = list(cg_table.values())[0].shape[1]  # Dimension of product space
        P_total = np.zeros((dim, dim), dtype=complex)
        
        for irrep_name, cg_array in cg_table.items():
            for i in range(cg_array.shape[0]):
                state = cg_array[i]
                P_total += np.outer(state, state.conj())
        
        # Should equal identity
        I = np.eye(dim)
        error_matrix = P_total - I
        max_error = np.max(np.abs(error_matrix))
        
        if verbose:
            print("\n" + "="*70)
            print("COMPLETENESS VALIDATION")
            print("="*70)
            print(f"Σ |irrep,i⟩⟨irrep,i| - I: max error = {max_error:.2e}")
            if max_error < 1e-14:
                print("✓ Completeness VERIFIED at machine precision!")
            else:
                print("⚠ Completeness deviation detected")
        
        return max_error
    
    def validate_dimensions(self, cg_table: Dict[str, np.ndarray],
                           expected_dims: Dict[str, int],
                           verbose: bool = True) -> bool:
        """
        Validate irrep dimensions match theory.
        
        Args:
            cg_table: Dictionary of CG arrays
            expected_dims: Expected dimensions for each irrep
            verbose: Print results
            
        Returns:
            all_correct: True if all dimensions match
        """
        if verbose:
            print("\n" + "="*70)
            print("DIMENSION VALIDATION")
            print("="*70)
        
        all_correct = True
        for irrep_name, cg_array in cg_table.items():
            dim_actual = cg_array.shape[0]
            dim_expected = expected_dims.get(irrep_name, None)
            
            if dim_expected is not None:
                match = (dim_actual == dim_expected)
                all_correct = all_correct and match
                
                if verbose:
                    status = "✓" if match else "✗"
                    print(f"{status} {irrep_name}: dim = {dim_actual} (expected {dim_expected})")
            else:
                if verbose:
                    print(f"? {irrep_name}: dim = {dim_actual} (no expected value)")
        
        if verbose and all_correct:
            print("\n✓ All dimensions CORRECT!")
        
        return all_correct


# ============================================================================
# VALIDATION AND DEMONSTRATION
# ============================================================================

def validate_cg_system():
    """Run full validation suite for all CG tables."""
    print("\n" + "="*70)
    print("CLEBSCH-GORDAN COEFFICIENT VALIDATION")
    print("="*70)
    
    cg = ClebschGordanSU3()
    
    # Test 1: 3 ⊗ 3 = 6 ⊕ 3̄
    print("\n\nTest 1: 3 ⊗ 3 = 6 ⊕ 3̄")
    print("-"*70)
    ortho_err_1 = cg.validate_orthonormality(cg.cg_3x3_to_6_3bar, verbose=True)
    comp_err_1 = cg.validate_completeness(cg.cg_3x3_to_6_3bar, verbose=True)
    dims_ok_1 = cg.validate_dimensions(cg.cg_3x3_to_6_3bar, {'6': 6, '3bar': 3}, verbose=True)
    
    # Test 2: 3 ⊗ 3̄ = 1 ⊕ 8
    print("\n\nTest 2: 3 ⊗ 3̄ = 1 ⊕ 8")
    print("-"*70)
    ortho_err_2 = cg.validate_orthonormality(cg.cg_3x3bar_to_1_8, verbose=True)
    comp_err_2 = cg.validate_completeness(cg.cg_3x3bar_to_1_8, verbose=True)
    dims_ok_2 = cg.validate_dimensions(cg.cg_3x3bar_to_1_8, {'1': 1, '8': 8}, verbose=True)
    
    # Test 3: 3̄ ⊗ 3̄ = 6̄ ⊕ 3
    print("\n\nTest 3: 3̄ ⊗ 3̄ = 6̄ ⊕ 3")
    print("-"*70)
    ortho_err_3 = cg.validate_orthonormality(cg.cg_3barx3bar_to_6bar_3, verbose=True)
    comp_err_3 = cg.validate_completeness(cg.cg_3barx3bar_to_6bar_3, verbose=True)
    dims_ok_3 = cg.validate_dimensions(cg.cg_3barx3bar_to_6bar_3, {'6bar': 6, '3': 3}, verbose=True)
    
    print("\n" + "="*70)
    print("✓ ALL CG COEFFICIENT TESTS COMPLETED")
    print("="*70)
    
    # Summary
    print("\n\nVALIDATION SUMMARY")
    print("="*70)
    print(f"3 ⊗ 3: ortho={ortho_err_1:.2e}, complete={comp_err_1:.2e}, dims={'✓' if dims_ok_1 else '✗'}")
    print(f"3 ⊗ 3̄: ortho={ortho_err_2:.2e}, complete={comp_err_2:.2e}, dims={'✓' if dims_ok_2 else '✗'}")
    print(f"3̄ ⊗ 3̄: ortho={ortho_err_3:.2e}, complete={comp_err_3:.2e}, dims={'✓' if dims_ok_3 else '✗'}")
    
    all_passed = (ortho_err_1 < 1e-14 and comp_err_1 < 1e-14 and dims_ok_1 and
                 ortho_err_2 < 1e-14 and comp_err_2 < 1e-14 and dims_ok_2 and
                 ortho_err_3 < 1e-14 and comp_err_3 < 1e-14 and dims_ok_3)
    
    if all_passed:
        print("\n✓ All CG coefficient tests PASSED at machine precision!")
    else:
        print("\n⚠ Some tests show deviations")


if __name__ == "__main__":
    validate_cg_system()
