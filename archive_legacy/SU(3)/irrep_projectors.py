# SU(3) Irrep Projection Operators
"""
Construct projection operators onto irreducible representations using
Clebsch-Gordan coefficients.

P_irrep = Σ_i |irrep,i⟩⟨irrep,i|
"""

import numpy as np
from typing import Dict, Tuple
from clebsch_gordan_su3 import ClebschGordanSU3


class IrrepProjector:
    """
    Projection operator builder for SU(3) irreps.
    """
    
    def __init__(self):
        """Initialize with CG coefficients."""
        self.cg = ClebschGordanSU3()
        
        # Build all projection operators
        self.projectors = {}
        self._build_all_projectors()
    
    def _build_all_projectors(self):
        """Build projection operators for all known decompositions."""
        # 3 ⊗ 3 = 6 ⊕ 3̄
        self.projectors['3x3'] = self._build_projectors_from_cg(
            self.cg.cg_3x3_to_6_3bar
        )
        
        # 3 ⊗ 3̄ = 1 ⊕ 8
        self.projectors['3x3bar'] = self._build_projectors_from_cg(
            self.cg.cg_3x3bar_to_1_8
        )
        
        # 3̄ ⊗ 3̄ = 6̄ ⊕ 3
        self.projectors['3barx3bar'] = self._build_projectors_from_cg(
            self.cg.cg_3barx3bar_to_6bar_3
        )
    
    def _build_projectors_from_cg(self, cg_table: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Build projection operators from CG table.
        
        P_irrep = Σ_i |irrep,i⟩⟨irrep,i|
        
        Args:
            cg_table: Dictionary with irrep CG coefficients
            
        Returns:
            projectors: Dictionary with projection matrices
        """
        projectors = {}
        
        for irrep_name, cg_array in cg_table.items():
            dim_product = cg_array.shape[1]
            P = np.zeros((dim_product, dim_product), dtype=complex)
            
            # Sum over all states in this irrep
            for i in range(cg_array.shape[0]):
                state = cg_array[i]
                P += np.outer(state, state.conj())
            
            projectors[irrep_name] = P
        
        return projectors
    
    def project_state(self, psi: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Project state onto irrep subspace.
        
        |ψ_irrep⟩ = P |ψ⟩
        
        Args:
            psi: State vector in product space
            P: Projection operator
            
        Returns:
            psi_projected: Projected state
        """
        return P @ psi
    
    def project_operator(self, O: np.ndarray, P: np.ndarray) -> np.ndarray:
        """
        Project operator onto irrep subspace.
        
        O_irrep = P O P
        
        Args:
            O: Operator in product space
            P: Projection operator
            
        Returns:
            O_projected: Projected operator
        """
        return P @ O @ P
    
    def validate_projector_properties(self, P: np.ndarray, irrep_name: str,
                                     expected_dim: int, verbose: bool = True) -> Dict[str, float]:
        """
        Validate projection operator properties.
        
        Tests:
        - P² = P (idempotency)
        - P† = P (Hermiticity)
        - Tr(P) = dim(irrep) (trace)
        
        Args:
            P: Projection operator
            irrep_name: Name of irrep
            expected_dim: Expected dimension
            verbose: Print results
            
        Returns:
            errors: Dictionary of error measures
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"PROJECTOR VALIDATION: {irrep_name}")
            print(f"{'='*70}")
        
        # Test 1: Idempotency P² = P
        P2 = P @ P
        idempotent_error = np.linalg.norm(P2 - P)
        
        # Test 2: Hermiticity P† = P
        hermitian_error = np.linalg.norm(P - P.conj().T)
        
        # Test 3: Trace Tr(P) = dim
        trace = np.trace(P)
        trace_error = abs(trace - expected_dim)
        
        if verbose:
            print(f"\nIdempotency: ||P² - P|| = {idempotent_error:.2e}")
            print(f"Hermiticity: ||P - P†|| = {hermitian_error:.2e}")
            print(f"Trace: Tr(P) = {trace.real:.6f} (expected {expected_dim})")
            print(f"Trace error: {trace_error:.2e}")
            
            if idempotent_error < 1e-14 and hermitian_error < 1e-14 and trace_error < 1e-14:
                print(f"\n✓ All projector properties VERIFIED at machine precision!")
            else:
                print(f"\n⚠ Some properties show deviations")
        
        return {
            'idempotent_error': idempotent_error,
            'hermitian_error': hermitian_error,
            'trace_error': trace_error
        }
    
    def validate_orthogonality(self, P1: np.ndarray, P2: np.ndarray,
                              name1: str, name2: str, verbose: bool = True) -> float:
        """
        Validate orthogonality between different irrep projectors.
        
        P1 P2 = 0 for different irreps
        
        Args:
            P1, P2: Projection operators
            name1, name2: Irrep names
            verbose: Print results
            
        Returns:
            ortho_error: ||P1 P2||
        """
        P1P2 = P1 @ P2
        ortho_error = np.linalg.norm(P1P2)
        
        if verbose:
            print(f"\nOrthogonality {name1} ⊥ {name2}:")
            print(f"  ||P_{name1} P_{name2}|| = {ortho_error:.2e}")
            if ortho_error < 1e-14:
                print(f"  ✓ Orthogonal at machine precision!")
        
        return ortho_error
    
    def validate_completeness_projectors(self, projectors: Dict[str, np.ndarray],
                                        verbose: bool = True) -> float:
        """
        Validate that projectors sum to identity.
        
        Σ P_irrep = I
        
        Args:
            projectors: Dictionary of projectors
            verbose: Print results
            
        Returns:
            completeness_error: ||Σ P - I||
        """
        # Sum all projectors
        dim = list(projectors.values())[0].shape[0]
        P_sum = np.zeros((dim, dim), dtype=complex)
        for P in projectors.values():
            P_sum += P
        
        # Compare to identity
        I = np.eye(dim)
        completeness_error = np.linalg.norm(P_sum - I)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"COMPLETENESS: Σ P_irrep = I")
            print(f"{'='*70}")
            print(f"||Σ P - I|| = {completeness_error:.2e}")
            if completeness_error < 1e-14:
                print("✓ Completeness VERIFIED at machine precision!")
        
        return completeness_error


# ============================================================================
# VALIDATION AND DEMONSTRATION
# ============================================================================

def validate_projection_operators():
    """Run full validation suite for projection operators."""
    print("\n" + "="*70)
    print("IRREP PROJECTION OPERATOR VALIDATION")
    print("="*70)
    
    proj = IrrepProjector()
    
    # Test 1: 3 ⊗ 3 projectors
    print("\n\nTest 1: 3 ⊗ 3 = 6 ⊕ 3̄ Projectors")
    print("-"*70)
    
    P_6 = proj.projectors['3x3']['6']
    P_3bar = proj.projectors['3x3']['3bar']
    
    errors_6 = proj.validate_projector_properties(P_6, '6', 6, verbose=True)
    errors_3bar = proj.validate_projector_properties(P_3bar, '3̄', 3, verbose=True)
    ortho_err_1 = proj.validate_orthogonality(P_6, P_3bar, '6', '3̄', verbose=True)
    comp_err_1 = proj.validate_completeness_projectors(proj.projectors['3x3'], verbose=True)
    
    # Test 2: 3 ⊗ 3̄ projectors
    print("\n\nTest 2: 3 ⊗ 3̄ = 1 ⊕ 8 Projectors")
    print("-"*70)
    
    P_1 = proj.projectors['3x3bar']['1']
    P_8 = proj.projectors['3x3bar']['8']
    
    errors_1 = proj.validate_projector_properties(P_1, '1', 1, verbose=True)
    errors_8 = proj.validate_projector_properties(P_8, '8', 8, verbose=True)
    ortho_err_2 = proj.validate_orthogonality(P_1, P_8, '1', '8', verbose=True)
    comp_err_2 = proj.validate_completeness_projectors(proj.projectors['3x3bar'], verbose=True)
    
    # Test 3: 3̄ ⊗ 3̄ projectors
    print("\n\nTest 3: 3̄ ⊗ 3̄ = 6̄ ⊕ 3 Projectors")
    print("-"*70)
    
    P_6bar = proj.projectors['3barx3bar']['6bar']
    P_3 = proj.projectors['3barx3bar']['3']
    
    errors_6bar = proj.validate_projector_properties(P_6bar, '6̄', 6, verbose=True)
    errors_3 = proj.validate_projector_properties(P_3, '3', 3, verbose=True)
    ortho_err_3 = proj.validate_orthogonality(P_6bar, P_3, '6̄', '3', verbose=True)
    comp_err_3 = proj.validate_completeness_projectors(proj.projectors['3barx3bar'], verbose=True)
    
    print("\n" + "="*70)
    print("✓ ALL PROJECTION OPERATOR TESTS COMPLETED")
    print("="*70)
    
    # Summary
    print("\n\nVALIDATION SUMMARY")
    print("="*70)
    
    all_errors = [
        errors_6['idempotent_error'], errors_6['hermitian_error'], errors_6['trace_error'],
        errors_3bar['idempotent_error'], errors_3bar['hermitian_error'], errors_3bar['trace_error'],
        ortho_err_1, comp_err_1,
        errors_1['idempotent_error'], errors_1['hermitian_error'], errors_1['trace_error'],
        errors_8['idempotent_error'], errors_8['hermitian_error'], errors_8['trace_error'],
        ortho_err_2, comp_err_2,
        errors_6bar['idempotent_error'], errors_6bar['hermitian_error'], errors_6bar['trace_error'],
        errors_3['idempotent_error'], errors_3['hermitian_error'], errors_3['trace_error'],
        ortho_err_3, comp_err_3
    ]
    
    max_error = max(all_errors)
    print(f"Maximum error across all tests: {max_error:.2e}")
    
    if max_error < 1e-14:
        print("\n✓ All projection operator tests PASSED at machine precision!")
    else:
        print("\n⚠ Some tests show deviations")


if __name__ == "__main__":
    validate_projection_operators()
