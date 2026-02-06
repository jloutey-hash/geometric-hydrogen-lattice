# SU(3) Irrep-Restricted Operators
"""
Construct SU(3) generators restricted to each irreducible representation.

T_a^(irrep) = P_irrep T_a^(product) P_irrep

Validates commutation relations inside each irrep.
"""

import numpy as np
from typing import Dict, List, Tuple
from weight_basis_gellmann import WeightBasisSU3
from irrep_projectors import IrrepProjector


class IrrepOperators:
    """
    Builder for SU(3) operators restricted to irreps.
    """
    
    def __init__(self):
        """Initialize with projectors and fundamental generators."""
        self.proj = IrrepProjector()
        self.fund = WeightBasisSU3(1, 0)
        self.antifund = WeightBasisSU3(0, 1)
        
        # Build all irrep operators
        self.irrep_generators = {}
        self._build_all_irrep_operators()
    
    def _tensor_product_operators(self, ops1: List[np.ndarray],
                                  ops2: List[np.ndarray]) -> List[np.ndarray]:
        """
        Tensor product of operator lists.
        
        T^(R1⊗R2) = T^(R1) ⊗ I + I ⊗ T^(R2)
        """
        dim1 = ops1[0].shape[0]
        dim2 = ops2[0].shape[0]
        I1 = np.eye(dim1, dtype=complex)
        I2 = np.eye(dim2, dtype=complex)
        
        ops_prod = []
        for T1, T2 in zip(ops1, ops2):
            T_prod = np.kron(T1, I2) + np.kron(I1, T2)
            ops_prod.append(T_prod)
        
        return ops_prod
    
    def _build_all_irrep_operators(self):
        """Build generators for all known irreps."""
        # Get fundamental generators
        ops_fund = [self.fund.T3, self.fund.T8, self.fund.E12, self.fund.E21,
                   self.fund.E23, self.fund.E32, self.fund.E13, self.fund.E31]
        ops_anti = [self.antifund.T3, self.antifund.T8, self.antifund.E12, self.antifund.E21,
                   self.antifund.E23, self.antifund.E32, self.antifund.E13, self.antifund.E31]
        
        # 3 ⊗ 3 = 6 ⊕ 3̄
        ops_3x3 = self._tensor_product_operators(ops_fund, ops_fund)
        self.irrep_generators['6'] = self._project_operators(
            ops_3x3, self.proj.projectors['3x3']['6']
        )
        self.irrep_generators['3bar'] = self._project_operators(
            ops_3x3, self.proj.projectors['3x3']['3bar']
        )
        
        # 3 ⊗ 3̄ = 1 ⊕ 8
        ops_3x3bar = self._tensor_product_operators(ops_fund, ops_anti)
        self.irrep_generators['1'] = self._project_operators(
            ops_3x3bar, self.proj.projectors['3x3bar']['1']
        )
        self.irrep_generators['8'] = self._project_operators(
            ops_3x3bar, self.proj.projectors['3x3bar']['8']
        )
        
        # 3̄ ⊗ 3̄ = 6̄ ⊕ 3
        ops_3barx3bar = self._tensor_product_operators(ops_anti, ops_anti)
        self.irrep_generators['6bar'] = self._project_operators(
            ops_3barx3bar, self.proj.projectors['3barx3bar']['6bar']
        )
        self.irrep_generators['3'] = self._project_operators(
            ops_3barx3bar, self.proj.projectors['3barx3bar']['3']
        )
    
    def _project_operators(self, ops_product: List[np.ndarray],
                          P: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Project operators onto irrep subspace and extract reduced matrix.
        
        First project: P T P
        Then extract subspace representation by similarity transform
        """
        ops_irrep = {}
        names = ['T3', 'T8', 'E12', 'E21', 'E23', 'E32', 'E13', 'E31']
        
        # Get eigenspace of P to extract irrep subspace
        # P has rank = dim(irrep), find the non-zero eigenvectors
        evals, evecs = np.linalg.eigh(P)
        
        # Tolerance for non-zero eigenvalues
        tol = 1e-10
        irrep_indices = np.where(evals > tol)[0]
        irrep_basis = evecs[:, irrep_indices]  # columns are basis vectors
        
        # Transform operators to irrep basis: T_irrep = V† T V
        # where V are the irrep basis vectors
        for i, name in enumerate(names):
            T_product = ops_product[i]
            # Transform to irrep basis
            T_irrep = irrep_basis.conj().T @ T_product @ irrep_basis
            ops_irrep[name] = T_irrep
        
        return ops_irrep
    
    def validate_commutation_relations(self, ops: Dict[str, np.ndarray],
                                      irrep_name: str, verbose: bool = True) -> float:
        """
        Validate SU(3) commutation relations inside irrep.
        
        Tests:
        - [E12, E23] = E13
        - [E23, E31] = E21
        - [E31, E12] = E32
        - [T3, E12] = E12
        - [T3, E23] = -E23/2
        
        Args:
            ops: Dictionary of generators
            irrep_name: Name of irrep
            verbose: Print results
            
        Returns:
            max_error: Maximum commutator error
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"COMMUTATION RELATIONS: {irrep_name}")
            print(f"{'='*70}")
        
        max_error = 0.0
        errors = []
        
        # Test [E12, E23] = E13
        comm = ops['E12'] @ ops['E23'] - ops['E23'] @ ops['E12']
        err = np.linalg.norm(comm - ops['E13'])
        errors.append(('E12', 'E23', 'E13', err))
        max_error = max(max_error, err)
        
        # Test [E23, E31] = E21
        comm = ops['E23'] @ ops['E31'] - ops['E31'] @ ops['E23']
        err = np.linalg.norm(comm - ops['E21'])
        errors.append(('E23', 'E31', 'E21', err))
        max_error = max(max_error, err)
        
        # Test [E31, E12] = E32
        comm = ops['E31'] @ ops['E12'] - ops['E12'] @ ops['E31']
        err = np.linalg.norm(comm - ops['E32'])
        errors.append(('E31', 'E12', 'E32', err))
        max_error = max(max_error, err)
        
        # Test [T3, E12] = E12
        comm = ops['T3'] @ ops['E12'] - ops['E12'] @ ops['T3']
        err = np.linalg.norm(comm - ops['E12'])
        errors.append(('T3', 'E12', 'E12', err))
        max_error = max(max_error, err)
        
        # Test [T3, E23] = -E23/2
        comm = ops['T3'] @ ops['E23'] - ops['E23'] @ ops['T3']
        err = np.linalg.norm(comm + ops['E23']/2)
        errors.append(('T3', 'E23', '-E23/2', err))
        max_error = max(max_error, err)
        
        if verbose:
            for (op1, op2, result, err) in errors:
                status = "✓" if err < 1e-10 else "✗"
                print(f"{status} ||[{op1}, {op2}] - {result}|| = {err:.2e}")
            
            print(f"\nMax commutator error: {max_error:.2e}")
            if max_error < 1e-10:
                print("✓ Commutation relations VERIFIED!")
            else:
                print("⚠ Commutation violations detected")
        
        return max_error
    
    def validate_hermiticity(self, ops: Dict[str, np.ndarray],
                            irrep_name: str, verbose: bool = True) -> float:
        """
        Validate Hermiticity of generators.
        
        T3, T8 should be Hermitian
        E_ij† = E_ji
        
        Args:
            ops: Dictionary of generators
            irrep_name: Name of irrep
            verbose: Print results
            
        Returns:
            max_error: Maximum Hermiticity error
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"HERMITICITY: {irrep_name}")
            print(f"{'='*70}")
        
        max_error = 0.0
        
        # Test T3 Hermitian
        err_T3 = np.linalg.norm(ops['T3'] - ops['T3'].conj().T)
        max_error = max(max_error, err_T3)
        
        # Test T8 Hermitian
        err_T8 = np.linalg.norm(ops['T8'] - ops['T8'].conj().T)
        max_error = max(max_error, err_T8)
        
        # Test E_ij† = E_ji
        err_E12 = np.linalg.norm(ops['E12'].conj().T - ops['E21'])
        err_E23 = np.linalg.norm(ops['E23'].conj().T - ops['E32'])
        err_E13 = np.linalg.norm(ops['E13'].conj().T - ops['E31'])
        max_error = max(max_error, err_E12, err_E23, err_E13)
        
        if verbose:
            print(f"||T3 - T3†|| = {err_T3:.2e}")
            print(f"||T8 - T8†|| = {err_T8:.2e}")
            print(f"||E12† - E21|| = {err_E12:.2e}")
            print(f"||E23† - E32|| = {err_E23:.2e}")
            print(f"||E13† - E31|| = {err_E13:.2e}")
            print(f"\nMax Hermiticity error: {max_error:.2e}")
            
            if max_error < 1e-14:
                print("✓ Hermiticity VERIFIED at machine precision!")
        
        return max_error
    
    def compute_casimir(self, ops: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute Casimir operator C₂ = Σ Tₐ².
        
        Uses Hermitian combinations:
        λ₁ = E12 + E21, λ₂ = -i(E12 - E21), etc.
        """
        lambda1 = ops['E12'] + ops['E21']
        lambda2 = -1j * (ops['E12'] - ops['E21'])
        lambda3 = 2 * ops['T3']
        lambda4 = ops['E23'] + ops['E32']
        lambda5 = -1j * (ops['E23'] - ops['E32'])
        lambda6 = ops['E13'] + ops['E31']
        lambda7 = -1j * (ops['E13'] - ops['E31'])
        lambda8 = 2 * ops['T8']
        
        C2 = (lambda1 @ lambda1 + lambda2 @ lambda2 + lambda3 @ lambda3 +
              lambda4 @ lambda4 + lambda5 @ lambda5 + lambda6 @ lambda6 +
              lambda7 @ lambda7 + lambda8 @ lambda8) / 4.0
        
        return C2
    
    def validate_casimir(self, ops: Dict[str, np.ndarray], irrep_name: str,
                        expected_value: float, verbose: bool = True) -> float:
        """
        Validate Casimir eigenvalue.
        
        Args:
            ops: Dictionary of generators
            irrep_name: Name of irrep
            expected_value: Expected C₂ eigenvalue
            verbose: Print results
            
        Returns:
            casimir_error: Deviation from expected value
        """
        C2 = self.compute_casimir(ops)
        eigs = np.linalg.eigvalsh(C2)
        
        # Should be proportional to identity in irrep
        mean_eig = np.mean(eigs)
        std_eig = np.std(eigs)
        casimir_error = abs(mean_eig - expected_value)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"CASIMIR: {irrep_name}")
            print(f"{'='*70}")
            print(f"C₂ eigenvalues: {np.unique(np.round(eigs, 6))}")
            print(f"Mean: {mean_eig:.6f}")
            print(f"Std: {std_eig:.2e}")
            print(f"Expected: {expected_value:.6f}")
            print(f"Error: {casimir_error:.2e}")
            
            if casimir_error < 1e-10 and std_eig < 1e-10:
                print("✓ Casimir CORRECT!")
        
        return casimir_error


# ============================================================================
# VALIDATION AND DEMONSTRATION
# ============================================================================

def validate_irrep_operators():
    """Run full validation suite for irrep-restricted operators."""
    print("\n" + "="*70)
    print("IRREP-RESTRICTED OPERATOR VALIDATION")
    print("="*70)
    
    irrep_ops = IrrepOperators()
    
    # Expected Casimir values: C₂(p,q) = (p² + q² + pq + 3p + 3q)/3
    expected_casimirs = {
        '1': 0.0,       # singlet
        '3': 4/3,       # fundamental (1,0)
        '3bar': 4/3,    # antifundamental (0,1)
        '6': 10/3,      # symmetric (2,0)
        '6bar': 10/3,   # (0,2)
        '8': 3.0        # adjoint (1,1)
    }
    
    all_errors = {}
    
    for irrep_name in ['6', '3bar', '1', '8', '6bar', '3']:
        print(f"\n\n{'='*70}")
        print(f"IRREP: {irrep_name}")
        print(f"{'='*70}")
        
        ops = irrep_ops.irrep_generators[irrep_name]
        
        # Test 1: Commutation relations
        comm_err = irrep_ops.validate_commutation_relations(ops, irrep_name, verbose=True)
        
        # Test 2: Hermiticity
        herm_err = irrep_ops.validate_hermiticity(ops, irrep_name, verbose=True)
        
        # Test 3: Casimir
        casimir_err = irrep_ops.validate_casimir(ops, irrep_name,
                                                 expected_casimirs[irrep_name], verbose=True)
        
        all_errors[irrep_name] = {
            'commutator': comm_err,
            'hermiticity': herm_err,
            'casimir': casimir_err
        }
    
    print("\n" + "="*70)
    print("✓ ALL IRREP OPERATOR TESTS COMPLETED")
    print("="*70)
    
    # Summary
    print("\n\nVALIDATION SUMMARY")
    print("="*70)
    for irrep_name, errs in all_errors.items():
        print(f"{irrep_name:6s}: comm={errs['commutator']:.2e}, herm={errs['hermiticity']:.2e}, C₂={errs['casimir']:.2e}")
    
    max_comm = max(e['commutator'] for e in all_errors.values())
    max_herm = max(e['hermiticity'] for e in all_errors.values())
    max_cas = max(e['casimir'] for e in all_errors.values())
    
    print(f"\nMax errors: comm={max_comm:.2e}, herm={max_herm:.2e}, C₂={max_cas:.2e}")
    
    if max_comm < 1e-10 and max_herm < 1e-14 and max_cas < 1e-10:
        print("\n✓ All irrep operator tests PASSED!")
    else:
        print("\n⚠ Some tests show deviations")


if __name__ == "__main__":
    validate_irrep_operators()
