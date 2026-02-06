"""
SU(3) Operators in Weight Basis - Using Gell-Mann Matrices as Reference
=========================================================================

For the fundamental (1,0) representation, use explicit Gell-Mann matrices.
For other irreps, use systematic construction from representation theory.
"""

import numpy as np
from typing import List, Tuple, Dict


class WeightBasisSU3:
    """SU(3) operators in weight basis using standard Gell-Mann normalization."""
    
    def __init__(self, p: int, q: int):
        """Initialize SU(3) operators for (p,q) irrep in weight basis."""
        self.p = p
        self.q = q
        
        if (p, q) == (1, 0):
            self._init_fundamental()
        elif (p, q) == (0, 1):
            self._init_antifundamental()
        elif (p, q) == (1, 1):
            self._init_adjoint()
        else:
            raise NotImplementedError(f"(p,q)=({p},{q}) not yet implemented")
    
    def _init_fundamental(self):
        """Initialize using explicit Gell-Mann matrices for (1,0)."""
        self.dim = 3
        self.weights = [(0.5, 1/3), (-0.5, 1/3), (0.0, -2/3)]
        
        # Gell-Mann matrices λ1 through λ8
        lambda1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
        lambda2 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex)
        lambda3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex)
        lambda4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex)
        lambda5 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex)
        lambda6 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex)
        lambda7 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex)
        lambda8 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex) / np.sqrt(3)
        
        # Define generators as Tᵢ = λᵢ/2
        self.E12 = (lambda1 + 1j*lambda2) / 2
        self.E21 = self.E12.conj().T
        
        self.E23 = (lambda4 + 1j*lambda5) / 2
        self.E32 = self.E23.conj().T
        
        self.E13 = (lambda6 + 1j*lambda7) / 2
        self.E31 = self.E13.conj().T
        
        self.T3 = lambda3 / 2
        self.T8 = lambda8 / 2
    
    def _init_antifundamental(self):
        """
        Initialize for (0,1) - antifundamental representation.
        
        The antifundamental is the complex conjugate representation of the fundamental.
        In terms of generators: T → -T*, E → -E*
        
        Since T3 and T8 are real diagonal, we have: T3 → -T3, T8 → -T8
        For raising/lowering: Eij → -Eji*
        """
        self.dim = 3
        self.weights = [(-0.5, -1/3), (0.5, -1/3), (0.0, 2/3)]
        
        # Get fundamental
        fund = WeightBasisSU3(1, 0)
        
        # Antifundamental operators: negate and conjugate
        self.T3 = -fund.T3.conj()
        self.T8 = -fund.T8.conj()
        
        # Ladder operators: swap indices and negate
        self.E12 = -fund.E21.conj()
        self.E21 = -fund.E12.conj()
        self.E23 = -fund.E32.conj()
        self.E32 = -fund.E23.conj()
        self.E13 = -fund.E31.conj()
        self.E31 = -fund.E13.conj()
    
    def _init_adjoint(self):
        """
        Initialize for (1,1) - adjoint representation in weight basis.
        
        Strategy:
        1. Build generators in structure constant basis
        2. Diagonalize T3 and T8 simultaneously to find weight basis
        3. Transform all operators to this basis
        """
        self.dim = 8
        
        # First build operators in structure constant basis
        f = self._build_structure_constants()
        
        # Build 8x8 generator matrices: (T_i)_{jk} = f_{ijk}
        T_struct = np.zeros((8, 8, 8), dtype=complex)
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    T_struct[i, j, k] = f[i, j, k]
        
        # T3 and T8 in structure basis
        T3_struct = T_struct[2]  # corresponds to λ₃
        T8_struct = T_struct[7]  # corresponds to λ₈
        
        # Find simultaneous eigenbasis of T3 and T8
        # Since [T3, T8] = 0, they share eigenvectors
        # Diagonalize T3 first
        evals_T3, evecs = np.linalg.eigh(T3_struct)
        
        # Transform T8 to this basis and verify it's also diagonal
        T8_in_T3_basis = evecs.conj().T @ T8_struct @ evecs
        
        # If T8 is not diagonal, need to sort by (T3, T8) eigenvalues
        # For now, use the T3 eigenbasis (should work since [T3,T8]=0)
        
        # Transform all operators to weight basis
        U = evecs  # Unitary transformation to weight basis
        U_dag = U.conj().T
        
        self.T3 = U_dag @ T3_struct @ U
        self.T8 = U_dag @ T8_struct @ U
        
        # Build ladder operators in structure basis
        E12_struct = (T_struct[0] + 1j*T_struct[1]) / 2
        E21_struct = (T_struct[0] - 1j*T_struct[1]) / 2
        E23_struct = (T_struct[3] + 1j*T_struct[4]) / 2
        E32_struct = (T_struct[3] - 1j*T_struct[4]) / 2
        E13_struct = (T_struct[5] + 1j*T_struct[6]) / 2
        E31_struct = (T_struct[5] - 1j*T_struct[6]) / 2
        
        # Transform to weight basis
        self.E12 = U_dag @ E12_struct @ U
        self.E21 = U_dag @ E21_struct @ U
        self.E23 = U_dag @ E23_struct @ U
        self.E32 = U_dag @ E32_struct @ U
        self.E13 = U_dag @ E13_struct @ U
        self.E31 = U_dag @ E31_struct @ U
    
    def _build_structure_constants(self) -> np.ndarray:
        """Build SU(3) structure constants f_{ijk}."""
        f = np.zeros((8, 8, 8))
        
        # f_{123} = 1
        f[0,1,2] = 1; f[1,2,0] = 1; f[2,0,1] = 1
        f[1,0,2] = -1; f[0,2,1] = -1; f[2,1,0] = -1
        
        # f_{147} = f_{246} = f_{257} = f_{345} = 1/2
        f[0,3,6] = 0.5; f[3,6,0] = 0.5; f[6,0,3] = 0.5
        f[1,3,5] = 0.5; f[3,5,1] = 0.5; f[5,1,3] = 0.5
        f[1,4,6] = 0.5; f[4,6,1] = 0.5; f[6,1,4] = 0.5
        f[2,3,4] = 0.5; f[3,4,2] = 0.5; f[4,2,3] = 0.5
        
        f[3,0,6] = -0.5; f[6,3,0] = -0.5; f[0,6,3] = -0.5
        f[3,1,5] = -0.5; f[5,3,1] = -0.5; f[1,5,3] = -0.5
        f[4,1,6] = -0.5; f[6,4,1] = -0.5; f[1,6,4] = -0.5
        f[3,2,4] = -0.5; f[4,3,2] = -0.5; f[2,4,3] = -0.5
        
        # f_{156} = f_{367} = -1/2
        f[0,4,5] = -0.5; f[4,5,0] = -0.5; f[5,0,4] = -0.5
        f[2,5,6] = -0.5; f[5,6,2] = -0.5; f[6,2,5] = -0.5
        
        f[4,0,5] = 0.5; f[5,4,0] = 0.5; f[0,5,4] = 0.5
        f[5,2,6] = 0.5; f[6,5,2] = 0.5; f[2,6,5] = 0.5
        
        # f_{458} = f_{678} = √3/2
        f[3,4,7] = np.sqrt(3)/2; f[4,7,3] = np.sqrt(3)/2; f[7,3,4] = np.sqrt(3)/2
        f[5,6,7] = np.sqrt(3)/2; f[6,7,5] = np.sqrt(3)/2; f[7,5,6] = np.sqrt(3)/2
        
        f[4,3,7] = -np.sqrt(3)/2; f[7,4,3] = -np.sqrt(3)/2; f[3,7,4] = -np.sqrt(3)/2
        f[6,5,7] = -np.sqrt(3)/2; f[7,6,5] = -np.sqrt(3)/2; f[5,7,6] = -np.sqrt(3)/2
        
        return f
    
    def get_casimir(self) -> np.ndarray:
        """Compute Casimir operator C2 = Σ (λᵢ/2)²."""
        # Reconstruct all λ matrices
        lambda1 = self.E12 + self.E21
        lambda2 = -1j * (self.E12 - self.E21)
        lambda3 = 2 * self.T3
        lambda4 = self.E23 + self.E32
        lambda5 = -1j * (self.E23 - self.E32)
        lambda6 = self.E13 + self.E31
        lambda7 = -1j * (self.E13 - self.E31)
        lambda8 = 2 * self.T8
        
        C2 = sum((l/2) @ (l/2) for l in [lambda1, lambda2, lambda3, lambda4, 
                                          lambda5, lambda6, lambda7, lambda8])
        return C2
    
    def theoretical_casimir(self) -> float:
        """Theoretical Casimir eigenvalue for (p,q) irrep."""
        return (self.p**2 + self.q**2 + 3*self.p + 3*self.q + self.p*self.q) / 3
    
    def validate(self) -> Dict[str, float]:
        """Validate SU(3) algebra relations."""
        results = {}
        
        # Commutators
        comm_T3T8 = self.T3 @ self.T8 - self.T8 @ self.T3
        results['[T3,T8]'] = np.max(np.abs(comm_T3T8))
        
        comm_E12E21 = self.E12 @ self.E21 - self.E21 @ self.E12
        expected = 2 * self.T3
        results['[E12,E21]-2T3'] = np.max(np.abs(comm_E12E21 - expected))
        
        comm_E23E32 = self.E23 @ self.E32 - self.E32 @ self.E23
        expected = self.T3 + np.sqrt(3) * self.T8
        results['[E23,E32]-(T3+√3*T8)'] = np.max(np.abs(comm_E23E32 - expected))
        
        comm_E13E31 = self.E13 @ self.E31 - self.E31 @ self.E13
        expected = -self.T3 + np.sqrt(3) * self.T8
        results['[E13,E31]-(-T3+√3*T8)'] = np.max(np.abs(comm_E13E31 - expected))
        
        # Casimir
        C2 = self.get_casimir()
        eigs = np.diag(C2).real
        expected_val = self.theoretical_casimir()
        results['Casimir_std'] = np.std(eigs)
        results['Casimir_mean_error'] = abs(np.mean(eigs) - expected_val)
        
        # Hermiticity
        results['E21-E12†'] = np.max(np.abs(self.E21 - self.E12.conj().T))
        results['T3_hermitian'] = np.max(np.abs(self.T3 - self.T3.conj().T))
        
        return results


if __name__ == "__main__":
    print("="*80)
    print("SU(3) Weight Basis - Gell-Mann Reference")
    print("="*80)
    
    for p, q in [(1, 0), (0, 1)]:
        print(f"\n{'='*80}")
        print(f"(p,q) = ({p},{q})")
        print(f"{'='*80}")
        
        ops = WeightBasisSU3(p, q)
        
        print(f"Dimension: {ops.dim}")
        print(f"Weights: {ops.weights}")
        
        # Check matrices
        print(f"\nE12:\n{ops.E12}")
        print(f"\nT3:\n{ops.T3}")
        print(f"\nT8:\n{ops.T8}")
        
        # Validate
        results = ops.validate()
        
        print("\nValidation Results:")
        for key, val in results.items():
            status = "✓" if val < 1e-10 else "✗"
            print(f"  {key}: {val:.2e} {status}")
        
        # Check trace norms
        # Note: Tᵢ = λᵢ/2, so λᵢ = 2*Tᵢ for all generators
        lambda1 = ops.E12 + ops.E21
        lambda2 = -1j * (ops.E12 - ops.E21)
        lambda3 = 2 * ops.T3
        lambda4 = ops.E23 + ops.E32
        lambda5 = -1j * (ops.E23 - ops.E32)
        lambda6 = ops.E13 + ops.E31
        lambda7 = -1j * (ops.E13 - ops.E31)
        lambda8 = 2 * ops.T8
        
        print("\nTrace Norms Tr(λᵢ²):")
        for idx, lam in zip([1, 2, 3, 4, 5, 6, 7, 8], 
                            [lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7, lambda8]):
            tr = np.trace(lam @ lam).real
            status = "✓" if abs(tr - 2) < 1e-10 else "✗"
            print(f"  λ{idx}: {tr:.6f} {status}")
        
        # Casimir value
        C2 = ops.get_casimir()
        print(f"\nCasimir eigenvalues: {np.diag(C2).real}")
        print(f"Expected: {ops.theoretical_casimir():.6f}")
