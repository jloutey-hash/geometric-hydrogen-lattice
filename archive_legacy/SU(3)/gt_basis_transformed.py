"""
GT Basis SU(3) Operators via Unitary Transformation
====================================================

Transform perfect weight-basis operators to GT basis using unitary transformation.

Strategy:
1. Generate GT patterns for (p,q)
2. Build T3_GT and T8_GT (diagonal in GT basis)
3. Construct unitary U: weight basis → GT basis
4. Transform all operators: O_GT = U† O_weight U
"""

import numpy as np
from typing import List, Tuple, Dict
from weight_basis_gellmann import WeightBasisSU3
from lattice import SU3Lattice


class GTBasisSU3:
    """SU(3) operators in GT basis, obtained by transforming weight-basis operators."""
    
    def __init__(self, p: int, q: int):
        """
        Initialize GT basis operators by transforming from weight basis.
        
        Args:
            p, q: Dynkin labels
        """
        self.p = p
        self.q = q
        
        # Get weight-basis operators (perfect reference)
        self.weight_ops = WeightBasisSU3(p, q)
        
        # Generate GT patterns
        self.lattice = SU3Lattice(max_p=p, max_q=q)
        self.gt_state_dicts = [s for s in self.lattice.states if s['p'] == p and s['q'] == q]
        self.gt_states = [(s['m13'], s['m23'], s['m33'], s['m12'], s['m22'], s['m11']) 
                          for s in self.gt_state_dicts]
        self.dim = len(self.gt_states)
        
        # Build T3_GT and T8_GT (diagonal in GT basis)
        self.T3_GT_diag = self._build_T3_GT_diagonal()
        self.T8_GT_diag = self._build_T8_GT_diagonal()
        
        # Construct unitary transformation
        self.U = self._build_unitary_transformation()
        
        # Transform all operators
        self._transform_operators()
        
    def _build_T3_GT_diagonal(self) -> np.ndarray:
        """Build diagonal of T3 in GT basis from lattice i3 values."""
        return np.array([s['i3'] for s in self.gt_state_dicts])
    
    def _build_T8_GT_diagonal(self) -> np.ndarray:
        """Build diagonal of T8 in GT basis from lattice y values: T8 = Y*√3/2."""
        return np.array([s['y'] * np.sqrt(3) / 2 for s in self.gt_state_dicts])
    
    def _build_unitary_transformation(self) -> np.ndarray:
        """
        Build unitary matrix U: weight basis → GT basis.
        
        Strategy: Match states by their (I3, Y) quantum numbers.
        U[i,j] = 1 if GT_state[i] and weight_state[j] have same (I3, Y)
        
        For states with same (I3,Y), need to diagonalize T3_weight, T8_weight
        to find the correct correspondence.
        """
        # Get weight basis T3, T8 (diagonal)
        T3_weight = self.weight_ops.T3
        T8_weight = self.weight_ops.T8
        
        # Extract eigenvalues
        I3_weight = np.diag(T3_weight).real
        T8_weight_diag = np.diag(T8_weight).real
        Y_weight = T8_weight_diag * 2 / np.sqrt(3)
        
        # Build matching matrix
        U = np.zeros((self.dim, self.dim), dtype=complex)
        
        # For each GT state, find corresponding weight state
        used_weight_indices = []
        
        for i, gt_dict in enumerate(self.gt_state_dicts):
            I3_gt = gt_dict['i3']
            Y_gt = gt_dict['y']
            
            # Find weight state with matching quantum numbers
            for j in range(self.dim):
                if j in used_weight_indices:
                    continue
                    
                I3_w = I3_weight[j]
                Y_w = Y_weight[j]
                
                # Check if quantum numbers match (within tolerance)
                if abs(I3_gt - I3_w) < 1e-10 and abs(Y_gt - Y_w) < 1e-10:
                    U[i, j] = 1.0
                    used_weight_indices.append(j)
                    break
        
        # Verify U is unitary (at least check dimensions and basic properties)
        if not np.allclose(U @ U.conj().T, np.eye(self.dim), atol=1e-10):
            print("WARNING: U is not unitary!")
            print(f"U @ U† =\n{U @ U.conj().T}")
        
        return U
    
    def _transform_operators(self):
        """Transform all weight-basis operators to GT basis: O_GT = U† O_weight U."""
        U_dag = self.U.conj().T
        
        # Transform Cartan operators
        self.T3 = U_dag @ self.weight_ops.T3 @ self.U
        self.T8 = U_dag @ self.weight_ops.T8 @ self.U
        
        # Transform ladder operators
        self.E12 = U_dag @ self.weight_ops.E12 @ self.U
        self.E21 = U_dag @ self.weight_ops.E21 @ self.U
        
        self.E23 = U_dag @ self.weight_ops.E23 @ self.U
        self.E32 = U_dag @ self.weight_ops.E32 @ self.U
        
        self.E13 = U_dag @ self.weight_ops.E13 @ self.U
        self.E31 = U_dag @ self.weight_ops.E31 @ self.U
    
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
        """Validate SU(3) algebra relations in GT basis."""
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
        results['E32-E23†'] = np.max(np.abs(self.E32 - self.E23.conj().T))
        results['E31-E13†'] = np.max(np.abs(self.E31 - self.E13.conj().T))
        results['T3_hermitian'] = np.max(np.abs(self.T3 - self.T3.conj().T))
        results['T8_hermitian'] = np.max(np.abs(self.T8 - self.T8.conj().T))
        
        # Check T3, T8 are diagonal
        T3_off_diag = self.T3 - np.diag(np.diag(self.T3))
        T8_off_diag = self.T8 - np.diag(np.diag(self.T8))
        results['T3_diagonal'] = np.max(np.abs(T3_off_diag))
        results['T8_diagonal'] = np.max(np.abs(T8_off_diag))
        
        return results


if __name__ == "__main__":
    print("="*80)
    print("GT Basis SU(3) via Unitary Transformation")
    print("="*80)
    
    for p, q in [(1, 0), (0, 1)]:
        print(f"\n{'='*80}")
        print(f"(p,q) = ({p},{q})")
        print(f"{'='*80}")
        
        ops = GTBasisSU3(p, q)
        
        print(f"Dimension: {ops.dim}")
        print(f"\nGT states: {ops.gt_states}")
        
        # Show T3, T8 diagonals
        print(f"\nT3 diagonal (GT basis): {np.diag(ops.T3).real}")
        print(f"T8 diagonal (GT basis): {np.diag(ops.T8).real}")
        
        # Check transformation preserved quantum numbers
        print(f"\nT3 GT pattern formula: {ops.T3_GT_diag}")
        print(f"T3 from transformed operator: {np.diag(ops.T3).real}")
        print(f"Match: {np.allclose(ops.T3_GT_diag, np.diag(ops.T3).real)}")
        
        # Validate
        results = ops.validate()
        
        print("\nValidation Results:")
        for key, val in results.items():
            status = "✓" if val < 1e-10 else "✗"
            print(f"  {key}: {val:.2e} {status}")
        
        # Casimir value
        C2 = ops.get_casimir()
        print(f"\nCasimir eigenvalues: {np.diag(C2).real}")
        print(f"Expected: {ops.theoretical_casimir():.6f}")
        
        # Compare with weight basis
        print(f"\nComparison with weight basis:")
        print(f"  Weight Casimir: {np.diag(ops.weight_ops.get_casimir()).real}")
        print(f"  GT Casimir: {np.diag(C2).real}")
