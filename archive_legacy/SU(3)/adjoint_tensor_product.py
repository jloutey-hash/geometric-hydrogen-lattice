"""
SU(3) Adjoint Representation via Tensor Product
================================================

Construct (1,1) adjoint representation from 3 ⊗ 3̄ = 1 ⊕ 8.

Strategy:
1. Build tensor product space (dimension 9)
2. Construct generators: T_a^(prod) = T_a^(fund) ⊗ I + I ⊗ T_a^(antifund)
3. Project out singlet → 8-dimensional adjoint
4. Validate in weight basis
5. Transform to GT basis
"""

import numpy as np
from typing import Tuple
from weight_basis_gellmann import WeightBasisSU3
from lattice import SU3Lattice


class AdjointSU3:
    """SU(3) adjoint (1,1) representation via tensor product 3 ⊗ 3̄."""
    
    def __init__(self):
        """
        Build adjoint representation from 3 ⊗ 3̄ = 1 ⊕ 8.
        
        Steps:
        1. Get (1,0) and (0,1) operators
        2. Build 9-dimensional product space
        3. Project out singlet
        4. Extract 8-dimensional adjoint
        """
        # Get fundamental and antifundamental
        self.fund = WeightBasisSU3(1, 0)
        self.antifund = WeightBasisSU3(0, 1)
        
        # Build product space (9-dimensional)
        self._build_product_operators()
        
        # Project to adjoint (8-dimensional)
        self._project_to_adjoint()
        
        self.dim = 8
        
    def _build_product_operators(self):
        """
        Build operators in 9D product space.
        
        For A ⊗ B representation:
        T^(prod) = T^A ⊗ I_B + I_A ⊗ T^B
        
        But for 3 ⊗ 3̄, the antifundamental transforms with complex conjugate:
        T^(prod) = T^(fund) ⊗ I - I ⊗ (T^(antifund))^T
        
        Actually, since we already built antifund with negated generators,
        we can use:
        T^(prod) = T^(fund) ⊗ I + I ⊗ T^(antifund)
        """
        I3 = np.eye(3, dtype=complex)
        
        # Tensor product: A ⊗ I + I ⊗ B
        def tensor_gen(A, B):
            """Construct generator in product space."""
            return np.kron(A, I3) + np.kron(I3, B)
        
        # Build all 8 generators in product space
        self.T3_prod = tensor_gen(self.fund.T3, self.antifund.T3)
        self.T8_prod = tensor_gen(self.fund.T8, self.antifund.T8)
        
        self.E12_prod = tensor_gen(self.fund.E12, self.antifund.E12)
        self.E21_prod = tensor_gen(self.fund.E21, self.antifund.E21)
        
        self.E23_prod = tensor_gen(self.fund.E23, self.antifund.E23)
        self.E32_prod = tensor_gen(self.fund.E32, self.antifund.E32)
        
        self.E13_prod = tensor_gen(self.fund.E13, self.antifund.E13)
        self.E31_prod = tensor_gen(self.fund.E31, self.antifund.E31)
        
    def _project_to_adjoint(self):
        """
        Project product space to 8D adjoint subspace.
        
        3 ⊗ 3̄ = 1 ⊕ 8
        
        The singlet is |singlet⟩ = (1/√3) Σᵢ |i⟩⊗|ī⟩
        where |ī⟩ is the antifundamental state conjugate to |i⟩.
        
        In the tensor product basis |α⟩⊗|β⟩ (α,β=0,1,2), the singlet is:
        |singlet⟩ = (1/√3)(|0⟩⊗|0⟩ + |1⟩⊗|1⟩ + |2⟩⊗|2⟩)
        
        This is the trace state: singlet_ij = δ_ij / √3
        """
        # Construct singlet state (9D vector)
        # In tensor product basis (i,j) where i=0,1,2 (fund), j=0,1,2 (antifund)
        # Flattened as: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)
        singlet = np.zeros(9, dtype=complex)
        singlet[0] = 1/np.sqrt(3)  # (0,0)
        singlet[4] = 1/np.sqrt(3)  # (1,1)
        singlet[8] = 1/np.sqrt(3)  # (2,2)
        
        print(f"Singlet state (explicit):")
        print(f"  |singlet⟩ = (|0⟩⊗|0⟩ + |1⟩⊗|1⟩ + |2⟩⊗|2⟩) / √3")
        
        # Check that singlet is eigenstate of T3 and T8 with eigenvalue 0
        T3_singlet = self.T3_prod @ singlet
        T8_singlet = self.T8_prod @ singlet
        print(f"  T3|singlet⟩: {np.linalg.norm(T3_singlet):.6f} (should be ~0)")
        print(f"  T8|singlet⟩: {np.linalg.norm(T8_singlet):.6f} (should be ~0)")
        
        # Build projector to adjoint: P_adj = I - |singlet⟩⟨singlet|
        I9 = np.eye(9, dtype=complex)
        P_singlet = np.outer(singlet, singlet.conj())
        P_adj_full = I9 - P_singlet
        
        # Find orthonormal basis for adjoint subspace
        # Use eigendecomposition of P_adj
        evals, evecs = np.linalg.eigh(P_adj_full)
        
        # The 8 eigenvectors with eigenvalue 1 form the adjoint subspace
        # (1 eigenvector has eigenvalue 0 - that's the singlet direction)
        adjoint_indices = np.where(np.abs(evals - 1.0) < 1e-10)[0]
        
        if len(adjoint_indices) != 8:
            print(f"Warning: Found {len(adjoint_indices)} adjoint states, expected 8")
            print(f"Eigenvalues: {evals}")
        
        # Projection matrix: columns are the 8 adjoint eigenvectors
        self.P_adj = evecs[:, adjoint_indices]
        
        # Get quantum numbers in adjoint basis
        # Transform T3 and T8 to adjoint basis and diagonalize
        P_dag = self.P_adj.conj().T
        T3_adj_temp = P_dag @ self.T3_prod @ self.P_adj
        T8_adj_temp = P_dag @ self.T8_prod @ self.P_adj
        
        # Diagonalize to find weight basis
        evals_T3, evecs_T3 = np.linalg.eigh(T3_adj_temp)
        
        # Update projection to go directly to weight basis
        self.P_adj = self.P_adj @ evecs_T3
        
        # Now recompute in weight basis
        P_dag = self.P_adj.conj().T
        T3_adj_temp = P_dag @ self.T3_prod @ self.P_adj
        T8_adj_temp = P_dag @ self.T8_prod @ self.P_adj
        
        # Get eigenvalues
        self.I3_adjoint = np.diag(T3_adj_temp).real
        self.T8_adjoint_diag = np.diag(T8_adj_temp).real
        self.Y_adjoint = self.T8_adjoint_diag * 2 / np.sqrt(3)
        
        print(f"\nAdjoint states (I3, Y) in weight basis:")
        for i, (i3, y) in enumerate(zip(self.I3_adjoint, self.Y_adjoint)):
            print(f"  State {i}: I3={i3:6.3f}, Y={y:6.3f}")
        
        # Project all operators to adjoint subspace
        # O_adj = P† O_prod P
        self.T3 = P_dag @ self.T3_prod @ self.P_adj
        self.T8 = P_dag @ self.T8_prod @ self.P_adj
        
        self.E12 = P_dag @ self.E12_prod @ self.P_adj
        self.E21 = P_dag @ self.E21_prod @ self.P_adj
        
        self.E23 = P_dag @ self.E23_prod @ self.P_adj
        self.E32 = P_dag @ self.E32_prod @ self.P_adj
        
        self.E13 = P_dag @ self.E13_prod @ self.P_adj
        self.E31 = P_dag @ self.E31_prod @ self.P_adj
        
    def get_casimir(self) -> np.ndarray:
        """Compute Casimir operator C2 = Σ Tᵢ²."""
        # Reconstruct λ matrices
        lambda1 = self.E12 + self.E21
        lambda2 = -1j * (self.E12 - self.E21)
        lambda3 = 2 * self.T3
        lambda4 = self.E23 + self.E32
        lambda5 = -1j * (self.E23 - self.E32)
        lambda6 = self.E13 + self.E31
        lambda7 = -1j * (self.E13 - self.E31)
        lambda8 = 2 * self.T8
        
        # C2 = Σ (λᵢ/2)² = Σ Tᵢ²
        C2 = (lambda1 @ lambda1 + lambda2 @ lambda2 + lambda3 @ lambda3 +
              lambda4 @ lambda4 + lambda5 @ lambda5 + lambda6 @ lambda6 +
              lambda7 @ lambda7 + lambda8 @ lambda8) / 4
        
        return C2
    
    def validate(self, verbose=True):
        """Validate all SU(3) relations."""
        errors = {}
        
        # [T3, T8] = 0
        comm = self.T3 @ self.T8 - self.T8 @ self.T3
        errors['[T3,T8]'] = np.max(np.abs(comm))
        
        # [E12, E21] = 2*T3
        comm = self.E12 @ self.E21 - self.E21 @ self.E12
        expected = 2 * self.T3
        errors['[E12,E21]-2T3'] = np.max(np.abs(comm - expected))
        
        # [E23, E32] = T3 + √3*T8
        comm = self.E23 @ self.E32 - self.E32 @ self.E23
        expected = self.T3 + np.sqrt(3) * self.T8
        errors['[E23,E32]-(T3+√3*T8)'] = np.max(np.abs(comm - expected))
        
        # [E13, E31] = -T3 + √3*T8
        comm = self.E13 @ self.E31 - self.E31 @ self.E13
        expected = -self.T3 + np.sqrt(3) * self.T8
        errors['[E13,E31]-(-T3+√3*T8)'] = np.max(np.abs(comm - expected))
        
        # Casimir
        C2 = self.get_casimir()
        eigenvalues = np.linalg.eigvalsh(C2)
        casimir_mean = np.mean(eigenvalues)
        casimir_std = np.std(eigenvalues)
        errors['Casimir_std'] = casimir_std
        
        # Expected Casimir for (1,1): C2 = (p² + q² + pq + 3p + 3q)/3 = (1+1+1+3+3)/3 = 3
        expected_casimir = 3.0
        errors['Casimir_mean_error'] = abs(casimir_mean - expected_casimir)
        
        # Hermiticity
        errors['E21-E12†'] = np.max(np.abs(self.E21 - self.E12.conj().T))
        errors['E32-E23†'] = np.max(np.abs(self.E32 - self.E23.conj().T))
        errors['E31-E13†'] = np.max(np.abs(self.E31 - self.E13.conj().T))
        errors['T3_hermitian'] = np.max(np.abs(self.T3 - self.T3.conj().T))
        errors['T8_hermitian'] = np.max(np.abs(self.T8 - self.T8.conj().T))
        
        # T3, T8 diagonal
        errors['T3_diagonal'] = np.max(np.abs(self.T3 - np.diag(np.diag(self.T3))))
        errors['T8_diagonal'] = np.max(np.abs(self.T8 - np.diag(np.diag(self.T8))))
        
        if verbose:
            print("\nValidation Results:")
            for key, val in errors.items():
                status = "✓" if val < 1e-12 else "✗"
                print(f"  {key:30s}: {val:.2e} {status}")
            
            print(f"\nCasimir eigenvalues: {eigenvalues}")
            print(f"Expected: {expected_casimir}")
        
        return errors


class AdjointSU3_GT:
    """SU(3) adjoint (1,1) in GT basis via transformation from weight basis."""
    
    def __init__(self):
        """
        Build adjoint in GT basis by transforming from weight basis.
        
        Steps:
        1. Get adjoint in weight basis
        2. Generate GT patterns for (1,1)
        3. Match states by (I3, Y) quantum numbers
        4. Build unitary transformation
        5. Transform all operators
        """
        # Get weight basis adjoint
        self.weight_adj = AdjointSU3()
        self.dim = 8
        
        # Generate GT patterns
        self.lattice = SU3Lattice(max_p=1, max_q=1)
        self.gt_state_dicts = [s for s in self.lattice.states if s['p'] == 1 and s['q'] == 1]
        
        if len(self.gt_state_dicts) != 8:
            raise ValueError(f"Expected 8 GT states for (1,1), got {len(self.gt_state_dicts)}")
        
        # Build unitary transformation
        self.U = self._build_unitary_transformation()
        
        # Transform operators
        self._transform_operators()
        
    def _build_unitary_transformation(self) -> np.ndarray:
        """
        Build unitary U: weight basis → GT basis.
        
        Match states by (I3, Y) quantum numbers.
        For degenerate states, need additional quantum numbers or ordering convention.
        """
        # Get weight basis quantum numbers
        I3_weight = self.weight_adj.I3_adjoint
        Y_weight = self.weight_adj.Y_adjoint
        
        print("\nBuilding transformation matrix...")
        print(f"Weight basis (I3, Y):")
        for i in range(self.dim):
            print(f"  State {i}: I3={I3_weight[i]:6.3f}, Y={Y_weight[i]:6.3f}")
        
        print(f"\nGT basis (I3, Y):")
        for i, s in enumerate(self.gt_state_dicts):
            print(f"  State {i}: I3={s['i3']:6.3f}, Y={s['y']:6.3f}, GT={s['gt']}")
        
        # Build permutation matrix
        U = np.zeros((self.dim, self.dim), dtype=complex)
        used_weight_indices = []
        
        for i_gt, gt_dict in enumerate(self.gt_state_dicts):
            I3_gt = gt_dict['i3']
            Y_gt = gt_dict['y']
            
            # Find matching weight state
            matched = False
            for i_w in range(self.dim):
                if i_w in used_weight_indices:
                    continue
                
                if abs(I3_gt - I3_weight[i_w]) < 1e-10 and abs(Y_gt - Y_weight[i_w]) < 1e-10:
                    U[i_gt, i_w] = 1.0
                    used_weight_indices.append(i_w)
                    print(f"  GT {i_gt} ← Weight {i_w}")
                    matched = True
                    break
            
            if not matched:
                print(f"  WARNING: No match for GT state {i_gt} (I3={I3_gt:.3f}, Y={Y_gt:.3f})")
        
        # Verify unitarity
        U_check = U @ U.conj().T
        if not np.allclose(U_check, np.eye(self.dim), atol=1e-10):
            print("\nWARNING: U is not unitary!")
            print(f"U @ U† =\n{U_check}")
        else:
            print("\n✓ U is unitary")
        
        return U
    
    def _transform_operators(self):
        """Transform all operators: O_GT = U† O_weight U."""
        U_dag = self.U.conj().T
        
        self.T3 = U_dag @ self.weight_adj.T3 @ self.U
        self.T8 = U_dag @ self.weight_adj.T8 @ self.U
        
        self.E12 = U_dag @ self.weight_adj.E12 @ self.U
        self.E21 = U_dag @ self.weight_adj.E21 @ self.U
        
        self.E23 = U_dag @ self.weight_adj.E23 @ self.U
        self.E32 = U_dag @ self.weight_adj.E32 @ self.U
        
        self.E13 = U_dag @ self.weight_adj.E13 @ self.U
        self.E31 = U_dag @ self.weight_adj.E31 @ self.U
    
    def get_casimir(self) -> np.ndarray:
        """Compute Casimir operator C2."""
        # Reconstruct λ matrices
        lambda1 = self.E12 + self.E21
        lambda2 = -1j * (self.E12 - self.E21)
        lambda3 = 2 * self.T3
        lambda4 = self.E23 + self.E32
        lambda5 = -1j * (self.E23 - self.E32)
        lambda6 = self.E13 + self.E31
        lambda7 = -1j * (self.E13 - self.E31)
        lambda8 = 2 * self.T8
        
        C2 = (lambda1 @ lambda1 + lambda2 @ lambda2 + lambda3 @ lambda3 +
              lambda4 @ lambda4 + lambda5 @ lambda5 + lambda6 @ lambda6 +
              lambda7 @ lambda7 + lambda8 @ lambda8) / 4
        
        return C2
    
    def validate(self, verbose=True):
        """Validate all SU(3) relations in GT basis."""
        errors = {}
        
        # Commutators
        comm = self.T3 @ self.T8 - self.T8 @ self.T3
        errors['[T3,T8]'] = np.max(np.abs(comm))
        
        comm = self.E12 @ self.E21 - self.E21 @ self.E12
        expected = 2 * self.T3
        errors['[E12,E21]-2T3'] = np.max(np.abs(comm - expected))
        
        comm = self.E23 @ self.E32 - self.E32 @ self.E23
        expected = self.T3 + np.sqrt(3) * self.T8
        errors['[E23,E32]-(T3+√3*T8)'] = np.max(np.abs(comm - expected))
        
        comm = self.E13 @ self.E31 - self.E31 @ self.E13
        expected = -self.T3 + np.sqrt(3) * self.T8
        errors['[E13,E31]-(-T3+√3*T8)'] = np.max(np.abs(comm - expected))
        
        # Casimir
        C2 = self.get_casimir()
        eigenvalues = np.linalg.eigvalsh(C2)
        casimir_mean = np.mean(eigenvalues)
        casimir_std = np.std(eigenvalues)
        errors['Casimir_std'] = casimir_std
        errors['Casimir_mean_error'] = abs(casimir_mean - 3.0)
        
        # Hermiticity
        errors['E21-E12†'] = np.max(np.abs(self.E21 - self.E12.conj().T))
        errors['E32-E23†'] = np.max(np.abs(self.E32 - self.E23.conj().T))
        errors['E31-E13†'] = np.max(np.abs(self.E31 - self.E13.conj().T))
        errors['T3_hermitian'] = np.max(np.abs(self.T3 - self.T3.conj().T))
        errors['T8_hermitian'] = np.max(np.abs(self.T8 - self.T8.conj().T))
        
        # Diagonal
        errors['T3_diagonal'] = np.max(np.abs(self.T3 - np.diag(np.diag(self.T3))))
        errors['T8_diagonal'] = np.max(np.abs(self.T8 - np.diag(np.diag(self.T8))))
        
        if verbose:
            print("\nValidation Results (GT basis):")
            for key, val in errors.items():
                status = "✓" if val < 1e-12 else "✗"
                print(f"  {key:30s}: {val:.2e} {status}")
            
            print(f"\nCasimir eigenvalues: {eigenvalues}")
            print(f"Expected: 3.0")
        
        return errors


def main():
    """Test adjoint representation in both weight and GT bases."""
    print("="*80)
    print("SU(3) Adjoint (1,1) via Tensor Product 3 ⊗ 3̄")
    print("="*80)
    
    # Weight basis
    adj = AdjointSU3()
    print(f"\nAdjoint dimension: {adj.dim}")
    adj.validate()
    
    T3_diag = np.diag(adj.T3).real
    T8_diag = np.diag(adj.T8).real
    Y_diag = T8_diag * 2 / np.sqrt(3)
    
    print(f"\nDiagonal quantum numbers (weight basis):")
    print(f"T3: {T3_diag}")
    print(f"T8: {T8_diag}")
    print(f"Y:  {Y_diag}")
    
    C2 = adj.get_casimir()
    eigenvalues = np.linalg.eigvalsh(C2)
    print(f"\nCasimir eigenvalues: {eigenvalues}")
    print(f"Mean: {np.mean(eigenvalues):.6f}")
    print(f"Std:  {np.std(eigenvalues):.2e}")
    
    # GT basis
    print("\n" + "="*80)
    print("Transforming to GT Basis")
    print("="*80)
    
    adj_gt = AdjointSU3_GT()
    adj_gt.validate()
    
    T3_diag_gt = np.diag(adj_gt.T3).real
    T8_diag_gt = np.diag(adj_gt.T8).real
    Y_diag_gt = T8_diag_gt * 2 / np.sqrt(3)
    
    print(f"\nDiagonal quantum numbers (GT basis):")
    print(f"T3: {T3_diag_gt}")
    print(f"T8: {T8_diag_gt}")
    print(f"Y:  {Y_diag_gt}")
    
    C2_gt = adj_gt.get_casimir()
    eigenvalues_gt = np.linalg.eigvalsh(C2_gt)
    print(f"\nCasimir eigenvalues: {eigenvalues_gt}")
    print(f"Mean: {np.mean(eigenvalues_gt):.6f}")
    print(f"Std:  {np.std(eigenvalues_gt):.2e}")


if __name__ == "__main__":
    main()
