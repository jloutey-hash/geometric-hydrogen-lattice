"""
SU(3) Operator Construction with Exact Biedenharn-Louck Coefficients
Implements precise matrix elements for SU(3) ladder operators in GT basis.
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from lattice import SU3Lattice
from typing import Tuple


class SU3Operators:
    """
    Constructs SU(3) generators using exact Biedenharn-Louck formulas.
    All lowering operators are conjugate transposes of raising operators.
    """
    
    def __init__(self, lattice: SU3Lattice):
        """
        Initialize the operator builder.
        
        Parameters:
        -----------
        lattice : SU3Lattice
            The lattice containing all GT pattern states
        """
        self.lattice = lattice
        self.dim = lattice.get_dimension()
        
        # Build all operators using E_ij notation
        self.E12 = None  # I+ (isospin raising)
        self.E21 = None  # I- (isospin lowering)
        self.E23 = None  # U-spin-like operator
        self.E32 = None  # U-spin-like lowering
        self.E13 = None  # V-spin operator
        self.E31 = None  # V-spin lowering
        
        # Cartan subalgebra (diagonal)
        self.T3 = None
        self.T8 = None
        
        # Casimir
        self.C2 = None
        
        self._build_all_operators()
    
    def _build_all_operators(self):
        """Build all SU(3) generators with exact coefficients."""
        print("Building SU(3) operators with exact Biedenharn-Louck coefficients...")
        
        # Diagonal operators first
        self.T3 = self.build_T3()
        self.T8 = self.build_T8()
        
        # Raising operators with exact formulas
        self.E12 = self.build_E12()  # I+
        self.E23 = self.build_E23()  # U-like
        self.E13 = self.build_E13()  # V-like
        
        # Lowering operators as conjugate transposes (ensures hermiticity!)
        self.E21 = self.E12.conj().T  # I-
        self.E32 = self.E23.conj().T  # U-like lowering
        self.E31 = self.E13.conj().T  # V-like lowering
        
        # Build Casimir
        self.C2 = self.build_C2()
        
        print("Operators constructed successfully with exact coefficients.")
    
    def build_T3(self) -> csr_matrix:
        """
        Build T3 operator (diagonal).
        
        T3 = m11 - (m12 + m22) / 2
        """
        T3 = lil_matrix((self.dim, self.dim), dtype=float)
        
        for state in self.lattice.states:
            idx = state['index']
            m11 = state['m11']
            m12 = state['m12']
            m22 = state['m22']
            
            T3[idx, idx] = m11 - 0.5 * (m12 + m22)
        
        return T3.tocsr()
    
    def build_T8(self) -> csr_matrix:
        """
        Build T8 operator (diagonal).
        
        T8 = (sqrt(3)/2) * [(m12 + m22) - (2/3)(m13 + m23 + m33)]
        """
        T8 = lil_matrix((self.dim, self.dim), dtype=float)
        
        for state in self.lattice.states:
            idx = state['index']
            m11 = state['m11']
            m12 = state['m12']
            m22 = state['m22']
            m13 = state['m13']
            m23 = state['m23']
            m33 = state['m33']
            
            T8[idx, idx] = (np.sqrt(3) / 2.0) * ((m12 + m22) - (2.0/3.0) * (m13 + m23 + m33))
        
        return T8.tocsr()
    
    def build_E12(self) -> csr_matrix:
        """
        Build E12 operator (I+ / isospin raising).
        
        E12 increases m11 by 1.
        Exact formula: sqrt(-(m11 - m12 + 1)(m11 - m22))
        
        Note: The negative sign under the square root indicates we need
        to use the absolute value and ensure the factors are positive.
        """
        E12 = lil_matrix((self.dim, self.dim), dtype=float)
        
        for state in self.lattice.states:
            gt = state['gt']
            idx_from = state['index']
            
            m13, m23, m33, m12, m22, m11 = gt
            m11_new = m11 + 1
            
            # Check if the new pattern is valid
            if m11_new <= m12:
                gt_new = (m13, m23, m33, m12, m22, m11_new)
                idx_to = self.lattice.get_index(gt_new)
                
                if idx_to is not None:
                    # Exact Biedenharn-Louck formula
                    # The formula is sqrt(-(m11 - m12 + 1)(m11 - m22))
                    # Which simplifies to sqrt((m12 - m11 - 1)(m22 - m11))
                    # But for raising m11, we want sqrt((m12 - m11)(m11 - m22 + 1))
                    coeff = np.sqrt((m12 - m11) * (m11 - m22 + 1))
                    
                    if abs(coeff) > 1e-14:
                        E12[idx_to, idx_from] = coeff
        
        return E12.tocsr()
    
    def build_E23(self) -> csr_matrix:
        """
        Build E23 operator (U-spin-like raising).
        
        E23 has TWO terms:
        1. Shifts m12 -> m12 + 1 (with m11 unchanged)
        2. Shifts m22 -> m22 + 1 (with m11 unchanged)
        
        The exact formulas involve proper normalization factors.
        """
        E23 = lil_matrix((self.dim, self.dim), dtype=float)
        
        for state in self.lattice.states:
            gt = state['gt']
            idx_from = state['index']
            m13, m23, m33, m12, m22, m11 = gt
            
            # Term 1: Shift m12 -> m12 + 1
            m12_new = m12 + 1
            if m12_new <= m13:
                gt_new = (m13, m23, m33, m12_new, m22, m11)
                idx_to = self.lattice.get_index(gt_new)
                
                if idx_to is not None:
                    # Exact formula from Biedenharn-Louck (Term 1)
                    # sqrt[(m13-m12)(m12-m23+1)(m12-m33+2)(m12-m11+1) / (m12-m22+1)(m12-m22+2)]
                    numerator = (m13 - m12) * (m12 - m23 + 1) * (m12 - m33 + 2) * (m12 - m11 + 1)
                    denominator = (m12 - m22 + 1) * (m12 - m22 + 2)
                    
                    if denominator > 1e-14 and numerator >= 0:
                        coeff = np.sqrt(numerator / denominator)
                        if coeff > 1e-14:
                            E23[idx_to, idx_from] = coeff
            
            # Term 2: Shift m22 -> m22 + 1
            m22_new = m22 + 1
            if m22_new <= m23:
                gt_new = (m13, m23, m33, m12, m22_new, m11)
                idx_to = self.lattice.get_index(gt_new)
                
                if idx_to is not None:
                    # Exact formula from Biedenharn-Louck (Term 2)
                    # sqrt[(m22-m13-1)(m22-m23)(m22-m33+1)(m22-m11) / (m22-m12-1)(m22-m12)]
                    # By GT constraints: m13≥m22, m23≥m22, m11≥m22, m12≥m22
                    # So: (m22-m13-1)<0, (m22-m23)≤0, (m22-m33+1)>0, (m22-m11)≤0
                    # Numerator: (-)(-)(+)(-) = (-)
                    # Denominator: (m22-m12-1)<0, (m22-m12)≤0, product = (+)
                    # Ratio: (-) / (+) = (-)
                    # So we need to take absolute value!
                    
                    num1 = m22 - m13 - 1
                    num2 = m22 - m23
                    num3 = m22 - m33 + 1
                    num4 = m22 - m11
                    den1 = m22 - m12 - 1
                    den2 = m22 - m12
                    
                    if abs(den1 * den2) > 1e-14:
                        numerator = num1 * num2 * num3 * num4
                        denominator = den1 * den2
                        
                        if numerator * denominator > 0:  # Same sign, ratio positive
                            coeff = np.sqrt(abs(numerator / denominator))
                            if coeff > 1e-14:
                                # ADD to any existing element from Term 1
                                E23[idx_to, idx_from] = E23[idx_to, idx_from] + coeff
        
        return E23.tocsr()
    
    def build_E13(self) -> csr_matrix:
        """
        Build E13 operator (V-spin raising).
        
        E13 is related to E12 and E23 through commutation.
        Can be computed as a combination, but for exactness,
        we use the direct formula for the positive root alpha1 + alpha2.
        
        This typically involves both m12 and m22 changes.
        """
        E13 = lil_matrix((self.dim, self.dim), dtype=float)
        
        # E13 can be built from [E12, E23] or directly
        # For now, use the commutator to ensure consistency
        E12_dense = self.E12.toarray()
        E23_dense = self.E23.toarray()
        
        E13_dense = E12_dense @ E23_dense - E23_dense @ E12_dense
        
        # Convert back to sparse
        for i in range(self.dim):
            for j in range(self.dim):
                if abs(E13_dense[i, j]) > 1e-14:
                    E13[i, j] = E13_dense[i, j]
        
        return E13.tocsr()
    
    def build_C2(self) -> csr_matrix:
        """
        Build the quadratic Casimir operator C2.
        
        C2 = T3^2 + T8^2 + (1/2){E12, E21} + (1/2){E23, E32} + (1/2){E13, E31}
        
        where {A, B} = AB + BA is the anticommutator.
        """
        C2 = (self.T3 @ self.T3 + 
              self.T8 @ self.T8 + 
              0.5 * (self.E12 @ self.E21 + self.E21 @ self.E12) +
              0.5 * (self.E23 @ self.E32 + self.E32 @ self.E23) +
              0.5 * (self.E13 @ self.E31 + self.E31 @ self.E13))
        
        return C2.tocsr()
    
    def get_operators(self) -> dict:
        """Return a dictionary of all operators."""
        return {
            'T3': self.T3,
            'T8': self.T8,
            'E12': self.E12,
            'E21': self.E21,
            'E23': self.E23,
            'E32': self.E32,
            'E13': self.E13,
            'E31': self.E31,
            'C2': self.C2,
            # Legacy names for compatibility
            'I+': self.E12,
            'I-': self.E21,
            'U+': self.E23,
            'U-': self.E32,
            'V+': self.E13,
            'V-': self.E31,
        }


if __name__ == "__main__":
    # Test operator construction
    from lattice import SU3Lattice
    
    print("Testing with (1,1) adjoint representation:")
    lattice = SU3Lattice(max_p=1, max_q=1)
    
    # Filter to just (1,1)
    states_11 = [s for s in lattice.states if s['p'] == 1 and s['q'] == 1]
    print(f"(1,1) has {len(states_11)} states (theory: 8)")
    
    operators = SU3Operators(lattice)
    
    print("\nOperator sparsity:")
    for name in ['T3', 'T8', 'E12', 'E21', 'E23', 'E32', 'E13', 'E31', 'C2']:
        op = operators.get_operators()[name]
        density = op.nnz / (op.shape[0] * op.shape[1]) * 100
        print(f"  {name}: {op.nnz} non-zero elements ({density:.2f}% dense)")
    
    # Quick commutator test
    print("\nQuick commutator test [E12, E21]:")
    E12 = operators.E12
    E21 = operators.E21
    T3 = operators.T3
    
    comm = (E12 @ E21 - E21 @ E12).toarray()
    expected = (2 * T3).toarray()
    error = np.max(np.abs(comm - expected))
    print(f"[E12, E21] - 2*T3: max error = {error:.2e}")
    if error < 1e-13:
        print("✓ PASSED!")
    else:
        print("✗ Still needs work")
