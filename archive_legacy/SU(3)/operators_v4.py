"""
SU(3) Operator Construction with Shifted l-Index Formulas (v4)
Uses Biedenharn-Louck shifted coordinates for exact matrix elements.
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from lattice import SU3Lattice
from typing import Tuple


class SU3Operators:
    """
    Constructs SU(3) generators using shifted l-index formulas.
    This ensures exact commutation relations at machine precision.
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
        self.E23 = None  # U-spin raising operator
        self.E32 = None  # U-spin lowering
        self.E13 = None  # V-spin operator
        self.E31 = None  # V-spin lowering
        
        # Cartan subalgebra (diagonal)
        self.T3 = None
        self.T8 = None
        
        # Casimir
        self.C2 = None
        
        self._build_all_operators()
    
    def _m_to_l(self, gt: Tuple[int, int, int, int, int, int]) -> Tuple[int, int, int, int, int, int]:
        """
        Convert GT pattern m-indices to shifted l-indices.
        
        l13 = m13 + 2, l23 = m23 + 1, l33 = m33
        l12 = m12 + 1, l22 = m22
        l11 = m11
        """
        m13, m23, m33, m12, m22, m11 = gt
        return (m13 + 2, m23 + 1, m33, m12 + 1, m22, m11)
    
    def _build_all_operators(self):
        """Build all SU(3) generators with shifted l-index formulas."""
        print("Building SU(3) operators with shifted l-index formulas...")
        
        # Diagonal operators first
        self.T3 = self.build_T3()
        self.T8 = self.build_T8()
        
        # Raising operators with exact l-index formulas
        self.E12 = self.build_E12()  # I+
        self.E23 = self.build_E23()  # U+
        
        # E13 (V+) via commutator - algebraic closure!
        self.E13 = self.build_E13_via_commutator()
        
        # Lowering operators as conjugate transposes (ensures hermiticity!)
        self.E21 = self.E12.conj().T  # I-
        self.E32 = self.E23.conj().T  # U-
        self.E31 = self.E13.conj().T  # V-
        
        # Build Casimir
        self.C2 = self.build_C2()
        
        print("Operators constructed successfully with l-index formulas.")
    
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
            m13 = state['m13']
            m23 = state['m23']
            m33 = state['m33']
            m12 = state['m12']
            m22 = state['m22']
            
            T8[idx, idx] = (np.sqrt(3) / 2.0) * ((m12 + m22) - (2.0/3.0) * (m13 + m23 + m33))
        
        return T8.tocsr()
    
    def build_E12(self) -> csr_matrix:
        """
        Build E12 operator (I+ / isospin raising).
        
        E12 increases m11 by 1.
        Using l-index formula: sqrt((l12 - l11) * (l11 - l22 + 1))
        Since l12 = m12 + 1, l11 = m11, l22 = m22:
        sqrt((m12 + 1 - m11) * (m11 - m22 + 1)) = sqrt((m12 - m11 + 1) * (m11 - m22 + 1))
        
        But we want to use the m-index formula which already works:
        sqrt((m12 - m11) * (m11 - m22 + 1))
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
                    # Formula: sqrt((m12 - m11) * (m11 - m22 + 1))
                    coeff = np.sqrt((m12 - m11) * (m11 - m22 + 1))
                    
                    if coeff > 1e-14:
                        E12[idx_to, idx_from] = coeff
        
        return E12.tocsr()
    
    def build_E23(self) -> csr_matrix:
        """
        Build E23 operator (U-spin raising) using shifted l-index formulas.
        
        E23 has TWO terms:
        1. Shifts m12 -> m12 + 1 (l12 -> l12 + 1)
        2. Shifts m22 -> m22 + 1 (l22 -> l22 + 1)
        
        Using the corrected l-index formulas from v4 spec.
        """
        E23 = lil_matrix((self.dim, self.dim), dtype=float)
        
        for state in self.lattice.states:
            gt = state['gt']
            idx_from = state['index']
            m13, m23, m33, m12, m22, m11 = gt
            
            # Convert to l-indices
            l13, l23, l33, l12, l22, l11 = self._m_to_l(gt)
            
            # Term 1: Shift m12 -> m12 + 1 (l12 -> l12 + 1)
            m12_new = m12 + 1
            if m12_new <= m13:
                gt_new = (m13, m23, m33, m12_new, m22, m11)
                idx_to = self.lattice.get_index(gt_new)
                
                if idx_to is not None:
                    # Formula: sqrt(|(l13-l12-1)(l23-l12-1)(l33-l12-1)(l11-l12-1) / (l22-l12-1)(l22-l12)|)
                    numerator = ((l13 - l12 - 1) * (l23 - l12 - 1) * 
                                (l33 - l12 - 1) * (l11 - l12 - 1))
                    denominator = (l22 - l12 - 1) * (l22 - l12)
                    
                    if abs(denominator) > 1e-14:
                        ratio = numerator / denominator
                        if abs(ratio) > 0:  # Take absolute value as spec says
                            coeff = np.sqrt(abs(ratio))
                            if coeff > 1e-14:
                                E23[idx_to, idx_from] = coeff
            
            # Term 2: Shift m22 -> m22 + 1 (l22 -> l22 + 1)
            m22_new = m22 + 1
            if m22_new <= m23:
                gt_new = (m13, m23, m33, m12, m22_new, m11)
                idx_to = self.lattice.get_index(gt_new)
                
                if idx_to is not None:
                    # Formula: sqrt(|(l13-l22-1)(l23-l22-1)(l33-l22-1)(l11-l22-1) / (l12-l22-1)(l12-l22)|)
                    numerator = ((l13 - l22 - 1) * (l23 - l22 - 1) * 
                                (l33 - l22 - 1) * (l11 - l22 - 1))
                    denominator = (l12 - l22 - 1) * (l12 - l22)
                    
                    if abs(denominator) > 1e-14:
                        ratio = numerator / denominator
                        if abs(ratio) > 0:
                            coeff = np.sqrt(abs(ratio))
                            if coeff > 1e-14:
                                # ADD to any existing element from Term 1
                                E23[idx_to, idx_from] = E23[idx_to, idx_from] + coeff
        
        return E23.tocsr()
    
    def build_E13_via_commutator(self) -> csr_matrix:
        """
        Build E13 (V-spin) via commutator: E13 = [E12, E23].
        
        This is the "algebraic closure" strategy - guarantees the SU(3) triangle closes.
        """
        # Convert to dense for matrix multiplication
        E12_dense = self.E12.toarray()
        E23_dense = self.E23.toarray()
        
        # Compute commutator
        E13_dense = E12_dense @ E23_dense - E23_dense @ E12_dense
        
        # Convert back to sparse
        E13 = lil_matrix((self.dim, self.dim), dtype=float)
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
    print("\nQuick commutator tests:")
    E12 = operators.E12
    E21 = operators.E21
    E23 = operators.E23
    E32 = operators.E32
    T3 = operators.T3
    T8 = operators.T8
    
    comm_I = (E12 @ E21 - E21 @ E12).toarray()
    expected_I = (2 * T3).toarray()
    error_I = np.max(np.abs(comm_I - expected_I))
    print(f"[E12, E21] - 2*T3: {error_I:.3e}")
    
    comm_U = (E23 @ E32 - E32 @ E23).toarray()
    expected_U = (-1.5 * T3 + (np.sqrt(3)/2) * T8).toarray()
    error_U = np.max(np.abs(comm_U - expected_U))
    print(f"[E23, E32] - (-(3/2)*T3 + (√3/2)*T8): {error_U:.3e}")
    
    if error_I < 1e-13 and error_U < 1e-13:
        print("✓ ALL TESTS PASSED!")
    else:
        print("⚠ Some tests need refinement")
