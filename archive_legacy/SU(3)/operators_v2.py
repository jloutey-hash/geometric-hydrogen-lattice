"""
SU(3) Operator Construction using Gelfand-Tsetlin Patterns
Implements exact ladder operator matrix elements.
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from lattice import SU3Lattice
from typing import Tuple


class SU3Operators:
    """
    Constructs SU(3) generators using Gelfand-Tsetlin pattern basis.
    Uses exact formulas for ladder operator matrix elements.
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
        
        # Build all operators
        self.T3 = None
        self.T8 = None
        self.I_plus = None
        self.I_minus = None
        self.U_plus = None
        self.U_minus = None
        self.V_plus = None
        self.V_minus = None
        self.C2 = None
        
        self._build_all_operators()
    
    def _build_all_operators(self):
        """Build all SU(3) generators."""
        print("Building SU(3) operators using GT patterns...")
        self.T3 = self.build_T3()
        self.T8 = self.build_T8()
        self.I_plus = self.build_I_plus()
        self.I_minus = self.build_I_minus()
        self.U_plus = self.build_U_plus()
        self.U_minus = self.build_U_minus()
        self.V_plus = self.build_V_plus()
        self.V_minus = self.build_V_minus()
        self.C2 = self.build_C2()
        print("Operators constructed successfully.")
    
    def build_T3(self) -> csr_matrix:
        """
        Build the T3 operator (diagonal, third component of isospin).
        
        T3 = I3 for each state.
        """
        T3 = lil_matrix((self.dim, self.dim), dtype=complex)
        
        for state in self.lattice.states:
            idx = state['index']
            i3 = state['i3']
            T3[idx, idx] = i3
        
        return T3.tocsr()
    
    def build_T8(self) -> csr_matrix:
        """
        Build the T8 operator (diagonal, related to hypercharge).
        
        T8 = (sqrt(3)/2) * Y for each state.
        """
        T8 = lil_matrix((self.dim, self.dim), dtype=complex)
        
        for state in self.lattice.states:
            idx = state['index']
            y = state['y']
            T8[idx, idx] = np.sqrt(3) / 2.0 * y
        
        return T8.tocsr()
    
    def build_I_plus(self) -> csr_matrix:
        """
        Build the I+ operator (isospin raising operator).
        
        I+ changes m11 -> m11 + 1 in the GT pattern.
        Matrix element is computed using the exact SU(3) formula.
        """
        I_plus = lil_matrix((self.dim, self.dim), dtype=complex)
        
        for state in self.lattice.states:
            gt = state['gt']
            idx_from = state['index']
            p, q = state['p'], state['q']
            
            # I+ raises m11 by 1
            m13, m23, m33, m12, m22, m11 = gt
            m11_new = m11 + 1
            
            # Check if the new pattern is valid
            if m11_new <= m12:
                gt_new = (m13, m23, m33, m12, m22, m11_new)
                idx_to = self.lattice.get_index(gt_new)
                
                if idx_to is not None:
                    # Compute matrix element
                    coeff = self._i_plus_coefficient(gt)
                    if abs(coeff) > 1e-14:
                        I_plus[idx_to, idx_from] = coeff
        
        return I_plus.tocsr()
    
    def build_I_minus(self) -> csr_matrix:
        """
        Build the I- operator (isospin lowering operator).
        
        I- changes m11 -> m11 - 1 in the GT pattern.
        """
        I_minus = lil_matrix((self.dim, self.dim), dtype=complex)
        
        for state in self.lattice.states:
            gt = state['gt']
            idx_from = state['index']
            
            # I- lowers m11 by 1
            m13, m23, m33, m12, m22, m11 = gt
            m11_new = m11 - 1
            
            # Check if the new pattern is valid
            if m11_new >= m22:
                gt_new = (m13, m23, m33, m12, m22, m11_new)
                idx_to = self.lattice.get_index(gt_new)
                
                if idx_to is not None:
                    # Compute matrix element
                    coeff = self._i_minus_coefficient(gt)
                    if abs(coeff) > 1e-14:
                        I_minus[idx_to, idx_from] = coeff
        
        return I_minus.tocsr()
    
    def build_U_plus(self) -> csr_matrix:
        """
        Build the U+ operator (U-spin raising operator).
        
        U+ changes m12 -> m12 + 1 and m11 -> m11 - 1.
        """
        U_plus = lil_matrix((self.dim, self.dim), dtype=complex)
        
        for state in self.lattice.states:
            gt = state['gt']
            idx_from = state['index']
            
            m13, m23, m33, m12, m22, m11 = gt
            m12_new = m12 + 1
            m11_new = m11 - 1
            
            # Check if the new pattern is valid
            if m12_new <= m13 and m11_new >= m22:
                gt_new = (m13, m23, m33, m12_new, m22, m11_new)
                idx_to = self.lattice.get_index(gt_new)
                
                if idx_to is not None:
                    # Compute matrix element
                    coeff = self._u_plus_coefficient(gt)
                    if abs(coeff) > 1e-14:
                        U_plus[idx_to, idx_from] = coeff
        
        return U_plus.tocsr()
    
    def build_U_minus(self) -> csr_matrix:
        """
        Build the U- operator (U-spin lowering operator).
        
        U- changes m12 -> m12 - 1 and m11 -> m11 + 1.
        """
        U_minus = lil_matrix((self.dim, self.dim), dtype=complex)
        
        for state in self.lattice.states:
            gt = state['gt']
            idx_from = state['index']
            
            m13, m23, m33, m12, m22, m11 = gt
            m12_new = m12 - 1
            m11_new = m11 + 1
            
            # Check if the new pattern is valid
            if m12_new >= m23 and m11_new <= m12:
                gt_new = (m13, m23, m33, m12_new, m22, m11_new)
                idx_to = self.lattice.get_index(gt_new)
                
                if idx_to is not None:
                    # Compute matrix element
                    coeff = self._u_minus_coefficient(gt)
                    if abs(coeff) > 1e-14:
                        U_minus[idx_to, idx_from] = coeff
        
        return U_minus.tocsr()
    
    def build_V_plus(self) -> csr_matrix:
        """
        Build the V+ operator (V-spin raising operator).
        
        V+ changes m22 -> m22 + 1 and m11 -> m11 - 1.
        """
        V_plus = lil_matrix((self.dim, self.dim), dtype=complex)
        
        for state in self.lattice.states:
            gt = state['gt']
            idx_from = state['index']
            
            m13, m23, m33, m12, m22, m11 = gt
            m22_new = m22 + 1
            m11_new = m11 - 1
            
            # Check if the new pattern is valid
            if m22_new <= m23 and m11_new >= m22_new:
                gt_new = (m13, m23, m33, m12, m22_new, m11_new)
                idx_to = self.lattice.get_index(gt_new)
                
                if idx_to is not None:
                    # Compute matrix element
                    coeff = self._v_plus_coefficient(gt)
                    if abs(coeff) > 1e-14:
                        V_plus[idx_to, idx_from] = coeff
        
        return V_plus.tocsr()
    
    def build_V_minus(self) -> csr_matrix:
        """
        Build the V- operator (V-spin lowering operator).
        
        V- changes m22 -> m22 - 1 and m11 -> m11 + 1.
        """
        V_minus = lil_matrix((self.dim, self.dim), dtype=complex)
        
        for state in self.lattice.states:
            gt = state['gt']
            idx_from = state['index']
            
            m13, m23, m33, m12, m22, m11 = gt
            m22_new = m22 - 1
            m11_new = m11 + 1
            
            # Check if the new pattern is valid
            if m22_new >= m33 and m11_new <= m12:
                gt_new = (m13, m23, m33, m12, m22_new, m11_new)
                idx_to = self.lattice.get_index(gt_new)
                
                if idx_to is not None:
                    # Compute matrix element
                    coeff = self._v_minus_coefficient(gt)
                    if abs(coeff) > 1e-14:
                        V_minus[idx_to, idx_from] = coeff
        
        return V_minus.tocsr()
    
    def _i_plus_coefficient(self, gt: Tuple[int, ...]) -> float:
        """
        Calculate the coefficient for I+ acting on a GT pattern.
        
        Formula: sqrt((m12 - m11)(m11 - m22 + 1))
        """
        m13, m23, m33, m12, m22, m11 = gt
        coeff = np.sqrt((m12 - m11) * (m11 - m22 + 1))
        return coeff
    
    def _i_minus_coefficient(self, gt: Tuple[int, ...]) -> float:
        """
        Calculate the coefficient for I- acting on a GT pattern.
        
        Formula: sqrt((m12 - m11 + 1)(m11 - m22))
        """
        m13, m23, m33, m12, m22, m11 = gt
        coeff = np.sqrt((m12 - m11 + 1) * (m11 - m22))
        return coeff
    
    def _u_plus_coefficient(self, gt: Tuple[int, ...]) -> float:
        """
        Calculate the coefficient for U+ acting on a GT pattern.
        
        U+ involves changing both m12 and m11.
        Uses the exact SU(3) Gelfand-Tsetlin formula.
        """
        m13, m23, m33, m12, m22, m11 = gt
        
        # Formula from Biedenharn-Louck for E_alpha2^+
        coeff = np.sqrt((m13 - m12) * (m12 - m23 + 1) * (m11 - m22) * (m12 - m11 + 1)) / np.sqrt((m12 - m22 + 1))
        
        return coeff
    
    def _u_minus_coefficient(self, gt: Tuple[int, ...]) -> float:
        """
        Calculate the coefficient for U- acting on a GT pattern.
        """
        m13, m23, m33, m12, m22, m11 = gt
        
        # Formula from Biedenharn-Louck for E_alpha2^-
        coeff = np.sqrt((m13 - m12 + 1) * (m12 - m23) * (m11 - m22 + 1) * (m12 - m11)) / np.sqrt((m12 - m22))
        
        return coeff
    
    def _v_plus_coefficient(self, gt: Tuple[int, ...]) -> float:
        """
        Calculate the coefficient for V+ acting on a GT pattern.
        """
        m13, m23, m33, m12, m22, m11 = gt
        
        # Formula for E_(alpha1+alpha2)^+
        coeff = np.sqrt((m23 - m22) * (m22 - m33 + 1) * (m11 - m22) * (m12 - m11 + 1)) / np.sqrt((m12 - m22))
        
        return coeff
    
    def _v_minus_coefficient(self, gt: Tuple[int, ...]) -> float:
        """
        Calculate the coefficient for V- acting on a GT pattern.
        """
        m13, m23, m33, m12, m22, m11 = gt
        
        # Formula for E_(alpha1+alpha2)^-
        coeff = np.sqrt((m23 - m22 + 1) * (m22 - m33) * (m11 - m22 + 1) * (m12 - m11)) / np.sqrt((m12 - m22 + 1))
        
        return coeff
    
    def build_C2(self) -> csr_matrix:
        """
        Build the quadratic Casimir operator C2.
        
        C2 = T3^2 + T8^2 + (1/2){I+, I-} + (1/2){U+, U-} + (1/2){V+, V-}
        """
        C2 = (self.T3 @ self.T3 + 
              self.T8 @ self.T8 + 
              0.5 * (self.I_plus @ self.I_minus + self.I_minus @ self.I_plus) +
              0.5 * (self.U_plus @ self.U_minus + self.U_minus @ self.U_plus) +
              0.5 * (self.V_plus @ self.V_minus + self.V_minus @ self.V_plus))
        
        return C2.tocsr()
    
    def get_operators(self) -> dict:
        """Return a dictionary of all operators."""
        return {
            'T3': self.T3,
            'T8': self.T8,
            'I+': self.I_plus,
            'I-': self.I_minus,
            'U+': self.U_plus,
            'U-': self.U_minus,
            'V+': self.V_plus,
            'V-': self.V_minus,
            'C2': self.C2
        }


if __name__ == "__main__":
    # Test operator construction
    from lattice import SU3Lattice
    
    lattice = SU3Lattice(max_p=1, max_q=1)
    lattice.print_summary()
    
    print("\nBuilding operators...")
    operators = SU3Operators(lattice)
    
    print("\nOperator sparsity:")
    for name, op in operators.get_operators().items():
        density = op.nnz / (op.shape[0] * op.shape[1]) * 100
        print(f"  {name}: {op.nnz} non-zero elements ({density:.2f}% dense)")
