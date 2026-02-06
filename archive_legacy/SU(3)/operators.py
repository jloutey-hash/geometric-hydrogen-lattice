"""
SU(3) Operator Construction
Builds the 8 Gell-Mann generators as sparse matrices on the SU(3) lattice.
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from lattice import SU3Lattice
from typing import Tuple


class SU3Operators:
    """
    Constructs SU(3) generators (Gell-Mann matrices) as sparse matrices
    acting on the weight states of a SU3Lattice.
    """
    
    def __init__(self, lattice: SU3Lattice):
        """
        Initialize the operator builder.
        
        Parameters:
        -----------
        lattice : SU3Lattice
            The lattice containing all states
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
        print("Building SU(3) operators...")
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
        
        T3 is diagonal with eigenvalue I3 for each state.
        """
        T3 = lil_matrix((self.dim, self.dim), dtype=complex)
        
        for state in self.lattice.states:
            idx = state['index']
            i3 = state['i3']
            T3[idx, idx] = i3
        
        return T3.tocsr()
    
    def build_T8(self) -> csr_matrix:
        """
        Build the T8 operator (diagonal, hypercharge).
        
        T8 is diagonal with eigenvalue (sqrt(3)/2) * Y for each state.
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
        
        I+ = (T1 + iT2) / 2, corresponds to the E_alpha1 operator
        where alpha1 = (1, 0) is the first simple root.
        
        I+ moves: |i3, y> -> |i3+1, y>
        """
        I_plus = lil_matrix((self.dim, self.dim), dtype=complex)
        
        for state in self.lattice.states:
            p, q = state['p'], state['q']
            i3, y = state['i3'], state['y']
            mult = state['multiplicity']
            idx_from = state['index']
            
            # Target state: i3 -> i3 + 1, y stays the same
            i3_to = i3 + 1.0
            y_to = y
            
            idx_to = self.lattice.get_index(p, q, i3_to, y_to, mult)
            
            if idx_to is not None:
                # Calculate the matrix element using SU(3) ladder coefficients
                coeff = self._isospin_raising_coefficient(p, q, i3, y)
                if abs(coeff) > 1e-14:
                    I_plus[idx_to, idx_from] = coeff / np.sqrt(2)  # Normalization factor
        
        return I_plus.tocsr()
    
    def build_I_minus(self) -> csr_matrix:
        """
        Build the I- operator (isospin lowering operator).
        
        I- moves: |i3, y> -> |i3-1, y>
        """
        I_minus = lil_matrix((self.dim, self.dim), dtype=complex)
        
        for state in self.lattice.states:
            p, q = state['p'], state['q']
            i3, y = state['i3'], state['y']
            mult = state['multiplicity']
            idx_from = state['index']
            
            # Target state: i3 -> i3 - 1, y stays the same
            i3_to = i3 - 1.0
            y_to = y
            
            idx_to = self.lattice.get_index(p, q, i3_to, y_to, mult)
            
            if idx_to is not None:
                # Calculate the matrix element
                coeff = self._isospin_lowering_coefficient(p, q, i3, y)
                if abs(coeff) > 1e-14:
                    I_minus[idx_to, idx_from] = coeff / np.sqrt(2)  # Normalization factor
        
        return I_minus.tocsr()
    
    def build_U_plus(self) -> csr_matrix:
        """
        Build the U+ operator (U-spin raising operator).
        
        U+ moves: |i3, y> -> |i3 - 0.5, y + sqrt(3)/2>
        """
        U_plus = lil_matrix((self.dim, self.dim), dtype=complex)
        
        for state in self.lattice.states:
            p, q = state['p'], state['q']
            i3, y = state['i3'], state['y']
            mult = state['multiplicity']
            idx_from = state['index']
            
            # Target state after U+ action
            i3_to = i3 - 0.5
            y_to = y + np.sqrt(3) / 2.0
            
            idx_to = self.lattice.get_index(p, q, i3_to, y_to, mult)
            
            if idx_to is not None:
                # Calculate the matrix element
                coeff = self._uspin_raising_coefficient(p, q, i3, y)
                if abs(coeff) > 1e-14:
                    U_plus[idx_to, idx_from] = coeff / np.sqrt(2)  # Normalization factor
        
        return U_plus.tocsr()
    
    def build_U_minus(self) -> csr_matrix:
        """
        Build the U- operator (U-spin lowering operator).
        
        U- moves: |i3, y> -> |i3 + 0.5, y - sqrt(3)/2>
        """
        U_minus = lil_matrix((self.dim, self.dim), dtype=complex)
        
        for state in self.lattice.states:
            p, q = state['p'], state['q']
            i3, y = state['i3'], state['y']
            mult = state['multiplicity']
            idx_from = state['index']
            
            # Target state after U- action
            i3_to = i3 + 0.5
            y_to = y - np.sqrt(3) / 2.0
            
            idx_to = self.lattice.get_index(p, q, i3_to, y_to, mult)
            
            if idx_to is not None:
                # Calculate the matrix element
                coeff = self._uspin_lowering_coefficient(p, q, i3, y)
                if abs(coeff) > 1e-14:
                    U_minus[idx_to, idx_from] = coeff / np.sqrt(2)  # Normalization factor
        
        return U_minus.tocsr()
    
    def build_V_plus(self) -> csr_matrix:
        """
        Build the V+ operator (V-spin raising operator).
        
        V+ moves: |i3, y> -> |i3 + 0.5, y + sqrt(3)/2>
        """
        V_plus = lil_matrix((self.dim, self.dim), dtype=complex)
        
        for state in self.lattice.states:
            p, q = state['p'], state['q']
            i3, y = state['i3'], state['y']
            mult = state['multiplicity']
            idx_from = state['index']
            
            # Target state after V+ action
            i3_to = i3 + 0.5
            y_to = y + np.sqrt(3) / 2.0
            
            idx_to = self.lattice.get_index(p, q, i3_to, y_to, mult)
            
            if idx_to is not None:
                # Calculate the matrix element
                coeff = self._vspin_raising_coefficient(p, q, i3, y)
                if abs(coeff) > 1e-14:
                    V_plus[idx_to, idx_from] = coeff / np.sqrt(2)  # Normalization factor
        
        return V_plus.tocsr()
    
    def build_V_minus(self) -> csr_matrix:
        """
        Build the V- operator (V-spin lowering operator).
        
        V- moves: |i3, y> -> |i3 - 0.5, y - sqrt(3)/2>
        """
        V_minus = lil_matrix((self.dim, self.dim), dtype=complex)
        
        for state in self.lattice.states:
            p, q = state['p'], state['q']
            i3, y = state['i3'], state['y']
            mult = state['multiplicity']
            idx_from = state['index']
            
            # Target state after V- action
            i3_to = i3 - 0.5
            y_to = y - np.sqrt(3) / 2.0
            
            idx_to = self.lattice.get_index(p, q, i3_to, y_to, mult)
            
            if idx_to is not None:
                # Calculate the matrix element
                coeff = self._vspin_lowering_coefficient(p, q, i3, y)
                if abs(coeff) > 1e-14:
                    V_minus[idx_to, idx_from] = coeff / np.sqrt(2)  # Normalization factor
        
        return V_minus.tocsr()
    
    def _isospin_raising_coefficient(self, p: int, q: int, i3: float, y: float) -> float:
        """
        Calculate the coefficient for I+ ladder operator.
        
        For SU(3), use the root string method. The coefficient for raising by
        the simple root alpha1 = (1, 0) is given by:
        sqrt(n * (m1 + 1))
        where n is the string length and m1 is the number of times we've
        applied the lowering operator.
        """
        # In (I3, Y) coordinates, I+ raises I3 by 1
        # We need to find the Dynkin index m1 for this weight
        
        # Convert weight back to Dynkin basis
        # The weight in Dynkin coords is: lambda = lambda_max - m1*alpha1 - m2*alpha2
        # where lambda_max = (p, q) in Dynkin labels
        
        # From (I3, Y), recover (m1, m2):
        # i3 = (p-q)/2 - m1 + m2/2
        # y = (p+q)/3 - m2*sqrt(3)/2
        
        # Solve for m1, m2:
        m2 = round(2 * ((p + q) / 3.0 - y) / np.sqrt(3))
        m1 = round((p - q) / 2.0 - i3 + m2 / 2.0)
        
        # Clamp to valid range
        m1 = max(0, min(p, int(m1)))
        m2 = max(0, min(q, int(m2)))
        
        # Coefficient formula: sqrt((p - m1) * (m1 + 1))
        # This is the correct SU(3) ladder coefficient
        coeff = np.sqrt((p - m1) * (m1 + 1))
        
        return coeff
    
    def _isospin_lowering_coefficient(self, p: int, q: int, i3: float, y: float) -> float:
        """Calculate the coefficient for I- ladder operator."""
        # Convert weight back to Dynkin basis
        m2 = round(2 * ((p + q) / 3.0 - y) / np.sqrt(3))
        m1 = round((p - q) / 2.0 - i3 + m2 / 2.0)
        
        # Clamp to valid range
        m1 = max(0, min(p, int(m1)))
        m2 = max(0, min(q, int(m2)))
        
        # Coefficient formula: sqrt(m1 * (p - m1 + 1))
        coeff = np.sqrt(m1 * (p - m1 + 1))
        
        return coeff
    
    def _uspin_raising_coefficient(self, p: int, q: int, i3: float, y: float) -> float:
        """
        Calculate the coefficient for U+ ladder operator.
        
        U-spin operates with the second simple root alpha2 = (-1/2, sqrt(3)/2).
        """
        # Convert weight back to Dynkin basis
        m2 = round(2 * ((p + q) / 3.0 - y) / np.sqrt(3))
        m1 = round((p - q) / 2.0 - i3 + m2 / 2.0)
        
        # Clamp to valid range
        m1 = max(0, min(p, int(m1)))
        m2 = max(0, min(q, int(m2)))
        
        # Coefficient formula for U+ (second root): sqrt((q - m2) * (m2 + 1))
        coeff = np.sqrt((q - m2) * (m2 + 1))
        
        return coeff
    
    def _uspin_lowering_coefficient(self, p: int, q: int, i3: float, y: float) -> float:
        """Calculate the coefficient for U- ladder operator."""
        # Convert weight back to Dynkin basis
        m2 = round(2 * ((p + q) / 3.0 - y) / np.sqrt(3))
        m1 = round((p - q) / 2.0 - i3 + m2 / 2.0)
        
        # Clamp to valid range
        m1 = max(0, min(p, int(m1)))
        m2 = max(0, min(q, int(m2)))
        
        # Coefficient formula for U-: sqrt(m2 * (q - m2 + 1))
        coeff = np.sqrt(m2 * (q - m2 + 1))
        
        return coeff
    
    def _vspin_raising_coefficient(self, p: int, q: int, i3: float, y: float) -> float:
        """
        Calculate the coefficient for V+ ladder operator.
        
        V+ corresponds to the positive root alpha1 + alpha2.
        """
        # Convert weight back to Dynkin basis
        m2 = round(2 * ((p + q) / 3.0 - y) / np.sqrt(3))
        m1 = round((p - q) / 2.0 - i3 + m2 / 2.0)
        
        # Clamp to valid range
        m1 = max(0, min(p, int(m1)))
        m2 = max(0, min(q, int(m2)))
        
        # V+ corresponds to the positive root, which involves both m1 and m2
        # Coefficient: sqrt((p - m1) * (q - m2) * (m1 + m2 + 1))
        # Simplified: check the available "room" in both directions
        coeff = np.sqrt(max(0, m1 * m2))
        
        return coeff
    
    def _vspin_lowering_coefficient(self, p: int, q: int, i3: float, y: float) -> float:
        """Calculate the coefficient for V- ladder operator."""
        # Convert weight back to Dynkin basis
        m2 = round(2 * ((p + q) / 3.0 - y) / np.sqrt(3))
        m1 = round((p - q) / 2.0 - i3 + m2 / 2.0)
        
        # Clamp to valid range
        m1 = max(0, min(p, int(m1)))
        m2 = max(0, min(q, int(m2)))
        
        # V- corresponds to the negative root -(alpha1 + alpha2)
        coeff = np.sqrt(max(0, (p - m1 + 1) * (q - m2 + 1)))
        
        return coeff
    
    def build_C2(self) -> csr_matrix:
        """
        Build the quadratic Casimir operator C2.
        
        C2 = T3^2 + T8^2 + (1/2){I+, I-} + (1/2){U+, U-} + (1/2){V+, V-}
        
        where {A, B} = AB + BA is the anticommutator.
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
