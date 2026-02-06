"""
SU(3) Generators v12: Algebraic Closure with Correct Normalization
=======================================================================
v5 + normalization to match Tr(λᵢ²) = 2

Key changes:
- E23: multiply by √2 (so λ4, λ5 have trace norm 2)
- T3: from [E12, E21]/2 (so λ3 has trace norm 2)
- T8: from ([E23, E32] - T3)/√3 (so λ8 has trace norm 2)

This ensures Gell-Mann normalization and constant Casimir eigenvalues.
"""

import numpy as np
from lattice import SU3Lattice


class SU3OperatorsV12:
    """SU(3) operators via algebraic closure with Gell-Mann normalization."""
    
    def __init__(self, p, q):
        """
        Initialize operators for the (p,q) irrep.
        
        Args:
            p, q: Dynkin labels for the SU(3) representation
        """
        # Create lattice with just this representation
        self.lattice = SU3Lattice(p, q)
        self.p = p
        self.q = q
        
        # Extract states for this specific (p,q)
        self.states = [(s['m13'], s['m23'], s['m33'], s['m12'], s['m22'], s['m11']) 
                       for s in self.lattice.states if s['p'] == p and s['q'] == q]
        self.dim = len(self.states)
        
        # Build the two fundamental raising operators
        self._E12 = self._build_E12()
        self._E23 = self._build_E23() * np.sqrt(2)  # Normalization correction
        
        # Derive all other operators algebraically
        self._E21 = self._E12.T.conj()  # Hermitian conjugate
        self._E32 = self._E23.T.conj()
        
        # E13 via commutator [E12, E23]
        self._E13 = self._E12 @ self._E23 - self._E23 @ self._E12
        self._E31 = self._E13.T.conj()
        
        # T3 from I-spin commutator [E12, E21] / 2
        self._T3 = 0.5 * (self._E12 @ self._E21 - self._E21 @ self._E12)
        
        # T8 from [E23, E32] = T3 + √3*T8  →  T8 = ([E23, E32] - T3)/√3
        comm_23_32 = self._E23 @ self._E32 - self._E32 @ self._E23
        self._T8 = (comm_23_32 - self._T3) / np.sqrt(3)
        
    def _m_to_l(self, m13, m23, m33, m12, m22, m11):
        """Convert m-indices to l-indices using the standard shift."""
        return (m13 + 2, m23 + 1, m33, m12 + 1, m22, m11)
    
    def _build_E12(self):
        """
        Build I+ operator (E12) using GT formula.
        Shifts m11 -> m11 + 1.
        """
        E12 = np.zeros((self.dim, self.dim), dtype=complex)
        
        for i, (m13, m23, m33, m12, m22, m11) in enumerate(self.states):
            # Target state: m11 + 1
            m11_new = m11 + 1
            
            # Check constraint: m12 >= m11_new >= m22
            if not (m12 >= m11_new >= m22):
                continue
            
            # Find target state index
            target_state = (m13, m23, m33, m12, m22, m11_new)
            if target_state not in self.states:
                continue
            j = self.states.index(target_state)
            
            # Convert to l-indices
            l13, l23, l33, l12, l22, l11 = self._m_to_l(m13, m23, m33, m12, m22, m11)
            
            # GT formula: sqrt(|(l12 - l11) * (l22 - l11 - 1)|)  / √2
            factor1 = l12 - l11
            factor2 = l22 - l11 - 1
            
            coeff = np.sqrt(abs(factor1 * factor2)) / np.sqrt(2)
            E12[j, i] = coeff
        
        return E12
    
    def _build_E23(self):
        """
        Build U+ operator (E23) using GT formula.
        Two terms: shift m12 or m22.
        """
        E23 = np.zeros((self.dim, self.dim), dtype=complex)
        
        for i, (m13, m23, m33, m12, m22, m11) in enumerate(self.states):
            # Convert to l-indices for current state
            l13, l23, l33, l12, l22, l11 = self._m_to_l(m13, m23, m33, m12, m22, m11)
            
            # Term 1: Shift m12 -> m12 + 1
            m12_new = m12 + 1
            if m13 >= m12_new >= m23:  # Constraint check
                target_state = (m13, m23, m33, m12_new, m22, m11)
                if target_state in self.states:
                    j = self.states.index(target_state)
                    
                    # GT formula for Term 1, with 1/√2 correction
                    numerator = (l13 - l12 - 1) * (l23 - l12 - 1) * (l33 - l12 - 1) * (l11 - l12)
                    denominator = (l12 - l22) * (l12 - l22 + 1)
                    
                    if denominator != 0:
                        coeff = np.sqrt(abs(numerator / denominator)) / np.sqrt(2)
                        E23[j, i] += coeff
            
            # Term 2: Shift m22 -> m22 + 1
            m22_new = m22 + 1
            if m12 >= m11 >= m22_new:  # Constraint check
                target_state = (m13, m23, m33, m12, m22_new, m11)
                if target_state in self.states:
                    j = self.states.index(target_state)
                    
                    # GT formula for Term 2, with 1/√2 correction
                    numerator = (l13 - l22 - 1) * (l23 - l22 - 1) * (l33 - l22 - 1) * (l11 - l22)
                    denominator = (l22 - l12) * (l22 - l12 + 1)
                    
                    if denominator != 0:
                        coeff = np.sqrt(abs(numerator / denominator)) / np.sqrt(2)
                        E23[j, i] += coeff
        
        return E23
    
    # Public interface
    @property
    def E12(self):
        """I+ raising operator."""
        return self._E12
    
    @property
    def E21(self):
        """I- lowering operator (E12†)."""
        return self._E21
    
    @property
    def E23(self):
        """U+ raising operator."""
        return self._E23
    
    @property
    def E32(self):
        """U- lowering operator (E23†)."""
        return self._E32
    
    @property
    def E13(self):
        """V+ raising operator ([E12, E23])."""
        return self._E13
    
    @property
    def E31(self):
        """V- lowering operator (E13†)."""
        return self._E31
    
    @property
    def T3(self):
        """Cartan generator T3 (from [E12, E21])."""
        return self._T3
    
    @property
    def T8(self):
        """Cartan generator T8 (from T3 + 2*T_U)."""
        return self._T8


if __name__ == "__main__":
    """Quick sanity check."""
    print("SU(3) Operators v5: Algebraic Closure\n")
    
    # Test on (1,1) adjoint representation
    ops = SU3OperatorsV5(1, 1)
    
    # Check hermiticity of Cartan operators
    T3_hermitian = np.allclose(ops.T3, ops.T3.T.conj())
    T8_hermitian = np.allclose(ops.T8, ops.T8.T.conj())
    print(f"T3 Hermitian: {T3_hermitian}")
    print(f"T8 Hermitian: {T8_hermitian}")
    
    # Check [T3, T8] = 0
    comm_T3_T8 = ops.T3 @ ops.T8 - ops.T8 @ ops.T3
    error_cartans = np.max(np.abs(comm_T3_T8))
    print(f"[T3, T8] error: {error_cartans:.3e}")
    
    # Check I-spin commutator [E12, E21] = 2*T3
    comm_I = ops.E12 @ ops.E21 - ops.E21 @ ops.E12
    error_I = np.max(np.abs(comm_I - 2 * ops.T3))
    print(f"[E12, E21] - 2*T3 error: {error_I:.3e}")
    
    # Check U-spin commutator [E23, E32] = -(3/2)*T3 + (√3/2)*T8
    comm_U = ops.E23 @ ops.E32 - ops.E32 @ ops.E23
    expected_U = -(3.0/2.0) * ops.T3 + (np.sqrt(3.0)/2.0) * ops.T8
    error_U = np.max(np.abs(comm_U - expected_U))
    print(f"[E23, E32] - (-(3/2)*T3 + (√3/2)*T8) error: {error_U:.3e}")
    
    # Check V-spin commutator [E13, E31] = (3/2)*T3 + (√3/2)*T8
    comm_V = ops.E13 @ ops.E31 - ops.E31 @ ops.E13
    expected_V = (3.0/2.0) * ops.T3 + (np.sqrt(3.0)/2.0) * ops.T8
    error_V = np.max(np.abs(comm_V - expected_V))
    print(f"[E13, E31] - ((3/2)*T3 + (√3/2)*T8) error: {error_V:.3e}")
    
    print(f"\nMaximum error: {max(error_I, error_U, error_V):.3e}")
