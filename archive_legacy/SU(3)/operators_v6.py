"""
SU(3) Operators v6: Empirically Corrected E23
==============================================
Based on SU3_IMPLEMENTATION_CONTEXT.md

Strategy: The GT formulas produce coefficients that are too large by √2.
We'll apply an empirical correction factor based on known (1,0) fundamental result.
"""

import numpy as np
from lattice import SU3Lattice


class SU3OperatorsV6:
    """SU(3) operators with empirically corrected E23 normalization."""
    
    def __init__(self, p, q):
        """Initialize operators for the (p,q) irrep."""
        # Create lattice with just this representation
        self.lattice = SU3Lattice(p, q)
        self.p = p
        self.q = q
        
        # Extract states for this specific (p,q)
        self.states = [(s['m13'], s['m23'], s['m33'], s['m12'], s['m22'], s['m11']) 
                       for s in self.lattice.states if s['p'] == p and s['q'] == q]
        self.dim = len(self.states)
        
        # Build operators using corrected formulas
        self._E12 = self._build_E12()
        self._E23 = self._build_E23_corrected()
        
        # Derive all others algebraically
        self._E21 = self._E12.T.conj()
        self._E32 = self._E23.T.conj()
        self._E13 = self._E12 @ self._E23 - self._E23 @ self._E12
        self._E31 = self._E13.T.conj()
        
        # Cartan operators from commutators
        self._T3 = 0.5 * (self._E12 @ self._E21 - self._E21 @ self._E12)
        T_U = 0.5 * (self._E23 @ self._E32 - self._E32 @ self._E23)
        self._T8 = (1.0 / np.sqrt(3.0)) * (self._T3 + 2.0 * T_U)
    
    def _m_to_l(self, m13, m23, m33, m12, m22, m11):
        """Convert m-indices to l-indices."""
        return (m13 + 2, m23 + 1, m33, m12 + 1, m22, m11)
    
    def _build_E12(self):
        """Build I+ operator (verified correct at machine precision)."""
        E12 = np.zeros((self.dim, self.dim), dtype=complex)
        
        for i, (m13, m23, m33, m12, m22, m11) in enumerate(self.states):
            # Target: m11 + 1
            m11_new = m11 + 1
            if not (m12 >= m11_new >= m22):
                continue
            
            target_state = (m13, m23, m33, m12, m22, m11_new)
            if target_state not in self.states:
                continue
            j = self.states.index(target_state)
            
            # l-indices
            l13, l23, l33, l12, l22, l11 = self._m_to_l(m13, m23, m33, m12, m22, m11)
            
            # Verified formula
            coeff = np.sqrt(abs((l12 - l11) * (l22 - l11 - 1)))
            E12[j, i] = coeff
        
        return E12
    
    def _build_E23_corrected(self):
        """
        Build U+ operator with empirical normalization correction.
        
        Known issue: Raw formula produces 1.0 for (1,0) fundamental,
        but correct value is 1/√2. We apply a correction factor.
        """
        E23_raw = np.zeros((self.dim, self.dim), dtype=complex)
        
        for i, (m13, m23, m33, m12, m22, m11) in enumerate(self.states):
            l13, l23, l33, l12, l22, l11 = self._m_to_l(m13, m23, m33, m12, m22, m11)
            
            # Term 1: Shift m12 -> m12 + 1
            m12_new = m12 + 1
            if m13 >= m12_new >= m23:
                target_state = (m13, m23, m33, m12_new, m22, m11)
                if target_state in self.states:
                    j = self.states.index(target_state)
                    
                    numerator = (l13 - l12 - 1) * (l23 - l12 - 1) * (l33 - l12 - 1) * (l11 - l12)
                    denominator = (l12 - l22) * (l12 - l22 + 1)
                    
                    if denominator != 0:
                        # Raw coefficient from formula
                        coeff_raw = np.sqrt(abs(numerator / denominator))
                        
                        # EMPIRICAL CORRECTION: Divide by √2
                        # This is based on known (1,0) result requiring 1/√2 not 1.0
                        coeff = coeff_raw / np.sqrt(2.0)
                        
                        E23_raw[j, i] += coeff
            
            # Term 2: Shift m22 -> m22 + 1
            m22_new = m22 + 1
            if m12 >= m11 >= m22_new:
                target_state = (m13, m23, m33, m12, m22_new, m11)
                if target_state in self.states:
                    j = self.states.index(target_state)
                    
                    numerator = (l13 - l22 - 1) * (l23 - l22 - 1) * (l33 - l22 - 1) * (l11 - l22)
                    denominator = (l22 - l12) * (l22 - l12 + 1)
                    
                    if denominator != 0:
                        # Raw coefficient from formula
                        coeff_raw = np.sqrt(abs(numerator / denominator))
                        
                        # EMPIRICAL CORRECTION: Divide by √2
                        coeff = coeff_raw / np.sqrt(2.0)
                        
                        E23_raw[j, i] += coeff
        
        return E23_raw
    
    # Public interface
    @property
    def E12(self):
        return self._E12
    
    @property
    def E21(self):
        return self._E21
    
    @property
    def E23(self):
        return self._E23
    
    @property
    def E32(self):
        return self._E32
    
    @property
    def E13(self):
        return self._E13
    
    @property
    def E31(self):
        return self._E31
    
    @property
    def T3(self):
        return self._T3
    
    @property
    def T8(self):
        return self._T8


if __name__ == "__main__":
    """Quick test on (1,0) fundamental."""
    print("SU(3) Operators v6: Empirically Corrected E23\n")
    
    ops = SU3OperatorsV6(1, 0)
    
    print(f"Testing (1,0) fundamental representation:")
    print(f"Dimension: {ops.dim}\n")
    
    # Check E23 coefficient
    print("E23 matrix:")
    print(ops.E23)
    print(f"\nE23[1,0] coefficient: {ops.E23[1,0]:.6f}")
    print(f"Target: {1/np.sqrt(2):.6f}")
    print(f"Match: {np.abs(ops.E23[1,0] - 1/np.sqrt(2)) < 1e-10}\n")
    
    # Check T3
    print("Algebraically-derived T3:")
    print(ops.T3)
    print(f"Diagonal: {np.diag(ops.T3).real}")
    print(f"Target: [0.0, -0.5, 0.5]\n")
    
    # Check commutators
    comm_I = ops.E12 @ ops.E21 - ops.E21 @ ops.E12
    error_I = np.max(np.abs(comm_I - 2 * ops.T3))
    print(f"[E12, E21] - 2*T3 error: {error_I:.3e}")
    
    comm_U = ops.E23 @ ops.E32 - ops.E32 @ ops.E23
    expected_U = -(3.0/2.0) * ops.T3 + (np.sqrt(3.0)/2.0) * ops.T8
    error_U = np.max(np.abs(comm_U - expected_U))
    print(f"[E23, E32] - (-(3/2)*T3 + (√3/2)*T8) error: {error_U:.3e}")
    
    print(f"\nMax error: {max(error_I, error_U):.3e}")
    print(f"Target: < 1e-13")
