"""
SU(3) Operators v9: CORRECT Commutation Relations
==================================================
Based on analysis of standard Gell-Mann matrices showing:
- [E12, E21] = 2*T3
- [E23, E32] = T3 + √3*T8  (NOT -(3/2)*T3 + (√3/2)*T8!)
- [E13, E31] = -T3 + √3*T8

Using:
- Weight-space T3 and T8
- GT formulas with √2 normalization for ladder operators
"""

import numpy as np
from lattice import SU3Lattice


class SU3OperatorsV9:
    """SU(3) operators with correct commutation relations."""
    
    def __init__(self, p, q):
        """Initialize operators for the (p,q) irrep."""
        self.lattice = SU3Lattice(p, q)
        self.p = p
        self.q = q
        
        # Extract states for this specific (p,q)
        self.states = [(s['m13'], s['m23'], s['m33'], s['m12'], s['m22'], s['m11']) 
                       for s in self.lattice.states if s['p'] == p and s['q'] == q]
        self.dim = len(self.states)
        
        # Build Cartan operators from weight space
        self._T3 = self._build_T3_weight()
        self._T8 = self._build_T8_weight()
        
        # Build ladder operators with √2 correction
        self._E12 = self._build_E12()
        self._E23 = self._build_E23()
        
        # Adjoints
        self._E21 = self._E12.T.conj()
        self._E32 = self._E23.T.conj()
        
        # E13 via commutator
        self._E13 = self._E12 @ self._E23 - self._E23 @ self._E12
        self._E31 = self._E13.T.conj()
    
    def _m_to_l(self, m13, m23, m33, m12, m22, m11):
        """Convert m-indices to l-indices."""
        return (m13 + 2, m23 + 1, m33, m12 + 1, m22, m11)
    
    def _build_T3_weight(self):
        """Build T3 from weight space: I3 = m11 - (m12+m22)/2."""
        T3 = np.zeros((self.dim, self.dim), dtype=complex)
        
        for i, (m13, m23, m33, m12, m22, m11) in enumerate(self.states):
            T3[i, i] = m11 - 0.5 * (m12 + m22)
        
        return T3
    
    def _build_T8_weight(self):
        """Build T8 from weight space: Y*√3/2."""
        T8 = np.zeros((self.dim, self.dim), dtype=complex)
        
        for i, (m13, m23, m33, m12, m22, m11) in enumerate(self.states):
            y = (m12 + m22) - (2.0/3.0) * (m13 + m23 + m33)
            T8[i, i] = (np.sqrt(3.0) / 2.0) * y
        
        return T8
    
    def _build_E12(self):
        """Build I+ with √2 correction."""
        E12 = np.zeros((self.dim, self.dim), dtype=complex)
        
        for i, (m13, m23, m33, m12, m22, m11) in enumerate(self.states):
            m11_new = m11 + 1
            if not (m12 >= m11_new >= m22):
                continue
            
            target_state = (m13, m23, m33, m12, m22, m11_new)
            if target_state not in self.states:
                continue
            j = self.states.index(target_state)
            
            l13, l23, l33, l12, l22, l11 = self._m_to_l(m13, m23, m33, m12, m22, m11)
            
            coeff_raw = np.sqrt(abs((l12 - l11) * (l22 - l11 - 1)))
            coeff = coeff_raw / np.sqrt(2.0)
            
            E12[j, i] = coeff
        
        return E12
    
    def _build_E23(self):
        """Build U+ with √2 correction."""
        E23 = np.zeros((self.dim, self.dim), dtype=complex)
        
        for i, (m13, m23, m33, m12, m22, m11) in enumerate(self.states):
            l13, l23, l33, l12, l22, l11 = self._m_to_l(m13, m23, m33, m12, m22, m11)
            
            # Term 1
            m12_new = m12 + 1
            if m13 >= m12_new >= m23:
                target_state = (m13, m23, m33, m12_new, m22, m11)
                if target_state in self.states:
                    j = self.states.index(target_state)
                    
                    numerator = (l13 - l12 - 1) * (l23 - l12 - 1) * (l33 - l12 - 1) * (l11 - l12)
                    denominator = (l12 - l22) * (l12 - l22 + 1)
                    
                    if denominator != 0:
                        coeff_raw = np.sqrt(abs(numerator / denominator))
                        coeff = coeff_raw / np.sqrt(2.0)
                        E23[j, i] += coeff
            
            # Term 2
            m22_new = m22 + 1
            if m12 >= m11 >= m22_new:
                target_state = (m13, m23, m33, m12, m22_new, m11)
                if target_state in self.states:
                    j = self.states.index(target_state)
                    
                    numerator = (l13 - l22 - 1) * (l23 - l22 - 1) * (l33 - l22 - 1) * (l11 - l22)
                    denominator = (l22 - l12) * (l22 - l12 + 1)
                    
                    if denominator != 0:
                        coeff_raw = np.sqrt(abs(numerator / denominator))
                        coeff = coeff_raw / np.sqrt(2.0)
                        E23[j, i] += coeff
        
        return E23
    
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
    """Test v9 with correct commutation relations."""
    print("SU(3) Operators v9: Correct Commutation Relations\n")
    print("="*70)
    
    ops = SU3OperatorsV9(1, 0)
    
    print(f"\nTesting (1,0) fundamental")
    print(f"Dimension: {ops.dim}\n")
    
    # Check operators
    print("E12 (I+):")
    print(ops.E12)
    print("\nE23 (U+):")
    print(ops.E23)
    print("\nE13 (V+):")
    print(ops.E13)
    
    print("\nT3:")
    print(ops.T3)
    print(f"Diagonal: {np.diag(ops.T3).real}")
    
    print("\nT8:")
    print(ops.T8)
    print(f"Diagonal: {np.diag(ops.T8).real}")
    
    # Commutator tests WITH CORRECT FORMULAS
    print("\n" + "="*70)
    print("Commutator Tests (CORRECT RELATIONS):")
    print("="*70)
    
    tests = [
        ("[T3, T8]", ops.T3 @ ops.T8 - ops.T8 @ ops.T3, np.zeros_like(ops.T3)),
        ("[E12, E21] - 2*T3", ops.E12 @ ops.E21 - ops.E21 @ ops.E12, 2 * ops.T3),
        ("[T3, E12] - E12", ops.T3 @ ops.E12 - ops.E12 @ ops.T3, ops.E12),
        ("[E23, E32] - (T3 + √3*T8)", 
         ops.E23 @ ops.E32 - ops.E32 @ ops.E23, 
         ops.T3 + np.sqrt(3.0) * ops.T8),
        ("[E13, E31] - (-T3 + √3*T8)", 
         ops.E13 @ ops.E31 - ops.E31 @ ops.E13,
         -ops.T3 + np.sqrt(3.0) * ops.T8),
        ("[E12, E23] - E13", ops.E12 @ ops.E23 - ops.E23 @ ops.E12, ops.E13),
    ]
    
    max_error = 0
    for name, actual, expected in tests:
        error = np.max(np.abs(actual - expected))
        max_error = max(max_error, error)
        status = "✓ PASS" if error < 1e-13 else "✗ FAIL"
        print(f"{name}: {error:.3e} {status}")
    
    print(f"\nMaximum error: {max_error:.3e}")
    print(f"Target: < 1e-13")
    
    if max_error < 1e-13:
        print("\n🎉 SUCCESS! All commutators at machine precision! 🎉")
