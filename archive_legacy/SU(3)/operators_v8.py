"""
SU(3) Operators v8: Hybrid Approach
====================================
- Use GT formulas with √2 correction for E12, E23
- Use weight-space definitions for T3, T8 (NOT algebraic closure)
- Derive E13 via commutator

This tests whether the issue is the algebraic closure philosophy.
"""

import numpy as np
from lattice import SU3Lattice


class SU3OperatorsV8:
    """SU(3) operators: GT formulas for ladders, weight-space for Cartans."""
    
    def __init__(self, p, q):
        """Initialize operators for the (p,q) irrep."""
        self.lattice = SU3Lattice(p, q)
        self.p = p
        self.q = q
        
        # Extract states for this specific (p,q)
        self.states = [(s['m13'], s['m23'], s['m33'], s['m12'], s['m22'], s['m11']) 
                       for s in self.lattice.states if s['p'] == p and s['q'] == q]
        self.dim = len(self.states)
        
        # Build Cartan operators FIRST (from weight space, not algebra)
        self._T3 = self._build_T3_weight()
        self._T8 = self._build_T8_weight()
        
        # Build ladder operators with√2 correction
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
    """Test v8 hybrid approach."""
    print("SU(3) Operators v8: Hybrid (Weight-Space Cartans)\n")
    print("="*70)
    
    ops = SU3OperatorsV8(1, 0)
    
    print(f"\nTesting (1,0) fundamental")
    print(f"Dimension: {ops.dim}\n")
    
    # Check T3 and T8
    print("T3 (from weight space):")
    print(ops.T3)
    print(f"Diagonal: {np.diag(ops.T3).real}")
    
    print("\nT8 (from weight space):")
    print(ops.T8)
    print(f"Diagonal: {np.diag(ops.T8).real}")
    
    # Commutator tests
    print("\n" + "="*70)
    print("Commutator Tests:")
    print("="*70)
    
    tests = [
        ("[T3, T8]", ops.T3 @ ops.T8 - ops.T8 @ ops.T3, np.zeros_like(ops.T3)),
        ("[E12, E21] - 2*T3", ops.E12 @ ops.E21 - ops.E21 @ ops.E12, 2 * ops.T3),
        ("[T3, E12] - E12", ops.T3 @ ops.E12 - ops.E12 @ ops.T3, ops.E12),
        ("[E23, E32] - (-(3/2)*T3 + (√3/2)*T8)", 
         ops.E23 @ ops.E32 - ops.E32 @ ops.E23, 
         -(3.0/2.0) * ops.T3 + (np.sqrt(3.0)/2.0) * ops.T8),
        ("[E13, E31] - ((3/2)*T3 + (√3/2)*T8)", 
         ops.E13 @ ops.E31 - ops.E31 @ ops.E13,
         (3.0/2.0) * ops.T3 + (np.sqrt(3.0)/2.0) * ops.T8),
        ("[E12, E23] - E13", ops.E12 @ ops.E23 - ops.E23 @ ops.E12, ops.E13),
    ]
    
    for name, actual, expected in tests:
        error = np.max(np.abs(actual - expected))
        status = "✓ PASS" if error < 1e-13 else "✗ FAIL"
        print(f"{name}: {error:.3e} {status}")
    
    print()
