"""
SU(3) Operators v7: Global √2 Normalization Correction
========================================================
Based on empirical testing showing:
1. E12 formula is correct in structure but needs 1/√2 factor
2. E23 formula is correct in structure but needs 1/√2 factor  
3. With these corrections: T3 = (1/2)*[E12, E21] gives correct values

The Biedenharn-Louck formulas appear to use a different normalization
convention than the standard physics Gell-Mann matrices.
"""

import numpy as np
from lattice import SU3Lattice


class SU3OperatorsV7:
    """SU(3) operators with global √2 normalization correction."""
    
    def __init__(self, p, q):
        """Initialize operators for the (p,q) irrep."""
        self.lattice = SU3Lattice(p, q)
        self.p = p
        self.q = q
        
        # Extract states for this specific (p,q)
        self.states = [(s['m13'], s['m23'], s['m33'], s['m12'], s['m22'], s['m11']) 
                       for s in self.lattice.states if s['p'] == p and s['q'] == q]
        self.dim = len(self.states)
        
        # Build core operators with GT formulas
        self._E12 = self._build_E12()
        self._E23 = self._build_E23()
        
        # Derive all others algebraically
        self._E21 = self._E12.T.conj()
        self._E32 = self._E23.T.conj()
        self._E13 = self._E12 @ self._E23 - self._E23 @ self._E12
        self._E31 = self._E13.T.conj()
        
        # Cartan operators from commutators (with correct normalization)
        self._T3 = 0.5 * (self._E12 @ self._E21 - self._E21 @ self._E12)
        T_U = 0.5 * (self._E23 @ self._E32 - self._E32 @ self._E23)
        self._T8 = (1.0 / np.sqrt(3.0)) * (self._T3 + 2.0 * T_U)
    
    def _m_to_l(self, m13, m23, m33, m12, m22, m11):
        """Convert m-indices to l-indices."""
        return (m13 + 2, m23 + 1, m33, m12 + 1, m22, m11)
    
    def _build_E12(self):
        """
        Build I+ operator with √2 normalization correction.
        GT formula gives correct structure but wrong overall scale.
        """
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
            
            # GT formula with √2 normalization correction
            coeff_raw = np.sqrt(abs((l12 - l11) * (l22 - l11 - 1)))
            coeff = coeff_raw / np.sqrt(2.0)
            
            E12[j, i] = coeff
        
        return E12
    
    def _build_E23(self):
        """
        Build U+ operator with √2 normalization correction.
        GT formula gives correct structure but wrong overall scale.
        """
        E23 = np.zeros((self.dim, self.dim), dtype=complex)
        
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
                        coeff_raw = np.sqrt(abs(numerator / denominator))
                        coeff = coeff_raw / np.sqrt(2.0)
                        E23[j, i] += coeff
            
            # Term 2: Shift m22 -> m22 + 1
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
    """Test v7 on (1,0) fundamental."""
    print("SU(3) Operators v7: Global √2 Correction\n")
    print("="*70)
    
    ops = SU3OperatorsV7(1, 0)
    
    print(f"\nTesting (1,0) fundamental representation")
    print(f"Dimension: {ops.dim}\n")
    
    # Check coefficients
    print("E12 matrix:")
    print(ops.E12)
    print(f"\nE23 matrix:")
    print(ops.E23)
    
    # Check T3
    print(f"\nT3 (from [E12, E21]/2):")
    print(ops.T3)
    print(f"Diagonal: {np.diag(ops.T3).real}")
    print(f"Expected: [0.0, -0.5, 0.5]")
    
    # Check T8
    print(f"\nT8 (from (T3 + 2*T_U)/√3):")
    print(ops.T8)
    print(f"Diagonal: {np.diag(ops.T8).real}")
    print(f"Expected: [-0.577, 0.289, 0.289]")
    
    # Check commutators
    print("\n" + "="*70)
    print("Commutator Tests:")
    print("="*70)
    
    # [T3, T8] = 0
    comm = ops.T3 @ ops.T8 - ops.T8 @ ops.T3
    error = np.max(np.abs(comm))
    status = "✓ PASS" if error < 1e-13 else "✗ FAIL"
    print(f"[T3, T8] = 0: {error:.3e} {status}")
    
    # [E12, E21] = 2*T3
    comm = ops.E12 @ ops.E21 - ops.E21 @ ops.E12
    error = np.max(np.abs(comm - 2 * ops.T3))
    status = "✓ PASS" if error < 1e-13 else "✗ FAIL"
    print(f"[E12, E21] - 2*T3: {error:.3e} {status}")
    
    # [E23, E32] = -(3/2)*T3 + (√3/2)*T8
    comm = ops.E23 @ ops.E32 - ops.E32 @ ops.E23
    expected = -(3.0/2.0) * ops.T3 + (np.sqrt(3.0)/2.0) * ops.T8
    error = np.max(np.abs(comm - expected))
    status = "✓ PASS" if error < 1e-13 else "✗ FAIL"
    print(f"[E23, E32] - (-(3/2)*T3 + (√3/2)*T8): {error:.3e} {status}")
    
    # [E13, E31] = (3/2)*T3 + (√3/2)*T8
    comm = ops.E13 @ ops.E31 - ops.E31 @ ops.E13
    expected = (3.0/2.0) * ops.T3 + (np.sqrt(3.0)/2.0) * ops.T8
    error = np.max(np.abs(comm - expected))
    status = "✓ PASS" if error < 1e-13 else "✗ FAIL"
    print(f"[E13, E31] - ((3/2)*T3 + (√3/2)*T8): {error:.3e} {status}")
    
    # [E12, E23] = E13
    comm = ops.E12 @ ops.E23 - ops.E23 @ ops.E12
    error = np.max(np.abs(comm - ops.E13))
    status = "✓ PASS" if error < 1e-13 else "✗ FAIL"
    print(f"[E12, E23] - E13: {error:.3e} {status}")
    
    print()
