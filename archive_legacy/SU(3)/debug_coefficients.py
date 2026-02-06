"""
Debug script to understand the exact coefficient formulas.
"""

import numpy as np
from lattice import SU3Lattice


def test_i_plus_formula():
    """Test the I+ formula."""
    print("="*70)
    print("Testing I+ Formula")
    print("="*70)
    
    lattice = SU3Lattice(max_p=1, max_q=1)
    
    print("\nChecking I+ transitions:")
    for state in lattice.states:
        gt = state['gt']
        m13, m23, m33, m12, m22, m11 = gt
        
        # Try to raise m11
        m11_new = m11 + 1
        if m11_new <= m12:
            gt_new = (m13, m23, m33, m12, m22, m11_new)
            
            # Formula: sqrt(-(m11 - m12 + 1)(m11 - m22))
            # This should be sqrt((m12 - m11)(m11 - m22 + 1))
            factor1 = m12 - m11
            factor2 = m11 - m22 + 1
            
            if factor1 >= 0 and factor2 >= 0:
                coeff = np.sqrt(factor1 * factor2)
                print(f"  {gt} -> {gt_new}: coeff = {coeff:.4f}")
                print(f"    factors: ({m12}-{m11}) * ({m11}-{m22}+1) = {factor1} * {factor2}")


def test_e23_formula():
    """Test the E23 formula."""
    print("\n" + "="*70)
    print("Testing E23 Formula")
    print("="*70)
    
    lattice = SU3Lattice(max_p=2, max_q=1)
    
    # Filter to (2,1) which should have U+ transitions
    states_21 = [s for s in lattice.states if s['p'] == 2 and s['q'] == 1]
    print(f"\n(2,1) has {len(states_21)} states")
    
    print("\nChecking E23 transitions:")
    for state in states_21[:5]:  # Check first 5
        gt = state['gt']
        m13, m23, m33, m12, m22, m11 = gt
        
        print(f"\n  From state: {gt}")
        
        # Term 1: Shift m12 -> m12 + 1
        m12_new = m12 + 1
        if m12_new <= m13:
            gt_new = (m13, m23, m33, m12_new, m22, m11)
            
            # Check all factors
            num1 = m12 - m13
            num2 = m12 - m23 + 1
            num3 = m12 - m33 + 2
            num4 = m12 - m11 + 1
            den1 = m12 - m22 + 1
            den2 = m12 - m22 + 2
            
            print(f"    Term 1 (m12+1): {gt_new}")
            print(f"      Numerator: ({num1}) * ({num2}) * ({num3}) * ({num4})")
            print(f"      Denominator: ({den1}) * ({den2})")
            
            if den1 * den2 > 0:
                # All numerator factors should be ≥ 0 by GT constraints
                numerator = abs(num1) * num2 * num3 * num4
                denominator = den1 * den2
                if numerator >= 0:
                    coeff = np.sqrt(numerator / denominator)
                    print(f"      Coeff = {coeff:.4f}")
        
        # Term 2: Shift m22 -> m22 + 1  
        m22_new = m22 + 1
        if m22_new <= m23:
            gt_new = (m13, m23, m33, m12, m22_new, m11)
            
            # Check all factors
            num1 = m22 - m13 - 1
            num2 = m22 - m23
            num3 = m22 - m33 + 1
            num4 = m22 - m11
            den1 = m22 - m12 - 1
            den2 = m22 - m12
            
            print(f"    Term 2 (m22+1): {gt_new}")
            print(f"      Numerator: ({num1}) * ({num2}) * ({num3}) * ({num4})")
            print(f"      Denominator: ({den1}) * ({den2})")
            
            if abs(den1 * den2) > 0:
                # Use absolute values where needed
                numerator = abs(num1) * abs(num2) * num3 * abs(num4)
                denominator = abs(den1 * den2)
                if numerator >= 0:
                    coeff = np.sqrt(numerator / denominator)
                    print(f"      Coeff = {coeff:.4f}")


if __name__ == "__main__":
    test_i_plus_formula()
    test_e23_formula()
