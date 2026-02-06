"""
Debug E23 coefficients on (1,1) adjoint.
"""

import numpy as np
from lattice import SU3Lattice


def detailed_e23_check():
    """Check E23 formula on (1,1) adjoint."""
    print("="*70)
    print("Detailed E23 check on (1,1) adjoint")
    print("="*70)
    
    lattice = SU3Lattice(max_p=1, max_q=1)
    
    # Get (1,1) states
    states_11 = [s for s in lattice.states if s['p'] == 1 and s['q'] == 1]
    print(f"\n(1,1) has {len(states_11)} states:")
    for i, s in enumerate(states_11):
        gt = s['gt']
        print(f"  {i}: {gt}")
    
    print("\n" + "="*70)
    print("Checking all E23 transitions")
    print("="*70)
    
    for state in states_11:
        gt = state['gt']
        m13, m23, m33, m12, m22, m11 = gt
        idx = state['index']
        
        print(f"\nFrom: {gt} (index {idx})")
        print(f"  m13={m13}, m23={m23}, m33={m33}, m12={m12}, m22={m22}, m11={m11}")
        
        # Term 1: Shift m12 -> m12 + 1
        m12_new = m12 + 1
        print(f"\n  Term 1: Try m12={m12} -> {m12_new}")
        print(f"    Constraint: m12_new={m12_new} <= m13={m13}? {m12_new <= m13}")
        
        if m12_new <= m13:
            gt_new = (m13, m23, m33, m12_new, m22, m11)
            
            # Find if this state exists
            found = False
            for s in lattice.states:
                if s['gt'] == gt_new and s['p'] == 1 and s['q'] == 1:
                    found = True
                    idx_new = s['index']
                    print(f"    → Found: {gt_new} (index {idx_new})")
            
            if not found:
                print(f"    → State {gt_new} does NOT exist in (1,1)")
            
            # Compute coefficient
            num1 = m13 - m12
            num2 = m12 - m23 + 1
            num3 = m12 - m33 + 2
            num4 = m12 - m11 + 1
            den1 = m12 - m22 + 1
            den2 = m12 - m22 + 2
            
            print(f"    Numerator factors: ({num1}) * ({num2}) * ({num3}) * ({num4})")
            print(f"    Denominator factors: ({den1}) * ({den2})")
            
            numerator = abs(num1) * num2 * num3 * num4
            denominator = den1 * den2
            
            if denominator > 0:
                coeff = np.sqrt(numerator / denominator)
                print(f"    Coefficient: {coeff:.6f}")
            else:
                print(f"    Denominator is zero or negative!")
        else:
            print(f"    Constraint violated - no transition")
        
        # Term 2: Shift m22 -> m22 + 1
        m22_new = m22 + 1
        print(f"\n  Term 2: Try m22={m22} -> {m22_new}")
        print(f"    Constraint: m22_new={m22_new} <= m23={m23}? {m22_new <= m23}")
        
        if m22_new <= m23:
            gt_new = (m13, m23, m33, m12, m22_new, m11)
            
            # Find if this state exists
            found = False
            for s in lattice.states:
                if s['gt'] == gt_new and s['p'] == 1 and s['q'] == 1:
                    found = True
                    idx_new = s['index']
                    print(f"    → Found: {gt_new} (index {idx_new})")
            
            if not found:
                print(f"    → State {gt_new} does NOT exist in (1,1)")
            
            # Compute coefficient
            num1 = m22 - m13 - 1
            num2 = m22 - m23
            num3 = m22 - m33 + 1
            num4 = m22 - m11
            den1 = m22 - m12 - 1
            den2 = m22 - m12
            
            print(f"    Numerator factors: ({num1}) * ({num2}) * ({num3}) * ({num4})")
            print(f"    Denominator factors: ({den1}) * ({den2})")
            
            if abs(den1 * den2) > 0:
                numerator = abs(num1) * abs(num2) * num3 * abs(num4)
                denominator = abs(den1 * den2)
                
                if numerator >= 0:
                    coeff = np.sqrt(numerator / denominator)
                    print(f"    Coefficient: {coeff:.6f}")
                else:
                    print(f"    Numerator is negative!")
            else:
                print(f"    Denominator is zero!")
        else:
            print(f"    Constraint violated - no transition")


if __name__ == "__main__":
    detailed_e23_check()
