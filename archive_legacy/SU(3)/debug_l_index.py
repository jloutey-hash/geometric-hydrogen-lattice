"""
Debug l-index formula for E23.
"""

import numpy as np
from lattice import SU3Lattice


def test_l_index_formula():
    """Test l-index formula on specific examples."""
    print("="*70)
    print("Testing l-index formulas")
    print("="*70)
    
    lattice = SU3Lattice(max_p=1, max_q=1)
    
    # Get (1,1) states
    states_11 = [s for s in lattice.states if s['p'] == 1 and s['q'] == 1]
    
    print(f"\n(1,1) has {len(states_11)} states")
    
    # Test on SPECIFIC state that's missing: (2,1,0,2,0,1)
    for state in states_11:
        gt = state['gt']
        if gt != (2, 1, 0, 2, 0, 1):
            continue
            
        m13, m23, m33, m12, m22, m11 = gt
        
        # Convert to l-indices
        l13 = m13 + 2
        l23 = m23 + 1
        l33 = m33
        l12 = m12 + 1
        l22 = m22
        l11 = m11
        
        print(f"\nState: {gt}")
        print(f"  m-indices: m13={m13}, m23={m23}, m33={m33}, m12={m12}, m22={m22}, m11={m11}")
        print(f"  l-indices: l13={l13}, l23={l23}, l33={l33}, l12={l12}, l22={l22}, l11={l11}")
        
        # Term 1: Shift m12 -> m12 + 1
        m12_new = m12 + 1
        if m12_new <= m13:
            print(f"\n  Term 1: m12={m12} -> {m12_new} (l12={l12} -> {l12+1})")
            
            # v4 formula: sqrt(|(l13-l12-1)(l23-l12-1)(l33-l12-1)(l11-l12-1) / (l22-l12-1)(l22-l12)|)
            f1 = l13 - l12 - 1
            f2 = l23 - l12 - 1
            f3 = l33 - l12 - 1
            f4 = l11 - l12 - 1
            d1 = l22 - l12 - 1
            d2 = l22 - l12
            
            print(f"    Numerator: ({f1}) * ({f2}) * ({f3}) * ({f4}) = {f1*f2*f3*f4}")
            print(f"    Denominator: ({d1}) * ({d2}) = {d1*d2}")
            
            if abs(d1 * d2) > 0:
                ratio = (f1*f2*f3*f4) / (d1*d2)
                coeff = np.sqrt(abs(ratio))
                print(f"    Coefficient: {coeff:.6f}")
            else:
                print(f"    Denominator is ZERO! Skipping this term.")
        
        # Term 2: Shift m22 -> m22 + 1
        m22_new = m22 + 1
        print(f"\n  Term 2: m22={m22} -> {m22_new} (l22={l22} -> {l22+1})")
        print(f"  Constraint: m22_new={m22_new} <= m23={m23}? {m22_new <= m23}")
        
        if m22_new <= m23:
            # v4 formula: sqrt(|(l13-l22-1)(l23-l22-1)(l33-l22-1)(l11-l22-1) / (l12-l22-1)(l12-l22)|)
            f1 = l13 - l22 - 1
            f2 = l23 - l22 - 1
            f3 = l33 - l22 - 1
            f4 = l11 - l22 - 1
            d1 = l12 - l22 - 1
            d2 = l12 - l22
            
            print(f"    Numerator: ({f1}) * ({f2}) * ({f3}) * ({f4}) = {f1*f2*f3*f4}")
            print(f"    Denominator: ({d1}) * ({d2}) = {d1*d2}")
            
            if abs(d1 * d2) > 0:
                ratio = (f1*f2*f3*f4) / (d1*d2)
                coeff = np.sqrt(abs(ratio))
                print(f"    Coefficient: {coeff:.6f}")
            else:
                print(f"    Denominator is ZERO! Skipping this term.")
        else:
            print(f"    Constraint violated - no transition")


if __name__ == "__main__":
    test_l_index_formula()
