"""
Test l-index formulas on (1,0) fundamental representation.
"""

import numpy as np
from lattice import SU3Lattice
from operators_v3 import SU3Operators as SU3Operators_v3
from operators_v4 import SU3Operators as SU3Operators_v4


def test_fundamental():
    """Test on (1,0) which should be simpler."""
    print("="*70)
    print("Testing (1,0) fundamental representation")
    print("="*70)
    
    lattice = SU3Lattice(max_p=1, max_q=0)
    states_10 = [s for s in lattice.states if s['p'] == 1 and s['q'] == 0]
    
    print(f"\n(1,0) has {len(states_10)} states (theory: 3)")
    for i, s in enumerate(states_10):
        print(f"  {i}: {s['gt']}")
    
    # Build v3 and v4 operators
    ops_v3 = SU3Operators_v3(lattice)
    ops_v4 = SU3Operators_v4(lattice)
    
    # Extract (1,0) subspace
    indices = [s['index'] for s in states_10]
    
    E23_v3 = ops_v3.E23.toarray()[np.ix_(indices, indices)]
    E23_v4 = ops_v4.E23.toarray()[np.ix_(indices, indices)]
    
    print("\nE23 from v3 (m-index formula):")
    print(E23_v3)
    
    print("\nE23 from v4 (l-index formula):")
    print(E23_v4)
    
    print("\nDifference:")
    print(E23_v4 - E23_v3)
    
    # Test commutator
    E32_v4 = ops_v4.E32.toarray()[np.ix_(indices, indices)]
    T3_v4 = ops_v4.T3.toarray()[np.ix_(indices, indices)]
    T8_v4 = ops_v4.T8.toarray()[np.ix_(indices, indices)]
    
    comm = E23_v4 @ E32_v4 - E32_v4 @ E23_v4
    expected = -1.5 * T3_v4 + (np.sqrt(3)/2) * T8_v4
    
    error = np.max(np.abs(comm - expected))
    print(f"\n[E23,E32] error on (1,0): {error:.3e}")


if __name__ == "__main__":
    test_fundamental()
