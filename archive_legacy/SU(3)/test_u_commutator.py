"""
Compute [E23, E32] commutator for (1,1) irrep only.
"""

import numpy as np
from lattice import SU3Lattice
from operators_v3 import SU3Operators


def test_u_spin_commutator():
    """Test [E23, E32] on (1,1)."""
    print("="*70)
    print("Testing [E23, E32] on (1,1) adjoint")
    print("="*70)
    
    # Build operators
    lattice = SU3Lattice(max_p=1, max_q=1)
    operators = SU3Operators(lattice)
    
    # Extract (1,1) subspace
    indices = [s['index'] for s in lattice.states if s['p'] == 1 and s['q'] == 1]
    dim = len(indices)
    print(f"(1,1) dimension: {dim}")
    
    # Get operators
    E23 = operators.E23.toarray()[np.ix_(indices, indices)]
    E32 = operators.E32.toarray()[np.ix_(indices, indices)]
    T3 = operators.T3.toarray()[np.ix_(indices, indices)]
    T8 = operators.T8.toarray()[np.ix_(indices, indices)]
    
    # Compute commutator
    comm = E23 @ E32 - E32 @ E23
    expected = -1.5 * T3 + (np.sqrt(3)/2) * T8
    
    print("\n[E23, E32] matrix:")
    print(comm)
    
    print("\nExpected: -(3/2)*T3 + (√3/2)*T8:")
    print(expected)
    
    print("\nDifference:")
    print(comm - expected)
    
    error = np.max(np.abs(comm - expected))
    print(f"\nMaximum error: {error:.3e}")
    
    if error < 1e-13:
        print("✓ PASSED!")
    else:
        print("✗ FAILED")
        
        # Try to diagnose
        print("\nDiagonal elements of commutator:")
        for i in range(dim):
            state = lattice.states[indices[i]]
            gt = state['gt']
            print(f"  {i} {gt}: computed={comm[i,i]:.6f}, expected={expected[i,i]:.6f}, diff={comm[i,i]-expected[i,i]:.6f}")


if __name__ == "__main__":
    test_u_spin_commutator()
