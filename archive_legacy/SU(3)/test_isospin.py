"""
Verify [I+, I-] = 2*T3 more carefully.
"""

import numpy as np
from lattice import SU3Lattice
from operators_v3 import SU3Operators


def test_isospin_carefully():
    """Test [E12, E21] = 2*T3 on several irreps."""
    for (p, q) in [(1, 0), (1, 1), (2, 1)]:
        print("="*70)
        print(f"Testing [E12, E21] = 2*T3 on ({p},{q})")
        print("="*70)
        
        # Build operators
        lattice = SU3Lattice(max_p=p, max_q=q)
        operators = SU3Operators(lattice)
        
        # Extract subspace
        indices = [s['index'] for s in lattice.states if s['p'] == p and s['q'] == q]
        dim = len(indices)
        print(f"Dimension: {dim}")
        
        # Get matrices
        E12 = operators.E12.toarray()[np.ix_(indices, indices)]
        E21 = operators.E21.toarray()[np.ix_(indices, indices)]
        T3 = operators.T3.toarray()[np.ix_(indices, indices)]
        
        # Compute commutator
        comm = E12 @ E21 - E21 @ E12
        expected = 2 * T3
        
        error = np.max(np.abs(comm - expected))
        print(f"Error: {error:.3e}")
        
        if error < 1e-13:
            print("✓ PASSED")
        else:
            print("✗ FAILED")
            print("\nCommutator diagonal:")
            print(np.diag(comm))
            print("\nExpected diagonal:")
            print(np.diag(expected))
        
        print()


if __name__ == "__main__":
    test_isospin_carefully()
