"""
Test hermiticity on a single irrep.
"""

import numpy as np
from lattice import SU3Lattice
from operators_v3 import SU3Operators


def test_single_irrep(p, q):
    """Test operators on a single irrep."""
    print("="*70)
    print(f"Testing irrep ({p},{q})")
    print("="*70)
    
    # Build full lattice
    lattice = SU3Lattice(max_p=p, max_q=q)
    operators = SU3Operators(lattice)
    
    # Extract states for this irrep
    indices = [s['index'] for s in lattice.states if s['p'] == p and s['q'] == q]
    dim = len(indices)
    print(f"Dimension: {dim}")
    
    # Extract submatrices
    E12_full = operators.E12.toarray()
    E21_full = operators.E21.toarray()
    E23_full = operators.E23.toarray()
    E32_full = operators.E32.toarray()
    
    E12 = E12_full[np.ix_(indices, indices)]
    E21 = E21_full[np.ix_(indices, indices)]
    E23 = E23_full[np.ix_(indices, indices)]
    E32 = E32_full[np.ix_(indices, indices)]
    
    # Check hermiticity
    print("\nHermiticity checks:")
    print(f"E12 vs E12†: {np.max(np.abs(E12 - E12.T)):.3e}")
    print(f"E21 vs E21†: {np.max(np.abs(E21 - E21.T)):.3e}")
    print(f"E23 vs E23†: {np.max(np.abs(E23 - E23.T)):.3e}")
    print(f"E32 vs E32†: {np.max(np.abs(E32 - E32.T)):.3e}")
    
    # Check adjoint relationship
    print("\nAdjoint relationship checks:")
    print(f"E21 vs E12†: {np.max(np.abs(E21 - E12.T)):.3e}")
    print(f"E32 vs E23†: {np.max(np.abs(E32 - E23.T)):.3e}")
    
    # Print matrices for small cases
    if dim <= 8:
        print("\nE12 matrix:")
        print(E12)
        print("\nE12† matrix:")
        print(E12.T)
        print("\nE21 matrix:")
        print(E21)
        
        print("\nE23 matrix:")
        print(E23)
        print("\nE23† matrix:")
        print(E23.T)
        print("\nE32 matrix:")
        print(E32)


if __name__ == "__main__":
    # Test (1,1) adjoint
    test_single_irrep(1, 1)
    print("\n")
    # Test (2,1)
    test_single_irrep(2, 1)
