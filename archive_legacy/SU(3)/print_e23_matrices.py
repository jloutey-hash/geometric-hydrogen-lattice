"""
Print E23 and E32 matrices for (1,1).
"""

import numpy as np
from lattice import SU3Lattice
from operators_v4 import SU3Operators


def print_matrices():
    """Print E23 and E32 for (1,1)."""
    # Build operators
    lattice = SU3Lattice(max_p=1, max_q=1)
    operators = SU3Operators(lattice)
    
    # Get (1,1) states
    states_11 = [(i, s) for i, s in enumerate(lattice.states) if s['p'] == 1 and s['q'] == 1]
    print(f"(1,1) states (8 total):")
    for i, (idx, s) in enumerate(states_11):
        gt = s['gt']
        print(f"  {i}: index={idx:2d}, GT={gt}")
    
    # Extract indices
    indices = [idx for idx, s in states_11]
    
    # Get matrices
    E23 = operators.E23.toarray()[np.ix_(indices, indices)]
    E32 = operators.E32.toarray()[np.ix_(indices, indices)]
    
    print("\nE23 matrix:")
    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    print(E23)
    
    print("\nE32 matrix:")
    print(E32)
    
    print("\nE32 - E23†:")
    print(E32 - E23.T)
    
    print("\nNon-zero elements of E23:")
    for i in range(8):
        for j in range(8):
            if abs(E23[i, j]) > 1e-10:
                print(f"  E23[{i},{j}] = {E23[i,j]:.6f}")
    
    print("\nNon-zero elements of E32:")
    for i in range(8):
        for j in range(8):
            if abs(E32[i, j]) > 1e-10:
                print(f"  E32[{i},{j}] = {E32[i,j]:.6f}")


if __name__ == "__main__":
    print_matrices()
