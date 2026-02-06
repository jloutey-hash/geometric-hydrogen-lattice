"""
Analyze the commutator problem more carefully.
"""

import numpy as np
from lattice import SU3Lattice
from operators_v3 import SU3Operators


def analyze_commutator():
    """Analyze [E23, E32] problem."""
    print("="*70)
    print("Analyzing [E23, E32] on (1,1)")
    print("="*70)
    
    # Build operators
    lattice = SU3Lattice(max_p=1, max_q=1)
    operators = SU3Operators(lattice)
    
    # Get (1,1) states
    states_11 = [(i, s) for i, s in enumerate(lattice.states) if s['p'] == 1 and s['q'] == 1]
    indices = [idx for idx, s in states_11]
    
    print("\nStates:")
    for i, (idx, s) in enumerate(states_11):
        gt = s['gt']
        m13, m23, m33, m12, m22, m11 = gt
        
        # Compute T3 and T8
        t3 = m11 - 0.5 * (m12 + m22)
        t8 = (np.sqrt(3) / 2) * ((m12 + m22) - (2.0/3) * (m13 + m23 + m33))
        
        # Expected commutator diagonal
        expected_diag = -1.5 * t3 + (np.sqrt(3)/2) * t8
        
        print(f"  {i}: {gt}")
        print(f"      T3 = {t3:.3f}, T8 = {t8:.3f}")
        print(f"      -(3/2)*T3 + (√3/2)*T8 = {expected_diag:.3f}")
    
    # Get matrices
    E23 = operators.E23.toarray()[np.ix_(indices, indices)]
    E32 = operators.E32.toarray()[np.ix_(indices, indices)]
    
    # Compute commutator
    comm = E23 @ E32 - E32 @ E23
    
    print("\n[E23, E32] diagonal:")
    for i in range(len(indices)):
        print(f"  {i}: {comm[i,i]:.3f}")
    
    # Check E23 @ E32
    print("\nE23 @ E32:")
    product1 = E23 @ E32
    for i in range(len(indices)):
        print(f"  Diagonal {i}: {product1[i,i]:.3f}")
    
    # Check E32 @ E23
    print("\nE32 @ E23:")
    product2 = E32 @ E23
    for i in range(len(indices)):
        print(f"  Diagonal {i}: {product2[i,i]:.3f}")


if __name__ == "__main__":
    analyze_commutator()
