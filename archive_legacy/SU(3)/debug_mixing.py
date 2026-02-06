"""
Debug: Check if operators mix irreps
"""

import numpy as np
from lattice import SU3Lattice
from operators_v3 import SU3Operators


def check_irrep_mixing():
    """Check if operators connect states from different irreps."""
    print("="*70)
    print("Checking for irrep mixing")
    print("="*70)
    
    lattice = SU3Lattice(max_p=2, max_q=2)
    operators = SU3Operators(lattice)
    
    ops_to_check = {
        'E12': operators.E12,
        'E23': operators.E23,
        'E13': operators.E13,
    }
    
    for op_name, op in ops_to_check.items():
        print(f"\n{op_name}:")
        op_lil = op.tolil()
        
        mixing_count = 0
        total_nonzero = 0
        
        for i in range(lattice.get_dimension()):
            state_i = lattice.states[i]
            p_i, q_i = state_i['p'], state_i['q']
            
            # Check all non-zero elements in row i
            row = op_lil.rows[i]
            for j in row:
                state_j = lattice.states[j]
                p_j, q_j = state_j['p'], state_j['q']
                
                total_nonzero += 1
                
                if (p_i, q_i) != (p_j, q_j):
                    mixing_count += 1
                    if mixing_count <= 5:  # Print first 5 examples
                        print(f"  Mixes ({p_i},{q_i}) -> ({p_j},{q_j})")
        
        print(f"  Total non-zero: {total_nonzero}, Mixing: {mixing_count}")
        
        if mixing_count > 0:
            print(f"  ⚠️ WARNING: {op_name} mixes irreps!")
        else:
            print(f"  ✓ {op_name} respects irrep structure")


if __name__ == "__main__":
    check_irrep_mixing()
