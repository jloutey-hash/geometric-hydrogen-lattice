"""
Test script to debug a single SU(3) representation
"""

import numpy as np
from scipy.sparse import lil_matrix
from lattice import SU3Lattice

# Test with just the (2, 1) representation
p, q = 2, 1
lattice = SU3Lattice(max_p=p, max_q=q)

# Filter to only (2,1) states
states_21 = [s for s in lattice.states if s['p'] == p and s['q'] == q]

print(f"(2,1) representation has {len(states_21)} states")
print(f"Theory predicts: {(p+1)*(q+1)*(p+q+2)//2} states")

print("\nStates:")
for s in states_21:
    print(f"  Index {s['index']}: I3={s['i3']:.2f}, Y={s['y']:.2f}, mult={s['multiplicity']}")

# Check uniqueness
coords = [(s['i3'], s['y']) for s in states_21]
print(f"\nUnique (I3, Y) pairs: {len(set(coords))}")
print(f"Total states: {len(states_21)}")

# Build test operators for just this representation
dim = len(states_21)
index_map = {s['index']: i for i, s in enumerate(states_21)}

T3 = lil_matrix((dim, dim), dtype=complex)
for i, state in enumerate(states_21):
    T3[i, i] = state['i3']

# Check if [I+, I-] = 2*T3 structure equation holds
I_plus = lil_matrix((dim, dim), dtype=complex)
for i, state in enumerate(states_21):
    i3, y = state['i3'], state['y']
    # Find target state
    target = None
    for j, s2 in enumerate(states_21):
        if abs(s2['i3'] - (i3 + 1)) < 0.01 and abs(s2['y'] - y) < 0.01:
            if s2['multiplicity'] == state['multiplicity']:
                target = j
                break
    
    if target is not None:
        # Compute coefficient using Dynkin labels
        m2 = round(2 * ((p + q) / 3.0 - y) / np.sqrt(3))
        m1 = round((p - q) / 2.0 - i3 + m2 / 2.0)
        m1 = max(0, min(p, int(m1)))
        m2 = max(0, min(q, int(m2)))
        
        coeff = np.sqrt((p - m1) * (m1 + 1))
        I_plus[target, i] = coeff / np.sqrt(2)
        print(f"I+ connects state {i} (I3={i3:.1f}) to {target} with coeff {coeff/np.sqrt(2):.3f}")

I_minus = I_plus.T.conj()

# Test commutator
comm = I_plus @ I_minus - I_minus @ I_plus
expected = 2 * T3

diff = (comm - expected).toarray()
print(f"\n[I+, I-] - 2*T3 max error: {np.max(np.abs(diff)):.2e}")
print("\nCommutator matrix:")
print(comm.toarray())
print("\nExpected (2*T3):")
print(expected.toarray())
