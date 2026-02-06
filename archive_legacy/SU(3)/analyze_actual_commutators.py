"""
Determine the ACTUAL commutation relations for SU(3) generators.
"""

import numpy as np

np.set_printoptions(precision=6, suppress=True, linewidth=100)

print("Determining Actual SU(3) Commutation Relations\n")
print("="*70)

# Standard Gell-Mann ladder operators
E12 = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=complex)
E21 = E12.T.conj()

E23 = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=complex)
E32 = E23.T.conj()

E13 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=complex)
E31 = E13.T.conj()

T3 = np.array([[0.5, 0, 0], [0, -0.5, 0], [0, 0, 0]], dtype=complex)
T8 = (1/np.sqrt(3)) * np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, -1]], dtype=complex)

# Hmm wait, T8 normalization might be off. Let me recalculate
# For SU(3), hypercharge Y is related to T8 by T8 = Y*√3/2
# For (1,0) fundamental: u has Y=1/3, d has Y=1/3, s has Y=-2/3
# So T8 should be diag(√3/6, √3/6, -√3/3) = diag(0.2887, 0.2887, -0.5774)

T8_correct = np.diag([1/(2*np.sqrt(3)), 1/(2*np.sqrt(3)), -1/np.sqrt(3)])

print("Operators:")
print(f"E12: {E12[0]}")
print(f"E23: {E23[0]}")
print(f"E13: {E13[1]}")
print(f"T3 diagonal: {np.diag(T3).real}")
print(f"T8 diagonal: {np.diag(T8_correct).real}")

# Compute all commutators
commutators = {}

commutators['[E12, E21]'] = E12 @ E21 - E21 @ E12
commutators['[E23, E32]'] = E23 @ E32 - E32 @ E23
commutators['[E13, E31]'] = E13 @ E31 - E31 @ E13
commutators['[E12, E23]'] = E12 @ E23 - E23 @ E12
commutators['[E23, E31]'] = E23 @ E31 - E31 @ E23
commutators['[E31, E12]'] = E31 @ E12 - E12 @ E31
commutators['[T3, E12]'] = T3 @ E12 - E12 @ T3
commutators['[T3, E23]'] = T3 @ E23 - E23 @ T3
commutators['[T3, E13]'] = T3 @ E13 - E13 @ T3
commutators['[T8, E12]'] = T8_correct @ E12 - E12 @ T8_correct
commutators['[T8, E23]'] = T8_correct @ E23 - E23 @ T8_correct
commutators['[T8, E13]'] = T8_correct @ E13 - E13 @ T8_correct

print("\n" + "="*70)
print("Commutators:")
print("="*70)

# Check ladder commutators
for name in ['[E12, E21]', '[E23, E32]', '[E13, E31]']:
    comm = commutators[name]
    print(f"\n{name}:")
    print(comm)
    print(f"Diagonal: {np.diag(comm).real}")
    
    # Try to express as linear combination of T3 and T8
    # comm = a*T3 + b*T8
    # We have 3 equations (diagonal elements), 2 unknowns
    # Use first two: comm[0,0] = a*T3[0,0] + b*T8[0,0]
    #                comm[1,1] = a*T3[1,1] + b*T8[1,1]
    
    c0 = comm[0,0].real
    c1 = comm[1,1].real
    t3_0 = T3[0,0].real
    t3_1 = T3[1,1].real
    t8_0 = T8_correct[0,0].real
    t8_1 = T8_correct[1,1].real
    
    # Solve: c0 = a*t3_0 + b*t8_0
    #        c1 = a*t3_1 + b*t8_1
    det = t3_0*t8_1 - t3_1*t8_0
    if abs(det) > 1e-10:
        a = (c0*t8_1 - c1*t8_0) / det
        b = (t3_0*c1 - t3_1*c0) / det
        print(f"Best fit: {a:.3f}*T3 + {b:.3f}*T8")
        
        fitted = a*T3 + b*T8_correct
        error = np.max(np.abs(comm - fitted))
        print(f"Fit error: {error:.3e}")

# Check cross commutators
print("\n" + "="*70)
print("Cross Commutators:")
print("="*70)

for name in ['[E12, E23]', '[E23, E31]', '[E31, E12]']:
    comm = commutators[name]
    print(f"\n{name}:")
    print(comm)

# Cartan commutators
print("\n" + "="*70)
print("Cartan Action on Ladders:")
print("="*70)

for name in ['[T3, E12]', '[T3, E23]', '[T3, E13]']:
    comm = commutators[name]
    print(f"\n{name}:")
    print(comm)
