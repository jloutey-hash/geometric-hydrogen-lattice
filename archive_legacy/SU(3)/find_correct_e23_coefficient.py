"""
Work backwards: given correct T3, T8, what should E23 be?
"""

import numpy as np

np.set_printoptions(precision=6, suppress=True)

print("Working Backwards to Find Correct E23 Coefficient\n")
print("="*70)

# Known correct values for (1,0) fundamental
T3 = np.diag([0, -0.5, 0.5])
T8 = np.diag([-1/np.sqrt(3), 1/(2*np.sqrt(3)), 1/(2*np.sqrt(3))])

print("Correct T3:")
print(T3)
print(f"Diagonal: {np.diag(T3)}")

print("\nCorrect T8:")
print(T8)
print(f"Diagonal: {np.diag(T8)}")

# What should [E23, E32] be?
comm_target = -(3.0/2.0) * T3 + (np.sqrt(3.0)/2.0) * T8
print("\nTarget [E23, E32] = -(3/2)*T3 + (√3/2)*T8:")
print(comm_target)
print(f"Diagonal: {np.diag(comm_target)}")

# E23 has structure: [[0, 0, 0], [c, 0, 0], [0, 0, 0]] where c is unknown
# E32 = E23† = [[0, c*, 0], [0, 0, 0], [0, 0, 0]]
# [E23, E32] = E23@E32 - E32@E23 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]] @ [[0, c*, 0], [0, 0, 0], [0, 0, 0]]

# Actually compute [E23, E32] for arbitrary coefficient c
# E23 @ E32: (0,1) element = c * c* = |c|^2 (at position [0,0])
# E32 @ E23: (1,0) element = c* * c = |c|^2 (at position [1,1])
# So [E23, E32] = diag(-|c|^2, |c|^2, 0)

print("\n" + "="*70)
print("Analysis:")
print("="*70)
print("\nE23 structure: [[0, 0, 0], [c, 0, 0], [0, 0, 0]]")
print("E32 structure: [[0, c*, 0], [0, 0, 0], [0, 0, 0]]")
print("\n[E23, E32] = E23@E32 - E32@E23 = diag(-|c|^2, |c|^2, 0)")

# So we need: diag(-|c|^2, |c|^2, 0) = target
target_diag = np.diag(comm_target)
print(f"\nTarget diagonal: {target_diag}")

# From first element: -|c|^2 = target_diag[0]
c_squared = -target_diag[0]
c_required = np.sqrt(c_squared)

print(f"\nFrom -|c|^2 = {target_diag[0]}:")
print(f"|c|^2 = {c_squared}")
print(f"|c| = {c_required}")

# Current formula gives
c_formula = 1.0 / np.sqrt(2)  # after √2 correction
print(f"\nCurrent formula (with √2 correction): {c_formula}")

# Ratio
ratio = c_required / c_formula
print(f"Required / Formula ratio: {ratio}")

print(f"\nConclusion: E23 coefficient needs additional factor of {ratio:.6f}")
print(f"Total correction from raw formula: {ratio * np.sqrt(2):.6f}")
