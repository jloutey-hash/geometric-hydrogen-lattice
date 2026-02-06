"""Test if E23 needs a √2 normalization factor."""

import numpy as np

print("Testing E23 Normalization for (1,0) Fundamental")
print("="*70)

# Manual construction with and without √2 factor
E23_formula = np.array([[0, 0, 0],
                        [1, 0, 0],  # Coefficient from formula
                        [0, 0, 0]], dtype=complex)

E23_corrected = E23_formula / np.sqrt(2)

E32_formula = E23_formula.T.conj()
E32_corrected = E23_corrected.T.conj()

print("\nE23 from formula:")
print(E23_formula)

print("\nE23 with 1/√2 factor:")
print(E23_corrected)

print("\n[E23, E32] from formula:")
comm_formula = E23_formula @ E32_formula - E32_formula @ E23_formula
print(comm_formula)

print("\n[E23, E32] with 1/√2 factor:")
comm_corrected = E23_corrected @ E32_corrected - E32_corrected @ E23_corrected
print(comm_corrected)

# What should it be?
print("\nExpected [E23, E32] = -(3/2)*T3 + (√3/2)*T8")
print("For (1,0): T3 diagonal is (0, -1, 1), T8 diagonal is (-1/√3, 0, 1/√3)")

T3 = np.diag([0, -1, 1])
T8 = np.diag([-1/np.sqrt(3), 0, 1/np.sqrt(3)])
expected = -(3.0/2.0) * T3 + (np.sqrt(3.0)/2.0) * T8

print("\nExpected:")
print(expected)

print("\nDifference (formula):")
print(comm_formula - expected)
print(f"Error: {np.max(np.abs(comm_formula - expected)):.6f}")

print("\nDifference (corrected with 1/√2):")
print(comm_corrected - expected)
print(f"Error: {np.max(np.abs(comm_corrected - expected)):.6f}")
