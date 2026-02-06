"""Test adjoint representation."""

import numpy as np
from weight_basis_gellmann import WeightBasisSU3

wb = WeightBasisSU3(1, 1)

print("Adjoint (1,1) representation")
print(f"Dimension: {wb.dim}")

# Check if T3 and T8 are diagonal
T3_diag_error = np.max(np.abs(wb.T3 - np.diag(np.diag(wb.T3))))
T8_diag_error = np.max(np.abs(wb.T8 - np.diag(np.diag(wb.T8))))

print(f"\nT3 diagonality error: {T3_diag_error}")
print(f"T8 diagonality error: {T8_diag_error}")

print(f"\nT3 matrix:\n{wb.T3.real}")
print(f"\nT8 matrix:\n{wb.T8.real}")

# Check commutators
comm_T3T8 = wb.T3 @ wb.T8 - wb.T8 @ wb.T3
print(f"\n[T3,T8] error: {np.max(np.abs(comm_T3T8))}")

comm_E12E21 = wb.E12 @ wb.E21 - wb.E21 @ wb.E12
expected = 2 * wb.T3
error = np.max(np.abs(comm_E12E21 - expected))
print(f"[E12,E21] - 2T3 error: {error}")

comm_E23E32 = wb.E23 @ wb.E32 - wb.E32 @ wb.E23
expected = wb.T3 + np.sqrt(3) * wb.T8
error = np.max(np.abs(comm_E23E32 - expected))
print(f"[E23,E32] - (T3+√3*T8) error: {error}")

comm_E13E31 = wb.E13 @ wb.E31 - wb.E31 @ wb.E13
expected = -wb.T3 + np.sqrt(3) * wb.T8
error = np.max(np.abs(comm_E13E31 - expected))
print(f"[E13,E31] - (-T3+√3*T8) error: {error}")

# Check Casimir
C2 = wb.get_casimir()
eigenvalues = np.linalg.eigvalsh(C2)
print(f"\nCasimir eigenvalues: {eigenvalues}")
print(f"Expected for (1,1): 3.0")
print(f"Casimir mean: {np.mean(eigenvalues):.6f}")
print(f"Casimir std: {np.std(eigenvalues):.2e}")
