"""
Check standard Gell-Mann matrices for (1,0) fundamental representation.
These are the 3x3 matrices acting on (u, d, s) quarks.
"""

import numpy as np

np.set_printoptions(precision=6, suppress=True, linewidth=100)

print("Standard Gell-Mann Matrices for SU(3) Fundamental (1,0)\n")
print("="*70)

# Standard Gell-Mann matrices (λ1 through λ8)
# These satisfy Tr(λa λb) = 2δab

# λ1 (I+  + I-)
lambda1 = np.array([[0, 1, 0], 
                    [1, 0, 0], 
                    [0, 0, 0]], dtype=complex)

# λ2 (i(I- - I+))
lambda2 = np.array([[0, -1j, 0], 
                    [1j, 0, 0], 
                    [0, 0, 0]], dtype=complex)

# λ3 (2*T3 in standard normalization)
lambda3 = np.array([[1, 0, 0], 
                    [0, -1, 0], 
                    [0, 0, 0]], dtype=complex)

# λ4 (U+ + U-)
lambda4 = np.array([[0, 0, 1], 
                    [0, 0, 0], 
                    [1, 0, 0]], dtype=complex)

# λ5 (i(U- - U+))
lambda5 = np.array([[0, 0, -1j], 
                    [0, 0, 0], 
                    [1j, 0, 0]], dtype=complex)

# λ6 (V+ + V-)
lambda6 = np.array([[0, 0, 0], 
                    [0, 0, 1], 
                    [0, 1, 0]], dtype=complex)

# λ7 (i(V- - V+))
lambda7 = np.array([[0, 0, 0], 
                    [0, 0, -1j], 
                    [0, 1j, 0]], dtype=complex)

# λ8 (2*T8 / √3)
lambda8 = (1/np.sqrt(3)) * np.array([[1, 0, 0], 
                                      [0, 1, 0], 
                                      [0, 0, -2]], dtype=complex)

# Compute ladder operators from Gell-Mann matrices
# E12 = (λ1 + i*λ2) / 2
# E23 = (λ4 + i*λ5) / 2  
# E13 = (λ6 + i*λ7) / 2

E12_GM = 0.5 * (lambda1 + 1j * lambda2)
E21_GM = E12_GM.T.conj()

E23_GM = 0.5 * (lambda4 + 1j * lambda5)
E32_GM = E23_GM.T.conj()

E13_GM = 0.5 * (lambda6 + 1j * lambda7)
E31_GM = E13_GM.T.conj()

T3_GM = 0.5 * lambda3
T8_GM = (np.sqrt(3) / 2) * lambda8

print("Gell-Mann ladder operators:")
print("\nE12 (I+):")
print(E12_GM)

print("\nE23 (U+):")
print(E23_GM)

print("\nE13 (V+):")
print(E13_GM)

print("\nT3:")
print(T3_GM)
print(f"Diagonal: {np.diag(T3_GM).real}")

print("\nT8:")
print(T8_GM)
print(f"Diagonal: {np.diag(T8_GM).real}")

# Now check commutators
print("\n" + "="*70)
print("Commutator Tests:")
print("="*70)

# [E12, E21]
comm = E12_GM @ E21_GM - E21_GM @ E12_GM
print("\n[E12, E21]:")
print(comm)
print(f"Diagonal: {np.diag(comm).real}")
print(f"2*T3 diagonal: {np.diag(2*T3_GM).real}")
print(f"Match: {np.allclose(comm, 2*T3_GM)}")

# [E23, E32]
comm = E23_GM @ E32_GM - E32_GM @ E23_GM
print("\n[E23, E32]:")
print(comm)
print(f"Diagonal: {np.diag(comm).real}")

# Test different formulas
test1 = -(3.0/2.0) * T3_GM + (np.sqrt(3.0)/2.0) * T8_GM
print(f"\n-(3/2)*T3 + (√3/2)*T8 diagonal: {np.diag(test1).real}")
print(f"Match: {np.allclose(comm, test1)}")

# Maybe it's different for fundamental?
# In the fundamental, the U-spin generators act on different quantum numbers
# Let's just see what the diagonal IS
print(f"\nActual [E23, E32] diagonal from Gell-Mann: {np.diag(comm).real}")

# Check [E13, E31]
comm = E13_GM @ E31_GM - E31_GM @ E13_GM
print(f"\n[E13, E31] diagonal: {np.diag(comm).real}")

test2 = (3.0/2.0) * T3_GM + (np.sqrt(3.0)/2.0) * T8_GM
print(f"(3/2)*T3 + (√3/2)*T8 diagonal: {np.diag(test2).real}")
print(f"Match: {np.allclose(comm, test2)}")

# Check total structure constraint
print("\n" + "="*70)
print("Jacobi Identity Check:")
print("="*70)

#[E12, [E23, E31]] + [E23, [E31, E12]] + [E31, [E12, E23]] = 0
comm1 = E12_GM @ (E23_GM @ E31_GM - E31_GM @ E23_GM) - (E23_GM @ E31_GM - E31_GM @ E23_GM) @ E12_GM
comm2 = E23_GM @ (E31_GM @ E12_GM - E12_GM @ E31_GM) - (E31_GM @ E12_GM - E12_GM @ E31_GM) @ E23_GM
comm3 = E31_GM @ (E12_GM @ E23_GM - E23_GM @ E12_GM) - (E12_GM @ E23_GM - E23_GM @ E12_GM) @ E31_GM

jacobi = comm1 + comm2 + comm3
print(f"Jacobi identity error: {np.max(np.abs(jacobi)):.3e}")
