"""
Check Casimir operator for Gell-Mann matrices.
"""

import numpy as np

np.set_printoptions(precision=6, suppress=True)

print("Gell-Mann Matrices Casimir Check\n")
print("="*70)

# Standard Gell-Mann ladder operators for (1,0) fundamental
E12 = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=complex)
E21 = E12.T.conj()

E23 = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=complex)
E32 = E23.T.conj()

E13 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=complex)
E31 = E13.T.conj()

T3 = np.diag([0.5, -0.5, 0])
T8 = np.diag([1/(2*np.sqrt(3)), 1/(2*np.sqrt(3)), -1/np.sqrt(3)])

print("Operators:")
print(f"E12:\n{E12}")
print(f"\nT3 diagonal: {np.diag(T3).real}")
print(f"T8 diagonal: {np.diag(T8).real}")

# Casimir operator - there are different conventions!
print("\n" + "="*70)
print("Casimir Operator - Different Formulas:")
print("="*70)

# Formula 1: Sum of products
C2_v1 = (E12 @ E21 + E21 @ E12 +
         E23 @ E32 + E32 @ E23 +
         E13 @ E31 + E31 @ E13 +
         T3 @ T3 + T8 @ T8)

print("\nFormula 1: E12@E21 + E21@E12 + ... + T3² + T8²")
print(f"Diagonal: {np.diag(C2_v1).real}")
print(f"Mean: {np.mean(np.diag(C2_v1).real):.6f}")
print(f"Std: {np.std(np.diag(C2_v1).real):.6f}")

# Formula 2: Anticommutators {Ea, Ea†}
C2_v2 = (E12 @ E21 + E23 @ E32 + E13 @ E31 +
         E21 @ E12 + E32 @ E23 + E31 @ E13 +
         T3 @ T3 + T8 @ T8)

print("\nFormula 2: Same as Formula 1 (just reordered)")
print(f"Diagonal: {np.diag(C2_v2).real}")

# Formula 3: Using structure constants - sum over all 8 generators
# C2 = sum_a T_a² where T_a are the generators (including Cartan)
# But we need to include all 8 Gell-Mann matrices

# Let's build all 8 generators
lambda1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex)
lambda2 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex)
lambda3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex)
lambda4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex)
lambda5 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex)
lambda6 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex)
lambda7 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex)
lambda8 = (1/np.sqrt(3)) * np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex)

generators = [lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7, lambda8]

# Casimir as sum of generator squares
C2_v3 = sum(g @ g for g in generators)

print("\nFormula 3: Sum of λᵢ²")
print(f"Diagonal: {np.diag(C2_v3).real}")
print(f"Mean: {np.mean(np.diag(C2_v3).real):.6f}")
print(f"Std: {np.std(np.diag(C2_v3).real):.6f}")

# The generators Ta = λa/2 (factor of 1/2 for SU(3))
# So C2 with Ta = (1/4) sum λᵢ²
T = [g/2 for g in generators]
C2_v4 = sum(t @ t for t in T)

print("\nFormula 4: Sum of Tᵢ² where Tᵢ = λᵢ/2")
print(f"Diagonal: {np.diag(C2_v4).real}")
print(f"Mean: {np.mean(np.diag(C2_v4).real):.6f}")
print(f"Std: {np.std(np.diag(C2_v4).real):.6f}")

# Expected value for (1,0)
expected = (1**2 + 0**2 + 3*1 + 3*0 + 1*0) / 3.0
print(f"\nExpected Casimir for (1,0): {expected:.6f}")

# Check individual contributions
print("\n" + "="*70)
print("Detailed Contributions (Gell-Mann):")
print("="*70)

print(f"\nE12@E21 + E21@E12 diagonal: {np.diag(E12 @ E21 + E21 @ E12).real}")
print(f"E23@E32 + E32@E23 diagonal: {np.diag(E23 @ E32 + E32 @ E23).real}")
print(f"E13@E31 + E31@E13 diagonal: {np.diag(E13 @ E31 + E31 @ E13).real}")
print(f"T3² diagonal: {np.diag(T3 @ T3).real}")
print(f"T8² diagonal: {np.diag(T8 @ T8).real}")

# Check if Casimir commutes with all generators
print("\n" + "="*70)
print("Casimir Commutation Check:")
print("="*70)

for i, g in enumerate(generators):
    comm = C2_v3 @ g - g @ C2_v3
    error = np.max(np.abs(comm))
    print(f"[C2, λ{i+1}] error: {error:.3e}")
