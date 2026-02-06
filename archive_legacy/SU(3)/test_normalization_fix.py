"""
Test correcting the normalization: scale E23, E13 by √2 and T8 by √3
"""

import numpy as np
from operators_v8 import SU3OperatorsV8

np.set_printoptions(precision=6, suppress=True)

print("Testing Normalization Correction\n")
print("="*70)

ops = SU3OperatorsV8(1, 0)

# Apply normalization corrections
E12 = ops.E12
E21 = ops.E21
E23 = ops.E23 * np.sqrt(2)  # Scale by √2
E32 = ops.E32 * np.sqrt(2)  # Scale by √2
E13 = ops.E13 * np.sqrt(2)  # Scale by √2
E31 = ops.E31 * np.sqrt(2)  # Scale by √2
T3 = ops.T3
T8 = ops.T8 * np.sqrt(3)  # Scale by √3

print("Corrected operators:")
print(f"E23:\n{E23}")
print(f"\nT8 diagonal: {np.diag(T8).real}")

# Reconstruct λ matrices
lambda1 = E12 + E21
lambda2 = -1j * (E12 - E21)
lambda3 = 2 * T3
lambda4 = E23 + E32
lambda5 = -1j * (E23 - E32)
lambda6 = E13 + E31
lambda7 = -1j * (E13 - E31)
lambda8 = (2/np.sqrt(3)) * T8

lambdas = [lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7, lambda8]

print("\n" + "="*70)
print("Trace Normalization: Tr(λᵢ λⱼ)")
print("="*70)

print("\nDiagonal (should all be 2):")
for i in range(8):
    trace = np.trace(lambdas[i] @ lambdas[i]).real
    print(f"Tr(λ{i+1}²) = {trace:.6f}")

# Casimir with Tᵢ = λᵢ/2
Ts = [l/2 for l in lambdas]

print("\n" + "="*70)
print("Casimir with Tᵢ = λᵢ/2:")
print("="*70)

C2 = sum(T @ T for T in Ts)

eigs = np.diag(C2).real
print(f"Diagonal: {eigs}")
print(f"Mean: {np.mean(eigs):.6f}")
print(f"Std: {np.std(eigs):.6f}")
print(f"Expected: 1.333333")

# Also check Cartan commutator
print("\n" + "="*70)
print("Cartan Commutator:")
print("="*70)

comm_T3T8 = T3 @ T8 - T8 @ T3
print(f"[T3, T8] max element: {np.max(np.abs(comm_T3T8)):.2e}")

# Check ladder commutators
print("\n" + "="*70)
print("Ladder Commutators:")
print("="*70)

# [E23, E32] should = T3 + √3*T8
comm_E23E32 = E23 @ E32 - E32 @ E23
expected = T3 + np.sqrt(3) * T8

diff = comm_E23E32 - expected
print(f"\n[E23, E32]:\n{np.diag(comm_E23E32).real}")
print(f"Expected T3 + √3*T8:\n{np.diag(expected).real}")
print(f"Difference: {np.max(np.abs(diff)):.2e}")

# [E12, E21] should = 2*T3
comm_E12E21 = E12 @ E21 - E21 @ E12
expected12 = 2 * T3

diff12 = comm_E12E21 - expected12
print(f"\n[E12, E21]:\n{np.diag(comm_E12E21).real}")
print(f"Expected 2*T3:\n{np.diag(expected12).real}")
print(f"Difference: {np.max(np.abs(diff12)):.2e}")

# [E13, E31] should = -T3 + √3*T8
comm_E13E31 = E13 @ E31 - E31 @ E13
expected13 = -T3 + np.sqrt(3) * T8

diff13 = comm_E13E31 - expected13
print(f"\n[E13, E31]:\n{np.diag(comm_E13E31).real}")
print(f"Expected -T3 + √3*T8:\n{np.diag(expected13).real}")
print(f"Difference: {np.max(np.abs(diff13)):.2e}")
