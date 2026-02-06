"""
Test different normalization conventions for Cartan operators.
"""

import numpy as np
from operators_v6 import SU3OperatorsV6

print("Testing Cartan Normalization Conventions\n")
print("="*70)

ops = SU3OperatorsV6(1, 0)

# Commutator values
comm_I = ops.E12 @ ops.E21 - ops.E21 @ ops.E12
comm_U = ops.E23 @ ops.E32 - ops.E32 @ ops.E23

print("Commutators:")
print("\n[E12, E21] =")
print(comm_I)
print(f"Diagonal: {np.diag(comm_I).real}")

print("\n[E23, E32] =")
print(comm_U)
print(f"Diagonal: {np.diag(comm_U).real}")

# Try different normalization factors
print("\n" + "="*70)
print("Testing T3 normalizations:")
print("="*70)

# Expected T3 for (1,0)
T3_expected = np.diag([0, -0.5, 0.5])
print(f"\nExpected T3 diagonal: {np.diag(T3_expected)}")

# Convention 1: T3 = (1/2) * [E12, E21]
T3_v1 = 0.5 * comm_I
print(f"\nConvention 1: T3 = (1/2)*[E12, E21]")
print(f"Diagonal: {np.diag(T3_v1).real}")
print(f"Error vs expected: {np.max(np.abs(T3_v1 - T3_expected)):.6f}")

# Convention 2: T3 = (1/4) * [E12, E21]
T3_v2 = 0.25 * comm_I
print(f"\nConvention 2: T3 = (1/4)*[E12, E21]")
print(f"Diagonal: {np.diag(T3_v2).real}")
print(f"Error vs expected: {np.max(np.abs(T3_v2 - T3_expected)):.6f}")

# Now test what the commutator should give
print("\n" + "="*70)
print("Testing what [E23, E32] should equal:")
print("="*70)

# Standard convention: [E23, E32] = -(3/2)*T3 + (√3/2)*T8
T8_expected = np.diag([-1/np.sqrt(3), 0, 1/np.sqrt(3)])

expected_v1 = -(3.0/2.0) * T3_v1 + (np.sqrt(3.0)/2.0) * (1.0/np.sqrt(3.0)) * (T3_v1 + 2.0 * 0.5 * comm_U)
expected_v2 = -(3.0/2.0) * T3_v2 + (np.sqrt(3.0)/2.0) * T8_expected

print(f"\nActual [E23, E32] diagonal: {np.diag(comm_U).real}")
print(f"\nIf T3 = (1/2)*[E12,E21], expected diagonal: {np.diag(expected_v1).real}")
print(f"Error: {np.max(np.abs(comm_U - expected_v1)):.6f}")

print(f"\nIf T3 = (1/4)*[E12,E21], expected = -(3/2)*T3 + (√3/2)*T8:")
print(f"Expected diagonal: {np.diag(expected_v2).real}")
print(f"Error: {np.max(np.abs(comm_U - expected_v2)):.6f}")

# Maybe the issue is E12 needs normalization too?
print("\n" + "="*70)
print("Hypothesis: Maybe E12 also needs √2 correction?")
print("="*70)

E12_scaled = ops.E12 / np.sqrt(2)
E21_scaled = E12_scaled.T.conj()
comm_I_scaled = E12_scaled @ E21_scaled - E21_scaled @ E12_scaled

print(f"\nIf E12 scaled by 1/√2:")
print(f"[E12_scaled, E21_scaled] diagonal: {np.diag(comm_I_scaled).real}")
print(f"T3 = (1/2)*[E12_scaled, E21_scaled] diagonal: {np.diag(0.5*comm_I_scaled).real}")
