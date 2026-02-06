"""Detailed analysis of [E23, E32] for v7."""

import numpy as np
from operators_v7 import SU3OperatorsV7

np.set_printoptions(precision=6, suppress=True)

print("Detailed [E23, E32] Analysis for v7\n")
print("="*70)

ops = SU3OperatorsV7(1, 0)

print("\nE23:")
print(ops.E23)

print("\nE32:")
print(ops.E32)

print("\n[E23, E32] = E23@E32 - E32@E23:")
comm = ops.E23 @ ops.E32 - ops.E32 @ ops.E23
print(comm)
print(f"Diagonal: {np.diag(comm).real}")

print("\nT3:")
print(ops.T3)
print(f"Diagonal: {np.diag(ops.T3).real}")

print("\nT8:")
print(ops.T8)
print(f"Diagonal: {np.diag(ops.T8).real}")

print("\nExpected [E23, E32] = -(3/2)*T3 + (√3/2)*T8:")
expected = -(3.0/2.0) * ops.T3 + (np.sqrt(3.0)/2.0) * ops.T8
print(expected)
print(f"Diagonal: {np.diag(expected).real}")

print("\nDifference:")
diff = comm - expected
print(diff)
print(f"Diagonal: {np.diag(diff).real}")

# What should T8 be to make this work?
print("\n" + "="*70)
print("Working backwards: what T8 would make [E23, E32] correct?")
print("="*70)

# [E23, E32] = -(3/2)*T3 + (√3/2)*T8
# => T8 = (2/√3) * ([E23, E32] + (3/2)*T3)

T8_required = (2.0/np.sqrt(3.0)) * (comm + (3.0/2.0) * ops.T3)
print("\nRequired T8:")
print(T8_required)
print(f"Diagonal: {np.diag(T8_required).real}")

# Compare with theoretical T8 for (1,0)
print("\nTheoretical T8 for (1,0) (from weight space):")
T8_theory = np.diag([-1/np.sqrt(3), 0, 1/np.sqrt(3)])
print(T8_theory)
print(f"Diagonal: {np.diag(T8_theory)}")

print(f"\nT8 we're getting from algebra: {np.diag(ops.T8).real}")
print(f"T8 we need: {np.diag(T8_required).real}")
print(f"T8 from theory: {np.diag(T8_theory)}")

# Check T_U
T_U = 0.5 * comm
print(f"\nT_U = (1/2)*[E23, E32]:")
print(T_U)
print(f"Diagonal: {np.diag(T_U).real}")

print(f"\nT8 formula: (1/√3)*(T3 + 2*T_U)")
T8_from_formula = (1.0/np.sqrt(3.0)) * (ops.T3 + 2.0 * T_U)
print(f"Diagonal: {np.diag(T8_from_formula).real}")
