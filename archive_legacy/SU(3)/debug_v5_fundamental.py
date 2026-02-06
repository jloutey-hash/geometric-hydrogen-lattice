"""Debug E23 coefficients for (1,0) representation."""

import numpy as np
from operators_v5 import SU3OperatorsV5

np.set_printoptions(precision=6, suppress=True)

print("E23 Matrix Analysis for (1,0) Fundamental Representation\n")
print("="*70)

ops = SU3OperatorsV5(1, 0)

print(f"\nStates (GT patterns):")
for i, state in enumerate(ops.states):
    print(f"  {i}: {state}")

print(f"\nE23 matrix:")
print(ops.E23)

print(f"\nE32 matrix (E23†):")
print(ops.E32)

print(f"\n[E23, E32] actual:")
comm = ops.E23 @ ops.E32 - ops.E32 @ ops.E23
print(comm)

print(f"\nT3 (from [E12, E21]):")
print(ops.T3)

print(f"\nT8 (from T3 + 2*T_U):")
print(ops.T8)

print(f"\n[E23, E32] expected = -(3/2)*T3 + (√3/2)*T8:")
expected = -(3.0/2.0) * ops.T3 + (np.sqrt(3.0)/2.0) * ops.T8
print(expected)

print(f"\nDifference:")
print(comm - expected)

print(f"\nError: {np.max(np.abs(comm - expected)):.6f}")
