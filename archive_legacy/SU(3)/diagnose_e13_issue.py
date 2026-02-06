"""
Diagnose the E13 commutator issue
"""

import numpy as np
from operators_v12 import SU3OperatorsV12

np.set_printoptions(precision=6, suppress=True)

ops = SU3OperatorsV12(1, 0)

print("Diagnosing [E13, E31] commutator\n")
print("="*70)

# What we actually get
comm_13_31 = ops.E13 @ ops.E31 - ops.E31 @ ops.E13
print("Actual [E13, E31] diagonal:")
print(np.diag(comm_13_31).real)

# What we expect from Gell-Mann
expected = -ops.T3 + np.sqrt(3) * ops.T8
print("\nExpected -T3 + √3*T8 diagonal:")
print(np.diag(expected).real)

# The difference
diff = comm_13_31 - expected
print("\nDifference:")
print(np.diag(diff).real)

# Try to infer what T8' should be to satisfy this
# [E13, E31] = -T3 + √3*T8'
# So: T8' = ([E13, E31] + T3) / √3

T8_from_E13 = (comm_13_31 + ops.T3) / np.sqrt(3)
print("\nT8 inferred from [E13, E31]:")
print(np.diag(T8_from_E13).real)

print("\nCurrent T8:")
print(np.diag(ops.T8).real)

# Check T8 from E23
comm_23_32 = ops.E23 @ ops.E32 - ops.E32 @ ops.E23
T8_from_E23 = (comm_23_32 - ops.T3) / np.sqrt(3)
print("\nT8 inferred from [E23, E32]:")
print(np.diag(T8_from_E23).real)

# These should be equal!
print("\n" + "="*70)
print("Consistency Check:")
print("="*70)

diff_T8 = T8_from_E13 - T8_from_E23
print(f"T8(E13) - T8(E23) max: {np.max(np.abs(diff_T8)):.2e}")

if np.max(np.abs(diff_T8)) > 1e-10:
    print("\n⚠ WARNING: T8 from E13 and E23 are INCONSISTENT!")
    print("This means E13 = [E12, E23] doesn't satisfy the algebra.")
    print("\nThe GT formulas for E12, E23 may be incompatible with")
    print("deriving E13 via commutator while maintaining SU(3) structure.")
