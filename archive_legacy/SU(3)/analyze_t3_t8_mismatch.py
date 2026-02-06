"""
Compare weight-space T3/T8 with what commutators expect
"""

import numpy as np
from operators_v11 import SU3OperatorsV11

np.set_printoptions(precision=6, suppress=True, linewidth=120)

print("Comparing T3/T8 definitions\n")
print("="*70)

ops = SU3OperatorsV11(1, 0)

print("Current T3 (weight-space):")
print(np.diag(ops.T3).real)

print("\nCurrent T8 (weight-space * √3):")
print(np.diag(ops.T8).real)

# What [E23, E32] actually gives
comm_E23E32 = ops.E23 @ ops.E32 - ops.E32 @ ops.E23
print("\n[E23, E32] diagonal:")
print(np.diag(comm_E23E32).real)

# What [E13, E31] actually gives
comm_E13E31 = ops.E13 @ ops.E31 - ops.E31 @ ops.E13
print("\n[E13, E31] diagonal:")
print(np.diag(comm_E13E31).real)

# Expected from Gell-Mann
# T3 diagonal for (1,0) should be [1/2, -1/2, 0] or permutation
# T8 diagonal for (1,0) should be [1/(2√3), 1/(2√3), -1/√3] or permutation

print("\n" + "="*70)
print("Expected from Gell-Mann matrices:")
print("="*70)

# For fundamental (1,0), the weights should correspond to quarks (u, d, s):
# u: (I3, Y) = (1/2, 1/3)   → T3 = 1/2, T8 = 1/(2√3)
# d: (I3, Y) = (-1/2, 1/3)  → T3 = -1/2, T8 = 1/(2√3)  
# s: (I3, Y) = (0, -2/3)    → T3 = 0, T8 = -1/√3

# Since T8 = Y*√3/2:
# u: T8 = (1/3)*√3/2 = √3/6 ≈ 0.288675
# d: T8 = (1/3)*√3/2 = √3/6 ≈ 0.288675
# s: T8 = (-2/3)*√3/2 = -√3/3 ≈ -0.577350

print("Expected T3: [0.5, -0.5, 0] or permutation")
print("Expected T8: [0.288675, 0.288675, -0.57735] or permutation")

# But [E23, E32] = T3 + √3*T8 should give specific values
# T3 + √3*T8 for each state:
# u: 0.5 + √3*(√3/6) = 0.5 + 0.5 = 1
# d: -0.5 + √3*(√3/6) = -0.5 + 0.5 = 0
# s: 0 + √3*(-√3/3) = 0 - 1 = -1

print("\nExpected T3 + √3*T8: [1, 0, -1] or permutation")

# Similarly [E13, E31] = -T3 + √3*T8:
# u: -0.5 + √3*(√3/6) = -0.5 + 0.5 = 0
# d: 0.5 + √3*(√3/6) = 0.5 + 0.5 = 1
# s: 0 + √3*(-√3/3) = 0 - 1 = -1

print("Expected -T3 + √3*T8: [0, 1, -1] or permutation")

# Check if current values can be mapped to expected
print("\n" + "="*70)
print("Mapping Analysis:")
print("="*70)

T3_plus_sqr3_T8 = ops.T3 + np.sqrt(3) * ops.T8
print("\nCurrent T3 + √3*T8 diagonal:")
print(np.diag(T3_plus_sqr3_T8).real)

minus_T3_plus_sqr3_T8 = -ops.T3 + np.sqrt(3) * ops.T8
print("\nCurrent -T3 + √3*T8 diagonal:")
print(np.diag(minus_T3_plus_sqr3_T8).real)

# The problem: weight-space gives different state ordering/values than GT expects
print("\n" + "="*70)
print("Conclusion:")
print("="*70)
print("Weight-space T3/T8 give different values than GT ladder operators expect!")
print("Need to define T3/T8 from algebra closure, not weight space.")
