"""Compare v9 operators with Gell-Mann matrices."""

import numpy as np
from operators_v9 import SU3OperatorsV9

np.set_printoptions(precision=6, suppress=True)

print("Comparing v9 with Gell-Mann Matrices\n")
print("="*70)

ops = SU3OperatorsV9(1, 0)

# Gell-Mann operators
E12_GM = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=complex)
E23_GM = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=complex)
E13_GM = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=complex)
T3_GM = np.diag([0.5, -0.5, 0])
T8_GM = np.diag([1/(2*np.sqrt(3)), 1/(2*np.sqrt(3)), -1/np.sqrt(3)])

print("\nv9 operators:")
print(f"E12:\n{ops.E12}")
print(f"\nE23:\n{ops.E23}")
print(f"\nT3 diagonal: {np.diag(ops.T3).real}")
print(f"T8 diagonal: {np.diag(ops.T8).real}")

print("\nGell-Mann operators:")
print(f"E12:\n{E12_GM}")
print(f"\nE23:\n{E23_GM}")
print(f"\nT3 diagonal: {np.diag(T3_GM).real}")
print(f"T8 diagonal: {np.diag(T8_GM).real}")

print("\n" + "="*70)
print("Coefficient Comparison:")
print("="*70)

print(f"\nE12[2,1] - v9: {ops.E12[2,1]:.6f}, Gell-Mann: {E12_GM[2,1]:.6f}")
print(f"E23[1,0] - v9: {ops.E23[1,0]:.6f}, Gell-Mann: {E23_GM[1,0]:.6f}")
print(f"E13[2,1] - v9: {ops.E13[2,1]:.6f}, Gell-Mann: {E13_GM[2,1]:.6f}")

print(f"\nRatio E12 (v9/GM): {ops.E12[2,1] / E12_GM[2,1]:.6f}")
print(f"Ratio E23 (v9/GM): {ops.E23[1,0] / E23_GM[1,0]:.6f}")

print("\n" + "="*70)
print("Commutator Comparison:")
print("="*70)

# v9 commutators
comm_E23_v9 = ops.E23 @ ops.E32 - ops.E32 @ ops.E23
comm_E13_v9 = ops.E13 @ ops.E31 - ops.E31 @ ops.E13

# Gell-Mann commutators
E32_GM = E23_GM.T.conj()
E31_GM = E13_GM.T.conj()
comm_E23_GM = E23_GM @ E32_GM - E32_GM @ E23_GM
comm_E13_GM = E13_GM @ E31_GM - E31_GM @ E13_GM

print(f"\n[E23, E32]:")
print(f"v9 diagonal: {np.diag(comm_E23_v9).real}")
print(f"GM diagonal: {np.diag(comm_E23_GM).real}")

print(f"\n[E13, E31]:")
print(f"v9 diagonal: {np.diag(comm_E13_v9).real}")
print(f"GM diagonal: {np.diag(comm_E13_GM).real}")

# What does the formula give?
formula_E23 = ops.T3 + np.sqrt(3.0) * ops.T8
formula_E13 = -ops.T3 + np.sqrt(3.0) * ops.T8

print(f"\nFormula T3 + √3*T8 diagonal: {np.diag(formula_E23).real}")
print(f"Formula -T3 + √3*T8 diagonal: {np.diag(formula_E13).real}")

# Try doubling
formula_E23_doubled = 2*ops.T3 + 2*np.sqrt(3.0) * ops.T8
formula_E13_doubled = -2*ops.T3 + 2*np.sqrt(3.0) * ops.T8

print(f"\nFormula 2*T3 + 2√3*T8 diagonal: {np.diag(formula_E23_doubled).real}")
print(f"Formula -2*T3 + 2√3*T8 diagonal: {np.diag(formula_E13_doubled).real}")

print(f"\nGM [E23, E32] diagonal: {np.diag(comm_E23_GM).real}")
print(f"GM [E13, E31] diagonal: {np.diag(comm_E13_GM).real}")
