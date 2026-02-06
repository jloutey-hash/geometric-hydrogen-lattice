"""Compare algebraically-derived T3/T8 with theoretical values."""

import numpy as np
from operators_v5 import SU3OperatorsV5

np.set_printoptions(precision=6, suppress=True, linewidth=100)

print("T3 and T8 Analysis for (1,0) Fundamental")
print("="*70)

ops = SU3OperatorsV5(1, 0)

print("\nStates:")
for i, state in enumerate(ops.states):
    m13, m23, m33, m12, m22, m11 = state
    i3 = m11 - 0.5 * (m12 + m22)
    y = (m12 + m22) - (2.0/3.0) * (m13 + m23 + m33)
    print(f"  {i}: GT={state}, I3={i3:.1f}, Y={y:.3f}")

print("\nAlgebraically-derived T3 (from [E12, E21]/2):")
print(ops.T3)
print(f"Diagonal: {np.diag(ops.T3).real}")

print("\nTheoretical T3 (I3 = m11 - (m12+m22)/2):")
T3_theory = np.zeros((3, 3))
for i, state in enumerate(ops.states):
    m13, m23, m33, m12, m22, m11 = state
    T3_theory[i, i] = m11 - 0.5 * (m12 + m22)
print(T3_theory)
print(f"Diagonal: {np.diag(T3_theory)}")

print("\nAlgebraically-derived T8 (from (T3 + 2*T_U)/√3):")
print(ops.T8)
print(f"Diagonal: {np.diag(ops.T8).real}")

print("\nTheoretical T8 (Y*√3/2 where Y = (m12+m22) - 2(m13+m23+m33)/3):")
T8_theory = np.zeros((3, 3))
for i, state in enumerate(ops.states):
    m13, m23, m33, m12, m22, m11 = state
    y = (m12 + m22) - (2.0/3.0) * (m13 + m23 + m33)
    T8_theory[i, i] = (np.sqrt(3.0) / 2.0) * y
print(T8_theory)
print(f"Diagonal: {np.diag(T8_theory)}")

print("\n" + "="*70)
print("MISMATCH ANALYSIS")
print("="*70)

T3_diff = ops.T3 - T3_theory
T8_diff = ops.T8 - T8_theory

print(f"\nT3 difference:")
print(T3_diff)
print(f"Max error: {np.max(np.abs(T3_diff)):.6f}")

print(f"\nT8 difference:")
print(T8_diff)
print(f"Max error: {np.max(np.abs(T8_diff)):.6f}")

print("\nCONCLUSION:")
print("The algebraically-derived T3 and T8 do NOT match the theoretical")
print("weight space values. This is because the E23 coefficients are wrong,")
print("which causes [E23, E32] to be wrong, which makes T_U wrong, which")
print("propagates to T8 being wrong.")
