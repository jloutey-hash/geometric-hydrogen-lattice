"""
Check if v8 satisfies the trace normalization Tr(T_a T_b) = (1/2)δ_ab
"""

import numpy as np
from operators_v8 import SU3OperatorsV8

np.set_printoptions(precision=6, suppress=True)

print("v8 Operator Normalization Check\n")
print("="*70)

ops = SU3OperatorsV8(1, 0)

print("\nOperators:")
print(f"E12:\n{ops.E12}")
print(f"\nE23:\n{ops.E23}")
print(f"\nT3 diagonal: {np.diag(ops.T3).real}")
print(f"T8 diagonal: {np.diag(ops.T8).real}")

# Reconstruct λ matrices from ladder operators
# E12 = (λ1 + iλ2)/2, so λ1 = E12 + E21, λ2 = -i(E12 - E21)
lambda1_v8 = ops.E12 + ops.E21
lambda2_v8 = -1j * (ops.E12 - ops.E21)
lambda3_v8 = 2 * ops.T3
lambda4_v8 = ops.E23 + ops.E32
lambda5_v8 = -1j * (ops.E23 - ops.E32)
lambda6_v8 = ops.E13 + ops.E31
lambda7_v8 = -1j * (ops.E13 - ops.E31)
lambda8_v8 = (2/np.sqrt(3)) * ops.T8

lambdas = [lambda1_v8, lambda2_v8, lambda3_v8, lambda4_v8, lambda5_v8, lambda6_v8, lambda7_v8, lambda8_v8]

# Check trace normalization: Tr(λᵢ λⱼ) should be 2δᵢⱼ
print("\n" + "="*70)
print("Trace Normalization: Tr(λᵢ λⱼ)")
print("="*70)

print("\nDiagonal (should be 2):")
for i in range(8):
    trace = np.trace(lambdas[i] @ lambdas[i]).real
    print(f"Tr(λ{i+1}²) = {trace:.6f}")

print("\nOff-diagonal (should be 0):")
for i in range(8):
    for j in range(i+1, 8):
        trace = np.trace(lambdas[i] @ lambdas[j])
        if abs(trace) > 1e-10:
            print(f"Tr(λ{i+1} λ{j+1}) = {trace:.6f}")

# Now check with T = λ/2
print("\n" + "="*70)
print("With Tᵢ = λᵢ/2: Tr(Tᵢ Tⱼ) (should be 1/2 δᵢⱼ)")
print("="*70)

Ts = [l/2 for l in lambdas]

print("\nDiagonal (should be 0.5):")
for i in range(8):
    trace = np.trace(Ts[i] @ Ts[i]).real
    print(f"Tr(T{i+1}²) = {trace:.6f}")

# Casimir with this normalization
C2_corrected = sum(T @ T for T in Ts)

print("\n" + "="*70)
print("Casimir with T = λ/2:")
print("="*70)

print(f"Diagonal: {np.diag(C2_corrected).real}")
print(f"Mean: {np.mean(np.diag(C2_corrected).real):.6f}")
print(f"Std: {np.std(np.diag(C2_corrected).real):.6f}")
print(f"Expected: 1.333333")

# The issue is the reconstruction! Let me check the actual Casimir formula
print("\n" + "="*70)
print("Direct Casimir Calculation:")
print("="*70)

# The formula Sum(Ta²) = Sum(ladder products) + Cartan²
# But we need the RIGHT formula

# For SU(3), using Gell-Mann normalization:
# C2 = (1/2) * sum of anticommutators {Ta, Ta†}
# Actually, the formula should account for the normalization

# Let me try: if operators have wrong normalization by factor α,
# then to match Gell-Mann we need to scale by 1/α where α² = 1.25 (from ratio analysis)

alpha = np.sqrt(1.25)
print(f"\nScaling factor from ratio analysis: α = {alpha:.6f}")
print(f"Need to scale operators by 1/α = {1/alpha:.6f}")

# This matches what we found: v8 has √2 correction giving α = 1/√2, 
# but actual should be α = 1/sqrt(1.25) ≈ 0.894

print(f"\nv8 uses 1/√2 = {1/np.sqrt(2):.6f}")
print(f"Should use 1/√1.25 = {1/np.sqrt(1.25):.6f}")
print(f"Additional factor needed: {(1/np.sqrt(2))/(1/np.sqrt(1.25)):.6f}")
