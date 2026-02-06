"""
Comprehensive test of v12
"""

import numpy as np
from operators_v12 import SU3OperatorsV12

np.set_printoptions(precision=6, suppress=True, linewidth=120)

print("="*80)
print("SU(3) Operators v12: Comprehensive Validation")
print("="*80)

for p, q in [(1, 0), (0, 1), (1, 1), (2, 0)]:
    print(f"\n{'='*80}")
    print(f"(p,q) = ({p},{q})")
    print(f"{'='*80}")
    
    ops = SU3OperatorsV12(p, q)
    
    print(f"Dimension: {ops.dim}")
    
    # Reconstruct Gell-Mann matrices
    lambda1 = ops.E12 + ops.E21
    lambda2 = -1j * (ops.E12 - ops.E21)
    lambda3 = 2 * ops.T3
    lambda4 = ops.E23 + ops.E32
    lambda5 = -1j * (ops.E23 - ops.E32)
    lambda6 = ops.E13 + ops.E31
    lambda7 = -1j * (ops.E13 - ops.E31)
    lambda8 = (2/np.sqrt(3)) * ops.T8
    
    lambdas = [lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7, lambda8]
    
    # Check trace normalization
    print("\nTrace Norms Tr(λᵢ²):")
    traces = [np.trace(l @ l).real for l in lambdas]
    for i, t in enumerate(traces, 1):
        print(f"  λ{i}: {t:.6f}")
    
    all_2 = all(abs(t - 2) < 1e-10 for t in traces)
    print(f"All equal to 2? {all_2} {'✓' if all_2 else '✗'}")
    
    # Casimir operator
    C2 = (ops.E12 @ ops.E21 + ops.E21 @ ops.E12 + 
          ops.E23 @ ops.E32 + ops.E32 @ ops.E23 +
          ops.E13 @ ops.E31 + ops.E31 @ ops.E13 +
          ops.T3 @ ops.T3 + ops.T8 @ ops.T8) / 4
    
    eigs = np.diag(C2).real
    expected = (p**2 + q**2 + 3*p + 3*q + p*q) / 3
    
    print(f"\nCasimir Operator:")
    print(f"  Eigenvalues: {eigs}")
    print(f"  Expected: {expected:.6f}")
    print(f"  Mean: {np.mean(eigs):.6f}")
    print(f"  Std: {np.std(eigs):.2e}")
    print(f"  Max error: {np.max(np.abs(eigs - expected)):.2e}")
    
    constant = np.std(eigs) < 1e-10
    print(f"  Constant? {constant} {'✓' if constant else '✗'}")
    
    # Commutation relations
    print("\nCommutation Relations:")
    
    # [T3, T8] = 0
    comm = ops.T3 @ ops.T8 - ops.T8 @ ops.T3
    error = np.max(np.abs(comm))
    print(f"  [T3, T8] = 0: {error:.2e} {'✓' if error < 1e-13 else '✗'}")
    
    # [E12, E21] = 2*T3
    comm = ops.E12 @ ops.E21 - ops.E21 @ ops.E12
    expected_comm = 2 * ops.T3
    error = np.max(np.abs(comm - expected_comm))
    print(f"  [E12, E21] = 2*T3: {error:.2e} {'✓' if error < 1e-13 else '✗'}")
    
    # [E23, E32] = T3 + √3*T8
    comm = ops.E23 @ ops.E32 - ops.E32 @ ops.E23
    expected_comm = ops.T3 + np.sqrt(3) * ops.T8
    error = np.max(np.abs(comm - expected_comm))
    print(f"  [E23, E32] = T3 + √3*T8: {error:.2e} {'✓' if error < 1e-13 else '✗'}")
    
    # [E13, E31] = -T3 + √3*T8
    comm = ops.E13 @ ops.E31 - ops.E31 @ ops.E13
    expected_comm = -ops.T3 + np.sqrt(3) * ops.T8
    error = np.max(np.abs(comm - expected_comm))
    print(f"  [E13, E31] = -T3 + √3*T8: {error:.2e} {'✓' if error < 1e-13 else '✗'}")
    
    # Hermiticity
    print("\nHermiticity:")
    error_E12 = np.max(np.abs(ops.E21 - ops.E12.conj().T))
    error_E23 = np.max(np.abs(ops.E32 - ops.E23.conj().T))
    error_E13 = np.max(np.abs(ops.E31 - ops.E13.conj().T))
    error_T3 = np.max(np.abs(ops.T3 - ops.T3.conj().T))
    error_T8 = np.max(np.abs(ops.T8 - ops.T8.conj().T))
    
    print(f"  E21 = E12†: {error_E12:.2e} {'✓' if error_E12 < 1e-15 else '✗'}")
    print(f"  E32 = E23†: {error_E23:.2e} {'✓' if error_E23 < 1e-15 else '✗'}")
    print(f"  E31 = E13†: {error_E13:.2e} {'✓' if error_E13 < 1e-15 else '✗'}")
    print(f"  T3 Hermitian: {error_T3:.2e} {'✓' if error_T3 < 1e-15 else '✗'}")
    print(f"  T8 Hermitian: {error_T8:.2e} {'✓' if error_T8 < 1e-15 else '✗'}")
    
print("\n" + "="*80)
print("Validation Complete")
print("="*80)
