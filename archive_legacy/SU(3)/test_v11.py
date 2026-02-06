"""
Test v11 normalization and Casimir
"""

import numpy as np
from operators_v11 import SU3OperatorsV11

np.set_printoptions(precision=6, suppress=True)

print("Testing v11 Normalization and Casimir\n")
print("="*70)

for p, q in [(1, 0), (0, 1), (1, 1)]:
    print(f"\n(p,q) = ({p},{q})")
    print("="*70)
    
    ops = SU3OperatorsV11(p, q)
    
    # Check trace norms
    lambda1 = ops.E12 + ops.E21
    lambda2 = -1j * (ops.E12 - ops.E21)
    lambda3 = 2 * ops.T3
    lambda4 = ops.E23 + ops.E32
    lambda5 = -1j * (ops.E23 - ops.E32)
    lambda6 = ops.E13 + ops.E31
    lambda7 = -1j * (ops.E13 - ops.E31)
    lambda8 = (2/np.sqrt(3)) * ops.T8
    
    lambdas = [lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7, lambda8]
    traces = [np.trace(l @ l).real for l in lambdas]
    
    print(f"Trace norms Tr(λᵢ²): {traces}")
    print(f"All equal to 2? {all(abs(t - 2) < 1e-10 for t in traces)}")
    
    # Casimir
    C2 = (ops.E12 @ ops.E21 + ops.E21 @ ops.E12 + 
          ops.E23 @ ops.E32 + ops.E32 @ ops.E23 +
          ops.E13 @ ops.E31 + ops.E31 @ ops.E13 +
          ops.T3 @ ops.T3 + ops.T8 @ ops.T8) / 4
    
    eigs = np.diag(C2).real
    expected = (p**2 + q**2 + 3*p + 3*q + p*q) / 3
    
    print(f"\nCasimir eigenvalues: {eigs}")
    print(f"Expected: {expected:.6f}")
    print(f"Mean: {np.mean(eigs):.6f}")
    print(f"Std: {np.std(eigs):.2e}")
    print(f"Max error: {np.max(np.abs(eigs - expected)):.2e}")
    
    # Check commutators
    print(f"\nCommutator checks:")
    
    # [E12, E21] = 2*T3
    comm = ops.E12 @ ops.E21 - ops.E21 @ ops.E12
    expected_comm = 2 * ops.T3
    error = np.max(np.abs(comm - expected_comm))
    print(f"[E12, E21] - 2*T3: {error:.2e}")
    
    # [E23, E32] = T3 + √3*T8
    comm = ops.E23 @ ops.E32 - ops.E32 @ ops.E23
    expected_comm = ops.T3 + np.sqrt(3) * ops.T8
    error = np.max(np.abs(comm - expected_comm))
    print(f"[E23, E32] - (T3 + √3*T8): {error:.2e}")
    
    # [E13, E31] = -T3 + √3*T8
    comm = ops.E13 @ ops.E31 - ops.E31 @ ops.E13
    expected_comm = -ops.T3 + np.sqrt(3) * ops.T8
    error = np.max(np.abs(comm - expected_comm))
    print(f"[E13, E31] - (-T3 + √3*T8): {error:.2e}")
    
    # [T3, T8] = 0
    comm = ops.T3 @ ops.T8 - ops.T8 @ ops.T3
    error = np.max(np.abs(comm))
    print(f"[T3, T8]: {error:.2e}")
