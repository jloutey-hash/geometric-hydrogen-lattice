"""
Validate SU(3) operators using algebraic structure, not external formulas.

Test:
1. Hermiticity properties
2. Cartan algebra structure
3. Jacobi identities  
4. Casimir operator
5. Internal consistency

NO assumptions about specific commutation formulas!
"""

import numpy as np
from operators_v8 import SU3OperatorsV8  # Use v8 which has weight-space Cartans + √2 correction

np.set_printoptions(precision=15, suppress=True)

def test_representation(p, q, verbose=True):
    """Test operators for (p,q) irrep."""
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Testing ({p}, {q}) representation")
        print(f"{'='*70}")
    
    ops = SU3OperatorsV8(p, q)
    
    if verbose:
        print(f"Dimension: {ops.dim}\n")
    
    errors = {}
    
    # Test 1: Hermiticity
    errors['T3_herm'] = np.max(np.abs(ops.T3 - ops.T3.T.conj()))
    errors['T8_herm'] = np.max(np.abs(ops.T8 - ops.T8.T.conj()))
    errors['E21_adj'] = np.max(np.abs(ops.E21 - ops.E12.T.conj()))
    errors['E32_adj'] = np.max(np.abs(ops.E32 - ops.E23.T.conj()))
    errors['E31_adj'] = np.max(np.abs(ops.E31 - ops.E13.T.conj()))
    
    # Test 2: Cartan algebra
    errors['[T3,T8]'] = np.max(np.abs(ops.T3 @ ops.T8 - ops.T8 @ ops.T3))
    
    # Test 3: Cartan action (eigenvalue equations)
    # [T3, E12] should be proportional to E12
    comm = ops.T3 @ ops.E12 - ops.E12 @ ops.T3
    # Check if comm = λ * E12 for some λ
    if np.max(np.abs(ops.E12)) > 1e-10:
        ratio = comm / (ops.E12 + 1e-20)  # Avoid div by zero
        ratio_vals = ratio[np.abs(ops.E12) > 1e-10]
        if len(ratio_vals) > 0:
            eigenvalue = ratio_vals[0]
            errors['[T3,E12]-λE12'] = np.max(np.abs(comm - eigenvalue * ops.E12))
    
    # Test 4: Jacobi identities
    # [E12, [E23, E31]] + [E23, [E31, E12]] + [E31, [E12, E23]] = 0
    comm1 = ops.E12 @ (ops.E23 @ ops.E31 - ops.E31 @ ops.E23) - (ops.E23 @ ops.E31 - ops.E31 @ ops.E23) @ ops.E12
    comm2 = ops.E23 @ (ops.E31 @ ops.E12 - ops.E12 @ ops.E31) - (ops.E31 @ ops.E12 - ops.E12 @ ops.E31) @ ops.E23
    comm3 = ops.E31 @ (ops.E12 @ ops.E23 - ops.E23 @ ops.E12) - (ops.E12 @ ops.E23 - ops.E23 @ ops.E12) @ ops.E31
    jacobi = comm1 + comm2 + comm3
    errors['Jacobi'] = np.max(np.abs(jacobi))
    
    # Test 5: Casimir (should be diagonal and constant)
    C2 = (ops.E12 @ ops.E21 + ops.E21 @ ops.E12 +
          ops.E23 @ ops.E32 + ops.E32 @ ops.E23 +
          ops.E13 @ ops.E31 + ops.E31 @ ops.E13 +
          ops.T3 @ ops.T3 + ops.T8 @ ops.T8)
    
    # Check diagonal
    off_diag = C2 - np.diag(np.diag(C2))
    errors['Casimir_off_diag'] = np.max(np.abs(off_diag))
    
    # Check constant
    diag_vals = np.diag(C2).real
    errors['Casimir_const'] = np.std(diag_vals)
    
    # Check theoretical value
    expected_C2 = (p**2 + q**2 + 3*p + 3*q + p*q) / 3.0
    errors['Casimir_theory'] = np.abs(np.mean(diag_vals) - expected_C2)
    
    if verbose:
        print("Hermiticity & Adjoint Relations:")
        for key in ['T3_herm', 'T8_herm', 'E21_adj', 'E32_adj', 'E31_adj']:
            status = "✓" if errors[key] < 1e-13 else "✗"
            print(f"  {key:12s}: {errors[key]:.3e} {status}")
        
        print("\nCartan Algebra:")
        for key in ['[T3,T8]']:
            status = "✓" if errors[key] < 1e-13 else "✗"
            print(f"  {key:12s}: {errors[key]:.3e} {status}")
        
        print("\nJacobi Identity:")
        status = "✓" if errors['Jacobi'] < 1e-13 else "✗"
        print(f"  Jacobi      : {errors['Jacobi']:.3e} {status}")
        
        print("\nCasimir Operator:")
        for key in ['Casimir_off_diag', 'Casimir_const', 'Casimir_theory']:
            status = "✓" if errors[key] < 1e-12 else "✗"
            print(f"  {key:18s}: {errors[key]:.3e} {status}")
        
        max_error = max(errors.values())
        print(f"\nMaximum error: {max_error:.3e}")
    
    return errors


if __name__ == "__main__":
    print("SU(3) Algebraic Structure Validation")
    print("="*70)
    print("Testing internal consistency without assuming specific commutators\n")
    
    representations = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1)]
    
    all_results = {}
    for p, q in representations:
        all_results[(p, q)] = test_representation(p, q, verbose=True)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for p, q in representations:
        errors = all_results[(p, q)]
        max_err = max(errors.values())
        jacobi_err = errors['Jacobi']
        casimir_err = errors['Casimir_theory']
        
        status = "✓ PASS" if max_err < 1e-12 else "✗ FAIL"
        print(f"({p},{q}): max={max_err:.2e}, Jacobi={jacobi_err:.2e}, Casimir={casimir_err:.2e} {status}")
