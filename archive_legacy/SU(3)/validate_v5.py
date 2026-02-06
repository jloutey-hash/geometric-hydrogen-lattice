"""
Comprehensive validation for SU(3) Operators v5
Algebraic closure approach from 09_su3_v5_algebraic_closure.md
"""

import numpy as np
from operators_v5 import SU3OperatorsV5


def test_commutation_relations(ops, verbose=True):
    """Test all SU(3) commutation relations."""
    
    if verbose:
        print("="*70)
        print("Testing Commutation Relations")
        print("="*70)
    
    errors = {}
    
    # Test 1: Cartan subalgebra commutes [T3, T8] = 0
    comm = ops.T3 @ ops.T8 - ops.T8 @ ops.T3
    error = np.max(np.abs(comm))
    errors['[T3, T8]'] = error
    status = "✓ PASS" if error < 1e-13 else "✗ FAIL"
    if verbose:
        print(f"[T3, T8] = 0: {error:.3e} {status}")
    
    # Test 2: I-spin commutators
    # [E12, E21] = 2*T3
    comm = ops.E12 @ ops.E21 - ops.E21 @ ops.E12
    expected = 2 * ops.T3
    error = np.max(np.abs(comm - expected))
    errors['[E12, E21] - 2*T3'] = error
    status = "✓ PASS" if error < 1e-13 else "✗ FAIL"
    if verbose:
        print(f"[E12, E21] - 2*T3: {error:.3e} {status}")
    
    # [T3, E12] = E12
    comm = ops.T3 @ ops.E12 - ops.E12 @ ops.T3
    error = np.max(np.abs(comm - ops.E12))
    errors['[T3, E12] - E12'] = error
    status = "✓ PASS" if error < 1e-13 else "✗ FAIL"
    if verbose:
        print(f"[T3, E12] - E12: {error:.3e} {status}")
    
    # Test 3: U-spin commutators
    # [E23, E32] = -(3/2)*T3 + (√3/2)*T8
    comm = ops.E23 @ ops.E32 - ops.E32 @ ops.E23
    expected = -(3.0/2.0) * ops.T3 + (np.sqrt(3.0)/2.0) * ops.T8
    error = np.max(np.abs(comm - expected))
    errors['[E23, E32] - (-(3/2)*T3 + (√3/2)*T8)'] = error
    status = "✓ PASS" if error < 1e-13 else "✗ FAIL"
    if verbose:
        print(f"[E23, E32] - (-(3/2)*T3 + (√3/2)*T8): {error:.3e} {status}")
    
    # Test 4: V-spin commutators
    # [E13, E31] = (3/2)*T3 + (√3/2)*T8
    comm = ops.E13 @ ops.E31 - ops.E31 @ ops.E13
    expected = (3.0/2.0) * ops.T3 + (np.sqrt(3.0)/2.0) * ops.T8
    error = np.max(np.abs(comm - expected))
    errors['[E13, E31] - ((3/2)*T3 + (√3/2)*T8)'] = error
    status = "✓ PASS" if error < 1e-13 else "✗ FAIL"
    if verbose:
        print(f"[E13, E31] - ((3/2)*T3 + (√3/2)*T8): {error:.3e} {status}")
    
    # Test 5: Cross commutators
    # [E12, E23] = E13
    comm = ops.E12 @ ops.E23 - ops.E23 @ ops.E12
    error = np.max(np.abs(comm - ops.E13))
    errors['[E12, E23] - E13'] = error
    status = "✓ PASS" if error < 1e-13 else "✗ FAIL"
    if verbose:
        print(f"[E12, E23] - E13: {error:.3e} {status}")
    
    # [E12, E31] = -E32
    comm = ops.E12 @ ops.E31 - ops.E31 @ ops.E12
    error = np.max(np.abs(comm + ops.E32))
    errors['[E12, E31] + E32'] = error
    status = "✓ PASS" if error < 1e-13 else "✗ FAIL"
    if verbose:
        print(f"[E12, E31] + E32: {error:.3e} {status}")
    
    # [E23, E31] = E21
    comm = ops.E23 @ ops.E31 - ops.E31 @ ops.E23
    error = np.max(np.abs(comm - ops.E21))
    errors['[E23, E31] - E21'] = error
    status = "✓ PASS" if error < 1e-13 else "✗ FAIL"
    if verbose:
        print(f"[E23, E31] - E21: {error:.3e} {status}")
    
    # [E13, E21] = E23
    comm = ops.E13 @ ops.E21 - ops.E21 @ ops.E13
    error = np.max(np.abs(comm - ops.E23))
    errors['[E13, E21] - E23'] = error
    status = "✓ PASS" if error < 1e-13 else "✗ FAIL"
    if verbose:
        print(f"[E13, E21] - E23: {error:.3e} {status}")
    
    if verbose:
        print()
        max_error = max(errors.values())
        print(f"Maximum commutator error: {max_error:.3e}")
        print(f"Target: < 1e-13")
        print()
    
    return errors


def test_casimir_operator(ops, verbose=True):
    """
    Test Casimir operator C2.
    According to spec 09: C2 = E12@E21 + E21@E12 + E23@E32 + E32@E23 + E13@E31 + E31@E13 + T3@T3 + T8@T8
    Should be a multiple of Identity for each irrep.
    """
    
    if verbose:
        print("="*70)
        print("Testing Casimir Operator")
        print("="*70)
    
    # Build Casimir
    C2 = (ops.E12 @ ops.E21 + ops.E21 @ ops.E12 + 
          ops.E23 @ ops.E32 + ops.E32 @ ops.E23 + 
          ops.E13 @ ops.E31 + ops.E31 @ ops.E13 + 
          ops.T3 @ ops.T3 + ops.T8 @ ops.T8)
    
    # Check if diagonal
    off_diag = C2 - np.diag(np.diag(C2))
    off_diag_error = np.max(np.abs(off_diag))
    
    # Check if constant (multiple of identity)
    diag_vals = np.diag(C2)
    diag_std = np.std(diag_vals)
    diag_mean = np.mean(diag_vals)
    
    # Expected value from theory: C2 = (p^2 + q^2 + 3p + 3q + pq) / 3
    p, q = ops.p, ops.q
    expected_eigenvalue = (p**2 + q**2 + 3*p + 3*q + p*q) / 3.0
    eigenvalue_error = abs(diag_mean - expected_eigenvalue)
    
    if verbose:
        print(f"Representation: ({p}, {q})")
        print(f"Off-diagonal elements: max = {off_diag_error:.3e}")
        print(f"Diagonal values: mean = {diag_mean:.6f}, std = {diag_std:.3e}")
        print(f"Expected C2 eigenvalue: {expected_eigenvalue:.6f}")
        print(f"Eigenvalue error: {eigenvalue_error:.3e}")
        
        status_diag = "✓ PASS" if off_diag_error < 1e-12 else "✗ FAIL"
        status_const = "✓ PASS" if diag_std < 1e-12 else "✗ FAIL"
        status_eigen = "✓ PASS" if eigenvalue_error < 1e-12 else "✗ FAIL"
        
        print(f"Is diagonal: {status_diag}")
        print(f"Is constant: {status_const}")
        print(f"Matches theory: {status_eigen}")
        print()
    
    return {
        'off_diagonal_error': off_diag_error,
        'diagonal_std': diag_std,
        'eigenvalue_error': eigenvalue_error,
        'expected': expected_eigenvalue,
        'actual': diag_mean
    }


def test_hermiticity(ops, verbose=True):
    """Test that operators have correct hermiticity properties."""
    
    if verbose:
        print("="*70)
        print("Testing Hermiticity")
        print("="*70)
    
    errors = {}
    
    # Cartan operators should be Hermitian
    error_T3 = np.max(np.abs(ops.T3 - ops.T3.T.conj()))
    error_T8 = np.max(np.abs(ops.T8 - ops.T8.T.conj()))
    errors['T3_hermitian'] = error_T3
    errors['T8_hermitian'] = error_T8
    
    # Raising/lowering pairs should be adjoints
    error_E21 = np.max(np.abs(ops.E21 - ops.E12.T.conj()))
    error_E32 = np.max(np.abs(ops.E32 - ops.E23.T.conj()))
    error_E31 = np.max(np.abs(ops.E31 - ops.E13.T.conj()))
    errors['E21 = E12†'] = error_E21
    errors['E32 = E23†'] = error_E32
    errors['E31 = E13†'] = error_E31
    
    if verbose:
        for name, error in errors.items():
            status = "✓ PASS" if error < 1e-14 else "✗ FAIL"
            print(f"{name}: {error:.3e} {status}")
        print()
    
    return errors


def run_full_validation():
    """Run comprehensive validation on multiple representations."""
    
    print("\n" + "="*70)
    print("SU(3) Operators v5: Comprehensive Validation")
    print("Algebraic Closure Method (09_su3_v5_algebraic_closure.md)")
    print("="*70 + "\n")
    
    representations = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2)]
    
    all_results = {}
    
    for p, q in representations:
        print(f"\n{'='*70}")
        print(f"Testing ({p}, {q}) representation")
        print(f"{'='*70}\n")
        
        ops = SU3OperatorsV5(p, q)
        print(f"Dimension: {ops.dim}\n")
        
        # Run tests
        herm_errors = test_hermiticity(ops, verbose=True)
        comm_errors = test_commutation_relations(ops, verbose=True)
        casimir_results = test_casimir_operator(ops, verbose=True)
        
        all_results[(p, q)] = {
            'hermiticity': herm_errors,
            'commutators': comm_errors,
            'casimir': casimir_results
        }
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70 + "\n")
    
    for p, q in representations:
        results = all_results[(p, q)]
        max_comm_error = max(results['commutators'].values())
        casimir_error = results['casimir']['eigenvalue_error']
        
        print(f"({p}, {q}): max_comm_error = {max_comm_error:.3e}, casimir_error = {casimir_error:.3e}")
    
    print()


if __name__ == "__main__":
    run_full_validation()
