"""
Comprehensive Validation of SU(3) Operators v4 (l-index formulas)
Tests all commutation relations and Casimir eigenvalues.
"""

import numpy as np
from scipy.sparse.linalg import eigsh
from lattice import SU3Lattice
from operators_v4 import SU3Operators
import matplotlib.pyplot as plt


def test_commutation_relations(operators: SU3Operators, verbose=True):
    """
    Test all SU(3) commutation relations.
    
    Key commutators:
    [E12, E21] = 2*T3 (I-spin)
    [E23, E32] = -(3/2)*T3 + (sqrt(3)/2)*T8 (U-spin)
    [E13, E31] = (3/2)*T3 + (sqrt(3)/2)*T8 (V-spin)
    """
    ops = operators.get_operators()
    T3 = ops['T3']
    T8 = ops['T8']
    E12 = ops['E12']
    E21 = ops['E21']
    E23 = ops['E23']
    E32 = ops['E32']
    E13 = ops['E13']
    E31 = ops['E31']
    
    results = {}
    
    # Convert to dense for easier computation
    T3_d = T3.toarray()
    T8_d = T8.toarray()
    E12_d = E12.toarray()
    E21_d = E21.toarray()
    E23_d = E23.toarray()
    E32_d = E32.toarray()
    E13_d = E13.toarray()
    E31_d = E31.toarray()
    
    if verbose:
        print("="*70)
        print("COMMUTATION RELATION TESTS (Target: error < 1e-13)")
        print("="*70)
    
    # Test 1: [E12, E21] = 2*T3 (I-spin)
    comm = E12_d @ E21_d - E21_d @ E12_d
    expected = 2 * T3_d
    error = np.max(np.abs(comm - expected))
    results['[E12, E21] = 2*T3'] = error
    if verbose:
        status = "✓ PASS" if error < 1e-13 else "✗ FAIL"
        print(f"[E12, E21] - 2*T3: {error:.3e} {status}")
    
    # Test 2: [E23, E32] = -(3/2)*T3 + (sqrt(3)/2)*T8 (U-spin)
    comm = E23_d @ E32_d - E32_d @ E23_d
    expected = -1.5 * T3_d + (np.sqrt(3)/2) * T8_d
    error = np.max(np.abs(comm - expected))
    results['[E23, E32] = -(3/2)*T3 + (√3/2)*T8'] = error
    if verbose:
        status = "✓ PASS" if error < 1e-13 else "✗ FAIL"
        print(f"[E23, E32] - (-(3/2)*T3 + (√3/2)*T8): {error:.3e} {status}")
    
    # Test 3: [E13, E31] = (3/2)*T3 + (sqrt(3)/2)*T8 (V-spin)
    comm = E13_d @ E31_d - E31_d @ E13_d
    expected = 1.5 * T3_d + (np.sqrt(3)/2) * T8_d
    error = np.max(np.abs(comm - expected))
    results['[E13, E31] = (3/2)*T3 + (√3/2)*T8'] = error
    if verbose:
        status = "✓ PASS" if error < 1e-13 else "✗ FAIL"
        print(f"[E13, E31] - ((3/2)*T3 + (√3/2)*T8): {error:.3e} {status}")
    
    # Test 4: [T3, E12] = E12
    comm = T3_d @ E12_d - E12_d @ T3_d
    expected = E12_d
    error = np.max(np.abs(comm - expected))
    results['[T3, E12] = E12'] = error
    if verbose:
        status = "✓ PASS" if error < 1e-13 else "✗ FAIL"
        print(f"[T3, E12] - E12: {error:.3e} {status}")
    
    # Test 5: [T3, E21] = -E21
    comm = T3_d @ E21_d - E21_d @ T3_d
    expected = -E21_d
    error = np.max(np.abs(comm - expected))
    results['[T3, E21] = -E21'] = error
    if verbose:
        status = "✓ PASS" if error < 1e-13 else "✗ FAIL"
        print(f"[T3, E21] + E21: {error:.3e} {status}")
    
    # Test 6: [T8, E23]
    comm = T8_d @ E23_d - E23_d @ T8_d
    expected = (np.sqrt(3)/2) * E23_d
    error = np.max(np.abs(comm - expected))
    results['[T8, E23] = (√3/2)*E23'] = error
    if verbose:
        status = "✓ PASS" if error < 1e-13 else "✗ FAIL"
        print(f"[T8, E23] - (√3/2)*E23: {error:.3e} {status}")
    
    # Test 7: [T8, E32]
    comm = T8_d @ E32_d - E32_d @ T8_d
    expected = -(np.sqrt(3)/2) * E32_d
    error = np.max(np.abs(comm - expected))
    results['[T8, E32] = -(√3/2)*E32'] = error
    if verbose:
        status = "✓ PASS" if error < 1e-13 else "✗ FAIL"
        print(f"[T8, E32] + (√3/2)*E32: {error:.3e} {status}")
    
    # Test 8: [T3, T8] = 0 (Cartan subalgebra)
    comm = T3_d @ T8_d - T8_d @ T3_d
    error = np.max(np.abs(comm))
    results['[T3, T8] = 0'] = error
    if verbose:
        status = "✓ PASS" if error < 1e-13 else "✗ FAIL"
        print(f"[T3, T8]: {error:.3e} {status}")
    
    if verbose:
        print()
        max_error = max(results.values())
        if max_error < 1e-13:
            print("🎉 ALL COMMUTATION RELATIONS PASSED! 🎉")
        else:
            print(f"❌ Maximum error: {max_error:.3e} (target: < 1e-13)")
    
    return results


def test_casimir_eigenvalues(operators: SU3Operators, lattice: SU3Lattice, verbose=True):
    """Test Casimir eigenvalues for each irrep."""
    C2 = operators.C2
    
    # Get unique irreps
    irreps = set((s['p'], s['q']) for s in lattice.states)
    
    if verbose:
        print("="*70)
        print("CASIMIR EIGENVALUE TESTS (Target: error < 1e-12)")
        print("="*70)
    
    results = {}
    
    for (p, q) in sorted(irreps):
        # Get indices for this irrep
        indices = [s['index'] for s in lattice.states if s['p'] == p and s['q'] == q]
        dim = len(indices)
        
        # Extract submatrix
        C2_sub = C2.toarray()[np.ix_(indices, indices)]
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(C2_sub)
        
        # Theoretical value
        c2_theory = (p**2 + q**2 + 3*p + 3*q + p*q) / 3.0
        
        # Check all eigenvalues are close to theory
        errors = np.abs(eigenvalues - c2_theory)
        max_error = np.max(errors)
        mean_error = np.mean(errors)
        
        results[f'({p},{q})'] = {
            'max_error': max_error,
            'mean_error': mean_error,
            'dimension': dim,
            'theory': c2_theory,
            'computed': np.mean(eigenvalues)
        }
        
        if verbose:
            status = "✓ PASS" if max_error < 1e-12 else "✗ FAIL"
            print(f"  ({p},{q}): dim={dim:2d}, C2_theory={c2_theory:.4f}, "
                  f"error={max_error:.3e} {status}")
    
    if verbose:
        print()
        max_error_all = max(r['max_error'] for r in results.values())
        if max_error_all < 1e-12:
            print("🎉 ALL CASIMIR EIGENVALUES PASSED! 🎉")
        else:
            print(f"❌ Maximum error: {max_error_all:.3e} (target: < 1e-12)")
    
    return results


def test_hermiticity(operators: SU3Operators, verbose=True):
    """Test hermiticity and adjoint relationships."""
    ops = operators.get_operators()
    
    if verbose:
        print("="*70)
        print("HERMITICITY AND ADJOINT TESTS")
        print("="*70)
    
    results = {}
    
    # Test that diagonal operators are hermitian
    for name in ['T3', 'T8', 'C2']:
        op = ops[name].toarray()
        error = np.max(np.abs(op - op.conj().T))
        results[f'{name} hermitian'] = error
        
        if verbose:
            status = "✓ PASS" if error < 1e-14 else "✗ FAIL"
            print(f"  {name} hermitian: {error:.3e} {status}")
    
    # Test that lowering operators are adjoints of raising operators
    adjoint_pairs = [
        ('E21', 'E12'),
        ('E32', 'E23'),
        ('E31', 'E13'),
    ]
    
    for lower_name, raise_name in adjoint_pairs:
        lower = ops[lower_name].toarray()
        raise_op = ops[raise_name].toarray()
        error = np.max(np.abs(lower - raise_op.conj().T))
        results[f'{lower_name} = {raise_name}†'] = error
        
        if verbose:
            status = "✓ PASS" if error < 1e-14 else "✗ FAIL"
            print(f"  {lower_name} = {raise_name}†: {error:.3e} {status}")
    
    if verbose:
        print()
    
    return results


def main():
    """Run all validation tests."""
    print("Building SU(3) lattice with l-index operators (v4)...\n")
    
    # Build lattice
    lattice = SU3Lattice(max_p=2, max_q=2)
    print(f"Lattice contains {lattice.get_dimension()} states\n")
    
    # Build operators
    operators = SU3Operators(lattice)
    print()
    
    # Run tests
    comm_results = test_commutation_relations(operators)
    print()
    
    casimir_results = test_casimir_eigenvalues(operators, lattice)
    print()
    
    herm_results = test_hermiticity(operators)
    print()
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    
    max_comm_error = max(comm_results.values())
    max_casimir_error = max(r['max_error'] for r in casimir_results.values())
    max_adjoint_error = max(v for k, v in herm_results.items() if '†' in k)
    
    print(f"Maximum commutator error:      {max_comm_error:.3e} (target: < 1e-13)")
    print(f"Maximum Casimir error:         {max_casimir_error:.3e} (target: < 1e-12)")
    print(f"Maximum adjoint error:         {max_adjoint_error:.3e} (target: < 1e-14)")
    print()
    
    if max_comm_error < 1e-13 and max_casimir_error < 1e-12:
        print("🎊 🎉 VALIDATION SUCCESSFUL! 🎉 🎊")
        print("All operators satisfy exact SU(3) algebra!")
    else:
        print("⚠️  Some tests did not meet target precision")
        print("Further refinement may be needed.")


if __name__ == "__main__":
    main()
