"""
Validation tests for Phase 9.2: Hydrogen Atom on Discrete Lattice

Tests:
1. Lattice construction
2. Energy level ordering
3. Comparison with continuum
4. Geometric factor search
"""

import numpy as np
import sys
sys.path.append('src')

from hydrogen_lattice import HydrogenLattice

def test_lattice_construction():
    """Test that lattice is built correctly."""
    print("TEST 1: Lattice Construction")
    print("-" * 60)
    
    hydrogen = HydrogenLattice(ell_max=10, a_lattice=1.0)
    
    # Check radial lattice
    assert len(hydrogen.r_ell) == 11, "Wrong number of shells"
    assert hydrogen.r_ell[0] == 1.0, "r_0 should be 1.0"
    assert hydrogen.r_ell[1] == 3.0, "r_1 should be 3.0"
    assert hydrogen.r_ell[10] == 21.0, "r_10 should be 21.0"
    
    # Check angular momentum
    assert hydrogen.L2_ell[0] == 0, "L²_0 should be 0"
    assert hydrogen.L2_ell[1] == 2, "L²_1 should be 2"
    assert hydrogen.L2_ell[5] == 30, "L²_5 should be 30"
    
    print("  ✓ Radial lattice: r_ℓ = 1 + 2ℓ")
    print("  ✓ Angular momentum: L²_ℓ = ℓ(ℓ+1)")
    print("  ✓ Number of shells correct")
    print()
    return True

def test_energy_ordering():
    """Test that energy levels are properly ordered."""
    print("TEST 2: Energy Level Ordering")
    print("-" * 60)
    
    hydrogen = HydrogenLattice(ell_max=30, a_lattice=1.0)
    
    # Solve diagonal case
    E_diag, _ = hydrogen.solve_diagonal()
    
    # Check ordering (energies should increase)
    assert np.all(E_diag[:-1] <= E_diag[1:]), "Energies not sorted"
    
    # Check ground state is negative
    assert E_diag[0] < 0, "Ground state should be negative"
    
    print(f"  ✓ Ground state energy: {E_diag[0]:.6f} Ry")
    print(f"  ✓ Energies properly ordered")
    print(f"  ✓ All bound states negative: {np.all(E_diag < 0)}")
    print()
    return True

def test_continuum_comparison():
    """Test comparison with continuum hydrogen."""
    print("TEST 3: Continuum Comparison")
    print("-" * 60)
    
    hydrogen = HydrogenLattice(ell_max=50, a_lattice=1.0)
    results = hydrogen.compare_with_continuum(n_states=10)
    
    # Check that lattice energies are close to continuum
    # (exact match not expected due to discretization)
    max_error = np.max(results['error_diagonal'])
    
    print(f"  n=1 continuum: {results['E_continuum'][0]:.6f} Ry")
    print(f"  n=1 lattice:   {results['E_lattice_diagonal'][0]:.6f} Ry")
    print(f"  n=1 error:     {results['error_diagonal'][0]:.3f}%")
    print(f"  Max error (n=1-10): {max_error:.3f}%")
    
    # Errors should be reasonable (< 100% for low n)
    assert max_error < 100, f"Error too large: {max_error}%"
    
    # Ground state should be closest
    assert results['error_diagonal'][0] < results['error_diagonal'][-1], \
           "Ground state should have smaller error than excited states"
    
    print("  ✓ Lattice approximates continuum")
    print("  ✓ Ground state best approximated")
    print()
    return True

def test_geometric_factor():
    """Test search for 1/(4π) in energy corrections."""
    print("TEST 4: Geometric Factor Analysis")
    print("-" * 60)
    
    hydrogen = HydrogenLattice(ell_max=50, a_lattice=1.0)
    analysis = hydrogen.find_geometric_factor()
    
    # Check that analysis ran
    assert 'models' in analysis, "Models not found in analysis"
    assert 'one_over_4pi' in analysis, "Reference value not found"
    
    one_4pi = analysis['one_over_4pi']
    print(f"  Reference: 1/(4π) = {one_4pi:.10f}")
    print()
    
    # Check each model
    print("  Model fit coefficients:")
    for model_name, model_data in analysis['models'].items():
        A = model_data['A_diag']
        A_times_4pi = A * 4 * np.pi
        residual = model_data['residual']
        
        print(f"    {model_name:15s}: A = {A:10.6f}, A×4π = {A_times_4pi:10.6f}, R = {residual:.4f}")
        
        # Check that residuals are finite
        assert np.isfinite(residual), f"Residual not finite for {model_name}"
    
    print()
    print("  ✓ Geometric factor analysis complete")
    print("  ✓ Multiple scaling models tested")
    print()
    
    # Look for evidence of 1/(4π)
    best_model = min(analysis['models'].items(), key=lambda x: x[1]['residual'])
    print(f"  Best model: {best_model[0]} (lowest residual)")
    
    return True

def test_with_hopping():
    """Test Hamiltonian with radial hopping."""
    print("TEST 5: Radial Hopping")
    print("-" * 60)
    
    hydrogen = HydrogenLattice(ell_max=30, a_lattice=1.0)
    
    # Solve with different hopping strengths
    E_diag, _ = hydrogen.solve_diagonal()
    E_hop_weak, _ = hydrogen.solve_with_hopping(t_hop=0.01)
    E_hop_strong, _ = hydrogen.solve_with_hopping(t_hop=0.1)
    
    print(f"  Ground state energies:")
    print(f"    Diagonal:        {E_diag[0]:.6f} Ry")
    print(f"    Hopping (weak):  {E_hop_weak[0]:.6f} Ry")
    print(f"    Hopping (strong):{E_hop_strong[0]:.6f} Ry")
    print()
    
    # Hopping should lower ground state energy
    assert E_hop_weak[0] < E_diag[0], "Weak hopping should lower energy"
    assert E_hop_strong[0] < E_hop_weak[0], "Strong hopping should lower more"
    
    print("  ✓ Hopping lowers ground state energy")
    print("  ✓ Stronger hopping → lower energy")
    print()
    return True

def run_all_tests():
    """Run all validation tests."""
    print("=" * 80)
    print("VALIDATION TESTS: PHASE 9.2 - HYDROGEN ATOM ON DISCRETE LATTICE")
    print("=" * 80)
    print()
    
    tests = [
        ("Lattice Construction", test_lattice_construction),
        ("Energy Ordering", test_energy_ordering),
        ("Continuum Comparison", test_continuum_comparison),
        ("Geometric Factor", test_geometric_factor),
        ("Radial Hopping", test_with_hopping)
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            print()
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            print()
            failed += 1
    
    print("=" * 80)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("✓✓✓ ALL TESTS PASSED! ✓✓✓")
    else:
        print(f"✗ {failed} test(s) failed")
    print("=" * 80)
    print()
    
    if failed == 0:
        print("Phase 9.2 validation SUCCESSFUL!")
        print("Hydrogen atom solver is working correctly.")
        print()
        print("Next step: Run src/hydrogen_lattice.py to generate full analysis")
    
    return failed == 0

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
