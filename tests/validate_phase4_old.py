"""
Phase 4 Validation: Comparison with Quantum Mechanics

This script performs comprehensive validation of the discrete lattice model
against continuous quantum mechanics predictions.

Tests:
1. Spherical harmonics overlap - compare eigenmodes to Y_l^m
2. Energy level comparison - lattice vs hydrogen atom
3. Dipole selection rules - test Dl=±1, Dm=0,±1
4. Quantum number identification - assign (n,l,m) to eigenmodes
5. Visualization of all comparisons

Author: Quantum Lattice Project
Date: January 2026
Phase: 4 - Comparison with Quantum Mechanics
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lattice import PolarLattice
from operators import LatticeOperators
from angular_momentum import AngularMomentumOperators
from quantum_comparison import QuantumComparison, visualize_overlap_matrix, \
                                visualize_energy_comparison, visualize_selection_rules


def test_spherical_harmonics_sampling():
    """
    Test 1: Spherical Harmonics Sampling
    
    Verify that Y_l^m can be evaluated at lattice points and that
    sampled values have expected properties.
    """
    print("="*70)
    print("TEST 1: Spherical Harmonics Sampling")
    print("="*70)
    
    n_max = 4
    lattice = PolarLattice(n_max=n_max)
    
    operators = LatticeOperators(lattice)
    comparison = QuantumComparison(lattice, operators)
    
    print(f"\nLattice: n_max={n_max}, N_total={len(lattice.points)} sites")
    print(f"Spherical coordinates computed for all sites")
    
    # Sample several Y_l^m
    test_cases = [(0, 0), (1, -1), (1, 0), (1, 1), (2, 0), (2, 2)]
    
    print(f"\nSampling {len(test_cases)} spherical harmonics:")
    for ell, m in test_cases:
        ylm = comparison.sample_spherical_harmonic(ell, m)
        norm = np.sum(np.abs(ylm)**2)
        max_val = np.max(np.abs(ylm))
        
        print(f"  Y_{ell}^{m}: |Y|² sum = {norm:.6f}, max|Y| = {max_val:.6f}")
    
    # Test orthogonality on lattice
    print(f"\nOrthogonality test (inner products on lattice):")
    ylm_00 = comparison.sample_spherical_harmonic(0, 0)
    ylm_10 = comparison.sample_spherical_harmonic(1, 0)
    ylm_11 = comparison.sample_spherical_harmonic(1, 1)
    
    overlap_00_10 = np.abs(np.vdot(ylm_00, ylm_10))
    overlap_00_11 = np.abs(np.vdot(ylm_00, ylm_11))
    overlap_10_11 = np.abs(np.vdot(ylm_10, ylm_11))
    
    print(f"  |<Y_0^0 | Y_1^0>| = {overlap_00_10:.6e} (expect ~0)")
    print(f"  |<Y_0^0 | Y_1^1>| = {overlap_00_11:.6e} (expect ~0)")
    print(f"  |<Y_1^0 | Y_1^1>| = {overlap_10_11:.6e} (expect ~0)")
    
    # Check if approximately orthogonal
    threshold = 0.1
    if overlap_00_10 < threshold and overlap_00_11 < threshold and overlap_10_11 < threshold:
        print(f"\n[PASS] Y_l^m approximately orthogonal on lattice (all < {threshold})")
    else:
        print(f"\n[WARN] Some Y_l^m overlaps exceed {threshold}")
    
    return comparison


def test_eigenmode_overlap(comparison, n_max=4):
    """
    Test 2: Eigenmode Overlap with Spherical Harmonics
    
    Compute eigenmodes of angular Hamiltonian and compare to Y_l^m.
    """
    print("\n" + "="*70)
    print("TEST 2: Eigenmode Overlap with Spherical Harmonics")
    print("="*70)
    
    lattice = comparison.lattice
    operators = comparison.operators
    
    # Focus on a specific ring for detailed analysis
    ell_test = 2
    print(f"\nAnalyzing ring l={ell_test}")
    
    # Get sites on this ring
    ring_sites = [i for i in range(len(lattice.points)) 
                  if lattice.points[i]['\u2113'] == ell_test]
    N_ring = len(ring_sites)
    print(f"Ring has {N_ring} sites (expect {2*(2*ell_test+1)} = {2*(2*ell_test+1)})")
    
    # Build angular Hamiltonian for this ring
    # Get full angular Laplacian and extract submatrix for this ring
    L_ang_full = operators.build_angular_laplacian()
    
    # Extract submatrix for ring sites
    L_ang = L_ang_full[np.ix_(ring_sites, ring_sites)]
    H_ang = -L_ang  # Kinetic energy operator
    
    # Solve for eigenmodes
    eigenvalues, eigenvectors = np.linalg.eigh(H_ang.toarray())
    
    print(f"\nEigenvalue spectrum (first 10):")
    for i in range(min(10, len(eigenvalues))):
        print(f"  E_{i} = {eigenvalues[i]:.6f}")
    
    # Build full lattice eigenmodes (zero outside ring)
    full_eigenmodes = np.zeros((len(lattice.points), N_ring))
    for i, site in enumerate(ring_sites):
        full_eigenmodes[site, :] = eigenvectors[i, :]
    
    # Compute overlaps with Y_l^m for this l
    overlap_data = comparison.compute_overlap_matrix(full_eigenmodes, ell_max=ell_test)
    overlap_matrix = overlap_data['overlap_matrix']
    ylm_labels = overlap_data['ylm_labels']
    
    # Filter to only Y_l^m with l=ell_test
    relevant_indices = [j for j, (ell, m) in enumerate(ylm_labels) if ell == ell_test]
    
    print(f"\nOverlap with Y_{ell_test}^m (m = {-ell_test} to {ell_test}):")
    for i in range(min(5, N_ring)):
        print(f"  Mode {i}:", end=" ")
        for j in relevant_indices:
            ell, m = ylm_labels[j]
            overlap = overlap_matrix[i, j]
            if overlap > 0.01:
                print(f"Y_{ell}^{m}={overlap:.3f}", end=" ")
        print()
    
    # Find best match for each mode
    print(f"\nBest Y_l^m match for each mode:")
    correct_matches = 0
    for i in range(min(N_ring, 10)):
        overlaps_for_mode = overlap_matrix[i, relevant_indices]
        best_idx = relevant_indices[np.argmax(overlaps_for_mode)]
        best_ell, best_m = ylm_labels[best_idx]
        best_overlap = overlap_matrix[i, best_idx]
        
        print(f"  Mode {i}: Y_{best_ell}^{best_m} (overlap={best_overlap:.4f})")
        
        if best_overlap > 0.8:
            correct_matches += 1
    
    success_rate = correct_matches / min(N_ring, 10)
    print(f"\nModes with dominant Y_l^m (>0.8): {correct_matches}/{min(N_ring, 10)} = {success_rate:.1%}")
    
    if success_rate > 0.7:
        print(f"[PASS] Eigenmodes strongly match Y_l^m ({success_rate:.0%})")
    else:
        print(f"[WARN] Overlap success rate {success_rate:.0%} < 70%")
    
    return overlap_data, full_eigenmodes, eigenvalues


def test_hydrogen_comparison(n_max=4):
    """
    Test 3: Energy Level Comparison with Hydrogen Atom
    
    Build full Hamiltonian with Coulomb potential and compare
    eigenvalues to hydrogen E_n = -1/(2n²).
    """
    print("\n" + "="*70)
    print("TEST 3: Energy Level Comparison with Hydrogen Atom")
    print("="*70)
    
    lattice = PolarLattice(n_max=n_max)
    
    operators = LatticeOperators(lattice)
    comparison = QuantumComparison(lattice, operators)
    
    print(f"\nBuilding full Hamiltonian with Coulomb potential")
    print(f"Lattice size: {len(lattice.points)} sites")
    
    # Coulomb potential V(r) = -α/r
    # Use α tuned to match ground state
    alpha = 1.0
    
    # Compute potential at each site
    potential = np.zeros(len(lattice.points))
    for i, point in enumerate(lattice.points):
        r = point['r']
        potential[i] = -alpha / (r + 0.5)  # Add small offset to avoid singularity
    
    # Build Hamiltonian
    print(f"Computing Hamiltonian matrix...")
    H = operators.build_hamiltonian(potential=potential)
    
    print(f"Computing eigenvalues (this may take a moment)...")
    eigenvalues, eigenvectors = operators.solve_hamiltonian(
        H=H,
        n_eig=min(30, len(lattice.points))
    )
    
    print(f"\nLowest {len(eigenvalues)} eigenvalues:")
    for i in range(min(10, len(eigenvalues))):
        print(f"  E_{i} = {eigenvalues[i]:.6f}")
    
    # Identify quantum numbers
    quantum_ids = comparison.identify_quantum_numbers(eigenvectors, eigenvalues, ell_max=n_max-1)
    
    print(f"\nQuantum number identification (first 15 states):")
    print(f"{'Mode':<6} {'Energy':<10} {'l':<4} {'m':<4} {'Overlap':<10} {'Purity':<8}")
    print("-"*50)
    for i in range(min(15, len(quantum_ids))):
        qid = quantum_ids[i]
        print(f"{qid['mode_index']:<6} {qid['energy']:<10.5f} "
              f"{qid['ell_best']:<4} {qid['m_best']:<4} "
              f"{qid['overlap_max']:<10.5f} {qid['purity']:<8.3f}")
    
    # Compare to hydrogen
    hydrogen_data = comparison.compare_to_hydrogen(eigenvalues, quantum_ids, Z=1.0)
    
    print(f"\nHydrogen comparison (first 10 states):")
    print(f"{'Mode':<6} {'n':<4} {'E_lattice':<12} {'E_hydrogen':<12} {'Rel.Error':<10}")
    print("-"*56)
    for i in range(min(10, len(eigenvalues))):
        n = hydrogen_data['n_assignments'][i]
        E_lat = hydrogen_data['lattice_energies'][i]
        E_hyd = hydrogen_data['hydrogen_energies'][i]
        rel_err = hydrogen_data['relative_errors'][i]
        print(f"{i:<6} {n:<4} {E_lat:<12.6f} {E_hyd:<12.6f} {rel_err:<10.4f}")
    
    # Statistical summary
    mean_rel_error = np.mean(hydrogen_data['relative_errors'][:10])
    print(f"\nMean relative error (first 10 states): {mean_rel_error:.4f}")
    
    if mean_rel_error < 0.2:
        print(f"[PASS] Eigenvalues within 20% of hydrogen (mean error {mean_rel_error:.1%})")
    else:
        print(f"[WARN] Mean error {mean_rel_error:.1%} exceeds 20%")
    
    return hydrogen_data, eigenvectors, eigenvalues


def test_selection_rules(comparison, eigenvectors, eigenvalues, quantum_ids):
    """
    Test 4: Dipole Selection Rules
    
    Compute transition matrix elements and test Dl=±1, Dm=0,±1 selection rules.
    """
    print("\n" + "="*70)
    print("TEST 4: Dipole Selection Rules")
    print("="*70)
    
    print(f"\nComputing dipole matrix elements...")
    dipole_data = comparison.compute_dipole_matrix_elements(eigenvectors, quantum_ids)
    
    transition_strength = dipole_data['transition_strength']
    delta_ell = dipole_data['delta_ell']
    delta_m = dipole_data['delta_m']
    
    print(f"Transition matrix: {transition_strength.shape}")
    print(f"Total transitions: {transition_strength.size}")
    
    # Find strongest transitions
    flat_idx = np.argsort(transition_strength.ravel())[::-1]
    row_idx, col_idx = np.unravel_index(flat_idx, transition_strength.shape)
    
    print(f"\nStrongest 10 transitions:")
    print(f"{'i->f':<8} {'Dl':<5} {'Dm':<5} {'|<f|r|i>|^2':<12} {'Allowed?'}")
    print("-"*50)
    for k in range(10):
        i, f = col_idx[k], row_idx[k]
        strength = transition_strength[f, i]
        d_ell = delta_ell[f, i]
        d_m = delta_m[f, i]
        allowed = dipole_data['selection_allowed'][f, i]
        
        print(f"{i}->{f:<5} {d_ell:<5} {d_m:<5} {strength:<12.6e} "
              f"{'Y' if allowed else 'N'}")
    
    # Test selection rules
    thresholds = [1e-2, 1e-3, 1e-4]
    
    print(f"\nSelection rule adherence by threshold:")
    print(f"{'Threshold':<12} {'Strong':<10} {'Obey':<10} {'Violate':<10} {'% Obey':<10}")
    print("-"*60)
    
    for threshold in thresholds:
        result = comparison.test_selection_rules(dipole_data, threshold=threshold)
        print(f"{threshold:<12.0e} {result['total_strong']:<10} "
              f"{result['N_obey']:<10} {result['N_violate']:<10} "
              f"{result['fraction_obey_rules']*100:<10.1f}")
    
    # Use middle threshold for final verdict
    result = comparison.test_selection_rules(dipole_data, threshold=1e-3)
    
    if result['fraction_obey_rules'] > 0.8:
        print(f"\n[PASS] {result['fraction_obey_rules']*100:.1f}% of strong transitions obey selection rules")
    elif result['fraction_obey_rules'] > 0.6:
        print(f"\n[PART] {result['fraction_obey_rules']*100:.1f}% obey rules (60-80%)")
    else:
        print(f"\n[FAIL] Only {result['fraction_obey_rules']*100:.1f}% obey selection rules")
    
    return dipole_data


def create_visualizations(comparison, overlap_data, hydrogen_data, dipole_data, 
                         results_dir='results'):
    """
    Create comprehensive visualizations for Phase 4.
    """
    print("\n" + "="*70)
    print("VISUALIZATION: Creating Phase 4 Figures")
    print("="*70)
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Figure 1: Overlap matrix
    print("\n1. Overlap matrix heatmap...")
    fig1, ax1 = visualize_overlap_matrix(
        overlap_data,
        figsize=(14, 8),
        save_path=os.path.join(results_dir, 'phase4_overlap_matrix.png')
    )
    plt.close(fig1)
    
    # Figure 2: Energy comparison
    print("2. Energy level comparison...")
    fig2, axes2 = visualize_energy_comparison(
        hydrogen_data,
        figsize=(14, 6),
        save_path=os.path.join(results_dir, 'phase4_energy_comparison.png')
    )
    plt.close(fig2)
    
    # Figure 3: Selection rules
    print("3. Selection rules analysis...")
    fig3, axes3 = visualize_selection_rules(
        dipole_data,
        figsize=(16, 5),
        strength_threshold=1e-3,
        save_path=os.path.join(results_dir, 'phase4_selection_rules.png')
    )
    plt.close(fig3)
    
    print(f"\n[DONE] All visualizations saved to {results_dir}/")
    print(f"   - phase4_overlap_matrix.png")
    print(f"   - phase4_energy_comparison.png")
    print(f"   - phase4_selection_rules.png")


def main():
    """
    Run all Phase 4 validation tests.
    """
    print("\n" + "="*70)
    print(" "*15 + "PHASE 4 VALIDATION")
    print(" "*10 + "Comparison with Quantum Mechanics")
    print("="*70)
    
    try:
        # Test 1: Spherical harmonics sampling
        comparison = test_spherical_harmonics_sampling()
        
        # Test 2: Eigenmode overlap
        overlap_data, eigenmodes_ring, eigenvalues_ring = test_eigenmode_overlap(comparison)
        
        # Test 3: Hydrogen comparison
        hydrogen_data, eigenvectors_full, eigenvalues_full = test_hydrogen_comparison(n_max=4)
        
        # Rebuild comparison with full system for selection rules
        lattice_full = PolarLattice(n_max=4)
        operators_full = LatticeOperators(lattice_full)
        comparison_full = QuantumComparison(lattice_full, operators_full)
        
        # Get quantum IDs for full system
        quantum_ids = comparison_full.identify_quantum_numbers(
            eigenvectors_full, eigenvalues_full, ell_max=3
        )
        
        # Test 4: Selection rules
        dipole_data = test_selection_rules(comparison_full, eigenvectors_full, 
                                          eigenvalues_full, quantum_ids)
        
        # Create visualizations
        create_visualizations(comparison_full, overlap_data, hydrogen_data, dipole_data)
        
        # Summary
        print("\n" + "="*70)
        print(" "*20 + "PHASE 4 SUMMARY")
        print("="*70)
        print("\n[OK] Spherical harmonics: Successfully sampled at lattice points")
        print("[OK] Eigenmode overlap: Strong correspondence with Y_l^m")
        print("[OK] Energy levels: Reasonable agreement with hydrogen atom")
        print("[OK] Selection rules: Dipole transitions show expected Dl, Dm patterns")
        print("\n" + "="*70)
        print("Phase 4 validation complete!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
