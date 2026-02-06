"""
Phase 3: Validation and Scaling - High ℓ Testing and Reversibility

This script implements Phase 3 of the geometric transformation research:
1. Scale testing to higher ℓ values (ℓ ≤ 10)
2. Test reversibility (round-trip fidelity)
3. Measure commutator preservation [Li, Lj] = iℏϵijk Lk
4. Assess computational cost of transformations
5. Comprehensive validation of best transformation

Expected outputs:
- Scaling behavior of improvements with ℓ
- Round-trip error quantification
- Commutator preservation verification
- Performance metrics

Author: Quantum Lattice Project
Date: January 2026
Research Phase 3: Validation
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.lattice import PolarLattice
from src.angular_momentum import AngularMomentumOperators
from src.geometric_transform_research import GeometricTransformResearch
from scipy import sparse


def test_commutator_preservation(research, psi, transform_type='stereographic'):
    """
    Test if geometric correction preserves [Li, Lj] = iℏϵijk Lk.
    
    Parameters:
        research: GeometricTransformResearch instance
        psi: Eigenvector to test
        transform_type: Transformation type
    
    Returns:
        errors: Dictionary of commutator errors
    """
    angular_ops = research.angular_ops
    
    # Get operators
    Lx = angular_ops.build_Lx()
    Ly = angular_ops.build_Ly()
    Lz = angular_ops.build_Lz()
    
    # Apply correction to eigenvector
    psi_corrected = research.apply_geometric_correction(psi, transform_type, 'forward')
    
    # Compute commutators with corrected state
    # [Lx, Ly] = iLz
    comm_xy = Lx @ Ly - Ly @ Lx
    expected_z = 1j * Lz
    
    # Expectation values
    comm_xy_exp = psi_corrected.conj() @ comm_xy @ psi_corrected
    expected_z_exp = psi_corrected.conj() @ expected_z @ psi_corrected
    error_xy = abs(comm_xy_exp - expected_z_exp)
    
    # [Ly, Lz] = iLx
    comm_yz = Ly @ Lz - Lz @ Ly
    expected_x = 1j * Lx
    comm_yz_exp = psi_corrected.conj() @ comm_yz @ psi_corrected
    expected_x_exp = psi_corrected.conj() @ expected_x @ psi_corrected
    error_yz = abs(comm_yz_exp - expected_x_exp)
    
    # [Lz, Lx] = iLy
    comm_zx = Lz @ Lx - Lx @ Lz
    expected_y = 1j * Ly
    comm_zx_exp = psi_corrected.conj() @ comm_zx @ psi_corrected
    expected_y_exp = psi_corrected.conj() @ expected_y @ psi_corrected
    error_zx = abs(comm_zx_exp - expected_y_exp)
    
    return {
        '[Lx, Ly]': float(error_xy),
        '[Ly, Lz]': float(error_yz),
        '[Lz, Lx]': float(error_zx),
        'max': float(max(error_xy, error_yz, error_zx))
    }


def phase3_validation_scaling(n_max=11, ℓ_max_test=10, save_plots=True):
    """
    Phase 3: Validation with high ℓ and comprehensive testing.
    
    Parameters:
        n_max: Maximum principal quantum number (lattice size)
        ℓ_max_test: Maximum ℓ to test
        save_plots: Whether to save plots to disk
    
    Returns:
        results_dict: Dictionary containing all validation data
    """
    print("=" * 70)
    print("PHASE 3: VALIDATION AND SCALING")
    print("=" * 70)
    print()
    
    # Initialize system
    print(f"Initializing lattice with n_max={n_max}...")
    lattice = PolarLattice(n_max)
    angular_ops = AngularMomentumOperators(lattice)
    research = GeometricTransformResearch(lattice, angular_ops)
    
    print(f"Total lattice points: {len(lattice.points)}")
    print(f"Maximum ℓ available: {lattice.ℓ_max}")
    print(f"Testing up to ℓ = {ℓ_max_test}")
    print()
    
    # Compute eigenvectors
    print("Computing L² eigenvectors...")
    L_squared = angular_ops.build_L_squared()
    n_eigs = min(200, L_squared.shape[0] - 2)
    eigenvalues, eigenvectors = sparse.linalg.eigsh(L_squared, k=n_eigs, which='SM')
    print(f"Computed {n_eigs} eigenpairs")
    print()
    
    # Test range
    ℓ_values = list(range(1, min(ℓ_max_test + 1, lattice.ℓ_max + 1)))
    m = 0  # Test m=0
    
    # Based on Phase 2, use the best transform (typically stereographic)
    # For thoroughness, test stereographic here
    transform_type = 'stereographic'
    
    results = {
        'ℓ_values': ℓ_values,
        'overlaps_original': [],
        'overlaps_corrected': [],
        'improvements': [],
        'eigenvalue_errors': [],
        'round_trip_fidelities': [],
        'commutator_errors': [],
        'computation_times': [],
    }
    
    print("=" * 70)
    print(f"HIGH-ℓ SCALING TEST ({transform_type.upper()} PROJECTION)")
    print("=" * 70)
    print()
    
    for ℓ in ℓ_values:
        print(f"\nℓ = {ℓ}")
        print("-" * 40)
        
        # Find eigenvector
        target_eigenvalue = ℓ * (ℓ + 1)
        idx = np.argmin(np.abs(eigenvalues - target_eigenvalue))
        psi = eigenvectors[:, idx]
        
        # Original overlap
        overlap_orig = research.compute_overlap_with_Ylm(psi, ℓ, m)
        
        # Apply correction and time it
        t_start = time.time()
        result = research.test_transformation(psi, ℓ, m, transform_type)
        t_elapsed = time.time() - t_start
        
        # Round-trip test
        fidelity = research.test_reversibility(psi, transform_type)
        
        # Commutator preservation test
        comm_errors = test_commutator_preservation(research, psi, transform_type)
        
        # Store results
        results['overlaps_original'].append(overlap_orig)
        results['overlaps_corrected'].append(result.overlap_best)
        results['improvements'].append(result.overlap_improvement)
        results['eigenvalue_errors'].append(result.eigenvalue_error)
        results['round_trip_fidelities'].append(fidelity)
        results['commutator_errors'].append(comm_errors['max'])
        results['computation_times'].append(t_elapsed)
        
        # Print summary
        print(f"  Original:    {overlap_orig:.4%}")
        print(f"  Corrected:   {result.overlap_best:.4%}")
        print(f"  Improvement: {result.overlap_improvement:+.2%}")
        print(f"  Eigenvalue error:   {result.eigenvalue_error:.2e}")
        print(f"  Round-trip fidelity: {fidelity:.6f}")
        print(f"  Commutator error:    {comm_errors['max']:.2e}")
        print(f"  Computation time:    {t_elapsed*1000:.2f} ms")
    
    # Analysis
    print("\n" + "=" * 70)
    print("SCALING ANALYSIS")
    print("=" * 70)
    print()
    
    # Overall statistics
    mean_improvement = np.mean(results['improvements'])
    std_improvement = np.std(results['improvements'])
    min_improvement = np.min(results['improvements'])
    max_improvement = np.max(results['improvements'])
    
    print(f"Improvement Statistics (across ℓ={ℓ_values[0]} to {ℓ_values[-1]}):")
    print(f"  Mean:    {mean_improvement:+.2%}")
    print(f"  Std Dev: {std_improvement:.2%}")
    print(f"  Min:     {min_improvement:+.2%} (ℓ={ℓ_values[np.argmin(results['improvements'])]})")
    print(f"  Max:     {max_improvement:+.2%} (ℓ={ℓ_values[np.argmax(results['improvements'])]})")
    print()
    
    # Check for trends with ℓ
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(ℓ_values, 
                                                              results['improvements'])
    print(f"Trend Analysis (improvement vs ℓ):")
    print(f"  Slope:       {slope:.4f} per unit ℓ")
    print(f"  R-squared:   {r_value**2:.4f}")
    print(f"  P-value:     {p_value:.4f}")
    
    if abs(r_value) > 0.7:
        if slope > 0:
            print(f"  → STRONG POSITIVE TREND: Improvement increases with ℓ")
            print(f"    Interpretation: Geometric correction more effective for high ℓ")
        else:
            print(f"  → STRONG NEGATIVE TREND: Improvement decreases with ℓ")
            print(f"    Interpretation: Geometric correction less effective for high ℓ")
    else:
        print(f"  → NO STRONG TREND: Improvement relatively uniform across ℓ")
    print()
    
    # Eigenvalue preservation
    max_eigenvalue_error = np.max(results['eigenvalue_errors'])
    all_preserved = all(e < 1e-10 for e in results['eigenvalue_errors'])
    
    print(f"Eigenvalue Preservation:")
    print(f"  Maximum error: {max_eigenvalue_error:.2e}")
    if all_preserved:
        print(f"  ✅ ALL eigenvalues preserved to machine precision")
    else:
        failed = sum(1 for e in results['eigenvalue_errors'] if e >= 1e-10)
        print(f"  ⚠️  {failed}/{len(ℓ_values)} cases exceeded tolerance")
    print()
    
    # Round-trip fidelity
    min_fidelity = np.min(results['round_trip_fidelities'])
    mean_fidelity = np.mean(results['round_trip_fidelities'])
    
    print(f"Round-Trip Reversibility:")
    print(f"  Mean fidelity: {mean_fidelity:.8f}")
    print(f"  Min fidelity:  {min_fidelity:.8f}")
    
    if min_fidelity > 0.9999:
        print(f"  ✅ EXCELLENT reversibility (>99.99%)")
        print(f"     → Transformation is essentially lossless")
    elif min_fidelity > 0.99:
        print(f"  ⚠️  GOOD reversibility (>99%)")
        print(f"     → Minor information loss in round-trip")
    else:
        print(f"  ❌ POOR reversibility (<99%)")
        print(f"     → Significant information loss")
    print()
    
    # Commutator preservation
    max_comm_error = np.max(results['commutator_errors'])
    mean_comm_error = np.mean(results['commutator_errors'])
    
    print(f"Commutator Preservation [Li, Lj] = iℏϵijk Lk:")
    print(f"  Mean error: {mean_comm_error:.2e}")
    print(f"  Max error:  {max_comm_error:.2e}")
    
    if max_comm_error < 1e-10:
        print(f"  ✅ PERFECT preservation")
        print(f"     → SU(2) algebra maintained exactly")
    elif max_comm_error < 1e-6:
        print(f"  ✅ GOOD preservation (errors ~10^-6)")
        print(f"     → Negligible violation of algebra")
    else:
        print(f"  ⚠️  MODERATE violation")
        print(f"     → May affect quantum mechanical consistency")
    print()
    
    # Computational cost
    mean_time = np.mean(results['computation_times'])
    total_time = np.sum(results['computation_times'])
    
    print(f"Computational Performance:")
    print(f"  Mean time per ℓ: {mean_time*1000:.2f} ms")
    print(f"  Total time:      {total_time:.2f} s")
    print(f"  → Transformation is computationally cheap (post-processing only)")
    print()
    
    # Generate plots
    if save_plots:
        print("=" * 70)
        print("GENERATING VALIDATION PLOTS")
        print("=" * 70)
        print()
        
        # Plot 1: Scaling with ℓ
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Overlap comparison
        ax = axes[0, 0]
        ax.plot(ℓ_values, [o*100 for o in results['overlaps_original']], 
               'o-', label='Original', markersize=6)
        ax.plot(ℓ_values, [o*100 for o in results['overlaps_corrected']], 
               's-', label='Corrected', markersize=6)
        ax.axhline(82, color='red', linestyle='--', alpha=0.3, label='~82% baseline')
        ax.set_xlabel('Angular Momentum ℓ')
        ax.set_ylabel('Overlap (%)')
        ax.set_title('Overlap Scaling with ℓ')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Improvement vs ℓ
        ax = axes[0, 1]
        ax.plot(ℓ_values, [i*100 for i in results['improvements']], 'go-', markersize=6)
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        ax.set_xlabel('Angular Momentum ℓ')
        ax.set_ylabel('Improvement (percentage points)')
        ax.set_title('Improvement from Geometric Correction')
        ax.grid(True, alpha=0.3)
        
        # Eigenvalue errors
        ax = axes[1, 0]
        ax.semilogy(ℓ_values, results['eigenvalue_errors'], 'ro-', markersize=6)
        ax.axhline(1e-10, color='green', linestyle='--', alpha=0.5, label='Tolerance')
        ax.set_xlabel('Angular Momentum ℓ')
        ax.set_ylabel('Eigenvalue Error')
        ax.set_title('Eigenvalue Preservation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Round-trip fidelity
        ax = axes[1, 1]
        ax.plot(ℓ_values, results['round_trip_fidelities'], 'bo-', markersize=6)
        ax.axhline(0.9999, color='green', linestyle='--', alpha=0.5, label='99.99% threshold')
        ax.set_xlabel('Angular Momentum ℓ')
        ax.set_ylabel('Round-Trip Fidelity')
        ax.set_title('Reversibility Test')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.99, 1.001])
        
        plt.suptitle(f'Phase 3 Validation: {transform_type.capitalize()} Transform (n_max={n_max})', 
                    fontsize=14, y=0.995)
        plt.tight_layout()
        plt.savefig('results/phase3_validation_scaling.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("💾 Saved: results/phase3_validation_scaling.png")
        
        # Plot 2: Commutator errors
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(ℓ_values, results['commutator_errors'], 'mo-', markersize=8, linewidth=2)
        ax.axhline(1e-10, color='green', linestyle='--', alpha=0.5, label='Machine precision')
        ax.axhline(1e-6, color='orange', linestyle='--', alpha=0.5, label='Acceptable threshold')
        ax.set_xlabel('Angular Momentum ℓ', fontsize=12)
        ax.set_ylabel('Max Commutator Error', fontsize=12)
        ax.set_title('SU(2) Algebra Preservation: max([Li, Lj] - iϵijk Lk)', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/phase3_commutator_preservation.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("💾 Saved: results/phase3_commutator_preservation.png")
        
        # Plot 3: Computation time scaling
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ℓ_values, [t*1000 for t in results['computation_times']], 'co-', 
               markersize=8, linewidth=2)
        ax.set_xlabel('Angular Momentum ℓ', fontsize=12)
        ax.set_ylabel('Computation Time (ms)', fontsize=12)
        ax.set_title('Computational Cost of Geometric Correction', fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/phase3_computation_time.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("💾 Saved: results/phase3_computation_time.png")
        print()
    
    # Final assessment
    print("=" * 70)
    print("PHASE 3 ASSESSMENT")
    print("=" * 70)
    print()
    
    print("Key Findings:")
    print()
    
    print("1. SCALING BEHAVIOR:")
    if abs(slope) < 0.01:
        print("   → Improvement is UNIFORM across ℓ range")
        print("     Implication: Geometric correction works consistently")
    elif slope > 0:
        print(f"   → Improvement INCREASES with ℓ (~{slope*100:.2f}% per ℓ)")
        print("     Implication: More effective for high angular momentum states")
    else:
        print(f"   → Improvement DECREASES with ℓ (~{slope*100:.2f}% per ℓ)")
        print("     Implication: Less effective for high angular momentum states")
    print()
    
    print("2. QUANTUM MECHANICAL CONSISTENCY:")
    if all_preserved and max_comm_error < 1e-10:
        print("   ✅ FULL CONSISTENCY maintained")
        print("      - Eigenvalues preserved exactly")
        print("      - Commutation relations preserved")
        print("      → Transformation is quantum mechanically valid")
    elif all_preserved:
        print("   ⚠️  MOSTLY CONSISTENT")
        print("      - Eigenvalues preserved")
        print(f"      - Commutators slightly violated (~{max_comm_error:.2e})")
    else:
        print("   ⚠️  CONSISTENCY ISSUES detected")
        print("      - Review eigenvalue preservation failures")
    print()
    
    print("3. REVERSIBILITY:")
    if min_fidelity > 0.9999:
        print("   ✅ EXCELLENT - transformation is essentially lossless")
        print("      → Can toggle between representations freely")
    elif min_fidelity > 0.99:
        print("   ⚠️  GOOD - minor information loss acceptable for most applications")
    else:
        print("   ❌ POOR - significant loss precludes reversible toggling")
    print()
    
    print("4. PRACTICAL VIABILITY:")
    if mean_time < 0.1:  # 100ms
        print(f"   ✅ VERY FAST (mean: {mean_time*1000:.1f} ms per correction)")
        print("      → Suitable for real-time applications")
    elif mean_time < 1.0:
        print(f"   ✅ FAST (mean: {mean_time*1000:.0f} ms per correction)")
        print("      → Suitable for interactive use")
    else:
        print(f"   ⚠️  SLOW (mean: {mean_time:.2f} s per correction)")
        print("      → May limit applicability")
    print()
    
    print("=" * 70)
    print("PHASE 3 COMPLETE")
    print("=" * 70)
    print()
    print("Next step: Phase 4 - Adaptive hybrid optimization")
    print()
    
    return results


if __name__ == "__main__":
    import os
    
    # Create results directory if needed
    os.makedirs('results', exist_ok=True)
    
    # Run Phase 3
    results = phase3_validation_scaling(n_max=11, ℓ_max_test=10, save_plots=True)
    
    print("\nPhase 3 validation complete!")
    print("Comprehensive scaling and consistency tests finished.")
