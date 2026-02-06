"""
Phase 1: Diagnostic Analysis - Quantify Geometric Distortion

This script implements the first phase of the geometric transformation research:
1. Compute eigenvectors for ℓ = 1, 2, 3, 4, 5
2. Calculate overlap with continuous Y_ℓm
3. Analyze spatial distribution of errors
4. Generate heatmaps and identify systematic patterns

Expected outputs:
- Overlap percentages before any correction
- Error heatmaps showing where eigenvectors deviate
- Regional error analysis (poles vs equator, inner vs outer shells)
- Diagnostic plots for publication

Author: Quantum Lattice Project
Date: January 2026
Research Phase 1: Diagnostic
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.lattice import PolarLattice
from src.angular_momentum import AngularMomentumOperators
from src.geometric_transform_research import GeometricTransformResearch
from scipy import sparse


def phase1_diagnostic_analysis(n_max=6, save_plots=True):
    """
    Phase 1: Comprehensive diagnostic analysis of eigenvector-spherical harmonic mismatch.
    
    Parameters:
        n_max: Maximum principal quantum number (lattice size)
        save_plots: Whether to save plots to disk
    
    Returns:
        results_dict: Dictionary containing all diagnostic data
    """
    print("=" * 70)
    print("PHASE 1: DIAGNOSTIC ANALYSIS - QUANTIFYING GEOMETRIC DISTORTION")
    print("=" * 70)
    print()
    
    # Initialize system
    print(f"Initializing lattice with n_max={n_max}...")
    lattice = PolarLattice(n_max)
    angular_ops = AngularMomentumOperators(lattice)
    research = GeometricTransformResearch(lattice, angular_ops)
    
    print(f"Total lattice points: {len(lattice.points)}")
    print(f"Maximum ℓ: {lattice.ℓ_max}")
    print()
    
    # Compute eigenvectors
    print("Computing L² eigenvectors...")
    L_squared = angular_ops.build_L_squared()
    n_eigs = min(100, L_squared.shape[0] - 2)
    eigenvalues, eigenvectors = sparse.linalg.eigsh(L_squared, k=n_eigs, which='SM')
    print(f"Computed {n_eigs} eigenpairs")
    print()
    
    # Test ℓ values
    ℓ_values = [1, 2, 3, 4, 5]
    m = 0  # Test m=0 for simplicity (can extend to all m)
    
    results = {
        'ℓ_values': ℓ_values,
        'overlaps': [],
        'eigenvalue_errors': [],
        'regional_errors': [],
        'eigenvectors': {},
    }
    
    print("=" * 70)
    print("BASELINE OVERLAP MEASUREMENTS (No Geometric Correction)")
    print("=" * 70)
    print()
    
    for ℓ in ℓ_values:
        print(f"\n{'-' * 60}")
        print(f"ℓ = {ℓ}, m = {m}")
        print(f"{'-' * 60}")
        
        # Find eigenvector
        target_eigenvalue = ℓ * (ℓ + 1)
        idx = np.argmin(np.abs(eigenvalues - target_eigenvalue))
        actual_eigenvalue = eigenvalues[idx]
        psi = eigenvectors[:, idx]
        
        print(f"Target eigenvalue:  {target_eigenvalue}")
        print(f"Actual eigenvalue:  {actual_eigenvalue:.10f}")
        print(f"Error:              {abs(actual_eigenvalue - target_eigenvalue):.2e}")
        print()
        
        # Compute overlap with Y_ℓm
        overlap = research.compute_overlap_with_Ylm(psi, ℓ, m)
        print(f"Overlap |⟨ψ|Y_{ℓ}^{m}⟩|² = {overlap:.4%}")
        print(f"Deficit:              {(1 - overlap):.4%}")
        print()
        
        # Regional error analysis
        regional_errors = research.analyze_error_by_region(psi, ℓ, m)
        print("Regional Error Analysis:")
        print(f"  Total mean error:      {regional_errors['total_mean']:.6f}")
        print(f"  Total max error:       {regional_errors['total_max']:.6f}")
        print(f"  Inner shells mean:     {regional_errors['inner_shells_mean']:.6f}")
        print(f"  Outer shells mean:     {regional_errors['outer_shells_mean']:.6f}")
        print(f"  Polar region mean:     {regional_errors['polar_region_mean']:.6f}")
        print(f"  Equator region mean:   {regional_errors['equator_region_mean']:.6f}")
        
        # Check for systematic patterns
        if regional_errors['polar_region_mean'] > 1.5 * regional_errors['equator_region_mean']:
            print("\n  ⚠️  PATTERN: Error concentrated at poles (systematic geometric distortion?)")
        elif regional_errors['equator_region_mean'] > 1.5 * regional_errors['polar_region_mean']:
            print("\n  ⚠️  PATTERN: Error concentrated at equator (systematic geometric distortion?)")
        else:
            print("\n  ℹ️  PATTERN: Error relatively uniform (may be fundamental discretization)")
        
        # Store results
        results['overlaps'].append(overlap)
        results['eigenvalue_errors'].append(abs(actual_eigenvalue - target_eigenvalue))
        results['regional_errors'].append(regional_errors)
        results['eigenvectors'][ℓ] = psi
        
        # Generate error heatmap
        if save_plots:
            fig = research.plot_error_heatmap(psi, ℓ, m, 
                                             title=f"Error Distribution: ℓ={ℓ}, m={m}")
            plt.savefig(f'results/phase1_error_heatmap_ell{ℓ}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"\n  💾 Saved: results/phase1_error_heatmap_ell{ℓ}.png")
    
    # Summary plots
    print(f"\n{'=' * 70}")
    print("SUMMARY VISUALIZATIONS")
    print(f"{'=' * 70}\n")
    
    # Plot 1: Overlap vs ℓ
    fig, ax = plt.subplots(figsize=(10, 6))
    overlaps_pct = [o * 100 for o in results['overlaps']]
    ax.plot(ℓ_values, overlaps_pct, 'o-', markersize=10, linewidth=2, label='No correction')
    ax.axhline(82, color='red', linestyle='--', alpha=0.5, label='~82% baseline (literature)')
    ax.axhline(100, color='green', linestyle='--', alpha=0.5, label='Perfect match')
    ax.set_xlabel('Angular Momentum ℓ', fontsize=12)
    ax.set_ylabel('Overlap |⟨ψ|Y_ℓm⟩|² (%)', fontsize=12)
    ax.set_title('Baseline Eigenvector Overlap with Spherical Harmonics\n(Before Geometric Correction)', 
                 fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([70, 105])
    
    if save_plots:
        plt.savefig('results/phase1_overlap_summary.png', dpi=150, bbox_inches='tight')
        print("💾 Saved: results/phase1_overlap_summary.png")
    plt.close(fig)
    
    # Plot 2: Regional error comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(ℓ_values))
    width = 0.35
    
    inner_means = [r['inner_shells_mean'] for r in results['regional_errors']]
    outer_means = [r['outer_shells_mean'] for r in results['regional_errors']]
    
    ax.bar(x - width/2, inner_means, width, label='Inner shells', alpha=0.8)
    ax.bar(x + width/2, outer_means, width, label='Outer shells', alpha=0.8)
    
    ax.set_xlabel('Angular Momentum ℓ', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title('Error Distribution: Inner vs Outer Shells', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(ℓ_values)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    if save_plots:
        plt.savefig('results/phase1_regional_errors.png', dpi=150, bbox_inches='tight')
        print("💾 Saved: results/phase1_regional_errors.png")
    plt.close(fig)
    
    # Plot 3: Polar vs Equator comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    polar_means = [r['polar_region_mean'] for r in results['regional_errors']]
    equator_means = [r['equator_region_mean'] for r in results['regional_errors']]
    
    ax.bar(x - width/2, polar_means, width, label='Polar regions (θ < π/4 or θ > 3π/4)', alpha=0.8)
    ax.bar(x + width/2, equator_means, width, label='Equatorial region', alpha=0.8)
    
    ax.set_xlabel('Angular Momentum ℓ', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title('Error Distribution: Poles vs Equator\n(Indicates Coordinate Distortion Pattern)', 
                 fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(ℓ_values)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    if save_plots:
        plt.savefig('results/phase1_polar_equator_errors.png', dpi=150, bbox_inches='tight')
        print("💾 Saved: results/phase1_polar_equator_errors.png")
    plt.close(fig)
    
    # Print summary statistics
    print(f"\n{'=' * 70}")
    print("PHASE 1 SUMMARY STATISTICS")
    print(f"{'=' * 70}\n")
    
    mean_overlap = np.mean(results['overlaps'])
    std_overlap = np.std(results['overlaps'])
    min_overlap = np.min(results['overlaps'])
    max_overlap = np.max(results['overlaps'])
    
    print(f"Overlap Statistics:")
    print(f"  Mean:    {mean_overlap:.4%}")
    print(f"  Std Dev: {std_overlap:.4%}")
    print(f"  Min:     {min_overlap:.4%} (ℓ={ℓ_values[np.argmin(results['overlaps'])]})")
    print(f"  Max:     {max_overlap:.4%} (ℓ={ℓ_values[np.argmax(results['overlaps'])]})")
    print(f"\n  Average deficit: {(1 - mean_overlap):.4%}")
    print()
    
    # Determine if error is systematic
    polar_equator_ratios = [r['polar_region_mean'] / (r['equator_region_mean'] + 1e-10) 
                            for r in results['regional_errors']]
    mean_ratio = np.mean(polar_equator_ratios)
    
    print(f"Spatial Pattern Analysis:")
    print(f"  Mean polar/equator error ratio: {mean_ratio:.3f}")
    if mean_ratio > 1.3:
        print(f"  🔍 CONCLUSION: Errors are systematically concentrated at POLES")
        print(f"     → Suggests geometric distortion (stereographic-like?)")
    elif mean_ratio < 0.7:
        print(f"  🔍 CONCLUSION: Errors are systematically concentrated at EQUATOR")
        print(f"     → Suggests geometric distortion (Mercator-like?)")
    else:
        print(f"  🔍 CONCLUSION: Errors are relatively UNIFORM")
        print(f"     → May indicate fundamental discretization, not pure geometry")
    print()
    
    print("=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)
    print("\nNext step: Phase 2 - Test conformal transformations to correct distortion")
    print()
    
    return results


if __name__ == "__main__":
    import os
    
    # Create results directory if needed
    os.makedirs('results', exist_ok=True)
    
    # Run Phase 1
    results = phase1_diagnostic_analysis(n_max=6, save_plots=True)
    
    print("\nPhase 1 diagnostic analysis complete!")
    print("Review the plots in results/ directory for spatial error patterns.")
