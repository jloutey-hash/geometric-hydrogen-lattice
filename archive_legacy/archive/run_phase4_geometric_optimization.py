"""
Phase 4: Adaptive Hybrid Optimization - Finding Optimal λ Parameters

This script implements Phase 4 of the geometric transformation research:
1. Build adaptive hybrid transformation with parameter λ ∈ [0, 1]
2. Optimize λ for each ℓ to maximize overlap while preserving eigenvalues
3. Analyze whether optimal λ varies systematically with ℓ
4. Compare hybrid approach vs full correction
5. Generate optimization curves and recommendations

Expected outputs:
- Optimal λ values for each ℓ
- Overlap improvements from optimization
- Analysis of λ(ℓ) relationship
- Practical recommendations for parameter choice

Author: Quantum Lattice Project
Date: January 2026
Research Phase 4: Adaptive Optimization
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


def phase4_adaptive_hybrid_optimization(n_max=8, ℓ_values=None, save_plots=True):
    """
    Phase 4: Optimize hybrid transformation parameter for each ℓ.
    
    Parameters:
        n_max: Maximum principal quantum number (lattice size)
        ℓ_values: List of ℓ values to optimize (default: 1-8)
        save_plots: Whether to save plots to disk
    
    Returns:
        results_dict: Dictionary containing optimization results
    """
    if ℓ_values is None:
        ℓ_values = list(range(1, 9))
    
    print("=" * 70)
    print("PHASE 4: ADAPTIVE HYBRID OPTIMIZATION")
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
    n_eigs = min(150, L_squared.shape[0] - 2)
    eigenvalues, eigenvectors = sparse.linalg.eigsh(L_squared, k=n_eigs, which='SM')
    print(f"Computed {n_eigs} eigenpairs")
    print()
    
    # Use best transform from Phase 2 (stereographic)
    transform_type = 'stereographic'
    m = 0
    
    results = {
        'ℓ_values': ℓ_values,
        'lambda_optimal': [],
        'overlap_uncorrected': [],
        'overlap_full_correction': [],
        'overlap_optimized': [],
        'improvement_full': [],
        'improvement_optimized': [],
        'eigenvalue_errors_optimized': [],
        'lambda_curves': {},  # Will store full optimization curves
    }
    
    print("=" * 70)
    print(f"OPTIMIZING HYBRID PARAMETER λ ({transform_type.upper()} PROJECTION)")
    print("=" * 70)
    print()
    print("For each ℓ, finding λ* that maximizes overlap while preserving eigenvalue")
    print("  λ = 0: pure flat lattice (no correction)")
    print("  λ = 1: full geometric correction")
    print()
    
    for ℓ in ℓ_values:
        print(f"\n{'-' * 60}")
        print(f"ℓ = {ℓ}")
        print(f"{'-' * 60}")
        
        # Find eigenvector
        target_eigenvalue = ℓ * (ℓ + 1)
        idx = np.argmin(np.abs(eigenvalues - target_eigenvalue))
        psi = eigenvectors[:, idx]
        
        # Uncorrected overlap
        overlap_uncorr = research.compute_overlap_with_Ylm(psi, ℓ, m)
        
        # Full correction (λ=1)
        psi_full = research.hybrid_transform(psi, 1.0, transform_type)
        overlap_full = research.compute_overlap_with_Ylm(psi_full, ℓ, m)
        
        # Optimize λ
        print(f"  Optimizing λ...")
        lambda_optimal, overlap_optimal = research.optimize_lambda(psi, ℓ, m, 
                                                                    transform_type)
        
        # Verify eigenvalue preservation at optimal λ
        psi_optimal = research.hybrid_transform(psi, lambda_optimal, transform_type)
        eigenvalue_error, _ = research.verify_eigenvalue_preservation(psi_optimal, ℓ)
        
        # Compute full optimization curve for visualization
        lambda_values = np.linspace(0, 1, 51)
        overlap_curve = []
        eigenvalue_curve = []
        
        for lam in lambda_values:
            psi_test = research.hybrid_transform(psi, lam, transform_type)
            overlap = research.compute_overlap_with_Ylm(psi_test, ℓ, m)
            ev_error, _ = research.verify_eigenvalue_preservation(psi_test, ℓ)
            overlap_curve.append(overlap)
            eigenvalue_curve.append(ev_error)
        
        results['lambda_curves'][ℓ] = {
            'lambda': lambda_values,
            'overlap': overlap_curve,
            'eigenvalue_error': eigenvalue_curve
        }
        
        # Store results
        results['lambda_optimal'].append(lambda_optimal)
        results['overlap_uncorrected'].append(overlap_uncorr)
        results['overlap_full_correction'].append(overlap_full)
        results['overlap_optimized'].append(overlap_optimal)
        results['improvement_full'].append(overlap_full - overlap_uncorr)
        results['improvement_optimized'].append(overlap_optimal - overlap_uncorr)
        results['eigenvalue_errors_optimized'].append(eigenvalue_error)
        
        # Print summary
        print(f"  Uncorrected (λ=0):      {overlap_uncorr:.4%}")
        print(f"  Full correction (λ=1):  {overlap_full:.4%}  (Δ = {(overlap_full-overlap_uncorr):+.2%})")
        print(f"  Optimized (λ={lambda_optimal:.3f}): {overlap_optimal:.4%}  (Δ = {(overlap_optimal-overlap_uncorr):+.2%})")
        print(f"  Eigenvalue error at λ*: {eigenvalue_error:.2e}")
        
        # Compare optimal vs full
        if abs(lambda_optimal - 1.0) < 0.01:
            print(f"  → Optimal λ ≈ 1: Full correction is best")
        elif abs(lambda_optimal - 0.0) < 0.01:
            print(f"  → Optimal λ ≈ 0: No correction is best (unexpected!)")
        else:
            print(f"  → Optimal λ = {lambda_optimal:.3f}: Partial correction optimal")
    
    # Analysis
    print("\n" + "=" * 70)
    print("OPTIMIZATION ANALYSIS")
    print("=" * 70)
    print()
    
    # Statistics on λ
    lambda_array = np.array(results['lambda_optimal'])
    mean_lambda = np.mean(lambda_array)
    std_lambda = np.std(lambda_array)
    min_lambda = np.min(lambda_array)
    max_lambda = np.max(lambda_array)
    
    print(f"Optimal λ Statistics:")
    print(f"  Mean:    {mean_lambda:.3f}")
    print(f"  Std Dev: {std_lambda:.3f}")
    print(f"  Min:     {min_lambda:.3f} (ℓ={ℓ_values[np.argmin(lambda_array)]})")
    print(f"  Max:     {max_lambda:.3f} (ℓ={ℓ_values[np.argmax(lambda_array)]})")
    print()
    
    # Check for λ(ℓ) relationship
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(ℓ_values, lambda_array)
    
    print(f"λ(ℓ) Trend Analysis:")
    print(f"  Slope:     {slope:.4f} per unit ℓ")
    print(f"  R-squared: {r_value**2:.4f}")
    print(f"  P-value:   {p_value:.4f}")
    print()
    
    if abs(r_value) > 0.7:
        if slope > 0:
            print(f"  → STRONG POSITIVE TREND: Optimal λ increases with ℓ")
            print(f"    Interpretation: High-ℓ states benefit from more correction")
        else:
            print(f"  → STRONG NEGATIVE TREND: Optimal λ decreases with ℓ")
            print(f"    Interpretation: Low-ℓ states benefit from more correction")
    else:
        print(f"  → NO STRONG TREND: Optimal λ relatively constant")
        print(f"    Interpretation: Universal λ ≈ {mean_lambda:.2f} works for all ℓ")
    print()
    
    # Compare optimized vs full correction
    improvements_full = np.array(results['improvement_full'])
    improvements_opt = np.array(results['improvement_optimized'])
    
    opt_better_count = np.sum(improvements_opt > improvements_full)
    full_better_count = len(ℓ_values) - opt_better_count
    mean_advantage = np.mean(improvements_opt - improvements_full)
    
    print(f"Optimized vs Full Correction:")
    print(f"  Cases where optimized better:  {opt_better_count}/{len(ℓ_values)}")
    print(f"  Cases where full better:       {full_better_count}/{len(ℓ_values)}")
    print(f"  Mean advantage of optimization: {mean_advantage:+.3%}")
    print()
    
    if abs(mean_advantage) < 0.001:  # < 0.1%
        print(f"  → Optimization provides NO significant advantage")
        print(f"    Recommendation: Use full correction (λ=1) for simplicity")
    elif mean_advantage > 0.01:  # > 1%
        print(f"  → Optimization provides SUBSTANTIAL advantage")
        print(f"    Recommendation: Use ℓ-dependent λ(ℓ) for best results")
    else:
        print(f"  → Optimization provides MINOR advantage")
        print(f"    Recommendation: Use λ ≈ {mean_lambda:.2f} as compromise")
    print()
    
    # Eigenvalue preservation check
    all_preserved = all(e < 1e-10 for e in results['eigenvalue_errors_optimized'])
    max_error = np.max(results['eigenvalue_errors_optimized'])
    
    print(f"Eigenvalue Preservation at Optimal λ:")
    print(f"  Maximum error: {max_error:.2e}")
    if all_preserved:
        print(f"  ✅ All eigenvalues preserved (< 10^-10)")
    else:
        print(f"  ⚠️  Some violations detected")
    print()
    
    # Generate plots
    if save_plots:
        print("=" * 70)
        print("GENERATING OPTIMIZATION PLOTS")
        print("=" * 70)
        print()
        
        # Plot 1: λ(ℓ) relationship
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ℓ_values, lambda_array, 'o-', markersize=10, linewidth=2, color='purple')
        ax.axhline(mean_lambda, color='red', linestyle='--', alpha=0.5, 
                  label=f'Mean λ = {mean_lambda:.3f}')
        ax.axhline(1.0, color='green', linestyle='--', alpha=0.3, label='Full correction')
        ax.axhline(0.0, color='gray', linestyle='--', alpha=0.3, label='No correction')
        ax.set_xlabel('Angular Momentum ℓ', fontsize=12)
        ax.set_ylabel('Optimal λ', fontsize=12)
        ax.set_title('Optimal Hybrid Parameter vs Angular Momentum', fontsize=14)
        ax.set_ylim([-0.05, 1.05])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/phase4_lambda_vs_ell.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("💾 Saved: results/phase4_lambda_vs_ell.png")
        
        # Plot 2: Optimization curves for selected ℓ
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        plot_ells = ℓ_values[:6] if len(ℓ_values) >= 6 else ℓ_values
        
        for idx, ℓ in enumerate(plot_ells):
            if idx >= 6:
                break
            ax = axes[idx]
            
            curve_data = results['lambda_curves'][ℓ]
            lambda_vals = curve_data['lambda']
            overlap_vals = [o * 100 for o in curve_data['overlap']]
            
            ax.plot(lambda_vals, overlap_vals, 'b-', linewidth=2)
            
            # Mark optimal
            ℓ_idx = ℓ_values.index(ℓ)
            lambda_opt = results['lambda_optimal'][ℓ_idx]
            overlap_opt = results['overlap_optimized'][ℓ_idx] * 100
            ax.plot(lambda_opt, overlap_opt, 'r*', markersize=15, 
                   label=f'λ* = {lambda_opt:.3f}')
            
            ax.set_xlabel('λ Parameter')
            ax.set_ylabel('Overlap (%)')
            ax.set_title(f'ℓ = {ℓ}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
        
        # Hide unused subplots
        for idx in range(len(plot_ells), 6):
            axes[idx].axis('off')
        
        plt.suptitle('Optimization Curves: Overlap vs λ Parameter', fontsize=14, y=1.00)
        plt.tight_layout()
        plt.savefig('results/phase4_optimization_curves.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("💾 Saved: results/phase4_optimization_curves.png")
        
        # Plot 3: Improvement comparison (optimized vs full)
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(ℓ_values))
        width = 0.35
        
        ax.bar(x - width/2, [i*100 for i in improvements_full], width, 
              label='Full correction (λ=1)', alpha=0.8, color='steelblue')
        ax.bar(x + width/2, [i*100 for i in improvements_opt], width, 
              label='Optimized (λ*)', alpha=0.8, color='orange')
        
        ax.set_xlabel('Angular Momentum ℓ', fontsize=12)
        ax.set_ylabel('Improvement (percentage points)', fontsize=12)
        ax.set_title('Improvement Comparison: Full vs Optimized Correction', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(ℓ_values)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(0, color='black', linewidth=0.5)
        plt.tight_layout()
        plt.savefig('results/phase4_improvement_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("💾 Saved: results/phase4_improvement_comparison.png")
        
        # Plot 4: Full comparison (uncorrected, full, optimized)
        fig, ax = plt.subplots(figsize=(12, 7))
        
        ax.plot(ℓ_values, [o*100 for o in results['overlap_uncorrected']], 
               'o-', label='Uncorrected (λ=0)', markersize=8, linewidth=2)
        ax.plot(ℓ_values, [o*100 for o in results['overlap_full_correction']], 
               's-', label='Full correction (λ=1)', markersize=8, linewidth=2)
        ax.plot(ℓ_values, [o*100 for o in results['overlap_optimized']], 
               '^-', label='Optimized (λ*)', markersize=8, linewidth=2)
        ax.axhline(100, color='green', linestyle='--', alpha=0.3, label='Perfect (100%)')
        ax.axhline(82, color='red', linestyle='--', alpha=0.3, label='~82% baseline')
        
        ax.set_xlabel('Angular Momentum ℓ', fontsize=12)
        ax.set_ylabel('Overlap with Y_ℓm (%)', fontsize=12)
        ax.set_title('Complete Comparison: Uncorrected vs Full vs Optimized', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([75, 102])
        plt.tight_layout()
        plt.savefig('results/phase4_complete_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("💾 Saved: results/phase4_complete_comparison.png")
        print()
    
    # Final recommendations
    print("=" * 70)
    print("PHASE 4 RECOMMENDATIONS")
    print("=" * 70)
    print()
    
    print("Based on optimization results:\n")
    
    if std_lambda < 0.1 and abs(mean_lambda - 1.0) < 0.1:
        print("1. UNIVERSAL PARAMETER RECOMMENDATION:")
        print(f"   Use λ = 1.0 (full correction) for all ℓ")
        print(f"   → Optimal λ consistently near 1.0 (mean: {mean_lambda:.3f}, std: {std_lambda:.3f})")
        print(f"   → Simplest approach with near-optimal results")
    elif std_lambda < 0.15:
        print("1. UNIVERSAL PARAMETER RECOMMENDATION:")
        print(f"   Use λ ≈ {mean_lambda:.2f} for all ℓ")
        print(f"   → Optimal λ relatively constant (std: {std_lambda:.3f})")
        print(f"   → Single parameter works well across all ℓ")
    else:
        print("1. ℓ-DEPENDENT PARAMETER RECOMMENDATION:")
        print(f"   Use ℓ-specific λ values (see table below)")
        print(f"   → Optimal λ varies significantly with ℓ (std: {std_lambda:.3f})")
        print(f"   → ℓ-dependent optimization worthwhile")
        print()
        print("   ℓ | Optimal λ")
        print("   --|----------")
        for ℓ, lam in zip(ℓ_values, lambda_array):
            print(f"   {ℓ:2d} | {lam:.3f}")
    
    print()
    print("2. PRACTICAL IMPLEMENTATION:")
    if abs(mean_advantage) < 0.005:
        print(f"   → Optimization provides minimal benefit ({abs(mean_advantage)*100:.2f}%)")
        print(f"   → Recommend using λ = 1.0 for simplicity")
    else:
        print(f"   → Optimization provides {abs(mean_advantage)*100:.2f}% average benefit")
        print(f"   → Consider adaptive λ for precision-critical applications")
    
    print()
    print("3. EIGENVALUE PRESERVATION:")
    if all_preserved:
        print(f"   ✅ All optimized corrections preserve eigenvalues")
        print(f"   → Safe to use for quantum mechanical calculations")
    else:
        print(f"   ⚠️  Some violations detected (max: {max_error:.2e})")
        print(f"   → Monitor eigenvalue errors in applications")
    
    print()
    print("=" * 70)
    print("PHASE 4 COMPLETE")
    print("=" * 70)
    print()
    print("All four research phases completed. Ready for comprehensive write-up.")
    print()
    
    return results


if __name__ == "__main__":
    import os
    
    # Create results directory if needed
    os.makedirs('results', exist_ok=True)
    
    # Run Phase 4 (n_max=8 gives ℓ_max=7, so test ℓ=1 to 7)
    results = phase4_adaptive_hybrid_optimization(n_max=8, ℓ_values=list(range(1, 8)), 
                                                 save_plots=True)
    
    print("\nPhase 4 optimization complete!")
    print("Adaptive hybrid parameter optimization finished.")
    print("\nAll research phases complete. Review results/ directory for comprehensive analysis.")
