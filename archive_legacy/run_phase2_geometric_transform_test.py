"""
Phase 2: Transformation Testing - Apply and Evaluate Conformal Mappings

This script implements Phase 2 of the geometric transformation research:
1. Apply stereographic projection with Jacobian correction
2. Apply Lambert azimuthal equal-area projection
3. Apply Mercator projection
4. Test both forward and pullback directions
5. Verify eigenvalue preservation for all transformations
6. Compare improvements across methods

Expected outputs:
- Overlap improvements for each transformation type
- Eigenvalue preservation verification
- Identification of best transformation and direction
- Comprehensive comparison table

Author: Quantum Lattice Project
Date: January 2026
Research Phase 2: Transformation Testing
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.lattice import PolarLattice
from src.angular_momentum import AngularMomentumOperators
from src.geometric_transform_research import (GeometricTransformResearch, 
                                               GeometricTransformBenchmark)
from scipy import sparse


def phase2_transformation_testing(n_max=6, ℓ_values=None, save_plots=True):
    """
    Phase 2: Systematic testing of conformal transformations.
    
    Parameters:
        n_max: Maximum principal quantum number (lattice size)
        ℓ_values: List of ℓ values to test (default: [1, 2, 3, 4, 5])
        save_plots: Whether to save plots to disk
    
    Returns:
        results: List of TransformationResult objects
    """
    if ℓ_values is None:
        ℓ_values = [1, 2, 3, 4, 5]
    
    print("=" * 70)
    print("PHASE 2: TRANSFORMATION TESTING - CONFORMAL MAPPINGS")
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
    
    # Visualize Jacobians first
    print("=" * 70)
    print("JACOBIAN ANALYSIS")
    print("=" * 70)
    print()
    
    transform_types = ['stereographic', 'lambert', 'mercator']
    
    for transform_type in transform_types:
        J = research.compute_jacobian(transform_type)
        print(f"{transform_type.capitalize()} Projection:")
        print(f"  Jacobian range: [{np.min(J):.4f}, {np.max(J):.4f}]")
        print(f"  Mean: {np.mean(J):.4f}, Std: {np.std(J):.4f}")
        
        # Interpretation
        if transform_type == 'stereographic':
            print(f"  Interpretation: J > 1 everywhere → area expansion from sphere to plane")
            print(f"                 Maximum distortion at poles (θ=0, π)")
        elif transform_type == 'lambert':
            print(f"  Interpretation: J ≈ 1 → area-preserving transformation")
        elif transform_type == 'mercator':
            print(f"  Interpretation: J = sec(θ) → large distortion near poles")
        print()
        
        # Plot Jacobian distribution
        if save_plots:
            fig = research.plot_jacobian_distribution(transform_type)
            plt.savefig(f'results/phase2_jacobian_{transform_type}.png', 
                       dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  💾 Saved: results/phase2_jacobian_{transform_type}.png\n")
    
    # Run comprehensive benchmark
    print("=" * 70)
    print("TRANSFORMATION BENCHMARK")
    print("=" * 70)
    print()
    
    benchmark = GeometricTransformBenchmark(lattice, angular_ops)
    results = benchmark.run_comprehensive_benchmark(ℓ_values, transform_types)
    
    # Generate summary table
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY TABLE")
    print("=" * 70)
    print()
    print(benchmark.generate_summary_table())
    
    # Analyze results
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS")
    print("=" * 70)
    print()
    
    # Group by transformation
    by_transform = {}
    for r in results:
        if r.transform_type not in by_transform:
            by_transform[r.transform_type] = []
        by_transform[r.transform_type].append(r)
    
    for transform_type in transform_types:
        transform_results = by_transform[transform_type]
        
        print(f"\n{transform_type.upper()} PROJECTION:")
        print(f"{'-' * 60}")
        
        # Statistics
        improvements = [r.overlap_improvement for r in transform_results]
        mean_improvement = np.mean(improvements)
        max_improvement = np.max(improvements)
        min_improvement = np.min(improvements)
        
        # Best direction
        forward_better = sum(1 for r in transform_results if r.overlap_forward > r.overlap_pullback)
        pullback_better = len(transform_results) - forward_better
        
        print(f"  Mean improvement:     {mean_improvement:+.2%}")
        print(f"  Max improvement:      {max_improvement:+.2%} (ℓ={transform_results[np.argmax(improvements)].ℓ})")
        print(f"  Min improvement:      {min_improvement:+.2%} (ℓ={transform_results[np.argmin(improvements)].ℓ})")
        print()
        print(f"  Direction comparison:")
        print(f"    Forward (√J) better:   {forward_better}/{len(transform_results)} cases")
        print(f"    Pullback (1/√J) better: {pullback_better}/{len(transform_results)} cases")
        
        # Determine preferred direction
        if forward_better > pullback_better:
            print(f"    → Preferred: FORWARD (√J multiplication)")
        else:
            print(f"    → Preferred: PULLBACK (1/√J division)")
        print()
        
        # Eigenvalue preservation
        all_preserved = all(r.eigenvalue_preserved for r in transform_results)
        if all_preserved:
            print(f"  ✅ Eigenvalue preservation: PERFECT (all errors < 10^-10)")
        else:
            failed = sum(1 for r in transform_results if not r.eigenvalue_preserved)
            print(f"  ⚠️  Eigenvalue preservation: {failed}/{len(transform_results)} cases failed")
        
        # Round-trip fidelity
        fidelities = [r.round_trip_fidelity for r in transform_results if r.round_trip_fidelity]
        if fidelities:
            mean_fidelity = np.mean(fidelities)
            min_fidelity = np.min(fidelities)
            print(f"\n  Round-trip fidelity:")
            print(f"    Mean: {mean_fidelity:.6f}")
            print(f"    Min:  {min_fidelity:.6f}")
            if min_fidelity > 0.9999:
                print(f"    ✅ Excellent reversibility (>99.99%)")
            elif min_fidelity > 0.99:
                print(f"    ⚠️  Good reversibility (>99%)")
            else:
                print(f"    ❌ Poor reversibility (<99%)")
    
    # Find overall best transformation
    print("\n" + "=" * 70)
    print("OVERALL BEST TRANSFORMATION")
    print("=" * 70)
    print()
    
    best_result = max(results, key=lambda r: r.overlap_improvement)
    print(f"Best improvement: {best_result.overlap_improvement:+.2%}")
    print(f"  Transformation: {best_result.transform_type}")
    print(f"  Angular momentum: ℓ={best_result.ℓ}")
    print(f"  Original overlap: {best_result.overlap_original:.4%}")
    print(f"  Corrected overlap: {best_result.overlap_best:.4%}")
    print(f"  Direction: {'Forward (√J)' if best_result.overlap_forward > best_result.overlap_pullback else 'Pullback (1/√J)'}")
    print()
    
    # Average by transform type
    avg_improvements = {}
    for transform_type in transform_types:
        transform_results = by_transform[transform_type]
        avg_improvements[transform_type] = np.mean([r.overlap_improvement for r in transform_results])
    
    best_transform = max(avg_improvements, key=avg_improvements.get)
    print(f"Best transformation on average: {best_transform.upper()}")
    print(f"  Average improvement: {avg_improvements[best_transform]:+.2%}")
    print()
    
    # Generate comparison plots
    if save_plots:
        print("=" * 70)
        print("GENERATING COMPARISON PLOTS")
        print("=" * 70)
        print()
        
        fig = benchmark.plot_improvement_comparison()
        plt.savefig('results/phase2_improvement_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("💾 Saved: results/phase2_improvement_comparison.png")
        
        # Additional detailed plots
        
        # Plot 1: Heatmap of improvements
        fig, ax = plt.subplots(figsize=(10, 8))
        
        improvement_matrix = np.zeros((len(transform_types), len(ℓ_values)))
        for i, transform_type in enumerate(transform_types):
            for j, ℓ in enumerate(ℓ_values):
                matching = [r for r in results if r.transform_type == transform_type and r.ℓ == ℓ]
                if matching:
                    improvement_matrix[i, j] = matching[0].overlap_improvement * 100
        
        im = ax.imshow(improvement_matrix, aspect='auto', cmap='RdYlGn', 
                       vmin=-2, vmax=max(2, improvement_matrix.max()))
        ax.set_xticks(range(len(ℓ_values)))
        ax.set_xticklabels(ℓ_values)
        ax.set_yticks(range(len(transform_types)))
        ax.set_yticklabels([t.capitalize() for t in transform_types])
        ax.set_xlabel('Angular Momentum ℓ', fontsize=12)
        ax.set_title('Overlap Improvement Heatmap\n(Percentage Points)', fontsize=14)
        
        # Annotate cells
        for i in range(len(transform_types)):
            for j in range(len(ℓ_values)):
                text = ax.text(j, i, f'{improvement_matrix[i, j]:+.1f}',
                              ha="center", va="center", color="black", fontsize=10)
        
        plt.colorbar(im, ax=ax, label='Improvement (%)')
        plt.tight_layout()
        plt.savefig('results/phase2_improvement_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("💾 Saved: results/phase2_improvement_heatmap.png")
        
        # Plot 2: Direction comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, transform_type in enumerate(transform_types):
            ax = axes[idx]
            transform_results = by_transform[transform_type]
            
            ℓs = [r.ℓ for r in transform_results]
            forward = [r.overlap_forward * 100 for r in transform_results]
            pullback = [r.overlap_pullback * 100 for r in transform_results]
            original = [r.overlap_original * 100 for r in transform_results]
            
            ax.plot(ℓs, original, 'k--', label='Original', alpha=0.5)
            ax.plot(ℓs, forward, 'o-', label='Forward (√J)', markersize=6)
            ax.plot(ℓs, pullback, 's-', label='Pullback (1/√J)', markersize=6)
            
            ax.set_xlabel('Angular Momentum ℓ')
            ax.set_ylabel('Overlap (%)')
            ax.set_title(f'{transform_type.capitalize()}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Forward vs Pullback Correction Comparison', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig('results/phase2_direction_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("💾 Saved: results/phase2_direction_comparison.png")
        print()
    
    # Final assessment
    print("=" * 70)
    print("PHASE 2 ASSESSMENT")
    print("=" * 70)
    print()
    
    mean_of_all_improvements = np.mean([r.overlap_improvement for r in results])
    max_of_all_improvements = np.max([r.overlap_improvement for r in results])
    
    print(f"Overall Statistics:")
    print(f"  Mean improvement across all (ℓ, transform): {mean_of_all_improvements:+.2%}")
    print(f"  Maximum improvement achieved:               {max_of_all_improvements:+.2%}")
    print()
    
    # Hypothesis evaluation
    print("Hypothesis Evaluation:")
    print()
    
    if mean_of_all_improvements > 0.05:  # 5% improvement
        print("  ✅ HYPOTHESIS SUPPORTED")
        print(f"     → Geometric correction provides significant improvement (>{mean_of_all_improvements*100:.1f}%)")
        print(f"     → The ~18% deficit IS partially geometric in nature")
        print(f"     → Coordinate transformations can recover eigenvector accuracy")
    elif mean_of_all_improvements > 0.02:  # 2% improvement
        print("  ⚠️  HYPOTHESIS PARTIALLY SUPPORTED")
        print(f"     → Geometric correction provides modest improvement (~{mean_of_all_improvements*100:.1f}%)")
        print(f"     → The deficit is partly geometric, partly fundamental discretization")
    else:
        print("  ❌ HYPOTHESIS NOT SUPPORTED")
        print(f"     → Geometric correction provides minimal improvement (<{mean_of_all_improvements*100:.1f}%)")
        print(f"     → The ~18% deficit is primarily fundamental discretization, not geometry")
    print()
    
    if all(r.eigenvalue_preserved for r in results):
        print("  ✅ EIGENVALUE PRESERVATION: SUCCESS")
        print("     → Transformations maintain exact algebraic structure")
        print("     → Compatible with quantum mechanics requirements")
    else:
        print("  ⚠️  EIGENVALUE PRESERVATION: PARTIAL")
        print("     → Some transformations break eigenvalue exactness")
    print()
    
    print("=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)
    print()
    print(f"Next step: Phase 3 - Validation at higher ℓ and reversibility testing")
    print()
    
    return results


if __name__ == "__main__":
    import os
    
    # Create results directory if needed
    os.makedirs('results', exist_ok=True)
    
    # Run Phase 2
    results = phase2_transformation_testing(n_max=6, ℓ_values=[1, 2, 3, 4, 5], 
                                           save_plots=True)
    
    print("\nPhase 2 transformation testing complete!")
    print("Review the plots in results/ directory for detailed comparisons.")
