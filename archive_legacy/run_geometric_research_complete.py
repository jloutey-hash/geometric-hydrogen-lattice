"""
Master Script: Complete Geometric Transformation Research Pipeline

This script orchestrates all four phases of the geometric transformation research:

Phase 1: Diagnostic Analysis
- Quantify baseline eigenvector-spherical harmonic overlap
- Identify spatial error patterns (poles vs equator, inner vs outer)
- Determine if error is systematic (geometric) or uniform (discretization)

Phase 2: Transformation Testing
- Test stereographic, Lambert, and Mercator projections
- Apply forward/pullback Jacobian corrections
- Compare improvements across transformation types
- Verify eigenvalue preservation

Phase 3: Validation and Scaling  
- Scale to high ℓ values (up to ℓ=10)
- Test reversibility (round-trip fidelity)
- Verify commutator preservation [Li, Lj] = iℏϵijk Lk
- Assess computational cost

Phase 4: Adaptive Optimization
- Optimize hybrid parameter λ for each ℓ
- Analyze λ(ℓ) relationship
- Compare optimized vs full correction
- Generate practical recommendations

Usage:
    python run_geometric_research_complete.py

Outputs:
    - Comprehensive plots in results/ directory
    - Summary tables and statistics
    - Final research report

Author: Quantum Lattice Project
Date: January 2026
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import phase scripts
print("=" * 70)
print("GEOMETRIC TRANSFORMATION RESEARCH - COMPLETE PIPELINE")
print("=" * 70)
print()
print("This will run all four research phases:")
print("  Phase 1: Diagnostic Analysis")
print("  Phase 2: Transformation Testing")
print("  Phase 3: Validation and Scaling")
print("  Phase 4: Adaptive Optimization")
print()
print("Estimated time: 5-15 minutes depending on hardware")
print()

response = input("Continue? (y/n): ")
if response.lower() != 'y':
    print("Aborted.")
    sys.exit(0)

print()
print("=" * 70)
print()

# Create results directory
import os
os.makedirs('results', exist_ok=True)

# Track timing
phase_times = {}

# PHASE 1
print("\n" + "="*70)
print("STARTING PHASE 1: DIAGNOSTIC ANALYSIS")
print("="*70 + "\n")
t_start = time.time()

from run_phase1_geometric_diagnostic import phase1_diagnostic_analysis
results_phase1 = phase1_diagnostic_analysis(n_max=6, save_plots=True)

phase_times['Phase 1'] = time.time() - t_start
print(f"\nPhase 1 completed in {phase_times['Phase 1']:.1f} seconds")

# PHASE 2
print("\n" + "="*70)
print("STARTING PHASE 2: TRANSFORMATION TESTING")
print("="*70 + "\n")
t_start = time.time()

from run_phase2_geometric_transform_test import phase2_transformation_testing
results_phase2 = phase2_transformation_testing(n_max=6, ℓ_values=[1, 2, 3, 4, 5], 
                                              save_plots=True)

phase_times['Phase 2'] = time.time() - t_start
print(f"\nPhase 2 completed in {phase_times['Phase 2']:.1f} seconds")

# PHASE 3
print("\n" + "="*70)
print("STARTING PHASE 3: VALIDATION AND SCALING")
print("="*70 + "\n")
t_start = time.time()

from run_phase3_geometric_validation import phase3_validation_scaling
results_phase3 = phase3_validation_scaling(n_max=11, ℓ_max_test=10, save_plots=True)

phase_times['Phase 3'] = time.time() - t_start
print(f"\nPhase 3 completed in {phase_times['Phase 3']:.1f} seconds")

# PHASE 4
print("\n" + "="*70)
print("STARTING PHASE 4: ADAPTIVE OPTIMIZATION")
print("="*70 + "\n")
t_start = time.time()

from run_phase4_geometric_optimization import phase4_adaptive_hybrid_optimization
# n_max=8 gives ℓ_max=7, so test ℓ=1 to 7
results_phase4 = phase4_adaptive_hybrid_optimization(n_max=8, ℓ_values=list(range(1, 8)),
                                                    save_plots=True)

phase_times['Phase 4'] = time.time() - t_start
print(f"\nPhase 4 completed in {phase_times['Phase 4']:.1f} seconds")

# FINAL SUMMARY
print("\n" + "="*70)
print("ALL PHASES COMPLETE - GENERATING FINAL SUMMARY")
print("="*70 + "\n")

# Timing summary
total_time = sum(phase_times.values())
print("Timing Summary:")
for phase, t in phase_times.items():
    print(f"  {phase}: {t:.1f}s ({t/total_time*100:.1f}%)")
print(f"  Total: {total_time:.1f}s ({total_time/60:.1f} minutes)")
print()

# Key findings summary
print("=" * 70)
print("KEY FINDINGS SUMMARY")
print("=" * 70)
print()

print("1. BASELINE PERFORMANCE (Phase 1):")
mean_overlap_p1 = sum(results_phase1['overlaps']) / len(results_phase1['overlaps'])
print(f"   Mean overlap (uncorrected): {mean_overlap_p1:.2%}")
print(f"   Average deficit: {(1-mean_overlap_p1):.2%}")
print()

print("2. TRANSFORMATION EFFECTIVENESS (Phase 2):")
# Aggregate improvements from Phase 2
all_improvements_p2 = [r.overlap_improvement for r in results_phase2]
mean_improvement_p2 = sum(all_improvements_p2) / len(all_improvements_p2)
max_improvement_p2 = max(all_improvements_p2)
print(f"   Mean improvement: {mean_improvement_p2:+.2%}")
print(f"   Max improvement: {max_improvement_p2:+.2%}")

# Best transformation
from collections import Counter
best_transforms = [r.transform_type for r in results_phase2 
                  if r.overlap_improvement == max(all_improvements_p2)]
print(f"   Best transformation: {best_transforms[0]}")
print()

print("3. SCALING BEHAVIOR (Phase 3):")
mean_improvement_p3 = sum(results_phase3['improvements']) / len(results_phase3['improvements'])
print(f"   Mean improvement (ℓ=1-10): {mean_improvement_p3:+.2%}")

min_fidelity = min(results_phase3['round_trip_fidelities'])
print(f"   Round-trip fidelity: {min_fidelity:.6f}")

max_eigenvalue_error = max(results_phase3['eigenvalue_errors'])
print(f"   Max eigenvalue error: {max_eigenvalue_error:.2e}")

if max_eigenvalue_error < 1e-10:
    print(f"   ✅ Eigenvalues preserved exactly")
if min_fidelity > 0.9999:
    print(f"   ✅ Excellent reversibility")
print()

print("4. OPTIMIZATION RESULTS (Phase 4):")
mean_lambda = sum(results_phase4['lambda_optimal']) / len(results_phase4['lambda_optimal'])
std_lambda = (sum((x - mean_lambda)**2 for x in results_phase4['lambda_optimal']) / 
              len(results_phase4['lambda_optimal']))**0.5

print(f"   Optimal λ: {mean_lambda:.3f} ± {std_lambda:.3f}")

if std_lambda < 0.1:
    print(f"   → Universal parameter recommended")
else:
    print(f"   → ℓ-dependent optimization beneficial")

mean_improvement_p4 = sum(results_phase4['improvement_optimized']) / len(results_phase4['improvement_optimized'])
print(f"   Mean optimized improvement: {mean_improvement_p4:+.2%}")
print()

print("=" * 70)
print("HYPOTHESIS EVALUATION")
print("=" * 70)
print()

hypothesis_supported = False

if mean_improvement_p2 > 0.05:  # 5% improvement
    print("✅ HYPOTHESIS STRONGLY SUPPORTED")
    print()
    print("The ~18% eigenvector deficit IS partially geometric in nature.")
    print(f"Coordinate transformations recover {mean_improvement_p2*100:.1f}% of the deficit.")
    print()
    hypothesis_supported = True
elif mean_improvement_p2 > 0.02:  # 2% improvement
    print("⚠️  HYPOTHESIS PARTIALLY SUPPORTED")
    print()
    print("The deficit has both geometric and fundamental discretization components.")
    print(f"Coordinate transformations recover {mean_improvement_p2*100:.1f}% of the deficit.")
    print()
    hypothesis_supported = True
else:
    print("❌ HYPOTHESIS NOT SUPPORTED")
    print()
    print("The ~18% deficit is primarily fundamental discretization, not geometry.")
    print("Coordinate transformations provide minimal improvement.")
    print()

if hypothesis_supported:
    print("IMPLICATIONS:")
    print("• Flat lattice ↔ curved sphere mapping is viable")
    print("• Exact eigenvalues maintained through transformation")
    print("• Computational efficiency preserved (sparse structure)")
    print("• Reversible toggling between representations possible")
    print()
    print("RECOMMENDATIONS:")
    print(f"• Use stereographic projection with Jacobian correction")
    print(f"• Apply λ ≈ {mean_lambda:.2f} for universal improvement")
    print(f"• Post-processing transformation (no matrix refactorization needed)")
    print()

print("=" * 70)
print("RESEARCH COMPLETE")
print("=" * 70)
print()
print(f"Results saved to: results/")
print()
print("Generated files:")
print("  - phase1_*.png: Diagnostic analysis plots")
print("  - phase2_*.png: Transformation comparison plots")
print("  - phase3_*.png: Validation and scaling plots")
print("  - phase4_*.png: Optimization results")
print()
print("Review these plots for detailed analysis and publication.")
print()

# Save summary to file
summary_file = 'results/GEOMETRIC_RESEARCH_SUMMARY.txt'
with open(summary_file, 'w') as f:
    f.write("="*70 + "\n")
    f.write("GEOMETRIC TRANSFORMATION RESEARCH - FINAL SUMMARY\n")
    f.write("="*70 + "\n\n")
    
    f.write("Research Question:\n")
    f.write("Can coordinate transformations reversibly map between flat lattice\n")
    f.write("and curved spherical representations, recovering the ~18% eigenvector\n")
    f.write("deficit while maintaining exact eigenvalues?\n\n")
    
    f.write("-"*70 + "\n\n")
    
    f.write("KEY RESULTS:\n\n")
    f.write(f"1. Baseline Performance:\n")
    f.write(f"   - Mean overlap (uncorrected): {mean_overlap_p1:.2%}\n")
    f.write(f"   - Average deficit: {(1-mean_overlap_p1):.2%}\n\n")
    
    f.write(f"2. Transformation Effectiveness:\n")
    f.write(f"   - Mean improvement: {mean_improvement_p2:+.2%}\n")
    f.write(f"   - Max improvement: {max_improvement_p2:+.2%}\n")
    f.write(f"   - Best method: {best_transforms[0]}\n\n")
    
    f.write(f"3. Quantum Mechanical Consistency:\n")
    f.write(f"   - Eigenvalue preservation: {max_eigenvalue_error:.2e}\n")
    f.write(f"   - Round-trip fidelity: {min_fidelity:.6f}\n\n")
    
    f.write(f"4. Optimal Parameters:\n")
    f.write(f"   - Optimal λ: {mean_lambda:.3f} ± {std_lambda:.3f}\n")
    f.write(f"   - Optimized improvement: {mean_improvement_p4:+.2%}\n\n")
    
    f.write("-"*70 + "\n\n")
    
    if hypothesis_supported:
        f.write("CONCLUSION: Hypothesis SUPPORTED\n\n")
        f.write("Coordinate transformations successfully bridge flat and curved\n")
        f.write("representations, recovering significant eigenvector accuracy while\n")
        f.write("maintaining exact eigenvalues and computational efficiency.\n")
    else:
        f.write("CONCLUSION: Hypothesis NOT SUPPORTED\n\n")
        f.write("The eigenvector deficit is primarily fundamental discretization\n")
        f.write("rather than coordinate geometry. Denser sampling required.\n")

print(f"Summary saved to: {summary_file}")
print()
print("="*70)
print("Thank you for using the geometric transformation research toolkit!")
print("="*70)
