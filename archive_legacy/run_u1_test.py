"""
Test script for Phase 10.1: U(1) Gauge Theory

This is the critical universality test. If e² ≈ 1/(4π) for U(1) as well as SU(2),
we have evidence that the geometric constant appears across all gauge groups.

Expected outcomes:
- BEST CASE: e² ≈ 0.0796 with <1% error → Universality confirmed
- GOOD CASE: e² ≈ 0.0796 with <5% error → Strong evidence
- NEUTRAL: e² differs significantly → Gauge-group dependent
"""

import sys
import os
sys.path.append('src')

from u1_gauge_theory import U1GaugeTheory
import numpy as np

def main():
    print("="*70)
    print("PHASE 10.1: U(1) GAUGE THEORY TEST")
    print("="*70)
    print("\nCritical Test: Does e² ≈ 1/(4π) for electromagnetism?")
    print("If yes, geometric constant is universal across gauge groups.\n")
    
    # Initialize U(1) gauge theory
    u1 = U1GaugeTheory(n_max=8, seed=42)
    
    # Beta scan around expected value
    # β = 1/e², target e² = 1/(4π) ≈ 0.0796
    # So target β ≈ 12.57
    print("\nTarget:")
    target = 1.0 / (4 * np.pi)
    print(f"  e² = 1/(4π) = {target:.6f}")
    print(f"  β = 1/e² = {1/target:.2f}")
    
    # Scan range
    beta_min = 8.0
    beta_max = 18.0
    n_beta = 11
    
    beta_values = np.linspace(beta_min, beta_max, n_beta)
    
    print(f"\nScanning β from {beta_min} to {beta_max} ({n_beta} points)")
    print("This will take several minutes with Monte Carlo sampling...\n")
    
    # Run the scan
    results = u1.scan_beta(
        beta_values, 
        n_therm=1000,   # Thermalization sweeps
        n_measure=100   # Measurement samples
    )
    
    # Test for geometric factor
    print("\n" + "="*70)
    print("GEOMETRIC FACTOR ANALYSIS")
    print("="*70)
    
    geometric_test = u1.test_geometric_factor(results)
    
    if geometric_test['match']:
        print(f"\nBest Match Found:")
        print(f"  β = {geometric_test['best_beta']:.4f}")
        print(f"  e² (measured) = {geometric_test['best_e_squared']:.6f}")
        print(f"  1/(4π) (target) = {geometric_test['target']:.6f}")
        print(f"  Error: {geometric_test['error_pct']:.3f}%")
        print(f"\nStatus: {geometric_test['status']}")
        print(f"\n{geometric_test['interpretation']}")
        
        # Compare to SU(2) from Phase 9
        comparison = u1.compare_to_su2(geometric_test['best_e_squared'])
        
        print(f"\n" + "-"*70)
        print("COMPARISON TO PHASE 9 (SU(2))")
        print("-"*70)
        print(f"U(1) coupling:  e² = {comparison['e_squared_u1']:.6f} (error: {comparison['u1_error_pct']:.3f}%)")
        print(f"SU(2) coupling: g² = {comparison['g_squared_su2']:.6f} (error: {comparison['su2_error_pct']:.3f}%)")
        print(f"Coupling ratio: e²/g² = {comparison['ratio']:.4f}")
        print(f"Compatible: {'YES' if comparison['compatible'] else 'NO'}")
        
        # Major result assessment
        if comparison['compatible'] and comparison['u1_error_pct'] < 5 and comparison['su2_error_pct'] < 5:
            print("\n" + "="*70)
            print("*** MAJOR BREAKTHROUGH: UNIVERSALITY CONFIRMED! ***")
            print("="*70)
            print("\nBoth U(1) (Abelian) and SU(2) (non-Abelian) gauge theories")
            print("show couplings ≈ 1/(4π). This strongly suggests the geometric")
            print("constant is universal across all gauge groups.")
            print("\nImplications:")
            print("  • Gauge couplings emerge from lattice geometry")
            print("  • Fine structure constant α has geometric origin")
            print("  • Possible explanation for coupling unification")
            print("  • Major paper: 'Universal Gauge Coupling from Discrete Geometry'")
        elif comparison['u1_error_pct'] < 10:
            print("\n*** GOOD RESULT: Geometric constant appears in U(1) ***")
            print("Evidence for universality, though not as strong as SU(2).")
        else:
            print("\n*** INTERESTING: U(1) differs from SU(2) ***")
            print("Suggests gauge-group dependence. Geometric constant")
            print("may be specific to non-Abelian theories.")
    else:
        print("\nNo clear match found. May need larger lattice or different β range.")
    
    # Generate visualizations and report
    print("\n" + "="*70)
    print("GENERATING OUTPUTS")
    print("="*70)
    
    u1.plot_analysis(results, geometric_test)
    u1.generate_report(results, geometric_test)
    
    print("\nResults saved:")
    print("  * results/u1_gauge_analysis.png")
    print("  * results/u1_gauge_report.txt")
    
    print("\n" + "="*70)
    print("PHASE 10.1 COMPLETE")
    print("="*70)
    
    # Summary
    if geometric_test['match']:
        print(f"\nSUMMARY:")
        print(f"  Measured e² = {geometric_test['best_e_squared']:.6f}")
        print(f"  Target 1/(4π) = {geometric_test['target']:.6f}")
        print(f"  Match: {geometric_test['error_pct']:.3f}% error")
        print(f"  Status: {geometric_test['status']}")


if __name__ == '__main__':
    main()
