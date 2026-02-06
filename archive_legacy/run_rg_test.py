"""
Run Phase 9.5: Renormalization Group Flow Investigation

Tests whether beta-function involves 1/(4pi) and how
coupling g^2 evolves across energy scales.
"""

import sys
sys.path.append('src')

from rg_flow import run_rg_flow_investigation

if __name__ == "__main__":
    print("\n" + "="*80)
    print("PHASE 9.5: RENORMALIZATION GROUP FLOW")
    print("="*80)
    print()
    print("Hypothesis: Beta-function beta(g) involves 1/(4pi)")
    print("            and g^2 remains near 1/(4pi) across scales")
    print()
    
    # Run investigation
    # Start at ell_max=12, flow down to ell_max=2
    # Use beta=50 (where g^2 ≈ 1/(4pi))
    calc, results = run_rg_flow_investigation(
        ell_max_initial=12,
        beta=50.0,
        ell_step=1
    )
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print()
    
    import numpy as np
    
    mean_g2 = np.mean(results['g_squared'])
    std_g2 = np.std(results['g_squared'])
    one_over_4pi = 1/(4*np.pi)
    
    print(f"RG Flow Results:")
    print(f"  Scales analyzed: {len(results['ell_values'])}")
    print(f"  Mean g^2: {mean_g2:.6f}")
    print(f"  Std dev: {std_g2:.6f}")
    print(f"  Compare to 1/(4pi): {one_over_4pi:.6f}")
    print()
    
    deviation = abs(mean_g2 - one_over_4pi) / one_over_4pi * 100
    
    print(f"Beta-function:")
    print(f"  Fit: beta(g) = {results['A_fit']:.6f} * g^3")
    print(f"  Continuum (1-loop): beta(g) = {-results['b0_continuum']:.6f} * g^3")
    print()
    
    print(f"Mean deviation from 1/(4pi): {deviation:.2f}%")
    print()
    
    if deviation < 5:
        print("*** EXCELLENT: g^2 stable near 1/(4pi) across scales!")
        status = "STRONG_EVIDENCE"
    elif deviation < 10:
        print("** GOOD: g^2 remains close to 1/(4pi)")
        status = "MODERATE_EVIDENCE"
    elif deviation < 20:
        print("* FAIR: Some stability near 1/(4pi)")
        status = "WEAK_EVIDENCE"
    else:
        print("WEAK: Significant running away from 1/(4pi)")
        status = "NO_EVIDENCE"
    
    print()
    print(f"Status: {status}")
    print()
    print("Results saved to:")
    print("  * results/rg_flow_analysis.png")
    print("  * results/rg_flow_report.txt")
    print()
