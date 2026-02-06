"""
Run Phase 9.4: Vacuum Energy Investigation

Tests whether the geometric constant 1/(4π) appears in:
- Zero-point energy calculations
- Mode density ρ(ω)  
- UV cutoff behavior
- Energy density scaling
"""

import sys
sys.path.append('src')

from vacuum_energy import run_vacuum_energy_investigation

if __name__ == "__main__":
    print("\n" + "="*80)
    print("PHASE 9.4: VACUUM ENERGY ON DISCRETE LATTICE")
    print("="*80)
    print()
    print("Hypothesis: Discrete lattice provides natural UV regulator")
    print("            with cutoff involving geometric constant 1/(4*pi)")
    print()
    print("Testing massless scalar field...")
    print()
    
    # Run main investigation
    calc, results = run_vacuum_energy_investigation(ell_max=15, mass=0.0)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print(f"Vacuum energy: E_vac = {results['E_vac']:.6e}")
    print(f"Total modes: {results['N_modes']}")
    print()
    print("Search for 1/(4*pi) = 0.0795774715:")
    print(f"  • Energy per mode test: {results['test1_match']*100:.2f}% error")
    print(f"  • Cutoff scale test: {results['test2_match']*100:.2f}% error")  
    print(f"  • Energy density test: {results['test3_match']*100:.2f}% error")
    print()
    print(f"BEST MATCH: {results['best_match']*100:.2f}% error")
    print()
    
    if results['best_match'] < 0.01:
        print("*** EXCELLENT: Strong evidence for 1/(4*pi) factor!")
        status = "BREAKTHROUGH"
    elif results['best_match'] < 0.05:
        print("** GOOD: Clear signature of 1/(4*pi)")
        status = "SUCCESS"
    elif results['best_match'] < 0.15:
        print("* MODERATE: Suggestive evidence")
        status = "PROMISING"
    else:
        print("WEAK: No clear 1/(4*pi) signature")
        status = "NEEDS_REFINEMENT"
    
    print()
    print(f"Status: {status}")
    print()
    print("Results saved to:")
    print("  • results/vacuum_energy_analysis.png")
    print("  • results/vacuum_energy_report.txt")
    print()
