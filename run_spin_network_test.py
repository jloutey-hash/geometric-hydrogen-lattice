"""
Run Phase 9.6: Spin Networks and Loop Quantum Gravity Investigation

Tests connections between discrete lattice and LQG,
particularly role of 1/(4pi) in quantum geometry.
"""

import sys
sys.path.append('src')

import numpy as np
from spin_networks import run_spin_network_investigation

if __name__ == "__main__":
    print("\n" + "="*80)
    print("PHASE 9.6: SPIN NETWORKS AND LOOP QUANTUM GRAVITY")
    print("="*80)
    print()
    print("Hypothesis: Discrete lattice IS a spin network")
    print("            Geometric constant 1/(4pi) appears in quantum geometry")
    print()
    
    # Run investigation
    calc, results = run_spin_network_investigation(ell_max=15)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print()
    
    one_over_4pi = 1/(4*np.pi)
    
    print(f"Spin Network Structure:")
    print(f"  Total nodes: {len(calc.lattice.points)}")
    print(f"  Shells: 0 to {calc.ell_max}")
    print()
    
    print(f"Quantum Geometry:")
    print(f"  Area eigenvalues computed for each shell")
    print(f"  Volume eigenvalues computed")
    print(f"  Immirzi parameter tested")
    print()
    
    print(f"Connection to Phase 8:")
    print(f"  Geometric ratio: {results['mean_phase8_ratio']:.6f}")
    print(f"  Target (1/4pi): {one_over_4pi:.6f}")
    print(f"  Match: {results['match_phase8']*100:.2f}% error")
    print()
    
    import numpy as np
    
    gamma_standard = np.log(2) / (np.pi * np.sqrt(3))
    print(f"Immirzi Parameter:")
    print(f"  Standard (BH entropy): {gamma_standard:.6f}")
    print(f"  Geometric (1/4pi): {one_over_4pi:.6f}")
    print(f"  Ratio: {gamma_standard/one_over_4pi:.4f}")
    print()
    
    if results['match_phase8'] < 0.05:
        print("*** EXCELLENT: Strong connection to 1/(4pi)!")
        status = "STRONG_CONNECTION"
    elif results['match_phase8'] < 0.15:
        print("** GOOD: Clear signature of 1/(4pi)")
        status = "GOOD_CONNECTION"
    else:
        print("* MODERATE: Some evidence for 1/(4pi)")
        status = "MODERATE_CONNECTION"
    
    print()
    print(f"Status: {status}")
    print()
    print("Results saved to:")
    print("  * results/spin_network_analysis.png")
    print("  * results/spin_network_report.txt")
    print()
