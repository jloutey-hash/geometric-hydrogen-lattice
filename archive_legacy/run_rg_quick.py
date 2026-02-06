"""
Quick test of RG Flow with fewer measurements for faster execution
"""

import sys
sys.path.append('src')

from rg_flow import run_rg_flow_investigation

if __name__ == "__main__":
    print("\nQuick RG Flow Test (reduced measurements)")
    print("="*60)
    
    # Smaller range, fewer measurements
    calc, results = run_rg_flow_investigation(
        ell_max_initial=8,  # Smaller starting point
        beta=50.0,
        ell_step=2  # Bigger steps (fewer scales)
    )
    
    import numpy as np
    
    mean_g2 = np.mean(results['g_squared'])
    one_over_4pi = 1/(4*np.pi)
    deviation = abs(mean_g2 - one_over_4pi) / one_over_4pi * 100
    
    print(f"\nResult: Mean g^2 = {mean_g2:.6f}")
    print(f"Target: 1/(4pi) = {one_over_4pi:.6f}")
    print(f"Deviation: {deviation:.2f}%")
