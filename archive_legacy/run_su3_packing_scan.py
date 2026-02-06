"""
Complete SU(3) Impedance + Packing Scan

Generate comprehensive dataset of impedance and packing metrics for SU(3) representations.
Outputs to CSV file for analysis.
"""

import sys
sys.path.insert(0, 'SU(3)')

import numpy as np
import csv
from su3_spherical_embedding import SU3SphericalEmbedding
from su3_impedance import SU3SymplecticImpedance
from packing_metrics import compute_packing_metrics


def compute_impedance_and_packing_quiet(p: int, q: int) -> dict:
    """
    Compute impedance and packing metrics quietly (no console output).
    
    Parameters
    ----------
    p, q : int
        SU(3) representation labels
    
    Returns
    -------
    data : dict
        Combined metrics
    """
    try:
        # Redirect stdout to suppress verbose output
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            # Compute impedance
            impedance_calc = SU3SymplecticImpedance(p, q, verbose=False)
            impedance_result = impedance_calc.compute_impedance()
            
            # Compute packing metrics
            embedding = SU3SphericalEmbedding(p, q)
            packing_by_shell = compute_packing_metrics(embedding)
        
        # Average packing metrics over shells
        covering_radii = [m.covering_radius for m in packing_by_shell.values()]
        kissing_numbers = [m.kissing_number_mean for m in packing_by_shell.values()]
        packing_effs = [m.packing_efficiency for m in packing_by_shell.values()]
        
        # Compile data
        data = {
            'p': p,
            'q': q,
            'dim': impedance_result.dim,
            'C2': impedance_result.C2,
            'Z': impedance_result.Z_impedance,
            'Z_normalized': impedance_result.Z_normalized,
            'Z_dimensionless': impedance_result.Z_dimensionless,
            'C_matter': impedance_result.C_matter,
            'S_holonomy': impedance_result.S_holonomy,
            'covering_radius_mean': np.mean(covering_radii),
            'covering_radius_std': np.std(covering_radii),
            'kissing_number_mean': np.mean(kissing_numbers),
            'packing_efficiency_mean': np.mean(packing_effs),
            'packing_efficiency_std': np.std(packing_effs),
            'n_shells': len(packing_by_shell)
        }
        
        return data
        
    except Exception as e:
        print(f"ERROR for ({p},{q}): {e}")
        return None


def run_scan(max_pq_sum: int = 8, output_file: str = "su3_impedance_packing_scan_extended.csv"):
    """
    Scan all (p,q) representations with p+q <= max_pq_sum.
    
    Parameters
    ----------
    max_pq_sum : int
        Maximum value of p+q to include
    output_file : str
        Output CSV filename
    """
    results = []
    
    print(f"Scanning SU(3) representations with p+q <= {max_pq_sum}...")
    print(f"{'(p,q)':<10} {'Status':<10}")
    print("-" * 30)
    
    for p in range(max_pq_sum + 1):
        for q in range(max_pq_sum + 1):
            if p + q <= max_pq_sum and (p > 0 or q > 0):  # Skip (0,0)
                print(f"({p},{q}){'':<6}", end="", flush=True)
                
                data = compute_impedance_and_packing_quiet(p, q)
                
                if data is not None:
                    results.append(data)
                    print(f" OK")
                else:
                    print(f" FAILED")
    
    # Write to CSV
    if results:
        fieldnames = list(results[0].keys())
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\nResults written to {output_file}")
        print(f"Total representations: {len(results)}")
        
        # Print summary statistics
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        Z_values = [d['Z'] for d in results]
        packing_values = [d['packing_efficiency_mean'] for d in results]
        C2_values = [d['C2'] for d in results]
        
        print(f"\nImpedance Z:")
        print(f"  Range: [{min(Z_values):.4f}, {max(Z_values):.4f}]")
        print(f"  Mean: {np.mean(Z_values):.4f}")
        print(f"  Std: {np.std(Z_values):.4f}")
        
        print(f"\nPacking Efficiency:")
        print(f"  Range: [{min(packing_values):.3f}, {max(packing_values):.3f}]")
        print(f"  Mean: {np.mean(packing_values):.3f}")
        print(f"  Std: {np.std(packing_values):.3f}")
        
        print(f"\nCasimir C2:")
        print(f"  Range: [{min(C2_values):.4f}, {max(C2_values):.4f}]")
        
        # Correlation analysis
        if len(Z_values) > 2:
            corr_Z_packing = np.corrcoef(Z_values, packing_values)[0, 1]
            corr_Z_C2 = np.corrcoef(Z_values, C2_values)[0, 1]
            corr_packing_C2 = np.corrcoef(packing_values, C2_values)[0, 1]
            
            print(f"\nCorrelations:")
            print(f"  Corr(Z, PackingEff): {corr_Z_packing:+.3f}")
            print(f"  Corr(Z, C2):         {corr_Z_C2:+.3f}")
            print(f"  Corr(PackingEff, C2): {corr_packing_C2:+.3f}")
        
        print("\n" + "="*80)
        print(f"Next steps:")
        print(f"1. Load {output_file} in Excel/Python for analysis")
        print(f"2. Plot Z vs packing_efficiency_mean")
        print(f"3. Compare with U(1) hydrogen impedance (kappa_5 ~ 137.04)")
        print("="*80)
    
    else:
        print("\nNo successful calculations!")


if __name__ == "__main__":
    # Extended scan: p+q <= 8 for canonical representation search
    # This includes many more representations for comprehensive analysis
    run_scan(max_pq_sum=8, output_file="su3_impedance_packing_scan_extended.csv")
