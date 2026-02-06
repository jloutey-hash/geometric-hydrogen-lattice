"""
SU(3) Impedance + Packing Correlation

Combine SU(3) impedance calculations with geometric packing metrics.
This enables comparison with U(1) hydrogen in unified geometric language.
"""

import sys
sys.path.insert(0, 'SU(3)')

import numpy as np
from typing import Dict
from su3_spherical_embedding import SU3SphericalEmbedding
from su3_impedance import SU3SymplecticImpedance
from packing_metrics import compute_packing_metrics


def compute_impedance_and_packing(p: int, q: int) -> Dict:
    """
    Compute both impedance (Z = S/C) and packing metrics for (p,q) representation.
    
    Parameters
    ----------
    p, q : int
        SU(3) Dynkin labels
    
    Returns
    -------
    data : dict
        Combined impedance and packing data:
        - p, q: representation labels
        - dim: dimension
        - C2: Casimir eigenvalue
        - Z: impedance (S/C)
        - Z_over_4pi: normalized impedance
        - covering_radius_mean: average over shells
        - kissing_number_mean: average over shells
        - packing_efficiency_mean: average over shells
        - n_shells: number of radial shells
    """
    # Compute impedance
    impedance_calc = SU3SymplecticImpedance(p, q, verbose=False)
    impedance_result = impedance_calc.compute_impedance()
    
    # Compute packing metrics
    embedding = SU3SphericalEmbedding(p, q)
    packing_by_shell = compute_packing_metrics(embedding)
    
    # Average packing metrics over all shells
    covering_radii = [m.covering_radius for m in packing_by_shell.values()]
    kissing_numbers = [m.kissing_number_mean for m in packing_by_shell.values()]
    packing_effs = [m.packing_efficiency for m in packing_by_shell.values()]
    
    # Combine data
    data = {
        'p': p,
        'q': q,
        'dim': impedance_result.dim,
        'C2': impedance_result.C2,
        'Z': impedance_result.Z_impedance,
        'Z_over_4pi': impedance_result.Z_impedance / (4 * np.pi),
        'covering_radius_mean': np.mean(covering_radii),
        'covering_radius_std': np.std(covering_radii),
        'kissing_number_mean': np.mean(kissing_numbers),
        'packing_efficiency_mean': np.mean(packing_effs),
        'n_shells': len(packing_by_shell)
    }
    
    return data


if __name__ == "__main__":
    # Test on a few representations
    test_reps = [(1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]
    
    print("\n" + "="*80)
    print("SU(3) Impedance + Packing Correlation")
    print("="*80)
    
    print(f"\n{'(p,q)':<8} {'dim':<5} {'C2':<8} {'Z':<12} {'Z/4pi':<10} "
          f"{'CoverRad':<10} {'PackEff':<10}")
    print("-"*80)
    
    results = []
    for p, q in test_reps:
        data = compute_impedance_and_packing(p, q)
        results.append(data)
        
        print(f"({data['p']},{data['q']}){'':<4} {data['dim']:<5} "
              f"{data['C2']:<8.3f} {data['Z']:<12.4f} {data['Z_over_4pi']:<10.4f} "
              f"{data['covering_radius_mean']:<10.4f} {data['packing_efficiency_mean']:<10.3f}")
    
    print("\n" + "="*80)
    print("Key Observations:")
    print("="*80)
    
    # Analyze correlation between Z and packing
    Z_values = [d['Z'] for d in results]
    packing_values = [d['packing_efficiency_mean'] for d in results]
    
    if len(Z_values) > 1:
        correlation = np.corrcoef(Z_values, packing_values)[0, 1]
        print(f"Correlation(Z, PackingEff): {correlation:.3f}")
        print(f"\nImpedance range: Z ∈ [{min(Z_values):.4f}, {max(Z_values):.4f}]")
        print(f"Packing eff range: [{min(packing_values):.3f}, {max(packing_values):.3f}]")
