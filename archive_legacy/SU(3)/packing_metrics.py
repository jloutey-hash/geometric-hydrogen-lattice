"""
SU(3) Packing Metrics Module

Compute geometric packing efficiency on spherical shells for SU(3) representations.

Given an SU3SphericalEmbedding (states positioned on spherical shells), compute:
1. Covering radius: Maximum distance to nearest neighbor
2. Kissing number: Average number of close neighbors  
3. Voronoi volume variance: Inhomogeneity of state distribution

These metrics quantify how efficiently states "pack" on the sphere, analogous to
the packing efficiency of helical windings in U(1) hydrogen.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class PackingMetrics:
    """Packing efficiency metrics for a spherical shell."""
    shell_index: int
    r_value: float
    n_states: int
    covering_radius: float  # Max distance to nearest neighbor (radians)
    kissing_number_mean: float  # Average number of close neighbors
    kissing_number_std: float
    angular_distance_mean: float  # Average angular separation (radians)
    angular_distance_std: float
    packing_efficiency: float  # Empirical: small covering_radius = good packing


def angular_distance_on_sphere(theta1: float, phi1: float, 
                                theta2: float, phi2: float) -> float:
    """
    Compute angular distance between two points on unit sphere.
    
    Uses spherical law of cosines:
    cos(d) = cos(θ1)cos(θ2) + sin(θ1)sin(θ2)cos(φ1 - φ2)
    
    Parameters
    ----------
    theta1, phi1 : float
        Polar and azimuthal angles of point 1 (radians)
    theta2, phi2 : float
        Polar and azimuthal angles of point 2 (radians)
    
    Returns
    -------
    distance : float
        Angular distance in radians, range [0, π]
    """
    cos_dist = (np.cos(theta1) * np.cos(theta2) + 
                np.sin(theta1) * np.sin(theta2) * np.cos(phi1 - phi2))
    
    # Clamp to [-1, 1] to handle numerical errors
    cos_dist = np.clip(cos_dist, -1.0, 1.0)
    
    return np.arccos(cos_dist)


def compute_shell_packing_metrics(states_on_shell: List, 
                                   shell_index: int = 0) -> PackingMetrics:
    """
    Compute packing metrics for states on a single spherical shell.
    
    Parameters
    ----------
    states_on_shell : list of SphericalState
        All states at a given radius r
    shell_index : int
        Index identifying the shell
    
    Returns
    -------
    metrics : PackingMetrics
        Packing efficiency metrics
    """
    n = len(states_on_shell)
    
    if n == 0:
        raise ValueError("Empty shell")
    
    if n == 1:
        # Single state - no meaningful packing metrics
        return PackingMetrics(
            shell_index=shell_index,
            r_value=states_on_shell[0].r,
            n_states=1,
            covering_radius=np.pi,  # Worst case
            kissing_number_mean=0.0,
            kissing_number_std=0.0,
            angular_distance_mean=0.0,
            angular_distance_std=0.0,
            packing_efficiency=0.0
        )
    
    # Extract coordinates
    r = states_on_shell[0].r  # All have same r on this shell
    theta = np.array([s.theta for s in states_on_shell])
    phi = np.array([s.phi for s in states_on_shell])
    
    # Compute pairwise angular distances
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = angular_distance_on_sphere(theta[i], phi[i], theta[j], phi[j])
            distances[i, j] = d
            distances[j, i] = d
    
    # For each state, find distance to nearest neighbor
    nearest_neighbor_distances = []
    for i in range(n):
        # Mask out self-distance (which is 0)
        other_distances = distances[i, distances[i, :] > 1e-10]
        if len(other_distances) > 0:
            nearest_neighbor_distances.append(np.min(other_distances))
    
    # Covering radius: maximum distance to nearest neighbor
    # (Larger = worse packing, states are more spread out)
    covering_radius = np.max(nearest_neighbor_distances) if nearest_neighbor_distances else np.pi
    
    # Average angular separation
    # Extract upper triangle (i < j) to avoid double-counting
    upper_tri_distances = distances[np.triu_indices(n, k=1)]
    angular_distance_mean = np.mean(upper_tri_distances)
    angular_distance_std = np.std(upper_tri_distances)
    
    # Kissing number: count neighbors within angular cutoff
    # Use 1.5 * median distance as cutoff for "close neighbor"
    median_distance = np.median(nearest_neighbor_distances) if nearest_neighbor_distances else np.pi
    kissing_cutoff = 1.5 * median_distance
    
    kissing_numbers = []
    for i in range(n):
        # Count states within kissing_cutoff (excluding self)
        close_neighbors = np.sum((distances[i, :] < kissing_cutoff) & (distances[i, :] > 1e-10))
        kissing_numbers.append(close_neighbors)
    
    kissing_number_mean = np.mean(kissing_numbers)
    kissing_number_std = np.std(kissing_numbers)
    
    # Packing efficiency: empirical metric (0 = bad, 1 = good)
    # Smaller covering_radius = better packing
    # Normalize by theoretical minimum for n points on sphere
    # (This is a rough heuristic, not rigorous)
    theoretical_min_covering = np.pi / np.sqrt(n) if n > 1 else np.pi
    packing_efficiency = min(1.0, theoretical_min_covering / (covering_radius + 1e-10))
    
    return PackingMetrics(
        shell_index=shell_index,
        r_value=r,
        n_states=n,
        covering_radius=covering_radius,
        kissing_number_mean=kissing_number_mean,
        kissing_number_std=kissing_number_std,
        angular_distance_mean=angular_distance_mean,
        angular_distance_std=angular_distance_std,
        packing_efficiency=packing_efficiency
    )


def compute_packing_metrics(embedding) -> Dict[str, PackingMetrics]:
    """
    Compute packing metrics for all shells in an SU3SphericalEmbedding.
    
    Parameters
    ----------
    embedding : SU3SphericalEmbedding
        The spherical embedding to analyze
    
    Returns
    -------
    metrics_by_shell : dict
        {shell_index: PackingMetrics} for each shell
    """
    # Get states grouped by shell
    shells = embedding.get_states_by_shell()
    
    metrics_by_shell = {}
    for shell_idx, states in shells.items():
        metrics = compute_shell_packing_metrics(states, shell_index=shell_idx)
        metrics_by_shell[shell_idx] = metrics
    
    return metrics_by_shell


def print_packing_report(p: int, q: int, metrics_by_shell: Dict[int, PackingMetrics]):
    """
    Print formatted packing metrics report.
    
    Parameters
    ----------
    p, q : int
        SU(3) representation labels
    metrics_by_shell : dict
        Output from compute_packing_metrics
    """
    print(f"\n{'='*80}")
    print(f"SU(3) Packing Metrics: (p, q) = ({p}, {q})")
    print(f"{'='*80}")
    
    print(f"\n{'Shell':<6} {'r':<8} {'N':<4} {'Cover_rad':<12} {'Kiss#':<10} "
          f"{'AngDist':<12} {'PackEff':<8}")
    print(f"{'-'*80}")
    
    for shell_idx in sorted(metrics_by_shell.keys()):
        m = metrics_by_shell[shell_idx]
        print(f"{m.shell_index:<6} {m.r_value:<8.3f} {m.n_states:<4} "
              f"{m.covering_radius:<12.4f} {m.kissing_number_mean:<10.2f} "
              f"{m.angular_distance_mean:<12.4f} {m.packing_efficiency:<8.3f}")
    
    # Overall statistics
    all_covering_radii = [m.covering_radius for m in metrics_by_shell.values()]
    all_packing_effs = [m.packing_efficiency for m in metrics_by_shell.values()]
    
    print(f"\n{'Summary':<6}")
    print(f"  Average covering radius:   {np.mean(all_covering_radii):.4f} rad")
    print(f"  Average packing efficiency: {np.mean(all_packing_effs):.3f}")
    print(f"  Total states:              {sum(m.n_states for m in metrics_by_shell.values())}")


# Example usage
if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'SU(3)')
    
    from su3_spherical_embedding import SU3SphericalEmbedding
    
    # Test on a few representations
    test_reps = [(1, 0), (0, 1), (1, 1), (2, 0)]
    
    for p, q in test_reps:
        embedding = SU3SphericalEmbedding(p, q)
        metrics = compute_packing_metrics(embedding)
        print_packing_report(p, q, metrics)
