"""
Phase 15.2: Angular Laplacian Coupling

Implement full angular Laplacian operator on S² lattice with off-diagonal couplings.

Key difference from Phase 15.1:
- Phase 15.1: Used diagonal L² = ℓ(ℓ+1) eigenvalue
- Phase 15.2: Build full angular Laplacian with nearest-neighbor couplings

The angular Laplacian on S² is:
    ∇²_S² = (1/sin²θ)[∂²/∂φ² + sin(θ)∂/∂θ(sin(θ)∂/∂θ)]

On a discrete lattice, this creates couplings between neighboring angular sites.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from lattice import PolarLattice


class AngularLaplacian:
    """
    Build angular Laplacian operator on S² for a given ℓ channel.
    
    The discrete angular Laplacian couples neighboring sites on the sphere.
    For a fixed ℓ, we have 2(2ℓ+1) sites distributed on S².
    """
    
    def __init__(self, ℓ):
        """
        Parameters
        ----------
        ℓ : int
            Angular momentum quantum number
        """
        self.ℓ = ℓ
        
        # Build angular lattice for this ℓ
        lattice = PolarLattice(n_max=ℓ+1)
        self.points = [p for p in lattice.points if p.get('ℓ', 0) == ℓ]
        self.n_sites = len(self.points)
        
        # Build site index map
        self.site_index = {i: i for i in range(self.n_sites)}
    
    def build_laplacian(self, method='graph'):
        """
        Build discrete angular Laplacian.
        
        Parameters
        ----------
        method : str
            'graph': Simple graph Laplacian (nearest neighbors)
            'finite_diff': Finite difference on sphere (more accurate)
        
        Returns
        -------
        L_ang : sparse matrix
            Angular Laplacian operator
        """
        if method == 'graph':
            return self._build_graph_laplacian()
        elif method == 'finite_diff':
            return self._build_finite_difference_laplacian()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _build_graph_laplacian(self):
        """
        Build graph Laplacian: L = D - A
        
        For each site, connect to k nearest neighbors on S².
        """
        L = lil_matrix((self.n_sites, self.n_sites), dtype=float)
        
        # For each site, find nearest neighbors
        k_neighbors = 4  # Connect to 4 nearest neighbors
        
        for i in range(self.n_sites):
            p_i = self.points[i]
            pos_i = np.array([p_i['x_3d'], p_i['y_3d'], p_i['z_3d']])
            
            # Find k nearest neighbors
            distances = []
            for j in range(self.n_sites):
                if i == j:
                    continue
                p_j = self.points[j]
                pos_j = np.array([p_j['x_3d'], p_j['y_3d'], p_j['z_3d']])
                
                # Angular distance on sphere
                cos_angle = np.dot(pos_i, pos_j)  # Both on unit sphere
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                
                distances.append((j, angle))
            
            # Sort by distance and take k nearest
            distances.sort(key=lambda x: x[1])
            neighbors = [j for j, d in distances[:k_neighbors]]
            
            # Build graph Laplacian
            for j in neighbors:
                L[i, j] = -1.0
                L[i, i] += 1.0
        
        return L.tocsr()
    
    def _build_finite_difference_laplacian(self):
        """
        Build finite difference Laplacian on sphere.
        
        Uses weighted connections based on geodesic distances.
        """
        L = lil_matrix((self.n_sites, self.n_sites), dtype=float)
        
        # For each site, connect to neighbors with weights
        for i in range(self.n_sites):
            p_i = self.points[i]
            pos_i = np.array([p_i['x_3d'], p_i['y_3d'], p_i['z_3d']])
            
            # Find all neighbors within angular distance threshold
            threshold = np.pi / (2 * self.ℓ + 2)  # Adaptive to lattice density
            
            total_weight = 0.0
            for j in range(self.n_sites):
                if i == j:
                    continue
                    
                p_j = self.points[j]
                pos_j = np.array([p_j['x_3d'], p_j['y_3d'], p_j['z_3d']])
                
                # Angular distance
                cos_angle = np.dot(pos_i, pos_j)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                
                if angle < threshold:
                    # Weight inversely proportional to distance²
                    weight = 1.0 / (angle**2 + 1e-10)
                    L[i, j] = -weight
                    total_weight += weight
            
            L[i, i] = total_weight
        
        return L.tocsr()
    
    def get_eigenvalue_ℓ(self):
        """
        Return theoretical L² eigenvalue: ℓ(ℓ+1).
        
        For comparison with computed Laplacian eigenvalues.
        """
        return self.ℓ * (self.ℓ + 1)


class Lattice3D_AngularCoupling:
    """
    3D lattice with FULL angular Laplacian coupling.
    
    Extends Phase 15.1 by including off-diagonal angular terms.
    """
    
    def __init__(self, n_radial=80, ℓ_max=3, r_max=50.0):
        """
        Parameters
        ----------
        n_radial : int
            Number of radial points (including r=0)
        ℓ_max : int
            Maximum angular momentum
        r_max : float
            Maximum radius (Bohr radii)
        """
        self.n_radial = n_radial
        self.ℓ_max = ℓ_max
        self.r_max = r_max
        
        # Build radial grid
        self.r_grid = np.linspace(0.0, r_max, n_radial)
        self.dr = self.r_grid[1] - self.r_grid[0]
        
        # Build angular Laplacians for each ℓ
        self._build_angular_laplacians()
        
        # Build site list
        self._build_sites()
    
    def _build_angular_laplacians(self):
        """Build angular Laplacian operators for each ℓ."""
        self.angular_ops = {}
        
        for ℓ in range(self.ℓ_max + 1):
            ang_lap = AngularLaplacian(ℓ)
            L_ang = ang_lap.build_laplacian(method='graph')
            
            self.angular_ops[ℓ] = {
                'operator': L_ang,
                'points': ang_lap.points,
                'n_sites': ang_lap.n_sites,
                'L2_eigenvalue': ℓ * (ℓ + 1)
            }
    
    def _build_sites(self):
        """Build list of all (r, ℓ, angular) sites."""
        self.sites = []
        self.site_index = {}
        idx = 0
        
        # Only use interior radial points: r[1], r[2], ..., r[n-1]
        for i_r in range(1, self.n_radial):
            r = self.r_grid[i_r]
            
            # For each angular momentum channel
            for ℓ in range(self.ℓ_max + 1):
                ang_info = self.angular_ops[ℓ]
                n_ang = ang_info['n_sites']
                points = ang_info['points']
                
                # Add sites for this (r, ℓ) combination
                for i_ang in range(n_ang):
                    point = points[i_ang]
                    
                    site = {
                        'global_idx': idx,
                        'i_r': i_r,
                        'i_ang': i_ang,
                        'r': r,
                        'ℓ': ℓ,
                        'm': point.get('m_ℓ', 0),
                        'theta': point['θ'],
                        'phi': point.get('φ', 0.0),
                        'x_3d': r * point['x_3d'],
                        'y_3d': r * point['y_3d'],
                        'z_3d': r * point['z_3d']
                    }
                    self.sites.append(site)
                    self.site_index[(i_r, ℓ, i_ang)] = idx
                    idx += 1
        
        self.n_sites = len(self.sites)
    
    def build_hamiltonian(self, potential='hydrogen', angular_coupling_strength=1.0):
        """
        Build 3D Hamiltonian with FULL angular Laplacian.
        
        H = -(1/2)d²/dr² + (1/2r²)∇²_angular + V(r)
        
        Parameters
        ----------
        potential : str
            'hydrogen' for -1/r
        angular_coupling_strength : float
            Scaling factor for angular Laplacian (default=1.0)
        
        Returns
        -------
        H : sparse matrix
        """
        N = self.n_sites
        H = lil_matrix((N, N), dtype=float)
        
        for site in self.sites:
            i = site['global_idx']
            i_r = site['i_r']
            i_ang = site['i_ang']
            r = site['r']
            ℓ = site['ℓ']
            
            # Potential energy
            if potential == 'hydrogen':
                V = -1.0 / r
            else:
                V = 0.0
            
            H[i, i] += V
            
            # Radial kinetic: -(1/2)d²u/dr²
            if i_r == 1:
                # First interior: u(0) = 0
                H[i, i] += 1.0 / self.dr**2
                if (i_r+1, ℓ, i_ang) in self.site_index:
                    j_p = self.site_index[(i_r+1, ℓ, i_ang)]
                    H[i, j_p] += -0.5 / self.dr**2
            
            elif i_r == self.n_radial - 1:
                # Last point: u(r_max) ≈ 0
                if (i_r-1, ℓ, i_ang) in self.site_index:
                    j_m = self.site_index[(i_r-1, ℓ, i_ang)]
                    H[i, j_m] += -0.5 / self.dr**2
                H[i, i] += 1.0 / self.dr**2
            
            else:
                # Interior: standard 3-point
                if (i_r-1, ℓ, i_ang) in self.site_index:
                    j_m = self.site_index[(i_r-1, ℓ, i_ang)]
                    H[i, j_m] += -0.5 / self.dr**2
                
                H[i, i] += 1.0 / self.dr**2
                
                if (i_r+1, ℓ, i_ang) in self.site_index:
                    j_p = self.site_index[(i_r+1, ℓ, i_ang)]
                    H[i, j_p] += -0.5 / self.dr**2
        
        # Angular kinetic: (1/2r²)∇²_angular
        # This adds off-diagonal couplings!
        print(f"Adding angular Laplacian couplings...")
        for i_r in range(1, self.n_radial):
            r = self.r_grid[i_r]
            coeff = angular_coupling_strength / (2 * r**2)
            
            for ℓ in range(self.ℓ_max + 1):
                ang_info = self.angular_ops[ℓ]
                L_ang = ang_info['operator']
                n_ang = ang_info['n_sites']
                
                # Add angular Laplacian for all sites at this (r, ℓ)
                for i_ang in range(n_ang):
                    if (i_r, ℓ, i_ang) not in self.site_index:
                        continue
                    i = self.site_index[(i_r, ℓ, i_ang)]
                    
                    for j_ang in range(n_ang):
                        if (i_r, ℓ, j_ang) not in self.site_index:
                            continue
                        j = self.site_index[(i_r, ℓ, j_ang)]
                        
                        # Add coupling
                        H[i, j] += coeff * L_ang[i_ang, j_ang]
        
        return H.tocsr()
    
    def solve_spectrum(self, n_states=20, angular_coupling_strength=1.0):
        """Solve for energy eigenvalues."""
        print(f"\nBuilding Hamiltonian ({self.n_sites} sites)...")
        H = self.build_hamiltonian(angular_coupling_strength=angular_coupling_strength)
        
        print(f"Solving {self.n_sites} × {self.n_sites} Hamiltonian...")
        
        try:
            E, psi = eigsh(H, k=n_states, which='SA')
        except:
            print("Warning: eigsh failed, using dense solver")
            E, psi = np.linalg.eigh(H.toarray())
            E = E[:n_states]
            psi = psi[:, :n_states]
        
        # Sort
        idx = np.argsort(E)
        E = E[idx]
        psi = psi[:, idx]
        
        return E, psi


def compare_with_phase15_1():
    """
    Compare Phase 15.2 (angular coupling) with Phase 15.1 (diagonal only).
    """
    print("="*80)
    print("PHASE 15.2: ANGULAR LAPLACIAN COUPLING")
    print("="*80)
    print()
    print("Comparing diagonal L² vs full angular Laplacian")
    print()
    
    # Build lattice
    lattice = Lattice3D_AngularCoupling(n_radial=80, ℓ_max=2, r_max=50.0)
    
    print(f"Lattice configuration:")
    print(f"  Radial: {lattice.n_radial} points, r ∈ [0, {lattice.r_max}], dr = {lattice.dr:.3f}")
    print(f"  Angular: ℓ ≤ {lattice.ℓ_max}")
    print(f"  Total sites: {lattice.n_sites}")
    
    # Test different angular coupling strengths
    coupling_factors = [0.0, 0.5, 1.0, 2.0]
    
    results = {}
    for factor in coupling_factors:
        print(f"\n{'-'*80}")
        print(f"Angular coupling strength: {factor}")
        print(f"{'-'*80}")
        
        E, psi = lattice.solve_spectrum(n_states=15, angular_coupling_strength=factor)
        
        # Ground state
        E0_error = abs(E[0] + 0.5) / 0.5 * 100
        print(f"Ground state: E₀ = {E[0]:.6f} (theory: -0.5)")
        print(f"Error: {E0_error:.2f}%")
        
        results[factor] = E
    
    # Compare
    print(f"\n{'-'*80}")
    print("COMPARISON OF COUPLING STRENGTHS")
    print(f"{'-'*80}")
    print(f"{'State':>6} ", end="")
    for factor in coupling_factors:
        print(f"{'α='+str(factor):>12}", end="")
    print(f" {'Theory':>12}")
    print(f"{'-'*80}")
    
    # Theoretical energies
    theory = [-0.5, -0.125, -0.125, -0.125, -0.0556, -0.0556]
    
    for i in range(min(6, len(results[1.0]))):
        print(f"{i:>6} ", end="")
        for factor in coupling_factors:
            print(f"{results[factor][i]:>12.6f}", end="")
        if i < len(theory):
            print(f" {theory[i]:>12.6f}")
        else:
            print()
    
    # Summary
    print(f"\n{'-'*80}")
    print("SUMMARY")
    print(f"{'-'*80}")
    
    print("\nGround state errors:")
    for factor in coupling_factors:
        E0 = results[factor][0]
        error = abs(E0 + 0.5) / 0.5 * 100
        print(f"  α={factor}: {error:.2f}%")
    
    print("\nBest coupling strength:", end=" ")
    best_factor = min(coupling_factors, key=lambda f: abs(results[f][0] + 0.5))
    print(f"α={best_factor}")
    
    return lattice, results


if __name__ == '__main__':
    lattice, results = compare_with_phase15_1()
