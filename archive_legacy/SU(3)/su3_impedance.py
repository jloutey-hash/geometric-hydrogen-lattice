"""
SU(3) Symplectic Impedance: Information-Theoretic Coupling Constants

Extends the U(1) impedance framework (α = S_photon / C_electron) to SU(3) color symmetry.
Treats coupling constants as geometric information conversion ratios between manifolds.

Mathematical Framework:
- Matter Capacity: C_SU3 = ∫ ω_matter (symplectic volume of color phase space)
- Gauge Action: S_SU3 = ∑ Tr[U_plaquette] (holonomy around closed loops)
- Impedance: Z_SU3 = S_SU3 / C_SU3 (information conversion ratio)

CRITICAL DISCLAIMER: This is a geometric/information-theoretic probe. We do NOT claim:
- Derivation of QCD coupling α_s from first principles
- Direct physical interpretation of Z_SU3 as running coupling
- Exact correspondence with lattice QCD renormalization

We explore whether geometric impedance exhibits coupling-like behavior and scaling.

Author: Unified Geometry Framework
Date: February 5, 2026
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.spatial import distance_matrix
from scipy.special import comb

from su3_spherical_embedding import SU3SphericalEmbedding, SphericalState
from general_rep_builder import GeneralRepBuilder


@dataclass
class ImpedanceData:
    """Container for impedance calculation results"""
    p: int
    q: int
    dim: int
    C2: float
    
    # Matter capacity components
    C_matter: float
    C_berry: float          # Berry curvature contribution
    C_symplectic: float     # Direct symplectic form
    
    # Gauge action components
    S_wilson: float         # Wilson loop contribution
    S_plaquette: float      # Plaquette action
    S_holonomy: float       # Total holonomy
    
    # Impedance ratios
    Z_impedance: float      # S_total / C_total
    Z_normalized: float     # Normalized by C2
    Z_dimensionless: float  # Scaled to O(1)
    
    # Geometric properties
    num_shells: int
    states_per_shell: List[int]
    shell_radii: List[float]
    
    # Information-theoretic interpretations
    entropy_matter: float
    entropy_gauge: float
    info_conversion_rate: float


class SU3SymplecticImpedance:
    """
    Compute symplectic impedance for SU(3) representations on spherical shells.
    
    Treats coupling constants as information conversion ratios between:
    - Matter manifold (SU(3) phase space on shells)
    - Gauge manifold (SU(3) connection space)
    
    Philosophy: Z_SU3 measures "resistance" to color charge flow on spherical shells.
    """
    
    def __init__(self, p: int, q: int, verbose: bool = True):
        """
        Initialize impedance calculator for representation (p,q).
        
        Parameters
        ----------
        p, q : int
            Dynkin labels
        verbose : bool
            Print diagnostic information
        """
        self.p = p
        self.q = q
        self.verbose = verbose
        
        # Create spherical embedding
        self.embedding = SU3SphericalEmbedding(p, q, r0=1.0, 
                                               scaling_mode='casimir',
                                               height_mode='linear')
        
        # Get representation properties
        self.dim = self.embedding.dim
        self.C2 = self.embedding.C2
        
        # Get operators if available
        self.rep_builder = GeneralRepBuilder()
        self.operators = self.rep_builder.get_irrep_operators(p, q)
        
        # Generate states
        self.states = self.embedding.create_spherical_states()
        self.shells = self.embedding.get_states_by_shell()
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"SU(3) Symplectic Impedance Calculator: (p,q)=({p},{q})")
            print(f"{'='*80}")
            print(f"  Dimension: {self.dim}")
            print(f"  Casimir C2: {self.C2:.4f}")
            print(f"  Number of shells: {len(self.shells)}")
            print(f"  States per shell: {[len(s) for s in self.shells.values()]}")
    
    def compute_matter_capacity(self) -> Dict[str, float]:
        """
        Compute matter capacity C_matter using Berry curvature and symplectic forms.
        
        C_matter measures the "volume" of color phase space that matter can occupy.
        
        Components:
        1. Berry curvature: Integrated over closed loops on shells
        2. Symplectic form: ω = Im⟨ψ|dψ⟩ integrated over plaquettes
        3. Casimir weighting: Each shell weighted by local C2 contribution
        
        Returns
        -------
        capacity : dict
            {'total': C_total, 'berry': C_berry, 'symplectic': C_symplectic}
        """
        if self.verbose:
            print(f"\nComputing Matter Capacity...")
        
        C_berry = 0.0
        C_symplectic = 0.0
        
        # Base capacity: minimal contribution from dimension and Casimir
        C_base = self.dim * np.sqrt(max(self.C2, 0.1))  # Minimum capacity
        
        # For each shell, compute capacity contributions
        for shell_r, shell_states in self.shells.items():
            n_states = len(shell_states)
            
            if n_states < 2:
                # Single state: minimal capacity from phase space volume
                C_symplectic += 4 * np.pi * shell_r**2
                continue
            
            if n_states < 3:
                # Two states: compute symplectic form only
                symp_contrib = self._compute_symplectic_form_shell(shell_states, shell_r)
                C_symplectic += symp_contrib
                continue
            
            # Berry curvature contribution: sum over closed triangular loops
            berry_contrib = self._compute_berry_curvature_shell(shell_states)
            
            # Symplectic form contribution: area of plaquettes times phase-space density
            symp_contrib = self._compute_symplectic_form_shell(shell_states, shell_r)
            
            C_berry += berry_contrib
            C_symplectic += symp_contrib
        
        # FIX (Feb 5, 2026): Ensure all contributions are finite before summing
        C_berry = C_berry if np.isfinite(C_berry) else 0.0
        C_symplectic = C_symplectic if np.isfinite(C_symplectic) else 0.0
        
        # Total capacity (add contributions plus base)
        C_total = C_base + C_berry + C_symplectic
        
        # Final safety check
        if not np.isfinite(C_total) or C_total <= 0:
            # Fallback to base capacity
            C_total = C_base
            if self.verbose:
                print(f"  WARNING: C_total not finite, using C_base = {C_base:.6f}")
        
        # Normalize by Casimir (dimensionless capacity per unit C2)
        if self.C2 > 1e-10:
            C_normalized = C_total / self.C2
        else:
            C_normalized = C_total
        
        if self.verbose:
            print(f"  Base capacity:     {C_base:.6f}")
            print(f"  Berry curvature:   {C_berry:.6f}")
            print(f"  Symplectic form:   {C_symplectic:.6f}")
            print(f"  Total capacity:    {C_total:.6f}")
            print(f"  Normalized by C2:  {C_normalized:.6f}")
        
        return {
            'total': C_total,
            'berry': C_berry,
            'symplectic': C_symplectic,
            'normalized': C_normalized
        }
    
    def _compute_berry_curvature_shell(self, shell_states: List[SphericalState]) -> float:
        """
        Compute Berry curvature contribution for one shell.
        
        For each closed triangular loop on the shell:
        γ = arg[⟨ψ₁|ψ₂⟩⟨ψ₂|ψ₃⟩⟨ψ₃|ψ₁⟩]
        
        Sum over all triangles, weighted by solid angle.
        """
        n = len(shell_states)
        if n < 3:
            return 0.0
        
        total_berry = 0.0
        
        # For computational efficiency, sample triangles rather than enumerate all
        # For small n, use all; for large n, sample
        max_triangles = 1000
        if comb(n, 3) <= max_triangles:
            # Use all triangles
            from itertools import combinations
            triangles = list(combinations(range(n), 3))
        else:
            # Random sample
            rng = np.random.RandomState(42)
            triangles = [tuple(rng.choice(n, 3, replace=False)) for _ in range(max_triangles)]
        
        # FIX (Feb 5, 2026): Filter out NaN and inf values during accumulation
        valid_berry_count = 0
        for i, j, k in triangles:
            # Get angular positions
            theta_i, phi_i = shell_states[i].theta, shell_states[i].phi
            theta_j, phi_j = shell_states[j].theta, shell_states[j].phi
            theta_k, phi_k = shell_states[k].theta, shell_states[k].phi
            
            # Compute solid angle of spherical triangle
            solid_angle = self._spherical_triangle_area(
                theta_i, phi_i, theta_j, phi_j, theta_k, phi_k
            )
            
            # Only accumulate finite values
            if np.isfinite(solid_angle):
                berry_phase = solid_angle
                total_berry += abs(berry_phase)
                valid_berry_count += 1
        
        # Normalize by number of valid triangles
        if valid_berry_count > 0:
            total_berry /= valid_berry_count
        else:
            # No valid triangles - use minimal contribution
            total_berry = 0.0
        
        # Scale by total number of possible triangles (coverage factor)
        if comb(n, 3) > 0:
            coverage = min(1.0, len(triangles) / comb(n, 3))
            total_berry *= (1.0 / coverage) if coverage > 0 else 1.0
        
        return total_berry
    
    def _spherical_triangle_area(self, theta1: float, phi1: float,
                                  theta2: float, phi2: float,
                                  theta3: float, phi3: float) -> float:
        """
        Compute solid angle of spherical triangle using spherical excess formula.
        
        Ω = E = A + B + C - π
        where A, B, C are the angles of the spherical triangle.
        
        FIX (Feb 5, 2026): Added robust handling of degenerate triangles to prevent NaN propagation.
        """
        # Convert to Cartesian coordinates on unit sphere
        def sph_to_cart(theta, phi):
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            return np.array([x, y, z])
        
        v1 = sph_to_cart(theta1, phi1)
        v2 = sph_to_cart(theta2, phi2)
        v3 = sph_to_cart(theta3, phi3)
        
        # Compute side lengths (great circle distances)
        a = np.arccos(np.clip(np.dot(v2, v3), -1, 1))  # opposite to vertex 1
        b = np.arccos(np.clip(np.dot(v1, v3), -1, 1))  # opposite to vertex 2
        c = np.arccos(np.clip(np.dot(v1, v2), -1, 1))  # opposite to vertex 3
        
        # Check for degenerate triangle (any side too small or collinear points)
        min_side = 1e-8
        if a < min_side or b < min_side or c < min_side:
            return 0.0
        
        # Check sines are not too small (prevents division by zero)
        sin_a = np.sin(a)
        sin_b = np.sin(b)
        sin_c = np.sin(c)
        
        if abs(sin_a) < 1e-8 or abs(sin_b) < 1e-8 or abs(sin_c) < 1e-8:
            return 0.0
        
        # Compute angles using spherical law of cosines
        try:
            cos_A = (np.cos(a) - np.cos(b) * np.cos(c)) / (sin_b * sin_c)
            cos_B = (np.cos(b) - np.cos(a) * np.cos(c)) / (sin_a * sin_c)
            cos_C = (np.cos(c) - np.cos(a) * np.cos(b)) / (sin_a * sin_b)
            
            # Clip to valid range
            cos_A = np.clip(cos_A, -1, 1)
            cos_B = np.clip(cos_B, -1, 1)
            cos_C = np.clip(cos_C, -1, 1)
            
            A = np.arccos(cos_A)
            B = np.arccos(cos_B)
            C = np.arccos(cos_C)
            
            # Spherical excess
            E = A + B + C - np.pi
            
            # Return 0 if E is NaN or negative
            if np.isnan(E) or E < 0:
                return 0.0
            
            return abs(E)
        except:
            # Degenerate triangle
            return 0.0
    
    def _compute_symplectic_form_shell(self, shell_states: List[SphericalState], 
                                       radius: float) -> float:
        """
        Compute symplectic form contribution for one shell.
        
        ω = r² sin(θ) dθ ∧ dφ (area form on sphere of radius r)
        
        Integrate over Voronoi cells of each state.
        """
        n = len(shell_states)
        if n < 2:
            return 0.0
        
        # Total area of sphere at this radius
        total_area = 4 * np.pi * radius**2
        
        # For uniform distribution, each state gets equal area
        # For non-uniform, weight by local density
        
        # Measure angular distribution
        thetas = np.array([s.theta for s in shell_states])
        phis = np.array([s.phi for s in shell_states])
        
        # Compute pairwise angular distances
        coords = np.column_stack([thetas, phis])
        
        # Symplectic capacity ~ area / number of states * angular variance
        mean_area_per_state = total_area / n
        
        # Angular spread (variance)
        theta_var = np.var(thetas)
        phi_var = np.var(phis)
        spread_factor = np.sqrt(theta_var + phi_var)
        
        # Symplectic contribution
        symp_contrib = mean_area_per_state * spread_factor * n
        
        return symp_contrib
    
    def compute_gauge_action(self) -> Dict[str, float]:
        """
        Compute gauge action S_gauge using Wilson loops and plaquettes.
        
        S_gauge measures the "stiffness" of SU(3) gauge connections on shells.
        
        Components:
        1. Wilson loops: Tr[U] around closed paths
        2. Plaquette action: Sum over elementary squares
        3. Holonomy: Path-ordered exponentials
        
        Returns
        -------
        action : dict
            {'total': S_total, 'wilson': S_wilson, 'plaquette': S_plaq}
        """
        if self.verbose:
            print(f"\nComputing Gauge Action...")
        
        S_wilson = 0.0
        S_plaquette = 0.0
        
        # Base action: minimal contribution from dimension
        S_base = np.sqrt(self.dim) * np.sqrt(max(self.C2, 0.1))
        
        # For each shell, compute gauge action contributions
        for shell_r, shell_states in self.shells.items():
            n_states = len(shell_states)
            
            if n_states < 2:
                # Single state: minimal action
                S_wilson += shell_r
                continue
            
            if n_states < 3:
                # Two states: pairwise Wilson loops
                wilson_contrib = self._compute_wilson_loops_shell(shell_states)
                S_wilson += wilson_contrib
                continue
            
            if n_states < 4:
                # Three states: Wilson loops only
                wilson_contrib = self._compute_wilson_loops_shell(shell_states)
                S_wilson += wilson_contrib
                continue
            
            # Wilson loop contribution
            wilson_contrib = self._compute_wilson_loops_shell(shell_states)
            
            # Plaquette action
            plaq_contrib = self._compute_plaquette_action_shell(shell_states, shell_r)
            
            S_wilson += wilson_contrib
            S_plaquette += plaq_contrib
        
        # Total action
        S_total = S_base + S_wilson + S_plaquette
        
        # Normalize by dimension (action per degree of freedom)
        S_normalized = S_total / self.dim if self.dim > 0 else S_total
        
        if self.verbose:
            print(f"  Base action:       {S_base:.6f}")
            print(f"  Wilson loops:      {S_wilson:.6f}")
            print(f"  Plaquette action:  {S_plaquette:.6f}")
            print(f"  Total action:      {S_total:.6f}")
            print(f"  Normalized by dim: {S_normalized:.6f}")
        
        return {
            'total': S_total,
            'wilson': S_wilson,
            'plaquette': S_plaquette,
            'normalized': S_normalized
        }
    
    def _compute_wilson_loops_shell(self, shell_states: List[SphericalState]) -> float:
        """
        Compute Wilson loop contribution for one shell.
        
        W = Tr[U_path] where U_path = U_1 U_2 ... U_n
        
        For SU(3), U_link ~ exp(iθ·T) where T are generators.
        Approximate using geodesic distance on sphere.
        """
        n = len(shell_states)
        if n < 3:
            return 0.0
        
        total_wilson = 0.0
        
        # Sample closed loops (triangles and squares)
        max_loops = 500
        rng = np.random.RandomState(42)
        
        num_loops = min(max_loops, n * (n-1) // 2)
        
        for _ in range(num_loops):
            # Random closed loop (3 or 4 vertices)
            loop_size = rng.choice([3, 4], p=[0.7, 0.3])
            loop_size = min(loop_size, n)  # Don't exceed available states
            
            if loop_size > n:
                loop_size = n
            
            loop_indices = rng.choice(n, loop_size, replace=False)
            
            # Compute total "phase" around loop (geodesic length)
            total_length = 0.0
            for i in range(loop_size):
                j = (i + 1) % loop_size
                idx_i, idx_j = loop_indices[i], loop_indices[j]
                
                # Angular distance
                dtheta = shell_states[idx_i].theta - shell_states[idx_j].theta
                dphi = shell_states[idx_i].phi - shell_states[idx_j].phi
                
                # Normalize dphi to [-π, π]
                dphi = np.arctan2(np.sin(dphi), np.cos(dphi))
                
                # Great circle distance
                angular_dist = np.sqrt(dtheta**2 + dphi**2)
                total_length += angular_dist
            
            # Wilson loop ~ exp(i * total_length)
            # For SU(3), Re[Tr(U)] = sum of eigenvalue real parts
            # Approximate: Tr(U) ~ 3 * cos(total_length / 3)
            wilson_value = 3 * np.cos(total_length / 3.0)
            
            total_wilson += abs(wilson_value)
        
        # Normalize
        if num_loops > 0:
            total_wilson /= num_loops
        
        return total_wilson
    
    def _compute_plaquette_action_shell(self, shell_states: List[SphericalState],
                                        radius: float) -> float:
        """
        Compute plaquette action for one shell.
        
        S_plaq = ∑_□ (1 - Re[Tr(U_□)] / 3)
        
        where U_□ is the product of link matrices around a plaquette.
        """
        n = len(shell_states)
        if n < 4:
            return 0.0
        
        total_action = 0.0
        
        # Sample plaquettes (quadrilaterals on sphere)
        max_plaq = 300
        rng = np.random.RandomState(42)
        
        # Calculate actual number of plaquettes we can sample
        num_plaq = min(max_plaq, max(1, n * (n-1) // 4))
        
        for _ in range(int(num_plaq)):
            # Random 4 vertices, but don't exceed n
            sample_size = min(4, n)
            indices = rng.choice(n, sample_size, replace=False)
            
            # Compute plaquette "area" (solid angle)
            theta_vals = [shell_states[i].theta for i in indices]
            phi_vals = [shell_states[i].phi for i in indices]
            
            # Approximate solid angle as variance of angles
            theta_span = np.ptp(theta_vals)  # max - min
            phi_span = np.ptp(phi_vals)
            
            solid_angle = theta_span * phi_span * np.sin(np.mean(theta_vals))
            
            # Plaquette action ~ solid_angle² (curvature ~ area)
            # For Yang-Mills, S ~ ∫ Tr(F²) ~ (solid angle)²
            plaq_action = solid_angle**2
            
            total_action += plaq_action
        
        # Normalize
        if num_plaq > 0:
            total_action /= num_plaq
        
        # Scale by radius² (larger shells have more action)
        total_action *= radius**2
        
        return total_action
    
    def compute_impedance(self) -> ImpedanceData:
        """
        Compute full impedance Z_SU3 = S_gauge / C_matter.
        
        This is the information-theoretic coupling constant analog for SU(3).
        
        Returns
        -------
        impedance : ImpedanceData
            Complete impedance data structure
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Computing SU(3) Impedance")
            print(f"{'='*80}")
        
        # Compute matter capacity
        capacity = self.compute_matter_capacity()
        
        # Compute gauge action
        action = self.compute_gauge_action()
        
        # Impedance ratios
        C_total = capacity['total']
        S_total = action['total']
        
        if C_total > 1e-10:
            Z_impedance = S_total / C_total
        else:
            Z_impedance = np.inf
        
        # Normalized by Casimir
        Z_normalized = Z_impedance / self.C2 if self.C2 > 1e-10 else Z_impedance
        
        # Dimensionless scaling (target O(0.1) like α)
        Z_dimensionless = Z_impedance / (4 * np.pi)
        
        # Information-theoretic quantities
        entropy_matter = np.log(C_total) if C_total > 0 else 0
        entropy_gauge = np.log(S_total) if S_total > 0 else 0
        info_conversion = entropy_gauge - entropy_matter
        
        # Shell statistics
        shell_radii = list(self.shells.keys())
        states_per_shell = [len(states) for states in self.shells.values()]
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"IMPEDANCE RESULTS")
            print(f"{'='*80}")
            print(f"  Matter capacity C:     {C_total:.6f}")
            print(f"  Gauge action S:        {S_total:.6f}")
            print(f"  Impedance Z = S/C:     {Z_impedance:.6f}")
            print(f"  Normalized by C2:      {Z_normalized:.6f}")
            print(f"  Dimensionless (Z/4π):  {Z_dimensionless:.6f}")
            print(f"  Info conversion Δ:     {info_conversion:.6f}")
            print(f"{'='*80}")
        
        return ImpedanceData(
            p=self.p, q=self.q, dim=self.dim, C2=self.C2,
            C_matter=C_total,
            C_berry=capacity['berry'],
            C_symplectic=capacity['symplectic'],
            S_wilson=action['wilson'],
            S_plaquette=action['plaquette'],
            S_holonomy=S_total,
            Z_impedance=Z_impedance,
            Z_normalized=Z_normalized,
            Z_dimensionless=Z_dimensionless,
            num_shells=len(self.shells),
            states_per_shell=states_per_shell,
            shell_radii=shell_radii,
            entropy_matter=entropy_matter,
            entropy_gauge=entropy_gauge,
            info_conversion_rate=info_conversion
        )


def scan_representations(max_sum: int = 5, verbose: bool = False) -> List[ImpedanceData]:
    """
    Scan impedance across multiple representations.
    
    Parameters
    ----------
    max_sum : int
        Maximum p+q to compute
    verbose : bool
        Print detailed output for each representation
    
    Returns
    -------
    results : List[ImpedanceData]
        Impedance data for all representations
    """
    print(f"\n{'#'*80}")
    print(f"# SU(3) IMPEDANCE SCAN: Representations with p+q ≤ {max_sum}")
    print(f"{'#'*80}\n")
    
    results = []
    rep_builder = GeneralRepBuilder()
    available = rep_builder.list_available_irreps(verbose=False)
    
    # Filter by p+q <= max_sum
    reps_to_scan = [(p, q) for (p, q) in available if p + q <= max_sum]
    reps_to_scan = sorted(reps_to_scan, key=lambda x: (x[0] + x[1], x[0]))
    
    for p, q in reps_to_scan:
        try:
            calculator = SU3SymplecticImpedance(p, q, verbose=verbose)
            impedance = calculator.compute_impedance()
            results.append(impedance)
            
            if not verbose:
                print(f"({p},{q}): Z = {impedance.Z_impedance:.4f}, "
                      f"Z/C2 = {impedance.Z_normalized:.4f}, "
                      f"Z/4π = {impedance.Z_dimensionless:.6f}")
        
        except Exception as e:
            print(f"({p},{q}): ERROR - {e}")
    
    return results


def analyze_scaling(results: List[ImpedanceData]) -> Dict[str, any]:
    """
    Analyze scaling laws and resonances in impedance data.
    
    Parameters
    ----------
    results : List[ImpedanceData]
        Impedance data from scan
    
    Returns
    -------
    analysis : dict
        Scaling fits, resonances, correlations
    """
    print(f"\n{'='*80}")
    print(f"SCALING ANALYSIS")
    print(f"{'='*80}\n")
    
    # Extract data arrays
    dims = np.array([r.dim for r in results])
    C2s = np.array([r.C2 for r in results])
    Zs = np.array([r.Z_impedance for r in results])
    Zs_norm = np.array([r.Z_normalized for r in results])
    Zs_dimless = np.array([r.Z_dimensionless for r in results])
    
    # Correlation with Casimir
    if len(C2s) > 1 and np.std(C2s) > 1e-10:
        corr_C2 = np.corrcoef(C2s, Zs)[0, 1]
        print(f"Correlation Z vs C2:     {corr_C2:.4f}")
    else:
        corr_C2 = 0
    
    # Correlation with dimension
    if len(dims) > 1 and np.std(dims) > 1e-10:
        corr_dim = np.corrcoef(dims, Zs)[0, 1]
        print(f"Correlation Z vs dim:    {corr_dim:.4f}")
    else:
        corr_dim = 0
    
    # Power law fits: Z ~ C2^α
    if len(C2s) > 2:
        log_C2 = np.log(C2s[C2s > 0])
        log_Z = np.log(Zs[C2s > 0])
        
        if len(log_C2) > 1:
            fit_coeffs = np.polyfit(log_C2, log_Z, 1)
            alpha_fit = fit_coeffs[0]
            print(f"Power law fit: Z ~ C2^{alpha_fit:.3f}")
        else:
            alpha_fit = None
    else:
        alpha_fit = None
    
    # Look for resonances (local minima/maxima)
    if len(Zs) > 2:
        # Find local extrema
        dZ = np.diff(Zs)
        sign_changes = np.where(np.diff(np.sign(dZ)))[0]
        
        if len(sign_changes) > 0:
            print(f"\nResonances found at indices: {sign_changes}")
            for idx in sign_changes:
                r = results[idx + 1]
                print(f"  ({r.p},{r.q}): Z = {r.Z_impedance:.4f}")
        else:
            print(f"\nNo resonances found (monotonic behavior)")
    
    # Statistics
    print(f"\nImpedance Statistics:")
    print(f"  Mean Z:        {np.mean(Zs):.4f}")
    print(f"  Std Z:         {np.std(Zs):.4f}")
    print(f"  Min Z:         {np.min(Zs):.4f} at ({results[np.argmin(Zs)].p},{results[np.argmin(Zs)].q})")
    print(f"  Max Z:         {np.max(Zs):.4f} at ({results[np.argmax(Zs)].p},{results[np.argmax(Zs)].q})")
    
    print(f"\nDimensionless Z/4π:")
    print(f"  Mean:          {np.mean(Zs_dimless):.6f}")
    print(f"  Range:         [{np.min(Zs_dimless):.6f}, {np.max(Zs_dimless):.6f}]")
    
    # Compare to U(1) α ≈ 1/137
    alpha_em = 1.0 / 137.0
    closest_idx = np.argmin(np.abs(Zs_dimless - alpha_em))
    print(f"\nComparison to U(1) α = 1/137 ≈ {alpha_em:.6f}:")
    print(f"  Closest: ({results[closest_idx].p},{results[closest_idx].q}) "
          f"with Z/4π = {Zs_dimless[closest_idx]:.6f}")
    
    return {
        'correlation_C2': corr_C2,
        'correlation_dim': corr_dim,
        'power_law_exponent': alpha_fit,
        'mean_Z': np.mean(Zs),
        'std_Z': np.std(Zs),
        'dimensionless_range': (np.min(Zs_dimless), np.max(Zs_dimless))
    }


def plot_impedance_scaling(results: List[ImpedanceData], 
                           save_path: Optional[str] = None):
    """
    Create comprehensive plots of impedance scaling.
    
    Parameters
    ----------
    results : List[ImpedanceData]
        Impedance data
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('SU(3) Symplectic Impedance Scaling Analysis', fontsize=14, fontweight='bold')
    
    # Extract data
    labels = [f"({r.p},{r.q})" for r in results]
    C2s = [r.C2 for r in results]
    dims = [r.dim for r in results]
    Zs = [r.Z_impedance for r in results]
    Zs_norm = [r.Z_normalized for r in results]
    Zs_dimless = [r.Z_dimensionless for r in results]
    
    # Plot 1: Z vs C2
    axes[0, 0].scatter(C2s, Zs, s=100, alpha=0.7, c=dims, cmap='viridis')
    axes[0, 0].set_xlabel('Casimir C₂', fontsize=11)
    axes[0, 0].set_ylabel('Impedance Z', fontsize=11)
    axes[0, 0].set_title('Z vs Casimir')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Z vs dim
    axes[0, 1].scatter(dims, Zs, s=100, alpha=0.7, c=C2s, cmap='plasma')
    axes[0, 1].set_xlabel('Dimension', fontsize=11)
    axes[0, 1].set_ylabel('Impedance Z', fontsize=11)
    axes[0, 1].set_title('Z vs Dimension')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Normalized Z
    axes[0, 2].bar(range(len(results)), Zs_norm, alpha=0.7)
    axes[0, 2].set_xlabel('Representation Index', fontsize=11)
    axes[0, 2].set_ylabel('Z / C₂', fontsize=11)
    axes[0, 2].set_title('Casimir-Normalized Impedance')
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Dimensionless Z/4π
    alpha_em = 1/137
    axes[1, 0].bar(range(len(results)), Zs_dimless, alpha=0.7, color='green')
    axes[1, 0].axhline(alpha_em, color='red', linestyle='--', linewidth=2, label='α(U(1)) = 1/137')
    axes[1, 0].set_xlabel('Representation Index', fontsize=11)
    axes[1, 0].set_ylabel('Z / 4π', fontsize=11)
    axes[1, 0].set_title('Dimensionless Impedance (cf. α)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Log-log: Z vs C2
    if len(C2s) > 1:
        C2_pos = [c for c in C2s if c > 0]
        Z_pos = [Zs[i] for i, c in enumerate(C2s) if c > 0 and np.isfinite(Zs[i])]
        C2_for_Z = [c for i, c in enumerate(C2s) if c > 0 and np.isfinite(Zs[i])]
        if len(C2_for_Z) > 1 and len(Z_pos) > 1:
            axes[1, 1].scatter(C2_for_Z, Z_pos, s=100, alpha=0.7)
            axes[1, 1].set_xscale('log')
            axes[1, 1].set_yscale('log')
            axes[1, 1].set_xlabel('log(C₂)', fontsize=11)
            axes[1, 1].set_ylabel('log(Z)', fontsize=11)
            axes[1, 1].set_title('Log-Log: Z vs C₂')
            axes[1, 1].grid(True, alpha=0.3, which='both')
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor log-log plot',
                           ha='center', va='center', transform=axes[1, 1].transAxes)
    
    # Plot 6: Information conversion
    info_conv = [r.info_conversion_rate for r in results]
    axes[1, 2].bar(range(len(results)), info_conv, alpha=0.7, color='purple')
    axes[1, 2].set_xlabel('Representation Index', fontsize=11)
    axes[1, 2].set_ylabel('ΔS = S_gauge - S_matter', fontsize=11)
    axes[1, 2].set_title('Information Conversion Rate')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.close()  # Close instead of showing to avoid hanging


def export_impedance_data(results: List[ImpedanceData], filepath: str):
    """Export impedance data to CSV for further analysis."""
    import csv
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'p', 'q', 'dim', 'C2', 'C_matter', 'C_berry', 'C_symplectic',
            'S_wilson', 'S_plaquette', 'S_total', 'Z_impedance', 'Z_normalized',
            'Z_dimensionless', 'num_shells', 'entropy_matter', 'entropy_gauge',
            'info_conversion'
        ])
        
        # Data
        for r in results:
            writer.writerow([
                r.p, r.q, r.dim, r.C2, r.C_matter, r.C_berry, r.C_symplectic,
                r.S_wilson, r.S_plaquette, r.S_holonomy, r.Z_impedance, r.Z_normalized,
                r.Z_dimensionless, r.num_shells, r.entropy_matter, r.entropy_gauge,
                r.info_conversion_rate
            ])
    
    print(f"\nData exported to: {filepath}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║  SU(3) SYMPLECTIC IMPEDANCE: Information-Theoretic Coupling Constants ║
    ║                                                                        ║
    ║  Treating gauge coupling as geometric information conversion ratio    ║
    ║  Z_SU3 = S_gauge / C_matter                                           ║
    ║                                                                        ║
    ║  DISCLAIMER: Geometric probe, NOT α_s derivation from first principles║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Scan representations
    results = scan_representations(max_sum=4, verbose=False)
    
    # Analyze scaling
    analysis = analyze_scaling(results)
    
    # Plot results
    plot_impedance_scaling(results, save_path='su3_impedance_scaling.png')
    
    # Export data
    export_impedance_data(results, 'su3_impedance_data.csv')
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Computed impedance for {len(results)} representations")
    print(f"Results saved to: su3_impedance_scaling.png, su3_impedance_data.csv")
