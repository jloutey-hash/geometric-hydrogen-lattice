"""
Discrete Spherical Harmonic Transform (DSHT) Module

This module implements forward and inverse discrete spherical harmonic transforms
for the discrete angular momentum lattice. Analogous to FFT but for S².

Key Features:
1. Forward transform: f(ℓ,m) → coefficients a_ℓ^m
2. Inverse transform: a_ℓ^m → f(ℓ,m)
3. Discrete orthogonality: ⟨Y_ℓ^m | Y_ℓ'^m'⟩_discrete ≈ δ_ℓℓ' δ_mm'
4. Fast algorithms using lattice symmetries
5. Bandlimited reconstruction and filtering

Research Direction 7.5: Discrete S² Harmonic Analysis
Author: Quantum Lattice Project
Date: January 2026
Phase: Extension 1 - DSHT
"""

import numpy as np
from scipy import special
from scipy import sparse
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from dataclasses import dataclass

try:
    from .lattice import PolarLattice
except ImportError:
    from lattice import PolarLattice


@dataclass
class SphericalHarmonicCoefficients:
    """
    Container for spherical harmonic expansion coefficients.
    
    Attributes:
        coeffs: Dictionary mapping (ℓ, m) → complex coefficient
        ℓ_max: Maximum angular momentum included
        normalization: Normalization convention used
    """
    coeffs: Dict[Tuple[int, int], complex]
    ℓ_max: int
    normalization: str = "quantum"  # or "geodesy" or "unnormalized"
    
    def __getitem__(self, key: Tuple[int, int]) -> complex:
        """Get coefficient for (ℓ, m)."""
        return self.coeffs.get(key, 0.0 + 0.0j)
    
    def __setitem__(self, key: Tuple[int, int], value: complex):
        """Set coefficient for (ℓ, m)."""
        self.coeffs[key] = value
    
    def total_power(self) -> float:
        """Compute total power: Σ |a_ℓ^m|²."""
        return sum(abs(c)**2 for c in self.coeffs.values())
    
    def power_spectrum(self) -> Dict[int, float]:
        """Compute power per ℓ: P(ℓ) = Σ_m |a_ℓ^m|²."""
        spectrum = {}
        for (ℓ, m), c in self.coeffs.items():
            spectrum[ℓ] = spectrum.get(ℓ, 0.0) + abs(c)**2
        return spectrum


class DiscreteSphericalHarmonicTransform:
    """
    Discrete Spherical Harmonic Transform for the quantum lattice.
    
    Implements forward and inverse transforms analogous to FFT:
    - Forward: f(lattice_point) → a_ℓ^m (frequency domain)
    - Inverse: a_ℓ^m → f(lattice_point) (spatial domain)
    
    Key properties:
    - Uses lattice points as sampling of S²
    - Integration weights from lattice geometry
    - Exploits spin structure for efficiency
    - Preserves orthogonality to high precision
    
    Usage:
        >>> lattice = PolarLattice(n_max=10)
        >>> dsht = DiscreteSphericalHarmonicTransform(lattice)
        >>> # Some function on lattice
        >>> f = np.array([...])  # shape (N_sites,)
        >>> # Forward transform
        >>> coeffs = dsht.forward_transform(f)
        >>> # Inverse transform
        >>> f_reconstructed = dsht.inverse_transform(coeffs)
        >>> # Check round-trip error
        >>> error = np.linalg.norm(f - f_reconstructed) / np.linalg.norm(f)
    """
    
    def __init__(self, lattice: PolarLattice, normalization: str = "quantum"):
        """
        Initialize DSHT for given lattice.
        
        Parameters:
            lattice: PolarLattice defining discrete sampling of S²
            normalization: Spherical harmonic normalization
                - "quantum": Y_ℓ^m = sqrt((2ℓ+1)/(4π) * (ℓ-m)!/(ℓ+m)!) P_ℓ^m e^(imφ)
                - "geodesy": 4π-normalized (geophysics convention)
                - "unnormalized": Raw associated Legendre polynomials
        """
        self.lattice = lattice
        self.normalization = normalization
        self.N_sites = len(lattice.points)
        self.ℓ_max = lattice.ℓ_max
        
        # Build quantum number mappings
        self._build_site_mapping()
        
        # Compute integration weights for forward transform
        self._compute_integration_weights()
        
        # Pre-compute spherical harmonics at all lattice points
        self._precompute_harmonics()
        
        # Build fast transform matrices
        self._build_transform_matrices()
    
    def _build_site_mapping(self):
        """
        Build mapping from site index to (ℓ, m) quantum numbers.
        
        Each site has (ℓ, m_ℓ, m_s). For DSHT, we work with orbital part (ℓ, m_ℓ).
        """
        self.site_to_quantum = {}
        self.quantum_to_sites = {}  # Maps (ℓ, m) → list of site indices
        
        for idx, point in enumerate(self.lattice.points):
            ℓ = point['ℓ']
            m = point['m_ℓ']
            m_s = point['m_s']
            
            self.site_to_quantum[idx] = (ℓ, m, m_s)
            
            # Group by (ℓ, m) - spin partners have same orbital state
            key = (ℓ, m)
            if key not in self.quantum_to_sites:
                self.quantum_to_sites[key] = []
            self.quantum_to_sites[key].append(idx)
    
    def _compute_integration_weights(self):
        """
        Compute integration weights for discrete inner products.
        
        For S² integrals: ∫_S² f(θ,φ) dΩ ≈ Σ_i w_i f(θ_i, φ_i)
        
        Uses area-based weights accounting for:
        1. Each ring ℓ represents a latitude band on S²
        2. Band has angular extent Δθ ~ 1/ℓ_max
        3. Solid angle element: dΩ = sin(θ) dθ dφ
        4. Points in ring share total band solid angle
        
        For hemisphere structure: each (ℓ, m) orbital appears twice (spin up/down),
        so we weight each point by half the orbital's solid angle.
        """
        self.weights = np.zeros(self.N_sites)
        
        # For each (ℓ, m) orbital, compute its solid angle contribution
        orbital_solid_angles = {}
        
        # Total solid angle of sphere
        total_solid_angle = 4.0 * np.pi
        
        # Each ℓ level gets equal solid angle (rough approximation)
        # Better: weight by number of m states (2ℓ+1)
        total_m_states = sum(2*ℓ + 1 for ℓ in range(self.ℓ_max + 1))
        
        for ℓ in range(self.ℓ_max + 1):
            n_m_states = 2*ℓ + 1  # Number of m values for this ℓ
            # Solid angle per ℓ level proportional to degeneracy
            solid_angle_per_ℓ = total_solid_angle * n_m_states / total_m_states
            # Each m state gets equal share
            solid_angle_per_m = solid_angle_per_ℓ / n_m_states
            
            for m in range(-ℓ, ℓ + 1):
                orbital_solid_angles[(ℓ, m)] = solid_angle_per_m
        
        # Assign weights: each site gets half the orbital solid angle (spin degeneracy)
        for idx, point in enumerate(self.lattice.points):
            ℓ = point['ℓ']
            m = point['m_ℓ']
            # Each point represents one spin state of the orbital
            # Two points (spin up/down) share the orbital's solid angle
            self.weights[idx] = 0.5 * orbital_solid_angles[(ℓ, m)]
    
    def _precompute_harmonics(self):
        """
        Pre-compute Y_ℓ^m(θ, φ) at all lattice points.
        
        Stores as dictionary: (site_idx, ℓ', m') → Y_ℓ'^m'(θ_site, φ_site)
        """
        self.Y_values = {}
        
        for idx, point in enumerate(self.lattice.points):
            # Get spherical coordinates from 3D position
            x, y, z = point['x_3d'], point['y_3d'], point['z_3d']
            r = np.sqrt(x**2 + y**2 + z**2)
            
            if r < 1e-10:  # Handle origin
                theta = 0.0
                phi = 0.0
            else:
                theta = np.arccos(np.clip(z / r, -1, 1))  # [0, π]
                phi = np.arctan2(y, x)  # [-π, π]
            
            # Compute Y_ℓ^m for all (ℓ, m) up to ℓ_max
            for ℓ in range(self.ℓ_max + 1):
                for m in range(-ℓ, ℓ + 1):
                    Y = self._spherical_harmonic(ℓ, m, theta, phi)
                    self.Y_values[(idx, ℓ, m)] = Y
    
    def _spherical_harmonic(self, ℓ: int, m: int, theta: float, phi: float) -> complex:
        """
        Compute spherical harmonic Y_ℓ^m(θ, φ).
        
        Uses scipy.special.sph_harm which implements:
        Y_ℓ^m(θ,φ) = sqrt((2ℓ+1)/(4π) * (ℓ-|m|)!/(ℓ+|m|)!) P_ℓ^|m|(cos θ) e^(imφ)
        
        Parameters:
            ℓ: Angular momentum quantum number
            m: Magnetic quantum number (-ℓ ≤ m ≤ ℓ)
            theta: Polar angle [0, π]
            phi: Azimuthal angle [0, 2π]
        
        Returns:
            Complex value Y_ℓ^m(θ, φ)
        """
        # scipy.special.sph_harm(m, ℓ, phi, theta) - note argument order!
        return special.sph_harm(m, ℓ, phi, theta)
    
    def _build_transform_matrices(self):
        """
        Build transform matrices for fast forward/inverse transforms.
        
        Forward matrix F: f → a where a_ℓ^m = Σ_i F_ℓm,i f_i
        Inverse matrix F†: a → f where f_i = Σ_ℓm (F†)_i,ℓm a_ℓ^m
        
        These are dense matrices but can exploit sparsity for large ℓ_max.
        """
        # Number of (ℓ, m) pairs
        self.N_modes = sum(2*ℓ + 1 for ℓ in range(self.ℓ_max + 1))
        
        # Forward transform matrix: (N_modes, N_sites)
        self.F_forward = np.zeros((self.N_modes, self.N_sites), dtype=complex)
        
        # Index mapping: (ℓ, m) → row index
        self.mode_to_index = {}
        mode_idx = 0
        for ℓ in range(self.ℓ_max + 1):
            for m in range(-ℓ, ℓ + 1):
                self.mode_to_index[(ℓ, m)] = mode_idx
                mode_idx += 1
        
        # Fill forward matrix: F_ℓm,i = w_i * conj(Y_ℓ^m(θ_i, φ_i))
        for ℓ in range(self.ℓ_max + 1):
            for m in range(-ℓ, ℓ + 1):
                row_idx = self.mode_to_index[(ℓ, m)]
                for site_idx in range(self.N_sites):
                    Y_val = self.Y_values[(site_idx, ℓ, m)]
                    w = self.weights[site_idx]
                    self.F_forward[row_idx, site_idx] = w * np.conj(Y_val)
        
        # Inverse transform matrix: (N_sites, N_modes)
        # f_i = Σ_ℓm Y_ℓ^m(θ_i, φ_i) a_ℓ^m
        self.F_inverse = np.zeros((self.N_sites, self.N_modes), dtype=complex)
        
        for ℓ in range(self.ℓ_max + 1):
            for m in range(-ℓ, ℓ + 1):
                col_idx = self.mode_to_index[(ℓ, m)]
                for site_idx in range(self.N_sites):
                    Y_val = self.Y_values[(site_idx, ℓ, m)]
                    self.F_inverse[site_idx, col_idx] = Y_val
    
    def forward_transform(self, f: np.ndarray) -> SphericalHarmonicCoefficients:
        """
        Forward discrete spherical harmonic transform.
        
        Computes expansion coefficients:
        a_ℓ^m = ∫_S² conj(Y_ℓ^m(θ,φ)) f(θ,φ) dΩ
              ≈ Σ_i w_i conj(Y_ℓ^m(θ_i, φ_i)) f_i
        
        Parameters:
            f: Function values at lattice sites, shape (N_sites,)
               Can be real or complex.
        
        Returns:
            SphericalHarmonicCoefficients object with coefficients a_ℓ^m
        """
        if len(f) != self.N_sites:
            raise ValueError(f"Expected f with {self.N_sites} values, got {len(f)}")
        
        # Matrix-vector product: a = F @ f
        a_vector = self.F_forward @ f
        
        # Convert to dictionary
        coeffs = {}
        for (ℓ, m), idx in self.mode_to_index.items():
            coeffs[(ℓ, m)] = a_vector[idx]
        
        return SphericalHarmonicCoefficients(
            coeffs=coeffs,
            ℓ_max=self.ℓ_max,
            normalization=self.normalization
        )
    
    def inverse_transform(self, coeffs: SphericalHarmonicCoefficients) -> np.ndarray:
        """
        Inverse discrete spherical harmonic transform.
        
        Reconstructs function from expansion coefficients:
        f(θ, φ) = Σ_ℓm a_ℓ^m Y_ℓ^m(θ, φ)
        
        Parameters:
            coeffs: SphericalHarmonicCoefficients with expansion coefficients
        
        Returns:
            Function values at lattice sites, shape (N_sites,)
        """
        # Convert coefficients to vector
        a_vector = np.zeros(self.N_modes, dtype=complex)
        for (ℓ, m), val in coeffs.coeffs.items():
            if (ℓ, m) in self.mode_to_index:
                idx = self.mode_to_index[(ℓ, m)]
                a_vector[idx] = val
        
        # Matrix-vector product: f = F† @ a
        f = self.F_inverse @ a_vector
        
        return f
    
    def round_trip_error(self, f: np.ndarray) -> float:
        """
        Compute round-trip transform error: ||f - IDSHT(DSHT(f))|| / ||f||.
        
        Parameters:
            f: Function values at lattice sites
        
        Returns:
            Relative L² error
        """
        coeffs = self.forward_transform(f)
        f_reconstructed = self.inverse_transform(coeffs)
        
        error = np.linalg.norm(f - f_reconstructed)
        norm_f = np.linalg.norm(f)
        
        return error / norm_f if norm_f > 0 else 0.0
    
    def test_orthogonality(self, ℓ1: int, m1: int, ℓ2: int, m2: int) -> complex:
        """
        Test discrete orthogonality: ⟨Y_ℓ1^m1 | Y_ℓ2^m2⟩_discrete.
        
        Should equal δ_ℓ1ℓ2 δ_m1m2 if discretization is exact.
        
        Parameters:
            ℓ1, m1: First mode
            ℓ2, m2: Second mode
        
        Returns:
            Discrete inner product (0 if orthogonal, 1 if same mode)
        """
        # Construct Y_ℓ1^m1 values at all sites
        Y1 = np.array([self.Y_values[(i, ℓ1, m1)] for i in range(self.N_sites)])
        
        # Construct Y_ℓ2^m2 values at all sites
        Y2 = np.array([self.Y_values[(i, ℓ2, m2)] for i in range(self.N_sites)])
        
        # Discrete inner product: ⟨Y1|Y2⟩ = Σ_i w_i conj(Y1_i) Y2_i
        inner_product = np.sum(self.weights * np.conj(Y1) * Y2)
        
        return inner_product
    
    def compute_orthogonality_matrix(self, ℓ_max_test: Optional[int] = None) -> np.ndarray:
        """
        Compute full orthogonality matrix for all modes up to ℓ_max_test.
        
        Matrix element (i,j): ⟨Y_i | Y_j⟩_discrete where i,j index (ℓ,m) pairs.
        Should be close to identity matrix.
        
        Parameters:
            ℓ_max_test: Maximum ℓ to include (default: self.ℓ_max)
        
        Returns:
            Orthogonality matrix, shape (N_modes_test, N_modes_test)
        """
        if ℓ_max_test is None:
            ℓ_max_test = min(self.ℓ_max, 10)  # Limit for computational cost
        
        # Build list of (ℓ, m) modes
        modes = []
        for ℓ in range(ℓ_max_test + 1):
            for m in range(-ℓ, ℓ + 1):
                modes.append((ℓ, m))
        
        N = len(modes)
        ortho_matrix = np.zeros((N, N), dtype=complex)
        
        for i, (ℓ1, m1) in enumerate(modes):
            for j, (ℓ2, m2) in enumerate(modes):
                ortho_matrix[i, j] = self.test_orthogonality(ℓ1, m1, ℓ2, m2)
        
        return ortho_matrix
    
    def bandlimit_filter(self, f: np.ndarray, ℓ_cutoff: int) -> np.ndarray:
        """
        Bandlimit filter: keep only modes with ℓ ≤ ℓ_cutoff.
        
        Useful for:
        - Smoothing noisy data
        - Testing convergence
        - Multiresolution analysis
        
        Parameters:
            f: Function values at lattice sites
            ℓ_cutoff: Maximum ℓ to keep
        
        Returns:
            Filtered function values
        """
        # Forward transform
        coeffs = self.forward_transform(f)
        
        # Filter: set a_ℓ^m = 0 for ℓ > ℓ_cutoff
        filtered_coeffs = {}
        for (ℓ, m), val in coeffs.coeffs.items():
            if ℓ <= ℓ_cutoff:
                filtered_coeffs[(ℓ, m)] = val
        
        filtered = SphericalHarmonicCoefficients(
            coeffs=filtered_coeffs,
            ℓ_max=ℓ_cutoff,
            normalization=self.normalization
        )
        
        # Inverse transform
        return self.inverse_transform(filtered)
    
    def plot_power_spectrum(self, f: np.ndarray, ax: Optional[plt.Axes] = None,
                           log_scale: bool = False):
        """
        Plot power spectrum P(ℓ) = Σ_m |a_ℓ^m|² vs ℓ.
        
        Parameters:
            f: Function values at lattice sites
            ax: Matplotlib axes (creates new if None)
            log_scale: Use log scale for power axis
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        
        # Compute coefficients
        coeffs = self.forward_transform(f)
        spectrum = coeffs.power_spectrum()
        
        # Plot
        ℓ_vals = sorted(spectrum.keys())
        power_vals = [spectrum[ℓ] for ℓ in ℓ_vals]
        
        ax.plot(ℓ_vals, power_vals, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('Angular momentum $\\ell$', fontsize=12)
        ax.set_ylabel('Power $P(\\ell) = \\sum_m |a_\\ell^m|^2$', fontsize=12)
        ax.set_title('Spherical Harmonic Power Spectrum', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        if log_scale:
            ax.set_yscale('log')
        
        return ax
    
    def visualize_mode(self, ℓ: int, m: int, real_part: bool = True):
        """
        Visualize a single spherical harmonic mode Y_ℓ^m on the lattice.
        
        Parameters:
            ℓ, m: Mode quantum numbers
            real_part: Plot Re(Y_ℓ^m) if True, else Im(Y_ℓ^m)
        """
        # Get Y_ℓ^m values at all sites
        Y_values = np.array([self.Y_values[(i, ℓ, m)] for i in range(self.N_sites)])
        
        if real_part:
            values = Y_values.real
            title = f'$\\Re[Y_{{{ℓ}}}^{{{m}}}]$'
        else:
            values = Y_values.imag
            title = f'$\\Im[Y_{{{ℓ}}}^{{{m}}}]$'
        
        # Extract 3D positions
        x = np.array([p['x_3d'] for p in self.lattice.points])
        y = np.array([p['y_3d'] for p in self.lattice.points])
        z = np.array([p['z_3d'] for p in self.lattice.points])
        
        # 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(x, y, z, c=values, cmap='RdBu_r', s=50,
                            vmin=-np.max(np.abs(values)),
                            vmax=np.max(np.abs(values)))
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title, fontsize=14)
        
        plt.colorbar(scatter, ax=ax, shrink=0.6)
        
        return fig, ax


def test_dsht_basic():
    """
    Basic test of DSHT: round-trip accuracy for simple functions.
    """
    print("=" * 60)
    print("BASIC DSHT TEST")
    print("=" * 60)
    
    # Create lattice
    n_max = 8
    lattice = PolarLattice(n_max=n_max)
    print(f"\nLattice: n_max = {n_max}, ℓ_max = {lattice.ℓ_max}")
    print(f"Number of sites: {len(lattice.points)}")
    
    # Create DSHT
    dsht = DiscreteSphericalHarmonicTransform(lattice)
    print(f"Number of modes: {dsht.N_modes}")
    
    # Test 1: Constant function (ℓ=0, m=0 mode)
    print("\n--- Test 1: Constant function ---")
    f_const = np.ones(dsht.N_sites)
    coeffs_const = dsht.forward_transform(f_const)
    
    print(f"a_0^0 = {coeffs_const[(0, 0)]:.6f} (expected: {1.0/np.sqrt(4*np.pi):.6f})")
    print(f"Total power: {coeffs_const.total_power():.6f}")
    
    f_reconstructed = dsht.inverse_transform(coeffs_const)
    error = dsht.round_trip_error(f_const)
    print(f"Round-trip error: {error:.6e}")
    
    # Test 2: Pure Y_2^1 mode
    print("\n--- Test 2: Pure Y_2^1 mode ---")
    ℓ_test, m_test = 2, 1
    f_Y21 = np.array([dsht.Y_values[(i, ℓ_test, m_test)] for i in range(dsht.N_sites)])
    coeffs_Y21 = dsht.forward_transform(f_Y21.real)  # Use real part
    
    print(f"Top 5 coefficients:")
    sorted_coeffs = sorted(coeffs_Y21.coeffs.items(), key=lambda x: abs(x[1]), reverse=True)
    for (ℓ, m), val in sorted_coeffs[:5]:
        print(f"  a_{ℓ}^{m} = {val:.6f}")
    
    error_Y21 = dsht.round_trip_error(f_Y21.real)
    print(f"Round-trip error: {error_Y21:.6e}")
    
    # Test 3: Random smooth function
    print("\n--- Test 3: Random smooth function (ℓ ≤ 3) ---")
    np.random.seed(42)
    f_random = np.zeros(dsht.N_sites, dtype=complex)
    for ℓ in range(4):  # ℓ = 0, 1, 2, 3
        for m in range(-ℓ, ℓ + 1):
            a_random = np.random.randn() + 1j * np.random.randn()
            Y_vals = np.array([dsht.Y_values[(i, ℓ, m)] for i in range(dsht.N_sites)])
            f_random += a_random * Y_vals
    
    f_random = f_random.real  # Take real part
    coeffs_random = dsht.forward_transform(f_random)
    error_random = dsht.round_trip_error(f_random)
    print(f"Round-trip error: {error_random:.6e}")
    
    print("\n" + "=" * 60)
    print("✓ BASIC TESTS COMPLETE")
    print("=" * 60)


def test_dsht_orthogonality():
    """
    Test discrete orthogonality of spherical harmonics.
    """
    print("\n" + "=" * 60)
    print("ORTHOGONALITY TEST")
    print("=" * 60)
    
    # Create lattice
    n_max = 6
    lattice = PolarLattice(n_max=n_max)
    dsht = DiscreteSphericalHarmonicTransform(lattice)
    
    # Test orthogonality for low ℓ
    print("\nTesting ⟨Y_ℓ^m | Y_ℓ'^m'⟩_discrete:")
    print("(Should be 1 for ℓ=ℓ', m=m', else 0)")
    
    test_cases = [
        ((0, 0), (0, 0)),   # Same mode
        ((1, 0), (1, 0)),   # Same mode
        ((1, 0), (1, 1)),   # Different m
        ((1, 0), (2, 0)),   # Different ℓ
        ((2, -1), (2, 1)),  # Different m
        ((2, 1), (2, 1)),   # Same mode
    ]
    
    for (ℓ1, m1), (ℓ2, m2) in test_cases:
        inner_prod = dsht.test_orthogonality(ℓ1, m1, ℓ2, m2)
        expected = 1.0 if (ℓ1 == ℓ2 and m1 == m2) else 0.0
        error = abs(inner_prod - expected)
        
        print(f"  ⟨Y_{ℓ1}^{m1} | Y_{ℓ2}^{m2}⟩ = {inner_prod:.6f}, "
              f"expected {expected:.6f}, error = {error:.6e}")
    
    # Compute full orthogonality matrix
    print("\nComputing orthogonality matrix for ℓ ≤ 3...")
    ortho_matrix = dsht.compute_orthogonality_matrix(ℓ_max_test=3)
    
    # Check how close to identity
    N = ortho_matrix.shape[0]
    identity = np.eye(N)
    deviation = np.linalg.norm(ortho_matrix - identity, 'fro') / np.sqrt(N)
    
    print(f"Matrix shape: {ortho_matrix.shape}")
    print(f"Frobenius deviation from identity: {deviation:.6e}")
    
    # Largest off-diagonal element
    ortho_abs = np.abs(ortho_matrix)
    np.fill_diagonal(ortho_abs, 0)
    max_off_diag = np.max(ortho_abs)
    print(f"Largest off-diagonal element: {max_off_diag:.6e}")
    
    print("\n" + "=" * 60)
    print("✓ ORTHOGONALITY TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    # Run tests
    test_dsht_basic()
    test_dsht_orthogonality()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
