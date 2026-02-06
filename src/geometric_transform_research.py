"""
Geometric Transformation Research Module for SU(2) Lattice Eigenvectors

This module investigates reversible geometric transformations that can map between:
- Flat lattice representation (exact eigenvalues, approximate eigenvectors)  
- Curved spherical representation (accurate spherical harmonics)

Research Goal: Recover the ~18% eigenvector deficit through coordinate corrections
while preserving exact eigenvalue properties and computational efficiency.

Key Components:
1. Conformal mappings (stereographic, Lambert, Mercator)
2. Jacobian corrections for wavefunction transformation
3. Diagnostic tools for spatial error analysis
4. Eigenvalue preservation verification
5. Adaptive hybrid transformations
6. Round-trip reversibility testing

Author: Quantum Lattice Project
Date: January 2026
Research Phase: Geometric Correction Investigation
"""

import numpy as np
from scipy import sparse, special
from scipy.optimize import minimize_scalar
from typing import Dict, List, Tuple, Optional, Callable
import matplotlib.pyplot as plt
from matplotlib import cm
from dataclasses import dataclass
import warnings

try:
    from .lattice import PolarLattice
    from .angular_momentum import AngularMomentumOperators
    from .spherical_harmonics_transform import DiscreteSphericalHarmonicTransform
except ImportError:
    from lattice import PolarLattice
    from angular_momentum import AngularMomentumOperators
    from spherical_harmonics_transform import DiscreteSphericalHarmonicTransform


@dataclass
class TransformationResult:
    """
    Container for results of geometric transformation experiments.
    
    Attributes:
        transform_type: Name of transformation applied
        overlap_original: Original overlap with Y_ℓm before correction
        overlap_forward: Overlap after forward transformation (√J correction)
        overlap_pullback: Overlap after pullback transformation (1/√J correction)
        overlap_best: Best overlap achieved
        overlap_improvement: Change in overlap percentage
        eigenvalue_error: Error in ⟨L²⟩ after transformation
        eigenvalue_preserved: Whether eigenvalue exactness maintained
        jacobian_stats: Statistics of Jacobian values
        round_trip_fidelity: Fidelity after forward+reverse transformation
    """
    transform_type: str
    ℓ: int
    m: int
    overlap_original: float
    overlap_forward: float
    overlap_pullback: float
    overlap_best: float
    overlap_improvement: float
    eigenvalue_error: float
    eigenvalue_preserved: bool
    jacobian_stats: Dict[str, float]
    round_trip_fidelity: Optional[float] = None
    
    def __str__(self):
        return (f"Transform: {self.transform_type} (ℓ={self.ℓ}, m={self.m})\n"
                f"  Original overlap:     {self.overlap_original:.4%}\n"
                f"  Forward overlap:      {self.overlap_forward:.4%}\n"
                f"  Pullback overlap:     {self.overlap_pullback:.4%}\n"
                f"  Best overlap:         {self.overlap_best:.4%}\n"
                f"  Improvement:          {self.overlap_improvement:+.2%}\n"
                f"  Eigenvalue error:     {self.eigenvalue_error:.2e}\n"
                f"  Eigenvalue preserved: {self.eigenvalue_preserved}\n"
                f"  Jacobian (min/max):   {self.jacobian_stats['min']:.4f} / "
                f"{self.jacobian_stats['max']:.4f}")


class GeometricTransformResearch:
    """
    Research toolkit for investigating geometric corrections to lattice eigenvectors.
    
    This class provides methods to:
    1. Apply conformal coordinate transformations (stereographic, Lambert, Mercator)
    2. Compute Jacobian corrections for wavefunction density
    3. Measure overlap improvements with spherical harmonics
    4. Verify eigenvalue preservation
    5. Test reversibility and round-trip fidelity
    6. Decompose error sources (geometric vs discretization)
    
    The central hypothesis: The 18% eigenvector deficit is partially geometric,
    arising from mismatch between flat polar coordinates and curved S² geometry.
    """
    
    def __init__(self, lattice: PolarLattice, angular_ops: AngularMomentumOperators,
                 dsht: Optional[DiscreteSphericalHarmonicTransform] = None):
        """
        Initialize geometric transformation research toolkit.
        
        Parameters:
            lattice: PolarLattice defining discrete angular momentum structure
            angular_ops: Angular momentum operators (for eigenvalue testing)
            dsht: Discrete spherical harmonic transform (for overlap computation)
        """
        self.lattice = lattice
        self.angular_ops = angular_ops
        self.dsht = dsht if dsht is not None else DiscreteSphericalHarmonicTransform(lattice)
        
        self.N_sites = len(lattice.points)
        self.ℓ_max = lattice.ℓ_max
        
        # Extract coordinates for all lattice points
        self._extract_coordinates()
        
        # Pre-compute spherical harmonics for overlap calculations
        self._precompute_spherical_harmonics()
        
        # Cache for Jacobians (expensive to recompute)
        self._jacobian_cache = {}
    
    def _extract_coordinates(self):
        """Extract and store lattice point coordinates in various systems."""
        self.coords_2d = np.array([(p['x_2d'], p['y_2d']) for p in self.lattice.points])
        self.coords_polar = np.array([(p['r'], p['θ']) for p in self.lattice.points])
        self.coords_3d = np.array([(p['x_3d'], p['y_3d'], p['z_3d']) 
                                    for p in self.lattice.points])
        
        # Quantum numbers
        self.quantum_numbers = np.array([(p['ℓ'], p['m_ℓ'], p['m_s']) 
                                          for p in self.lattice.points])
    
    def _precompute_spherical_harmonics(self):
        """Pre-compute Y_ℓm at all lattice points for overlap calculations."""
        self.Y_ℓm_cache = {}
        
        for ℓ in range(self.ℓ_max + 1):
            for m in range(-ℓ, ℓ + 1):
                Y_values = np.zeros(self.N_sites, dtype=complex)
                
                for i, point in enumerate(self.lattice.points):
                    θ = np.arccos(point['z_3d'])  # Polar angle from z-axis
                    φ = np.arctan2(point['y_3d'], point['x_3d'])  # Azimuthal angle
                    Y_values[i] = self._compute_Ylm(ℓ, m, θ, φ)
                
                self.Y_ℓm_cache[(ℓ, m)] = Y_values
    
    @staticmethod
    def _compute_Ylm(ℓ: int, m: int, θ: float, φ: float) -> complex:
        """
        Compute spherical harmonic Y_ℓ^m(θ, φ).
        
        Uses scipy's sph_harm with convention: Y_ℓ^m(θ, φ) = sph_harm(m, ℓ, φ, θ)
        """
        return special.sph_harm(m, ℓ, φ, θ)
    
    # =========================================================================
    # CONFORMAL TRANSFORMATIONS
    # =========================================================================
    
    def stereographic_projection(self, inverse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stereographic projection: S² ↔ ℝ² (angle-preserving).
        
        Forward (S² → ℝ²):
            x = 2R sin(θ)cos(φ) / (1 + cos(θ))
            y = 2R sin(θ)sin(φ) / (1 + cos(θ))
        
        Inverse (ℝ² → S²):
            θ = 2 arctan(r/2R)
            φ = arctan2(y, x)
        
        Jacobian: J = 4/(1 + cos(θ))²
        
        Parameters:
            inverse: If True, perform ℝ² → S²; otherwise S² → ℝ²
        
        Returns:
            (θ_new, φ_new): Transformed angular coordinates
        """
        if inverse:
            # Map flat lattice (x, y) → sphere (θ, φ)
            x, y = self.coords_2d[:, 0], self.coords_2d[:, 1]
            r = np.sqrt(x**2 + y**2)
            R = 1.0  # Nominal sphere radius
            
            θ = 2 * np.arctan(r / (2 * R))
            φ = np.arctan2(y, x)
        else:
            # Map sphere (θ, φ) → plane (x, y)
            # Use existing 3D spherical coords
            θ = np.arccos(self.coords_3d[:, 2])  # z = cos(θ)
            φ = np.arctan2(self.coords_3d[:, 1], self.coords_3d[:, 0])
            
            R = 1.0
            # This gives (x, y) coordinates, but we return (θ, φ) for consistency
            # Actually, for transformation purposes, we work in (θ, φ) space
        
        return θ, φ
    
    def lambert_azimuthal_projection(self, inverse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lambert azimuthal equal-area projection: S² ↔ ℝ² (area-preserving).
        
        Forward (S² → ℝ²):
            x = 2R sin(θ/2)cos(φ)
            y = 2R sin(θ/2)sin(φ)
        
        Inverse (ℝ² → S²):
            θ = 2 arcsin(r/2R)
            φ = arctan2(y, x)
        
        Jacobian: J = 1 (area-preserving by construction)
        
        Parameters:
            inverse: If True, perform ℝ² → S²; otherwise S² → ℝ²
        
        Returns:
            (θ_new, φ_new): Transformed angular coordinates
        """
        if inverse:
            x, y = self.coords_2d[:, 0], self.coords_2d[:, 1]
            r = np.sqrt(x**2 + y**2)
            R = 1.0
            
            # Clamp r to valid range to avoid numerical issues
            r = np.minimum(r, 2 * R)
            
            θ = 2 * np.arcsin(r / (2 * R))
            φ = np.arctan2(y, x)
        else:
            θ = np.arccos(self.coords_3d[:, 2])
            φ = np.arctan2(self.coords_3d[:, 1], self.coords_3d[:, 0])
        
        return θ, φ
    
    def mercator_projection(self, inverse: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mercator projection: S² ↔ ℝ² (angle-preserving, diverges at poles).
        
        Forward (S² → ℝ²):
            x = R·φ
            y = R·ln(tan(θ/2 + π/4))
        
        Inverse (ℝ² → S²):
            φ = x/R
            θ = 2·arctan(exp(y/R)) - π/2
        
        Jacobian: J = sec(θ) = 1/cos(θ)
        
        Parameters:
            inverse: If True, perform ℝ² → S²; otherwise S² → ℝ²
        
        Returns:
            (θ_new, φ_new): Transformed angular coordinates
        """
        R = 1.0
        
        if inverse:
            x, y = self.coords_2d[:, 0], self.coords_2d[:, 1]
            
            φ = x / R
            θ = 2 * np.arctan(np.exp(y / R)) - np.pi / 2
        else:
            θ = np.arccos(self.coords_3d[:, 2])
            φ = np.arctan2(self.coords_3d[:, 1], self.coords_3d[:, 0])
        
        # Wrap φ to [-π, π]
        φ = np.arctan2(np.sin(φ), np.cos(φ))
        
        return θ, φ
    
    def compute_jacobian(self, transform_type: str = 'stereographic') -> np.ndarray:
        """
        Compute Jacobian determinant for coordinate transformation.
        
        The Jacobian relates volume elements between coordinate systems:
            dV_sphere = J(θ, φ) · dV_flat
        
        For wavefunctions: |ψ_sphere|² dV_sphere = |ψ_flat|² dV_flat
        Therefore: ψ_sphere = (1/√J) · ψ_flat  or  ψ_flat = √J · ψ_sphere
        
        Parameters:
            transform_type: 'stereographic', 'lambert', or 'mercator'
        
        Returns:
            jacobian: Array of Jacobian values at each lattice point
        """
        # Check cache
        if transform_type in self._jacobian_cache:
            return self._jacobian_cache[transform_type]
        
        θ = np.arccos(self.coords_3d[:, 2])  # Polar angle
        
        if transform_type == 'stereographic':
            # J = 4/(1 + cos(θ))²
            jacobian = 4.0 / (1.0 + np.cos(θ))**2
        
        elif transform_type == 'lambert':
            # J = 1 (area-preserving)
            jacobian = np.ones(self.N_sites)
        
        elif transform_type == 'mercator':
            # J = sec(θ) = 1/cos(θ)
            # Avoid division by zero at poles
            cos_θ = np.cos(θ)
            cos_θ = np.clip(cos_θ, 1e-10, 1.0)
            jacobian = 1.0 / cos_θ
        
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")
        
        # Cache result
        self._jacobian_cache[transform_type] = jacobian
        
        return jacobian
    
    # =========================================================================
    # EIGENVECTOR CORRECTION AND TESTING
    # =========================================================================
    
    def apply_geometric_correction(self, psi: np.ndarray, transform_type: str = 'stereographic',
                                   direction: str = 'forward') -> np.ndarray:
        """
        Apply geometric correction to eigenvector using Jacobian.
        
        Two possible directions:
        - 'forward': ψ_corrected = √J · ψ_lattice (push forward from lattice)
        - 'pullback': ψ_corrected = ψ_lattice / √J (pull back from sphere)
        
        The correct direction depends on whether we're transforming from
        lattice→sphere or sphere→lattice. Both are tested empirically.
        
        Parameters:
            psi: Eigenvector on lattice (shape: N_sites)
            transform_type: 'stereographic', 'lambert', or 'mercator'
            direction: 'forward' (multiply by √J) or 'pullback' (divide by √J)
        
        Returns:
            psi_corrected: Geometrically corrected eigenvector (renormalized)
        """
        J = self.compute_jacobian(transform_type)
        
        if direction == 'forward':
            psi_corrected = np.sqrt(J) * psi
        elif direction == 'pullback':
            psi_corrected = psi / np.sqrt(J)
        else:
            raise ValueError(f"Unknown direction: {direction}")
        
        # Renormalize
        norm = np.linalg.norm(psi_corrected)
        if norm > 1e-10:
            psi_corrected /= norm
        
        return psi_corrected
    
    def compute_overlap_with_Ylm(self, psi: np.ndarray, ℓ: int, m: int) -> float:
        """
        Compute overlap |⟨ψ|Y_ℓm⟩|² between lattice eigenvector and spherical harmonic.
        
        Parameters:
            psi: Eigenvector on lattice
            ℓ, m: Quantum numbers for Y_ℓ^m
        
        Returns:
            overlap: |⟨ψ|Y_ℓm⟩|² (should be ~1 for perfect match)
        """
        if (ℓ, m) not in self.Y_ℓm_cache:
            raise ValueError(f"Y_{ℓ}^{m} not in cache (ℓ_max={self.ℓ_max})")
        
        Y_ℓm = self.Y_ℓm_cache[(ℓ, m)]
        
        # Inner product with integration weights
        # For now, use simple sum (could use DSHT weights)
        overlap = np.abs(np.vdot(psi, Y_ℓm))**2
        
        return overlap
    
    def verify_eigenvalue_preservation(self, psi: np.ndarray, ℓ: int,
                                      tolerance: float = 1e-10) -> Tuple[float, bool]:
        """
        Verify that geometric correction preserves exact eigenvalue.
        
        Computes ⟨ψ|L²|ψ⟩ and checks if it equals ℓ(ℓ+1) to machine precision.
        
        Parameters:
            psi: Eigenvector (possibly corrected)
            ℓ: Angular momentum quantum number
            tolerance: Acceptable relative error
        
        Returns:
            (error, preserved): Relative error and boolean indicating preservation
        """
        L_squared = self.angular_ops.build_L_squared()
        
        L2_expectation = np.real(psi.conj() @ L_squared @ psi)
        theoretical = ℓ * (ℓ + 1)
        
        if theoretical > 0:
            error = abs(L2_expectation - theoretical) / theoretical
        else:
            error = abs(L2_expectation - theoretical)
        
        preserved = error < tolerance
        
        return error, preserved
    
    def test_transformation(self, psi: np.ndarray, ℓ: int, m: int,
                           transform_type: str = 'stereographic') -> TransformationResult:
        """
        Comprehensive test of geometric transformation on eigenvector.
        
        Tests:
        1. Original overlap with Y_ℓm
        2. Forward correction (√J)
        3. Pullback correction (1/√J)
        4. Eigenvalue preservation
        5. Jacobian statistics
        
        Parameters:
            psi: Eigenvector to transform
            ℓ, m: Quantum numbers
            transform_type: 'stereographic', 'lambert', or 'mercator'
        
        Returns:
            TransformationResult: Comprehensive results object
        """
        # Original overlap
        overlap_original = self.compute_overlap_with_Ylm(psi, ℓ, m)
        
        # Forward correction
        psi_forward = self.apply_geometric_correction(psi, transform_type, 'forward')
        overlap_forward = self.compute_overlap_with_Ylm(psi_forward, ℓ, m)
        
        # Pullback correction
        psi_pullback = self.apply_geometric_correction(psi, transform_type, 'pullback')
        overlap_pullback = self.compute_overlap_with_Ylm(psi_pullback, ℓ, m)
        
        # Best result
        overlap_best = max(overlap_forward, overlap_pullback)
        overlap_improvement = overlap_best - overlap_original
        
        # Verify eigenvalue preservation (using best corrected version)
        psi_best = psi_forward if overlap_forward > overlap_pullback else psi_pullback
        eigenvalue_error, eigenvalue_preserved = self.verify_eigenvalue_preservation(psi_best, ℓ)
        
        # Jacobian statistics
        J = self.compute_jacobian(transform_type)
        jacobian_stats = {
            'min': float(np.min(J)),
            'max': float(np.max(J)),
            'mean': float(np.mean(J)),
            'std': float(np.std(J))
        }
        
        return TransformationResult(
            transform_type=transform_type,
            ℓ=ℓ,
            m=m,
            overlap_original=overlap_original,
            overlap_forward=overlap_forward,
            overlap_pullback=overlap_pullback,
            overlap_best=overlap_best,
            overlap_improvement=overlap_improvement,
            eigenvalue_error=eigenvalue_error,
            eigenvalue_preserved=eigenvalue_preserved,
            jacobian_stats=jacobian_stats
        )
    
    def test_reversibility(self, psi: np.ndarray, transform_type: str = 'stereographic') -> float:
        """
        Test round-trip reversibility: flat → curved → flat.
        
        The transformation should be reversible with high fidelity.
        
        Parameters:
            psi: Original eigenvector
            transform_type: Transformation to test
        
        Returns:
            fidelity: |⟨ψ_original|ψ_recovered⟩|² (should be ≈ 1)
        """
        # Forward
        psi_forward = self.apply_geometric_correction(psi, transform_type, 'forward')
        
        # Reverse (opposite direction)
        psi_recovered = self.apply_geometric_correction(psi_forward, transform_type, 'pullback')
        
        # Compute fidelity
        fidelity = np.abs(np.vdot(psi, psi_recovered))**2
        
        return fidelity
    
    # =========================================================================
    # DIAGNOSTIC ANALYSIS
    # =========================================================================
    
    def compute_spatial_error_distribution(self, psi: np.ndarray, ℓ: int, m: int) -> np.ndarray:
        """
        Compute spatial distribution of eigenvector error: |ψ_lattice - Y_ℓm|².
        
        Identifies where on the lattice the eigenvector deviates from ideal.
        
        Parameters:
            psi: Lattice eigenvector
            ℓ, m: Quantum numbers
        
        Returns:
            error_dist: Error at each lattice point (shape: N_sites)
        """
        Y_ℓm = self.Y_ℓm_cache[(ℓ, m)]
        
        # Normalize both for fair comparison
        psi_norm = psi / np.linalg.norm(psi)
        Y_norm = Y_ℓm / np.linalg.norm(Y_ℓm)
        
        # Pointwise squared error
        error_dist = np.abs(psi_norm - Y_norm)**2
        
        return error_dist
    
    def analyze_error_by_region(self, psi: np.ndarray, ℓ: int, m: int) -> Dict[str, float]:
        """
        Decompose error by lattice region (inner shells vs outer, poles vs equator).
        
        Parameters:
            psi: Lattice eigenvector
            ℓ, m: Quantum numbers
        
        Returns:
            error_breakdown: Dictionary with error statistics by region
        """
        error_dist = self.compute_spatial_error_distribution(psi, ℓ, m)
        
        # Get coordinates
        ℓ_values = self.quantum_numbers[:, 0]
        θ_values = np.arccos(self.coords_3d[:, 2])
        
        # Define regions
        inner_mask = ℓ_values < self.ℓ_max // 2
        outer_mask = ~inner_mask
        
        pole_mask = (θ_values < np.pi/4) | (θ_values > 3*np.pi/4)
        equator_mask = ~pole_mask
        
        return {
            'total_mean': float(np.mean(error_dist)),
            'total_max': float(np.max(error_dist)),
            'inner_shells_mean': float(np.mean(error_dist[inner_mask])) if inner_mask.any() else 0.0,
            'outer_shells_mean': float(np.mean(error_dist[outer_mask])) if outer_mask.any() else 0.0,
            'polar_region_mean': float(np.mean(error_dist[pole_mask])) if pole_mask.any() else 0.0,
            'equator_region_mean': float(np.mean(error_dist[equator_mask])) if equator_mask.any() else 0.0,
        }
    
    # =========================================================================
    # ADAPTIVE HYBRID TRANSFORMATION
    # =========================================================================
    
    def hybrid_transform(self, psi: np.ndarray, lambda_param: float,
                        transform_type: str = 'stereographic') -> np.ndarray:
        """
        Apply parameterized hybrid transformation interpolating between flat and curved.
        
        λ = 0: pure flat lattice (uncorrected)
        λ = 1: full geometric correction
        
        Uses fractional Jacobian: J^λ
        
        Parameters:
            psi: Eigenvector
            lambda_param: Interpolation parameter ∈ [0, 1]
            transform_type: Which transformation to use
        
        Returns:
            psi_hybrid: Interpolated eigenvector
        """
        if not 0 <= lambda_param <= 1:
            raise ValueError(f"lambda_param must be in [0, 1], got {lambda_param}")
        
        if lambda_param == 0:
            return psi.copy()
        
        J = self.compute_jacobian(transform_type)
        J_fractional = J ** lambda_param
        
        psi_hybrid = np.sqrt(J_fractional) * psi
        
        # Renormalize
        psi_hybrid /= np.linalg.norm(psi_hybrid)
        
        return psi_hybrid
    
    def optimize_lambda(self, psi: np.ndarray, ℓ: int, m: int,
                       transform_type: str = 'stereographic',
                       eigenvalue_tolerance: float = 1e-10) -> Tuple[float, float]:
        """
        Find optimal λ that maximizes overlap while preserving eigenvalue.
        
        Parameters:
            psi: Eigenvector
            ℓ, m: Quantum numbers
            transform_type: Which transformation to use
            eigenvalue_tolerance: Maximum acceptable eigenvalue error
        
        Returns:
            (lambda_optimal, overlap_optimal): Best λ and resulting overlap
        """
        def objective(lambda_param):
            """Negative overlap (for minimization)."""
            psi_hybrid = self.hybrid_transform(psi, lambda_param, transform_type)
            overlap = self.compute_overlap_with_Ylm(psi_hybrid, ℓ, m)
            
            # Penalize eigenvalue violation
            error, _ = self.verify_eigenvalue_preservation(psi_hybrid, ℓ, eigenvalue_tolerance)
            if error > eigenvalue_tolerance:
                penalty = 1000 * error  # Large penalty for violation
            else:
                penalty = 0.0
            
            return -overlap + penalty
        
        # Optimize
        result = minimize_scalar(objective, bounds=(0, 1), method='bounded')
        
        lambda_optimal = result.x
        overlap_optimal = -result.fun if result.fun < 0 else self.compute_overlap_with_Ylm(
            self.hybrid_transform(psi, lambda_optimal, transform_type), ℓ, m
        )
        
        return lambda_optimal, overlap_optimal
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    def plot_error_heatmap(self, psi: np.ndarray, ℓ: int, m: int,
                          title: str = "Eigenvector Error Distribution"):
        """
        Plot heatmap of spatial error distribution on lattice.
        
        Parameters:
            psi: Eigenvector
            ℓ, m: Quantum numbers
            title: Plot title
        """
        error_dist = self.compute_spatial_error_distribution(psi, ℓ, m)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 2D lattice view
        x_2d = self.coords_2d[:, 0]
        y_2d = self.coords_2d[:, 1]
        scatter1 = ax1.scatter(x_2d, y_2d, c=error_dist, cmap='hot', s=50)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(f'{title}\n(2D Lattice View)')
        ax1.axis('equal')
        plt.colorbar(scatter1, ax=ax1, label='|ψ - Y_ℓm|²')
        
        # 3D sphere view
        ax2 = fig.add_subplot(122, projection='3d')
        x_3d = self.coords_3d[:, 0]
        y_3d = self.coords_3d[:, 1]
        z_3d = self.coords_3d[:, 2]
        scatter2 = ax2.scatter(x_3d, y_3d, z_3d, c=error_dist, cmap='hot', s=30)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('z')
        ax2.set_title(f'{title}\n(3D Sphere View)')
        plt.colorbar(scatter2, ax=ax2, label='|ψ - Y_ℓm|²')
        
        plt.tight_layout()
        return fig
    
    def plot_jacobian_distribution(self, transform_type: str = 'stereographic'):
        """
        Visualize Jacobian distribution across lattice.
        
        Parameters:
            transform_type: Which transformation's Jacobian to plot
        """
        J = self.compute_jacobian(transform_type)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Spatial distribution
        x_2d = self.coords_2d[:, 0]
        y_2d = self.coords_2d[:, 1]
        scatter1 = ax1.scatter(x_2d, y_2d, c=J, cmap='viridis', s=50)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(f'Jacobian Distribution: {transform_type}')
        ax1.axis('equal')
        plt.colorbar(scatter1, ax=ax1, label='J(θ, φ)')
        
        # Histogram
        ax2.hist(J, bins=50, edgecolor='black')
        ax2.set_xlabel('Jacobian Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Jacobian Histogram\nmin={np.min(J):.3f}, max={np.max(J):.3f}')
        ax2.axvline(1.0, color='red', linestyle='--', label='J=1 (no distortion)')
        ax2.legend()
        
        plt.tight_layout()
        return fig


class GeometricTransformBenchmark:
    """
    Systematic benchmarking suite for geometric transformation research.
    
    Runs comprehensive tests across multiple ℓ values and transformation types,
    generating tables and plots for research documentation.
    """
    
    def __init__(self, lattice: PolarLattice, angular_ops: AngularMomentumOperators):
        """
        Initialize benchmark suite.
        
        Parameters:
            lattice: PolarLattice
            angular_ops: Angular momentum operators
        """
        self.lattice = lattice
        self.angular_ops = angular_ops
        self.research = GeometricTransformResearch(lattice, angular_ops)
        
        # Storage for results
        self.results = []
    
    def run_comprehensive_benchmark(self, ℓ_values: List[int],
                                   transform_types: List[str] = None) -> List[TransformationResult]:
        """
        Run benchmark across multiple ℓ and transformations.
        
        Parameters:
            ℓ_values: List of ℓ quantum numbers to test
            transform_types: List of transformations (default: all three)
        
        Returns:
            results: List of TransformationResult objects
        """
        if transform_types is None:
            transform_types = ['stereographic', 'lambert', 'mercator']
        
        # First, compute eigenvectors
        print("Computing eigenvectors...")
        L_squared = self.angular_ops.build_L_squared()
        eigenvalues, eigenvectors = sparse.linalg.eigsh(L_squared, k=min(50, L_squared.shape[0]-1),
                                                        which='SM')
        
        results = []
        
        for ℓ in ℓ_values:
            print(f"\n{'='*60}")
            print(f"Testing ℓ = {ℓ}")
            print(f"{'='*60}")
            
            # Find eigenvector for this ℓ
            target_eigenvalue = ℓ * (ℓ + 1)
            idx = np.argmin(np.abs(eigenvalues - target_eigenvalue))
            psi = eigenvectors[:, idx]
            
            # Test m=0 (can extend to all m later)
            m = 0
            
            for transform_type in transform_types:
                print(f"\nTransformation: {transform_type}")
                result = self.research.test_transformation(psi, ℓ, m, transform_type)
                print(result)
                
                # Test reversibility
                fidelity = self.research.test_reversibility(psi, transform_type)
                result.round_trip_fidelity = fidelity
                print(f"  Round-trip fidelity:  {fidelity:.6f}")
                
                results.append(result)
        
        self.results = results
        return results
    
    def generate_summary_table(self) -> str:
        """
        Generate formatted table of results.
        
        Returns:
            table: Markdown-formatted table string
        """
        if not self.results:
            return "No results available. Run benchmark first."
        
        # Header
        table = "| ℓ | Transform | Original | Forward | Pullback | Best | Δ | Eigenvalue Error | Preserved |\n"
        table += "|---|-----------|----------|---------|----------|------|---|------------------|----------|\n"
        
        # Rows
        for r in self.results:
            table += (f"| {r.ℓ} | {r.transform_type[:6]} | "
                     f"{r.overlap_original:.3%} | {r.overlap_forward:.3%} | "
                     f"{r.overlap_pullback:.3%} | {r.overlap_best:.3%} | "
                     f"{r.overlap_improvement:+.2%} | {r.eigenvalue_error:.2e} | "
                     f"{'✓' if r.eigenvalue_preserved else '✗'} |\n")
        
        return table
    
    def plot_improvement_comparison(self):
        """Generate comparison plot of overlap improvements."""
        if not self.results:
            print("No results available.")
            return None
        
        # Organize data
        transforms = sorted(set(r.transform_type for r in self.results))
        ℓ_values = sorted(set(r.ℓ for r in self.results))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Absolute overlaps
        for transform in transforms:
            data = [(r.ℓ, r.overlap_best) for r in self.results if r.transform_type == transform]
            data.sort()
            ℓs, overlaps = zip(*data)
            ax1.plot(ℓs, [o*100 for o in overlaps], 'o-', label=transform, markersize=8)
        
        ax1.set_xlabel('Angular Momentum ℓ')
        ax1.set_ylabel('Best Overlap (%)')
        ax1.set_title('Overlap with Y_ℓm after Geometric Correction')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Improvement over original
        for transform in transforms:
            data = [(r.ℓ, r.overlap_improvement) for r in self.results if r.transform_type == transform]
            data.sort()
            ℓs, improvements = zip(*data)
            ax2.plot(ℓs, [i*100 for i in improvements], 'o-', label=transform, markersize=8)
        
        ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Angular Momentum ℓ')
        ax2.set_ylabel('Improvement (percentage points)')
        ax2.set_title('Overlap Improvement from Geometric Correction')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# =========================================================================
# CONVENIENCE FUNCTIONS
# =========================================================================

def quick_test(n_max: int = 5, ℓ_test: int = 3):
    """
    Quick test of geometric transformation research.
    
    Parameters:
        n_max: Lattice size
        ℓ_test: Angular momentum to test
    """
    print(f"Initializing lattice with n_max={n_max}...")
    lattice = PolarLattice(n_max)
    angular_ops = AngularMomentumOperators(lattice)
    research = GeometricTransformResearch(lattice, angular_ops)
    
    print(f"Computing eigenvectors...")
    L_squared = angular_ops.build_L_squared()
    eigenvalues, eigenvectors = sparse.linalg.eigsh(L_squared, k=20, which='SM')
    
    # Find eigenvector for ℓ_test
    target = ℓ_test * (ℓ_test + 1)
    idx = np.argmin(np.abs(eigenvalues - target))
    psi = eigenvectors[:, idx]
    
    print(f"\n{'='*60}")
    print(f"Testing ℓ={ℓ_test}, m=0")
    print(f"{'='*60}\n")
    
    # Test all three transforms
    for transform_type in ['stereographic', 'lambert', 'mercator']:
        result = research.test_transformation(psi, ℓ_test, 0, transform_type)
        print(result)
        print()


if __name__ == "__main__":
    # Run quick test
    quick_test(n_max=5, ℓ_test=3)
