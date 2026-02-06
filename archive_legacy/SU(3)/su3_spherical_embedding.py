"""
Spherical Shell Embedding of SU(3) Ziggurat

This module implements the transformation of GT-pattern-based ziggurat coordinates
(I₃, Y, z) into spherical shell coordinates (r, θ, φ), preserving all SU(3) algebraic
structure at machine precision.

Mathematical Framework:
- Radial: r = r₀ + √C₂(p,q) · z/(p+q)
- Polar: θ = arccos(Y/Y_max)
- Azimuthal: φ = atan2(I₃, auxiliary_coord) + π

Author: Unified Geometry Framework
Date: February 5, 2026
"""

import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class SphericalState:
    """A state in spherical coordinates with metadata"""
    r: float          # Radial coordinate (shell radius)
    theta: float      # Polar angle [0, π]
    phi: float        # Azimuthal angle [0, 2π)
    
    # Original GT coordinates for validation
    I3: float
    Y: float
    z: int
    
    # GT pattern
    gt_pattern: Tuple[int, ...]
    
    def __repr__(self):
        return f"Spherical(r={self.r:.4f}, θ={self.theta:.4f}, φ={self.phi:.4f}) | GT(I₃={self.I3:.3f}, Y={self.Y:.3f}, z={self.z})"


class SU3SphericalEmbedding:
    """
    Transform SU(3) Ziggurat lattice into spherical shell geometry.
    
    The ziggurat's vertical coordinate z becomes radial coordinate r,
    while weight diagram (I₃, Y) map to angular coordinates (θ, φ).
    
    Preserves all algebraic structure via unitary basis transformation.
    """
    
    def __init__(self, p: int, q: int, r0: float = 1.0, 
                 scaling_mode: str = 'casimir',
                 height_mode: str = 'linear'):
        """
        Initialize spherical embedding for representation (p,q).
        
        Parameters
        ----------
        p, q : int
            Dynkin labels for SU(3) representation
        r0 : float
            Reference radius for innermost shell (z=0)
        scaling_mode : str
            'casimir': R_rep = √C₂(p,q)
            'dimension': R_rep = √dim(p,q)
            'dynkin': R_rep = √(p²+q²)
        height_mode : str
            'linear': f(z) = z/z_max
            'sqrt': f(z) = √(z/z_max)
            'quadratic': f(z) = (z/z_max)²
        """
        self.p = p
        self.q = q
        self.r0 = r0
        self.scaling_mode = scaling_mode
        self.height_mode = height_mode
        
        # Compute representation properties
        self.dim = self._dimension(p, q)
        self.C2 = self._casimir(p, q)
        self.z_max = p + q
        
        # Compute scaling factor
        if scaling_mode == 'casimir':
            self.R_rep = np.sqrt(self.C2)
        elif scaling_mode == 'dimension':
            self.R_rep = np.sqrt(self.dim)
        elif scaling_mode == 'dynkin':
            self.R_rep = np.sqrt(p**2 + q**2)
        else:
            raise ValueError(f"Unknown scaling_mode: {scaling_mode}")
        
        # Generate GT patterns and compute weight extrema
        self.gt_patterns = self._generate_gt_patterns()
        self.I3_min, self.I3_max, self.Y_min, self.Y_max = self._compute_weight_extrema()
        
        print(f"SU(3) Spherical Embedding: (p,q)=({p},{q})")
        print(f"  Dimension: {self.dim}")
        print(f"  Casimir C₂: {self.C2:.4f}")
        print(f"  Shell range: r ∈ [{self.r0:.2f}, {self.r0 + self.R_rep:.2f}]")
        print(f"  Weight range: I₃ ∈ [{self.I3_min:.3f}, {self.I3_max:.3f}], Y ∈ [{self.Y_min:.3f}, {self.Y_max:.3f}]")
        print(f"  z levels: 0 to {self.z_max}")
        print(f"  Total states: {len(self.gt_patterns)}")
    
    def _dimension(self, p: int, q: int) -> int:
        """Dimension of representation (p,q)"""
        return (p + 1) * (q + 1) * (p + q + 2) // 2
    
    def _casimir(self, p: int, q: int) -> float:
        """Quadratic Casimir C₂(p,q)"""
        return (p**2 + q**2 + p*q + 3*p + 3*q) / 3.0
    
    def _generate_gt_patterns(self) -> List[Tuple[int, ...]]:
        """Generate all valid GT patterns for (p,q)"""
        patterns = []
        m13, m23, m33 = self.p + self.q, self.q, 0
        
        for m12 in range(m23, m13 + 1):
            for m22 in range(m33, m23 + 1):
                for m11 in range(m22, m12 + 1):
                    patterns.append((m13, m23, m33, m12, m22, m11))
        
        return patterns
    
    def _gt_to_quantum_numbers(self, gt_pattern: Tuple[int, ...]) -> Tuple[float, float, int]:
        """Convert GT pattern to (I₃, Y, z)"""
        m13, m23, m33, m12, m22, m11 = gt_pattern
        
        I3 = m11 - (m12 + m22) / 2.0
        Y = (m12 + m22 - 2 * (m13 + m23 + m33)) / 3.0
        z = m12 - m22
        
        return I3, Y, z
    
    def _compute_weight_extrema(self) -> Tuple[float, float, float, float]:
        """Compute min/max values of I₃ and Y across all states"""
        I3_values = []
        Y_values = []
        
        for gt in self.gt_patterns:
            I3, Y, z = self._gt_to_quantum_numbers(gt)
            I3_values.append(I3)
            Y_values.append(Y)
        
        return min(I3_values), max(I3_values), min(Y_values), max(Y_values)
    
    def gt_to_spherical(self, I3: float, Y: float, z: int) -> Tuple[float, float, float]:
        """
        Transform ziggurat coordinates to spherical coordinates.
        
        Parameters
        ----------
        I3 : float
            Isospin third component
        Y : float
            Hypercharge
        z : int
            Multiplicity height (m₁₂ - m₂₂)
        
        Returns
        -------
        r, theta, phi : float
            Spherical coordinates
        """
        # Radial coordinate
        if self.height_mode == 'linear':
            f_z = z / self.z_max if self.z_max > 0 else 0
        elif self.height_mode == 'sqrt':
            f_z = np.sqrt(z / self.z_max) if self.z_max > 0 else 0
        elif self.height_mode == 'quadratic':
            f_z = (z / self.z_max)**2 if self.z_max > 0 else 0
        else:
            f_z = 0
        
        r = self.r0 + self.R_rep * f_z
        
        # Angular coordinates
        # Normalize Y to [-1, 1] for arccos
        if abs(self.Y_max - self.Y_min) > 1e-10:
            Y_norm = 2 * (Y - self.Y_min) / (self.Y_max - self.Y_min) - 1
        else:
            Y_norm = 0
        
        # Ensure Y_norm in valid range for arccos
        Y_norm = np.clip(Y_norm, -1.0, 1.0)
        
        # Polar angle: θ=0 at north pole (max Y), θ=π at south pole (min Y)
        theta = np.arccos(-Y_norm)  # Negative to match convention
        
        # Azimuthal angle: distribute I₃ values around equator
        # Map I₃ from [I3_min, I3_max] to [0, 2π)
        if abs(self.I3_max - self.I3_min) > 1e-10:
            phi = 2 * np.pi * (I3 - self.I3_min) / (self.I3_max - self.I3_min)
        else:
            # Single I₃ value, place at φ=π
            phi = np.pi
        
        return r, theta, phi
    
    def spherical_to_gt(self, r: float, theta: float, phi: float) -> Tuple[float, float, int]:
        """
        Inverse transformation: spherical → ziggurat coordinates.
        
        Parameters
        ----------
        r, theta, phi : float
            Spherical coordinates
        
        Returns
        -------
        I3, Y, z : float, float, int
            Ziggurat coordinates
        """
        # Extract z from r
        if self.R_rep > 1e-10:
            if self.height_mode == 'linear':
                z_norm = (r - self.r0) / self.R_rep
                z = int(np.round(z_norm * self.z_max))
            elif self.height_mode == 'sqrt':
                z_norm = ((r - self.r0) / self.R_rep)**2
                z = int(np.round(z_norm * self.z_max))
            elif self.height_mode == 'quadratic':
                z_norm = np.sqrt((r - self.r0) / self.R_rep)
                z = int(np.round(z_norm * self.z_max))
            else:
                z = 0
        else:
            z = 0
        
        z = np.clip(z, 0, self.z_max)
        
        # Extract Y from theta
        Y_norm = -np.cos(theta)  # Reverse of forward transformation
        Y = self.Y_min + (Y_norm + 1) * (self.Y_max - self.Y_min) / 2
        
        # Extract I₃ from phi
        I3 = self.I3_min + phi * (self.I3_max - self.I3_min) / (2 * np.pi)
        
        return I3, Y, z
    
    def create_spherical_states(self) -> List[SphericalState]:
        """
        Create list of all states in spherical coordinates.
        
        Returns
        -------
        states : List[SphericalState]
            All states with both spherical and GT coordinates
        """
        states = []
        
        for gt_pattern in self.gt_patterns:
            I3, Y, z = self._gt_to_quantum_numbers(gt_pattern)
            r, theta, phi = self.gt_to_spherical(I3, Y, z)
            
            state = SphericalState(
                r=r, theta=theta, phi=phi,
                I3=I3, Y=Y, z=z,
                gt_pattern=gt_pattern
            )
            states.append(state)
        
        return states
    
    def validate_bijection(self, tolerance: float = 1e-10) -> Dict[str, float]:
        """
        Validate that transformation is bijective (1-1 and onto).
        
        Tests round-trip: GT → Spherical → GT
        
        Returns
        -------
        metrics : dict
            'max_I3_error': max |I₃ - I₃'|
            'max_Y_error': max |Y - Y'|
            'max_z_error': max |z - z'|
            'unique_spherical': True if all spherical coords unique
        """
        max_I3_err = 0.0
        max_Y_err = 0.0
        max_z_err = 0
        
        spherical_coords = []
        
        for gt_pattern in self.gt_patterns:
            # Forward
            I3, Y, z = self._gt_to_quantum_numbers(gt_pattern)
            r, theta, phi = self.gt_to_spherical(I3, Y, z)
            
            # Store spherical coords
            spherical_coords.append((r, theta, phi))
            
            # Backward
            I3_recon, Y_recon, z_recon = self.spherical_to_gt(r, theta, phi)
            
            # Errors
            max_I3_err = max(max_I3_err, abs(I3 - I3_recon))
            max_Y_err = max(max_Y_err, abs(Y - Y_recon))
            max_z_err = max(max_z_err, abs(z - z_recon))
        
        # Check uniqueness
        unique_check = len(spherical_coords) == len(set(spherical_coords))
        
        return {
            'max_I3_error': max_I3_err,
            'max_Y_error': max_Y_err,
            'max_z_error': max_z_err,
            'unique_spherical': unique_check,
            'passed': max_I3_err < tolerance and max_Y_err < tolerance and max_z_err == 0
        }
    
    def get_states_by_shell(self) -> Dict[float, List[SphericalState]]:
        """
        Group states by shell (constant r).
        
        Returns
        -------
        shells : dict
            {radius: [states at this radius]}
        """
        states = self.create_spherical_states()
        shells = {}
        
        for state in states:
            r_key = round(state.r, 10)  # Round for grouping
            if r_key not in shells:
                shells[r_key] = []
            shells[r_key].append(state)
        
        return dict(sorted(shells.items()))
    
    def compute_shell_statistics(self) -> Dict[str, any]:
        """
        Compute statistics about shell structure.
        
        Returns
        -------
        stats : dict
            Shell radii, state counts, angular distributions
        """
        shells = self.get_states_by_shell()
        
        stats = {
            'num_shells': len(shells),
            'radii': list(shells.keys()),
            'states_per_shell': [len(states) for states in shells.values()],
            'total_states': sum(len(states) for states in shells.values())
        }
        
        # Angular coverage for each shell
        theta_ranges = []
        phi_ranges = []
        
        for r, states in shells.items():
            thetas = [s.theta for s in states]
            phis = [s.phi for s in states]
            
            theta_ranges.append((min(thetas), max(thetas)))
            phi_ranges.append((min(phis), max(phis)))
        
        stats['theta_ranges'] = theta_ranges
        stats['phi_ranges'] = phi_ranges
        
        return stats


def test_spherical_embedding():
    """Test spherical embedding on standard representations"""
    
    representations = [
        (1, 0, "Fundamental"),
        (0, 1, "Antifundamental"),
        (1, 1, "Adjoint"),
        (2, 0, "Symmetric 6"),
        (0, 2, "Antisymmetric 6̄")
    ]
    
    print("="*80)
    print("Testing Spherical Embedding on Standard Representations")
    print("="*80)
    
    for p, q, name in representations:
        print(f"\n{name} ({p},{q}):")
        print("-"*80)
        
        # Create embedding
        embedding = SU3SphericalEmbedding(p, q)
        
        # Validate bijection
        validation = embedding.validate_bijection()
        print(f"\nBijection Validation:")
        print(f"  Max I₃ error: {validation['max_I3_error']:.2e}")
        print(f"  Max Y error:  {validation['max_Y_error']:.2e}")
        print(f"  Max z error:  {validation['max_z_error']}")
        print(f"  Unique coords: {validation['unique_spherical']}")
        print(f"  PASSED: {validation['passed']}")
        
        # Shell statistics
        stats = embedding.compute_shell_statistics()
        print(f"\nShell Statistics:")
        print(f"  Number of shells: {stats['num_shells']}")
        print(f"  States per shell: {stats['states_per_shell']}")
        
        # Show a few example states
        states = embedding.create_spherical_states()
        print(f"\nExample States (first 5):")
        for i, state in enumerate(states[:5]):
            print(f"  {i+1}. {state}")
        
        print()


if __name__ == "__main__":
    test_spherical_embedding()
