"""
ALPHA DERIVATION: Spin-Orbit Holonomy and the Fine Structure Constant
=======================================================================
Hypothesis: α ≈ 1/137 is the "gear ratio" between orbital geometry (paraboloid)
            and spin geometry (quaternion fiber).

Method: Measure the holonomy deficit when parallel transporting a spinor
        around closed loops (plaquettes) on the lattice, and relate this
        to the geometric area enclosed.

Theory: On a curved surface, parallel transport around a loop causes
        rotation by angle θ = κ·Area, where κ is the Gaussian curvature.
        For spin-1/2 particles, the spinor picks up an additional phase
        factor related to the solid angle subtended.

Question: Is the ratio Area/Deficit = 137 or related constant?

Author: Geometric Constant Discovery Team
Date: February 2026
"""

import numpy as np
import scipy.sparse as sp
from scipy.spatial.transform import Rotation as R
from paraboloid_lattice_su11 import ParaboloidLattice
from typing import Dict, List, Tuple, Optional
import time


class SpinorGeometry:
    """
    Spin geometry on the paraboloid lattice.
    
    Each node has:
    - Position in 3D: (x, y, z) on paraboloid z = -(x²+y²)/(2r²)
    - Local frame: (tangent1, tangent2, normal)
    - Spinor: 2-component complex vector
    """
    
    def __init__(self, max_n: int = 20):
        """
        Initialize spinor geometry on paraboloid lattice.
        """
        self.max_n = max_n
        self.lattice = ParaboloidLattice(max_n=max_n)
        
        # Map quantum numbers to 3D positions
        self.positions = {}
        self.local_frames = {}
        self.compute_geometry()
        
    def quantum_to_cartesian(self, n: int, l: int, m: int) -> np.ndarray:
        """
        Map quantum numbers (n,l,m) to 3D Cartesian coordinates on paraboloid.
        
        Convention:
        - Radial: r = n² (parabolic radius)
        - Angular: θ = π·l/(n-1), φ = 2π·m/(2l+1)
        - Height: z = -1/n² (energy)
        """
        if n == 0:
            return np.array([0.0, 0.0, 0.0])
        
        # Radial distance scales with n²
        r = n**2
        
        # Angular position
        if n == 1:
            theta = 0.0  # Pole
        else:
            theta = np.pi * l / (n - 1)  # Colatitude [0, π]
        
        if l == 0:
            phi = 0.0  # Arbitrary for l=0
        else:
            # Azimuthal angle, map m ∈ [-l, l] to φ ∈ [0, 2π]
            phi = 2 * np.pi * (m + l) / (2 * l + 1)
        
        # Cartesian on paraboloid surface
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = -1 / (n**2)  # Energy level
        
        return np.array([x, y, z])
    
    def compute_normal(self, pos: np.ndarray) -> np.ndarray:
        """
        Compute unit normal to paraboloid surface at position (x,y,z).
        
        Paraboloid: z = -(x² + y²)/(2R²) where R ~ n²
        Gradient: ∇f = (∂z/∂x, ∂z/∂y, -1) = (-x/R², -y/R², -1)
        Normal: N = ∇f / |∇f|
        """
        x, y, z = pos
        
        # Estimate R² from z: z ≈ -1/n² ⟹ R ≈ n² ⟹ R² ≈ -1/z
        if abs(z) < 1e-10:
            # At origin
            return np.array([0.0, 0.0, 1.0])
        
        R_squared = max(1.0, -1.0 / z)  # Avoid division by zero
        
        # Gradient
        grad = np.array([-x / R_squared, -y / R_squared, -1.0])
        
        # Normalize
        normal = grad / np.linalg.norm(grad)
        
        return normal
    
    def compute_local_frame(self, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute orthonormal frame (e1, e2, N) at position.
        
        e1, e2: tangent vectors
        N: unit normal
        """
        normal = self.compute_normal(pos)
        
        # Choose tangent vectors orthogonal to normal
        # Use Gram-Schmidt to find orthonormal basis in tangent plane
        
        # Start with arbitrary vector not parallel to normal
        if abs(normal[2]) < 0.9:
            v1 = np.array([0.0, 0.0, 1.0])
        else:
            v1 = np.array([1.0, 0.0, 0.0])
        
        # Project onto tangent plane
        e1 = v1 - np.dot(v1, normal) * normal
        e1 = e1 / np.linalg.norm(e1)
        
        # Second tangent vector via cross product
        e2 = np.cross(normal, e1)
        e2 = e2 / np.linalg.norm(e2)
        
        return e1, e2, normal
    
    def compute_geometry(self):
        """
        Pre-compute positions and local frames for all nodes.
        """
        print(f"Computing spinor geometry for {len(self.lattice.nodes)} nodes...")
        
        for node in self.lattice.nodes:
            n, l, m = node
            pos = self.quantum_to_cartesian(n, l, m)
            frame = self.compute_local_frame(pos)
            
            self.positions[node] = pos
            self.local_frames[node] = frame
        
        print(f"[OK] Geometry computed.")
    
    def parallel_transport_operator(self, node1: Tuple[int,int,int], 
                                   node2: Tuple[int,int,int]) -> np.ndarray:
        """
        Compute 2×2 unitary matrix for parallel transport from node1 to node2.
        
        Theory: When moving along a curve, the spinor frame must rotate to
        remain aligned with the changing local coordinate system.
        
        For SU(2) spinors, the transport operator is:
        U = exp(-i·σ·n̂·θ/2)
        where θ is the rotation angle and n̂ is the rotation axis.
        """
        if node1 not in self.local_frames or node2 not in self.local_frames:
            # Identity if nodes not found
            return np.eye(2, dtype=complex)
        
        # Get local frames
        e1_1, e2_1, N1 = self.local_frames[node1]
        e1_2, e2_2, N2 = self.local_frames[node2]
        
        # Construct rotation matrices (frame -> world)
        R1 = np.column_stack([e1_1, e2_1, N1])
        R2 = np.column_stack([e1_2, e2_2, N2])
        
        # Rotation from frame1 to frame2
        R_12 = R2.T @ R1  # Transforms coordinates from frame1 to frame2
        
        # Convert to scipy Rotation
        try:
            rot = R.from_matrix(R_12)
            rotvec = rot.as_rotvec()
            angle = np.linalg.norm(rotvec)
            
            if angle < 1e-10:
                # No rotation
                return np.eye(2, dtype=complex)
            
            axis = rotvec / angle
            
        except Exception as e:
            # Fallback: use rotation vector directly
            # Compute axis and angle from rotation matrix
            trace = np.trace(R_12)
            angle = np.arccos((trace - 1) / 2)
            
            if angle < 1e-10:
                return np.eye(2, dtype=complex)
            
            # Axis from antisymmetric part
            axis = np.array([
                R_12[2,1] - R_12[1,2],
                R_12[0,2] - R_12[2,0],
                R_12[1,0] - R_12[0,1]
            ]) / (2 * np.sin(angle))
        
        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # σ·n̂ operator
        sigma_dot_n = axis[0]*sigma_x + axis[1]*sigma_y + axis[2]*sigma_z
        
        # Spinor transport: U = exp(-i·σ·n̂·θ/2)
        # For spin-1/2, rotation by θ → spinor rotates by θ/2
        U = sp.linalg.expm(-1j * sigma_dot_n * angle / 2)
        
        return U
    
    def plaquette_holonomy(self, corners: List[Tuple[int,int,int]]) -> float:
        """
        Compute holonomy deficit angle around a plaquette.
        
        Args:
            corners: List of 4 nodes forming a rectangular plaquette
        
        Returns:
            deficit_angle: Angle (in radians) by which spinor rotates
        """
        if len(corners) != 4:
            return 0.0
        
        # Check all corners exist
        if not all(c in self.local_frames for c in corners):
            return 0.0
        
        # Initial spinor: aligned with local z-axis (normal direction)
        spinor_init = np.array([1.0, 0.0], dtype=complex)
        
        # Transport around loop
        spinor = spinor_init.copy()
        
        for i in range(4):
            node1 = corners[i]
            node2 = corners[(i+1) % 4]
            
            U = self.parallel_transport_operator(node1, node2)
            spinor = U @ spinor
        
        # Measure deficit: angle between final and initial spinor
        # For complex vectors: cos(θ) = |⟨ψ₁|ψ₂⟩|
        overlap = np.abs(np.vdot(spinor_init, spinor))
        
        # Protect against numerical errors
        overlap = np.clip(overlap, 0.0, 1.0)
        
        # Deficit angle (full rotation for spin-1/2 is 4π, so factor of 2)
        deficit_angle = 2 * np.arccos(overlap)
        
        return deficit_angle
    
    def plaquette_area(self, corners: List[Tuple[int,int,int]]) -> float:
        """
        Compute geometric area of plaquette on paraboloid surface.
        
        Approximate as sum of two triangular areas.
        """
        if len(corners) != 4:
            return 0.0
        
        # Get positions
        positions = [self.positions.get(c) for c in corners]
        if any(p is None for p in positions):
            return 0.0
        
        p1, p2, p3, p4 = positions
        
        # Two triangles: (1,2,3) and (1,3,4)
        area1 = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
        area2 = 0.5 * np.linalg.norm(np.cross(p3 - p1, p4 - p1))
        
        total_area = area1 + area2
        
        return total_area


class AlphaDerivation:
    """
    Search for fine structure constant via spin-orbit coupling ratio.
    """
    
    def __init__(self, max_n: int = 20):
        self.max_n = max_n
        self.spinor_geom = SpinorGeometry(max_n=max_n)
        
        self.results = []
        
        # Known constants
        self.alpha = 0.0072973525693
        self.inv_alpha = 137.035999084
        self.alpha_over_4pi = self.alpha / (4 * np.pi)
        
    def find_plaquettes(self, n: int) -> List[List[Tuple[int,int,int]]]:
        """
        Find all rectangular plaquettes in shell n.
        
        Plaquette types:
        1. Radial-azimuthal: (n,l,m) → (n+1,l,m) → (n+1,l,m+1) → (n,l,m+1)
        2. Angular-azimuthal: (n,l,m) → (n,l+1,m) → (n,l+1,m+1) → (n,l,m+1)
        """
        plaquettes = []
        nodes = self.spinor_geom.lattice.nodes
        
        # Type 1: Radial-azimuthal rectangles (fixed l)
        for node_n, node_l, node_m in nodes:
            if node_n != n:
                continue
            
            # Build rectangle in (n,m) space at fixed l
            corners = [
                (n, node_l, node_m),
                (n+1, node_l, node_m),
                (n+1, node_l, node_m+1),
                (n, node_l, node_m+1)
            ]
            
            # Check all corners exist
            if all(c in nodes for c in corners):
                plaquettes.append(corners)
        
        # Type 2: Angular-azimuthal rectangles (fixed n)
        for node_n, node_l, node_m in nodes:
            if node_n != n:
                continue
            
            # Build rectangle in (l,m) space at fixed n
            corners = [
                (n, node_l, node_m),
                (n, node_l+1, node_m),
                (n, node_l+1, node_m+1),
                (n, node_l, node_m+1)
            ]
            
            # Check all corners exist
            if all(c in nodes for c in corners):
                plaquettes.append(corners)
        
        return plaquettes
    
    def analyze_shell(self, n: int) -> Dict:
        """
        Analyze spin-orbit coupling for shell n.
        """
        print(f"\n{'='*80}")
        print(f"Analyzing shell n = {n}")
        print(f"{'='*80}")
        
        plaquettes = self.find_plaquettes(n)
        
        if len(plaquettes) == 0:
            print(f"  No plaquettes found in shell {n}")
            return {'n': n, 'n_plaquettes': 0}
        
        print(f"  Found {len(plaquettes)} plaquettes")
        
        # Compute holonomy and area for each plaquette
        deficits = []
        areas = []
        ratios = []
        
        for i, corners in enumerate(plaquettes):
            deficit = self.spinor_geom.plaquette_holonomy(corners)
            area = self.spinor_geom.plaquette_area(corners)
            
            if deficit > 1e-10 and area > 1e-10:
                ratio = area / deficit
                deficits.append(deficit)
                areas.append(area)
                ratios.append(ratio)
                
                if i < 5:  # Show first few
                    print(f"    Plaquette {i}: Area={area:.6f}, Deficit={deficit:.6f} rad, Ratio={ratio:.6f}")
        
        if len(ratios) == 0:
            print(f"  [WARNING] No valid plaquettes (deficit=0 or area=0)")
            return {'n': n, 'n_plaquettes': len(plaquettes), 'valid': 0}
        
        # Statistics
        mean_deficit = np.mean(deficits)
        mean_area = np.mean(areas)
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        
        # Total quantities
        total_deficit = np.sum(deficits)
        total_area = np.sum(areas)
        total_ratio = total_area / total_deficit if total_deficit > 0 else 0
        
        result = {
            'n': n,
            'n_plaquettes': len(plaquettes),
            'valid_plaquettes': len(ratios),
            'mean_deficit': mean_deficit,
            'mean_area': mean_area,
            'mean_ratio': mean_ratio,
            'std_ratio': std_ratio,
            'total_deficit': total_deficit,
            'total_area': total_area,
            'total_ratio': total_ratio
        }
        
        print(f"\n  Summary:")
        print(f"    Valid plaquettes: {len(ratios)}")
        print(f"    Mean area: {mean_area:.6f}")
        print(f"    Mean deficit: {mean_deficit:.6f} rad")
        print(f"    Mean Area/Deficit: {mean_ratio:.6f}")
        print(f"    Total Area/Deficit: {total_ratio:.6f}")
        print(f"    Std dev: {std_ratio:.6f}")
        
        # Check proximity to alpha-related constants
        self.check_alpha_match("Mean Ratio", mean_ratio)
        self.check_alpha_match("Total Ratio", total_ratio)
        
        return result
    
    def check_alpha_match(self, name: str, value: float):
        """
        Check if value matches alpha-related constants.
        """
        targets = {
            '1/α': self.inv_alpha,
            'α': self.alpha,
            'α/(4π)': self.alpha_over_4pi,
            '(1/α)/(4π)': self.inv_alpha / (4 * np.pi),  # ≈ 10.9
            '(1/α)/2': self.inv_alpha / 2,  # ≈ 68.5
            '2·(1/α)': 2 * self.inv_alpha,  # ≈ 274
        }
        
        for target_name, target_value in targets.items():
            rel_error = abs(value - target_value) / target_value if target_value != 0 else float('inf')
            
            if rel_error < 0.05:  # Within 5%
                print(f"    *** MATCH: {name} ≈ {target_name} = {target_value:.6f} (error: {rel_error*100:.2f}%) ***")
            elif rel_error < 0.15:  # Within 15%
                print(f"    CLOSE: {name} ≈ {target_name} = {target_value:.6f} (error: {rel_error*100:.2f}%)")
    
    def run_derivation(self, n_start: int = 2, n_end: int = 15, n_step: int = 2):
        """
        Main derivation loop.
        """
        print("="*80)
        print("ALPHA DERIVATION: Spin-Orbit Holonomy Analysis")
        print("="*80)
        print(f"\nTarget constants:")
        print(f"  α = {self.alpha:.10f}")
        print(f"  1/α = {self.inv_alpha:.10f}")
        print(f"  α/(4π) = {self.alpha_over_4pi:.10f}")
        print(f"  (1/α)/(4π) = {self.inv_alpha/(4*np.pi):.10f}")
        print()
        
        for n in range(n_start, n_end + 1, n_step):
            result = self.analyze_shell(n)
            if result.get('valid_plaquettes', 0) > 0:
                self.results.append(result)
        
        # Convergence analysis
        self.analyze_convergence()
    
    def analyze_convergence(self):
        """
        Check if ratios converge to alpha-related values.
        """
        if len(self.results) < 3:
            return
        
        print("\n" + "="*80)
        print("CONVERGENCE ANALYSIS")
        print("="*80)
        
        # Extract mean ratios
        mean_ratios = [r['mean_ratio'] for r in self.results]
        total_ratios = [r['total_ratio'] for r in self.results]
        
        # Last 5 values
        if len(mean_ratios) >= 5:
            recent_mean = np.mean(mean_ratios[-5:])
            recent_std = np.std(mean_ratios[-5:])
            
            print(f"\nMean Area/Deficit (last 5 shells):")
            print(f"  Value: {recent_mean:.6f} ± {recent_std:.6f}")
            
            self.check_alpha_match("Converged Mean", recent_mean)
        
        if len(total_ratios) >= 5:
            recent_total = np.mean(total_ratios[-5:])
            recent_std_total = np.std(total_ratios[-5:])
            
            print(f"\nTotal Area/Deficit (last 5 shells):")
            print(f"  Value: {recent_total:.6f} ± {recent_std_total:.6f}")
            
            self.check_alpha_match("Converged Total", recent_total)
    
    def generate_report(self, filename: str = "alpha_derivation_report.txt"):
        """
        Generate comprehensive report.
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ALPHA DERIVATION REPORT: Spin-Orbit Holonomy\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("HYPOTHESIS:\n")
            f.write("The fine structure constant α ≈ 1/137 emerges as the ratio between\n")
            f.write("geometric area and spin holonomy deficit on the paraboloid lattice.\n\n")
            
            f.write("METHOD:\n")
            f.write("1. Construct spinor frames at each lattice node\n")
            f.write("2. Compute parallel transport operators between adjacent nodes\n")
            f.write("3. Measure holonomy deficit around closed plaquettes\n")
            f.write("4. Calculate ratio: Area / Deficit\n")
            f.write("5. Search for convergence to α-related constants\n\n")
            
            f.write("TARGET VALUES:\n")
            f.write(f"  α         = {self.alpha:.10f}\n")
            f.write(f"  1/α       = {self.inv_alpha:.10f}\n")
            f.write(f"  α/(4π)    = {self.alpha_over_4pi:.10f}\n")
            f.write(f"  (1/α)/(4π) = {self.inv_alpha/(4*np.pi):.10f}\n\n")
            
            f.write("="*80 + "\n")
            f.write("RESULTS BY SHELL\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"{'n':<5} {'Plaq':<8} {'Mean Area':<12} {'Mean Deficit':<15} {'Mean Ratio':<15} {'Total Ratio':<15}\n")
            f.write("-"*80 + "\n")
            
            for r in self.results:
                f.write(f"{r['n']:<5} {r['valid_plaquettes']:<8} {r['mean_area']:<12.6f} "
                       f"{r['mean_deficit']:<15.6e} {r['mean_ratio']:<15.6f} {r['total_ratio']:<15.6f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("CONCLUSION\n")
            f.write("="*80 + "\n\n")
            
            if self.results:
                # Check final convergence
                final_mean = self.results[-1]['mean_ratio']
                final_total = self.results[-1]['total_ratio']
                
                f.write(f"Final mean ratio: {final_mean:.6f}\n")
                f.write(f"Final total ratio: {final_total:.6f}\n\n")
                
                # Compare to targets
                for target_name, target_val in [('1/α', self.inv_alpha), 
                                               ('α', self.alpha),
                                               ('(1/α)/(4π)', self.inv_alpha/(4*np.pi))]:
                    error_mean = abs(final_mean - target_val) / target_val * 100
                    error_total = abs(final_total - target_val) / target_val * 100
                    
                    f.write(f"Comparison to {target_name} = {target_val:.6f}:\n")
                    f.write(f"  Mean ratio error: {error_mean:.2f}%\n")
                    f.write(f"  Total ratio error: {error_total:.2f}%\n\n")
        
        print(f"\n[OK] Report saved to {filename}")


def main():
    """
    Main execution: derive alpha from spin-orbit holonomy.
    """
    print("="*80)
    print("ALPHA DERIVATION FROM SPIN-ORBIT COUPLING")
    print("="*80)
    print("\nSearching for α ≈ 1/137 in the geometric coupling between")
    print("orbital motion (paraboloid base) and spin (quaternion fiber).")
    print()
    
    # Run derivation (limit to n<=15 for speed)
    derivation = AlphaDerivation(max_n=15)
    derivation.run_derivation(n_start=2, n_end=14, n_step=2)
    
    # Generate report
    derivation.generate_report()
    
    print("\n" + "="*80)
    print("DERIVATION COMPLETE")
    print("="*80)
    print("\nResults saved to: alpha_derivation_report.txt")


if __name__ == "__main__":
    main()
