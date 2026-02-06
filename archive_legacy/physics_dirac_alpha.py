"""
PHYSICS_DIRAC_ALPHA.PY
======================
Relativistic Correction to Fine Structure Constant Derivation

Hypothesis:
    The 0.48% error in κ₅ = 137.696 (vs 1/α = 137.036) arises from using
    non-relativistic (Schrödinger) geometry. Incorporating Dirac spinor structure
    and relativistic contraction should reduce the effective surface area.

Strategy:
    1. Model each lattice node with Dirac spinor (large + small components)
    2. Estimate local velocities from wavefunction gradients on lattice
    3. Apply Lorentz contraction: γ = 1/√(1-v²)
    4. Compute contracted area: S_Dirac = Σ A_plaquette · γ_local
    5. Check if κ_Dirac = S_Dirac/P → 137.036

Alternative Approach:
    Discrete Dirac Walk (Zitterbewegung) to measure diffusion area reduction
    due to spinor interference.
"""

import numpy as np
from paraboloid_lattice_su11 import ParaboloidLattice
import sys

# Physical constants
ALPHA = 0.0072973525693      # Fine structure constant
INV_ALPHA = 137.035999084    # Target value
C_LIGHT = 1.0                # Speed of light (natural units)

class DiracLatticeCorrection:
    """
    Incorporate relativistic Dirac spinor corrections to paraboloid geometry.
    
    Computes contracted surface areas accounting for local velocity fields
    estimated from lattice gradients.
    """
    
    def __init__(self, target_n=5):
        """
        Args:
            target_n: Principal quantum number for analysis
        """
        self.n = target_n
        # Need target_n + 1 to compute plaquettes involving transitions to next shell
        self.lattice = ParaboloidLattice(max_n=target_n + 1)
        
        print(f"Initialized Dirac Lattice Correction")
        print(f"  Target shell: n = {target_n}")
        print(f"  Lattice nodes: {self.lattice.dim}")
        
        # Store node positions
        self.positions = {}
        self._compute_all_positions()
        
    def quantum_to_cartesian(self, n, l, m):
        """
        Map quantum numbers to 3D Cartesian coordinates on paraboloid.
        
        Returns:
            (x, y, z) coordinates or None if invalid
        """
        if n < 1 or l >= n or abs(m) > l:
            return None
            
        # Radial coordinate (grows as n²)
        r = n * n
        
        # Energy depth
        z = -1.0 / (n * n)
        
        # Angular coordinates
        if n == 1:
            theta = 0
        else:
            theta = np.pi * l / (n - 1)
            
        if l == 0:
            phi = 0
        else:
            phi = 2 * np.pi * m / (2 * l + 1)
        
        # Cartesian coordinates
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        
        return np.array([x, y, z])
    
    def _compute_all_positions(self):
        """Precompute and cache all node positions."""
        # Include up to n+1 for plaquette calculations
        for n in range(1, self.n + 2):
            for l in range(n):
                for m in range(-l, l + 1):
                    pos = self.quantum_to_cartesian(n, l, m)
                    if pos is not None:
                        self.positions[(n, l, m)] = pos
    
    def estimate_local_velocity(self, n, l, m):
        """
        Estimate local velocity at node (n,l,m) from lattice gradients.
        
        Strategy:
            v_local ~ |∇ψ|/|ψ| ~ (characteristic length scale)^(-1)
            
            For hydrogen: v ~ Z/(n·a₀) ~ 1/n (atomic units)
            
            From virial theorem: <v²> = 1/n² (in atomic units where c=137)
            
            But we need dimensionless v/c. In atomic units:
                v_typical = 1/n (Bohr velocity)
                c = 137.036 (speed of light in atomic units)
                v/c ≈ 1/(137n)
        
        Returns:
            v_local: Local velocity magnitude (dimensionless, v/c)
        """
        # Method 1: Virial theorem estimate
        # <v²> = Z²/(n²) in atomic units, v_rms = Z/n
        # Converting to v/c: divide by c = 137
        v_rms_bohr = 1.0 / n  # Z=1 for hydrogen
        v_over_c = v_rms_bohr / INV_ALPHA
        
        # Method 2: Geometric gradient estimate
        # Measure local spacing to neighbors
        pos = self.positions.get((n, l, m))
        if pos is None:
            return 0.0
        
        # Find nearest neighbors (same shell)
        neighbors = []
        for m_neighbor in range(-l, l + 1):
            if m_neighbor != m:
                neighbor_pos = self.positions.get((n, l, m_neighbor))
                if neighbor_pos is not None:
                    neighbors.append(neighbor_pos)
        
        # Also check adjacent l values
        if l > 0:
            for m_neighbor in range(-(l-1), l):
                neighbor_pos = self.positions.get((n, l-1, m_neighbor))
                if neighbor_pos is not None:
                    neighbors.append(neighbor_pos)
        
        if l < n - 1:
            for m_neighbor in range(-(l+1), l+2):
                neighbor_pos = self.positions.get((n, l+1, m_neighbor))
                if neighbor_pos is not None:
                    neighbors.append(neighbor_pos)
        
        # Compute average spacing
        if len(neighbors) > 0:
            spacings = [np.linalg.norm(neighbor - pos) for neighbor in neighbors]
            avg_spacing = np.mean(spacings)
            
            # Velocity from gradient: v ~ λ/τ ~ spacing/period
            # Orbital period τ ~ n³ (Kepler's 3rd law)
            # Frequency ω ~ 1/n³
            # Velocity v ~ spacing · ω ~ spacing/n³
            
            # More direct: use uncertainty principle
            # Δx·Δp ~ ℏ → v ~ ℏ/(m·Δx) ~ 1/Δx (atomic units)
            # v/c ~ 1/(c·Δx) ~ 1/(137·Δx)
            
            v_over_c_geometric = 1.0 / (INV_ALPHA * avg_spacing)
        else:
            v_over_c_geometric = 0.0
        
        # Use weighted average of both methods
        # Weight virial theorem more heavily (it's exact)
        v_final = 0.8 * v_over_c + 0.2 * v_over_c_geometric
        
        # Sanity check: v/c should be << 1
        if v_final > 0.5:
            v_final = 0.5  # Cap at 0.5c (highly relativistic)
        
        return v_final
    
    def compute_lorentz_factor(self, v_over_c):
        """
        Compute Lorentz contraction factor γ = 1/√(1-v²/c²).
        
        Args:
            v_over_c: Velocity as fraction of speed of light
            
        Returns:
            gamma: Lorentz factor (γ ≥ 1)
        """
        v_squared = v_over_c ** 2
        
        # Avoid division by zero or imaginary numbers
        if v_squared >= 1.0:
            v_squared = 0.99  # Cap at 0.99c
        
        gamma = 1.0 / np.sqrt(1.0 - v_squared)
        
        return gamma
    
    def compute_contracted_plaquette_area(self, n, l, m):
        """
        Compute plaquette area with relativistic contraction correction.
        
        Plaquette corners:
            (n, l, m) → (n+1, l, m) → (n+1, l, m+1) → (n, l, m+1) → (n, l, m)
        
        Correction:
            - Compute geometric area (Schrödinger)
            - Estimate local velocity at plaquette center
            - Apply inverse Lorentz factor: A_Dirac = A_Schrodinger / γ
        
        Rationale:
            Moving clock runs slow → moving ruler contracts
            Proper area element: dA' = dA/γ (length contraction in direction of motion)
        
        Returns:
            area_contracted: Relativistically corrected area
            area_classical: Non-relativistic area
            gamma: Lorentz factor used
        """
        # Get four corners
        p0 = self.positions.get((n, l, m))
        p1 = self.positions.get((n + 1, l, m))
        p2 = self.positions.get((n + 1, l, m + 1))
        p3 = self.positions.get((n, l, m + 1))
        
        # Check validity
        if any(p is None for p in [p0, p1, p2, p3]):
            return 0.0, 0.0, 1.0
        
        # Classical area (two triangles)
        v1 = p1 - p0
        v2 = p2 - p0
        v3 = p3 - p0
        
        area1 = 0.5 * np.linalg.norm(np.cross(v1, v2))
        area2 = 0.5 * np.linalg.norm(np.cross(v2, v3))
        area_classical = area1 + area2
        
        # Estimate velocity at plaquette center
        # Average over four corners
        velocities = []
        for corner in [(n, l, m), (n+1, l, m), (n+1, l, m+1), (n, l, m+1)]:
            if corner in self.positions:
                v_local = self.estimate_local_velocity(*corner)
                velocities.append(v_local)
        
        if len(velocities) > 0:
            v_avg = np.mean(velocities)
        else:
            v_avg = 0.0
        
        # Compute Lorentz factor
        gamma = self.compute_lorentz_factor(v_avg)
        
        # Apply contraction
        # Length contracts as L' = L/γ in direction of motion
        # Area element perpendicular to motion: unchanged
        # Area element parallel to motion: contracts
        # For isotropic velocity distribution: A' ≈ A/γ (approximate)
        
        area_contracted = area_classical / gamma
        
        return area_contracted, area_classical, gamma
    
    def compute_shell_area_corrected(self, n):
        """
        Compute total surface area of shell n with Dirac corrections.
        
        Returns:
            S_Dirac: Contracted area
            S_classical: Non-relativistic area
            gamma_avg: Average Lorentz factor
        """
        S_Dirac = 0.0
        S_classical = 0.0
        gamma_values = []
        count = 0
        
        for l in range(n):
            for m in range(-l, l + 1):
                if m < l:  # Ensure plaquette is valid
                    area_cont, area_class, gamma = self.compute_contracted_plaquette_area(n, l, m)
                    if area_class > 0:
                        S_Dirac += area_cont
                        S_classical += area_class
                        gamma_values.append(gamma)
                        count += 1
        
        gamma_avg = np.mean(gamma_values) if len(gamma_values) > 0 else 1.0
        
        return S_Dirac, S_classical, gamma_avg, count
    
    def compute_impedance_ratio(self, n):
        """
        Compute geometric impedance κ = S/P for shell n.
        
        Returns both Dirac-corrected and classical ratios.
        
        Returns:
            dict with results
        """
        # Get areas
        S_Dirac, S_classical, gamma_avg, n_plaquettes = self.compute_shell_area_corrected(n)
        
        # Photon phase length (circumference-scaled model)
        P_photon = 2 * np.pi * n
        
        # Compute ratios
        kappa_Dirac = S_Dirac / P_photon if P_photon > 0 else 0
        kappa_classical = S_classical / P_photon if P_photon > 0 else 0
        
        # Errors relative to 1/α
        error_Dirac = abs(kappa_Dirac - INV_ALPHA) / INV_ALPHA
        error_classical = abs(kappa_classical - INV_ALPHA) / INV_ALPHA
        
        return {
            'n': n,
            'S_Dirac': S_Dirac,
            'S_classical': S_classical,
            'P_photon': P_photon,
            'gamma_avg': gamma_avg,
            'n_plaquettes': n_plaquettes,
            'kappa_Dirac': kappa_Dirac,
            'kappa_classical': kappa_classical,
            'error_Dirac_pct': error_Dirac * 100,
            'error_classical_pct': error_classical * 100,
            'correction_factor': S_Dirac / S_classical if S_classical > 0 else 1.0
        }


class DiracRandomWalk:
    """
    Alternative approach: Discrete Dirac walk with spinor interference.
    
    Models Zitterbewegung (jittery motion) effects on effective diffusion area.
    """
    
    def __init__(self, lattice_geometry, n_shell=5):
        """
        Args:
            lattice_geometry: DiracLatticeCorrection instance
            n_shell: Shell to analyze
        """
        self.geometry = lattice_geometry
        self.n = n_shell
        
        # Extract nodes on shell n
        self.shell_nodes = []
        for l in range(n_shell):
            for m in range(-l, l + 1):
                if (n_shell, l, m) in self.geometry.positions:
                    self.shell_nodes.append((n_shell, l, m))
        
        print(f"Dirac Random Walk initialized: {len(self.shell_nodes)} nodes on shell n={n_shell}")
    
    def simulate_walk(self, n_steps=1000, n_walkers=100):
        """
        Simulate Dirac walkers on shell with spinor interference.
        
        Each walker is a 2-component spinor that mixes at each step.
        
        Args:
            n_steps: Number of steps per walker
            n_walkers: Number of independent walkers
            
        Returns:
            diffusion_area: Effective area covered by walkers
        """
        if len(self.shell_nodes) == 0:
            return 0.0
        
        # Initialize walkers at random positions
        walker_positions = []
        walker_spinors = []
        
        for _ in range(n_walkers):
            # Random initial position
            node_idx = np.random.randint(len(self.shell_nodes))
            node = self.shell_nodes[node_idx]
            walker_positions.append(node)
            
            # Random initial spinor (normalized)
            psi_up = np.random.randn() + 1j * np.random.randn()
            psi_down = np.random.randn() + 1j * np.random.randn()
            norm = np.sqrt(abs(psi_up)**2 + abs(psi_down)**2)
            walker_spinors.append(np.array([psi_up/norm, psi_down/norm]))
        
        # Simulate steps
        for step in range(n_steps):
            for i in range(n_walkers):
                current_node = walker_positions[i]
                current_spinor = walker_spinors[i]
                
                # Find neighbors on same shell
                n, l, m = current_node
                neighbors = []
                
                # Azimuthal neighbors (same l)
                for dm in [-1, +1]:
                    m_new = m + dm
                    if -l <= m_new <= l:
                        neighbors.append((n, l, m_new))
                
                # Polar neighbors (different l)
                for dl in [-1, +1]:
                    l_new = l + dl
                    if 0 <= l_new < n:
                        # Random m in new l range
                        if l_new == 0:
                            neighbors.append((n, l_new, 0))
                        else:
                            m_new = np.random.randint(-l_new, l_new + 1)
                            neighbors.append((n, l_new, m_new))
                
                # Valid neighbors only
                neighbors = [node for node in neighbors if node in self.geometry.positions]
                
                if len(neighbors) > 0:
                    # Random step
                    next_node = neighbors[np.random.randint(len(neighbors))]
                    walker_positions[i] = next_node
                    
                    # Spinor mixing (simplified Dirac evolution)
                    # Apply Pauli matrix rotation
                    theta = np.random.uniform(0, 2*np.pi)
                    sigma_z = np.array([[1, 0], [0, -1]])
                    U = np.cos(theta/2) * np.eye(2) - 1j * np.sin(theta/2) * sigma_z
                    walker_spinors[i] = U @ current_spinor
        
        # Compute diffusion area
        # Find all unique positions visited
        unique_positions = set(walker_positions)
        positions_cartesian = [self.geometry.positions[node] for node in unique_positions]
        
        if len(positions_cartesian) < 3:
            return 0.0
        
        # Estimate area by convex hull (2D projection onto xy-plane)
        positions_2d = np.array([[pos[0], pos[1]] for pos in positions_cartesian])
        
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(positions_2d)
            area = hull.volume  # In 2D, volume = area
        except:
            # Fallback: bounding box area
            x_min, x_max = positions_2d[:, 0].min(), positions_2d[:, 0].max()
            y_min, y_max = positions_2d[:, 1].min(), positions_2d[:, 1].max()
            area = (x_max - x_min) * (y_max - y_min)
        
        return area


def main():
    """Execute Dirac correction analysis."""
    print("\n" + "="*70)
    print("DIRAC RELATIVISTIC CORRECTION TO α DERIVATION")
    print("="*70)
    print("")
    print("Objective: Refine κ₅ = 137.696 → 137.036 using Dirac spinor geometry")
    print("")
    
    # Initialize Dirac lattice
    dirac = DiracLatticeCorrection(target_n=5)
    
    print("\n" + "="*70)
    print("METHOD 1: LORENTZ CONTRACTION")
    print("="*70)
    
    # Compute for shell n=5
    result = dirac.compute_impedance_ratio(5)
    
    print(f"\nShell n = {result['n']}")
    print(f"  Plaquettes analyzed: {result['n_plaquettes']}")
    print(f"  Average Lorentz factor: γ = {result['gamma_avg']:.6f}")
    print(f"  Contraction factor: S_Dirac/S_classical = {result['correction_factor']:.6f}")
    print("")
    print(f"CLASSICAL (Schrödinger):")
    print(f"  Surface area: S = {result['S_classical']:.6e}")
    print(f"  Phase length: P = {result['P_photon']:.6e}")
    print(f"  Impedance: κ_classical = {result['kappa_classical']:.6f}")
    print(f"  Target: 1/α = {INV_ALPHA:.6f}")
    print(f"  Error: {result['error_classical_pct']:.4f}%")
    print("")
    print(f"DIRAC (Relativistic):")
    print(f"  Contracted area: S_Dirac = {result['S_Dirac']:.6e}")
    print(f"  Phase length: P = {result['P_photon']:.6e}")
    print(f"  Impedance: κ_Dirac = {result['kappa_Dirac']:.6f}")
    print(f"  Target: 1/α = {INV_ALPHA:.6f}")
    print(f"  Error: {result['error_Dirac_pct']:.4f}%")
    print("")
    
    # Improvement assessment
    error_improvement = result['error_classical_pct'] - result['error_Dirac_pct']
    print(f"CORRECTION SUMMARY:")
    print(f"  Classical error: {result['error_classical_pct']:.4f}%")
    print(f"  Dirac error: {result['error_Dirac_pct']:.4f}%")
    print(f"  Improvement: {error_improvement:.4f} percentage points")
    
    if result['error_Dirac_pct'] < result['error_classical_pct']:
        print(f"  ✓ Dirac correction IMPROVES accuracy")
    else:
        print(f"  ✗ Dirac correction WORSENS accuracy")
    
    print("\n" + "="*70)
    print("METHOD 2: DISCRETE DIRAC WALK")
    print("="*70)
    
    # Discrete walk simulation
    walker = DiracRandomWalk(dirac, n_shell=5)
    
    print("\nSimulating Dirac walkers with spinor interference...")
    area_walk = walker.simulate_walk(n_steps=1000, n_walkers=50)
    
    print(f"  Diffusion area (walk): {area_walk:.6e}")
    print(f"  Classical area: {result['S_classical']:.6e}")
    if result['S_classical'] > 0:
        print(f"  Ratio: {area_walk/result['S_classical']:.4f}")
    else:
        print(f"  Ratio: N/A (zero classical area)")
    
    print("\n" + "="*70)
    print("THEORETICAL ANALYSIS")
    print("="*70)
    print("")
    print("Expected Corrections:")
    print(f"  1. Velocity at n=5: v/c ≈ 1/(137·5) ≈ 0.00146 ≈ 0.15%")
    print(f"  2. Lorentz factor: γ ≈ 1 + v²/2 ≈ 1.0000011")
    print(f"  3. Area contraction: A' = A/γ ≈ 0.9999989·A")
    print(f"  4. Expected κ shift: ~0.001% (too small!)")
    print("")
    print("CONCLUSION:")
    print(f"  Required correction: 0.48% (to reach 137.036)")
    print(f"  Achieved correction: {abs(result['correction_factor'] - 1.0)*100:.6f}%")
    
    if abs(result['correction_factor'] - 1.0)*100 > 0.4:
        print("  Status: SIGNIFICANT relativistic correction detected")
    else:
        print("  Status: Correction too small to explain discrepancy")
    
    print("")
    print("Alternative Hypotheses:")
    print("  1. Higher-order curvature corrections (n⁴ vs n⁴·⁺ᵋ)")
    print("  2. Spin-orbit coupling modifies effective geometry")
    print("  3. Phase model requires refinement (P = f(n) more complex)")
    print("  4. Quantum corrections to classical area formula")
    
    # Generate report
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("DIRAC RELATIVISTIC CORRECTION TO α DERIVATION")
    report_lines.append("="*70)
    report_lines.append("")
    report_lines.append(f"Target: 1/α = {INV_ALPHA:.9f}")
    report_lines.append(f"Shell: n = {result['n']}")
    report_lines.append("")
    report_lines.append("RESULTS:")
    report_lines.append("-"*70)
    report_lines.append(f"Classical κ = {result['kappa_classical']:.9f} (error: {result['error_classical_pct']:.6f}%)")
    report_lines.append(f"Dirac κ     = {result['kappa_Dirac']:.9f} (error: {result['error_Dirac_pct']:.6f}%)")
    report_lines.append(f"Improvement = {error_improvement:.6f} percentage points")
    report_lines.append("")
    report_lines.append(f"Lorentz factor: γ_avg = {result['gamma_avg']:.9f}")
    report_lines.append(f"Contraction: {result['correction_factor']:.9f}")
    report_lines.append("")
    report_lines.append("="*70)
    
    report_text = "\n".join(report_lines)
    
    # Save report
    with open("dirac_correction_report.txt", 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("\n" + "="*70)
    print("Report saved to: dirac_correction_report.txt")
    print("="*70)
    
    return result


if __name__ == "__main__":
    try:
        result = main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
