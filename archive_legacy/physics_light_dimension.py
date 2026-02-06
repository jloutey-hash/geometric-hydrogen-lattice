"""
PHYSICS_LIGHT_DIMENSION.PY
===========================
Geometric Field Theory: Fine Structure from Electron-Photon Coupling

Hypothesis:
    α ≈ 1/137 emerges as the "Geometric Impedance" between:
    - Curved Electron Geometry: SO(4,2) Paraboloid (2D surface in 3D)
    - Flat Photon Geometry: U(1) Phase Circle (1D fiber)

Strategy:
    1. Build "Photon Fiber": U(1) phase circles attached to each paraboloid node
    2. Compute "Projection Mismatch": ratio of paraboloid area to phase length
    3. Search for convergence to 137, 1/137, or α-related constants

Geometric Concept:
    Electron emission of photon = "Unrolling" paraboloid patch onto phase circle
    Area (2D) / Length² (1D squared) = Dimensionless coupling constant
"""

import numpy as np
from paraboloid_lattice_su11 import ParaboloidLattice
from scipy.spatial.distance import cdist
import sys

# Alpha constants (exact values)
ALPHA = 0.0072973525693      # e²/(4πε₀ℏc)
INV_ALPHA = 137.035999084
ALPHA_OVER_2PI = 0.0011614048
ALPHA_OVER_4PI = 0.0005807024
SQRT_ALPHA = 0.0854024773

class PhotonFiber:
    """
    U(1) Phase Circle attached to each paraboloid node.
    
    Represents the electromagnetic phase degree of freedom.
    The "length" of the circle is 2π (one complete phase rotation).
    """
    
    def __init__(self, radius=1.0):
        """
        Args:
            radius: Photon fiber radius (default 1.0 for unit phase)
        """
        self.radius = radius
        self.circumference = 2 * np.pi * radius
        
    def phase_length(self, delta_phi):
        """
        Arc length for phase change delta_phi.
        
        Args:
            delta_phi: Phase change in radians
            
        Returns:
            Arc length on the U(1) circle
        """
        return self.radius * abs(delta_phi)
    
    def total_length(self, n_steps):
        """
        Total phase length for n_steps discrete phase jumps.
        
        For electron transition, each photon carries ħω ~ 1/n².
        Phase accumulated per transition: δφ ~ ω·t ~ 1/n²
        
        Args:
            n_steps: Number of phase steps
            
        Returns:
            Total length traversed on phase circle
        """
        # Unit phase step (minimal quantum)
        delta_phi_quantum = 2 * np.pi / n_steps if n_steps > 0 else 0
        return n_steps * self.phase_length(delta_phi_quantum)


class ElectronPhotonCoupling:
    """
    Geometric coupling between Paraboloid (electron) and U(1) Fiber (photon).
    
    Computes projection mismatch ratios to search for α.
    """
    
    def __init__(self, max_n=100):
        """
        Args:
            max_n: Maximum principal quantum number
        """
        self.max_n = max_n
        self.lattice = ParaboloidLattice(max_n=max_n)
        self.photon = PhotonFiber(radius=1.0)
        
        print(f"Initialized Electron-Photon Coupling")
        print(f"  Paraboloid nodes: {self.lattice.dim}")
        print(f"  Max shell: n = {max_n}")
        print(f"  Photon fiber: U(1) circle, R = {self.photon.radius}")
        
    def quantum_to_cartesian(self, n, l, m):
        """
        Map quantum numbers (n, l, m) to 3D Cartesian coordinates on paraboloid.
        
        Geometry:
            r² = n²  (radial shell)
            z = -1/n²  (energy depth)
            θ = π·l/(n-1)  (polar angle, 0 at pole, π at equator)
            φ = 2π·m/(2l+1)  (azimuthal angle)
        
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
            # Ground state at pole
            theta = 0
        else:
            theta = np.pi * l / (n - 1)
            
        if l == 0:
            # s-orbital at pole
            phi = 0
        else:
            phi = 2 * np.pi * m / (2 * l + 1)
        
        # Cartesian coordinates
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        # Note: z already defined as energy
        
        return np.array([x, y, z])
    
    def compute_plaquette_area(self, n, l, m):
        """
        Compute area of rectangular plaquette at (n, l, m).
        
        Plaquette corners (T+ and L+ transitions):
            (n, l, m) → (n+1, l, m) → (n+1, l, m+1) → (n, l, m+1) → (n, l, m)
        
        Returns:
            Plaquette area (sum of two triangles) or 0 if invalid
        """
        # Get four corners
        p0 = self.quantum_to_cartesian(n, l, m)
        p1 = self.quantum_to_cartesian(n + 1, l, m)
        p2 = self.quantum_to_cartesian(n + 1, l, m + 1)
        p3 = self.quantum_to_cartesian(n, l, m + 1)
        
        # Check validity
        if any(p is None for p in [p0, p1, p2, p3]):
            return 0.0
        
        # Area = sum of two triangles: (p0, p1, p2) + (p0, p2, p3)
        v1 = p1 - p0
        v2 = p2 - p0
        v3 = p3 - p0
        
        area1 = 0.5 * np.linalg.norm(np.cross(v1, v2))
        area2 = 0.5 * np.linalg.norm(np.cross(v2, v3))
        
        return area1 + area2
    
    def compute_shell_area(self, n):
        """
        Compute total surface area of paraboloid shell n.
        
        Sum all plaquette areas with base at shell n.
        
        Args:
            n: Principal quantum number
            
        Returns:
            Total area, number of plaquettes
        """
        total_area = 0.0
        count = 0
        
        # Iterate over all valid (l, m) at shell n
        for l in range(n):
            for m in range(-l, l + 1):
                # Check if plaquette is valid (can step m+1)
                if m < l:  # Ensure m+1 ≤ l
                    area = self.compute_plaquette_area(n, l, m)
                    if area > 0:
                        total_area += area
                        count += 1
        
        return total_area, count
    
    def compute_photon_phase_length(self, n):
        """
        Compute phase length for photon emission from shell n.
        
        Concept:
            - Electron transition n → n+1 emits photon ħω
            - ω ~ ΔE ~ 1/n² - 1/(n+1)² ~ 2/n³
            - Phase accumulated: δφ ~ ω·τ
            - For dimensional consistency: use n² as characteristic scale
        
        Strategy: Multiple hypotheses
        
        Args:
            n: Principal quantum number
            
        Returns:
            Dictionary of different phase length estimates
        """
        # Hypothesis 1: Linear phase (n steps around circle)
        L1 = self.photon.total_length(n)
        
        # Hypothesis 2: Quadratic phase (n² steps, matching area scaling)
        L2 = self.photon.total_length(n * n)
        
        # Hypothesis 3: Energy-weighted (ΔE ~ 1/n³)
        # Phase length ~ 1/n
        L3 = self.photon.circumference / n
        
        # Hypothesis 4: Circumference scaled by n
        L4 = self.photon.circumference * n
        
        return {
            'linear': L1,
            'quadratic': L2,
            'energy_weighted': L3,
            'circumference_scaled': L4
        }
    
    def compute_coupling_ratios(self, n):
        """
        Compute all geometric ratios for shell n.
        
        Search for convergence to α-related constants:
            - 137 (1/α)
            - 0.0073 (α)
            - 0.00116 (α/2π)
            - 0.0854 (√α)
        
        Returns:
            Dictionary of ratios and metadata
        """
        # Get paraboloid area
        S_n, n_plaquettes = self.compute_shell_area(n)
        
        # Get photon phase lengths (multiple hypotheses)
        P = self.compute_photon_phase_length(n)
        
        # Compute ratios
        ratios = {}
        
        for key, P_n in P.items():
            if P_n > 0:
                # Ratio 1: S/P (area / length)
                ratio_SP = S_n / P_n
                
                # Ratio 2: S/P² (dimensionless, area / length²)
                ratio_SP2 = S_n / (P_n ** 2)
                
                # Ratio 3: P²/S (inverse)
                ratio_P2S = (P_n ** 2) / S_n if S_n > 0 else 0
                
                # Ratio 4: P/S
                ratio_PS = P_n / S_n if S_n > 0 else 0
                
                ratios[key] = {
                    'S/P': ratio_SP,
                    'S/P²': ratio_SP2,
                    'P²/S': ratio_P2S,
                    'P/S': ratio_PS
                }
        
        return {
            'n': n,
            'area': S_n,
            'n_plaquettes': n_plaquettes,
            'phase_lengths': P,
            'ratios': ratios
        }
    
    def check_alpha_proximity(self, value, tolerance=0.10):
        """
        Check if value is within tolerance of any α-related constant.
        
        Returns:
            (is_match, constant_name, target_value, relative_error)
        """
        targets = {
            '1/α (137)': INV_ALPHA,
            'α (0.0073)': ALPHA,
            'α/(2π) (0.00116)': ALPHA_OVER_2PI,
            'α/(4π) (0.00058)': ALPHA_OVER_4PI,
            '√α (0.0854)': SQRT_ALPHA,
            '4π/α (1726)': 4 * np.pi / ALPHA,
            '2π/α (863)': 2 * np.pi / ALPHA,
            '1/(2πα) (21.8)': 1 / (2 * np.pi * ALPHA),
        }
        
        for name, target in targets.items():
            rel_error = abs(value - target) / target
            if rel_error < tolerance:
                return True, name, target, rel_error
        
        return False, None, None, None
    
    def run_full_analysis(self, n_min=1, n_max=None):
        """
        Run complete analysis across all shells.
        
        Args:
            n_min: Minimum shell
            n_max: Maximum shell (default self.max_n)
            
        Returns:
            results: List of dictionaries with all data
            report: String summary
        """
        if n_max is None:
            n_max = min(self.max_n, 100)
        
        print(f"\n{'='*70}")
        print(f"ELECTRON-PHOTON COUPLING ANALYSIS")
        print(f"{'='*70}")
        print(f"Shells: n = {n_min} to {n_max}")
        print(f"Searching for α ≈ {ALPHA:.10f} or 1/α ≈ {INV_ALPHA:.6f}\n")
        
        results = []
        matches_found = []
        
        for n in range(n_min, n_max + 1):
            data = self.compute_coupling_ratios(n)
            results.append(data)
            
            # Check for α proximity
            for phase_key, ratio_dict in data['ratios'].items():
                for ratio_name, ratio_value in ratio_dict.items():
                    is_match, const_name, target, rel_err = self.check_alpha_proximity(ratio_value)
                    if is_match:
                        matches_found.append({
                            'n': n,
                            'phase_model': phase_key,
                            'ratio': ratio_name,
                            'value': ratio_value,
                            'target': const_name,
                            'target_value': target,
                            'rel_error': rel_err
                        })
        
        # Generate report
        report = self._generate_report(results, matches_found, n_min, n_max)
        
        return results, report, matches_found
    
    def _generate_report(self, results, matches, n_min, n_max):
        """Generate comprehensive text report."""
        lines = []
        lines.append("="*70)
        lines.append("ELECTRON-PHOTON COUPLING: SEARCH FOR α")
        lines.append("="*70)
        lines.append("")
        lines.append("HYPOTHESIS:")
        lines.append("  α emerges from geometric mismatch between:")
        lines.append("    - Curved Electron: SO(4,2) Paraboloid (2D surface)")
        lines.append("    - Flat Photon: U(1) Phase Circle (1D fiber)")
        lines.append("")
        lines.append("METHODOLOGY:")
        lines.append("  1. Compute paraboloid plaquette area S_n (shell n)")
        lines.append("  2. Compute photon phase length P_n (multiple models)")
        lines.append("  3. Calculate ratios: S/P, S/P², P²/S, P/S")
        lines.append("  4. Check convergence to α-related constants")
        lines.append("")
        lines.append(f"ANALYSIS RANGE: n = {n_min} to {n_max}")
        lines.append(f"Total shells analyzed: {len(results)}")
        lines.append("")
        lines.append("="*70)
        lines.append("RESULTS SUMMARY")
        lines.append("="*70)
        lines.append("")
        
        # Sample data from key shells
        sample_shells = [1, 2, 5, 10, 20, 50, 100]
        sample_shells = [n for n in sample_shells if n_min <= n <= n_max]
        
        lines.append("SAMPLE SHELLS:")
        lines.append("-" * 70)
        lines.append(f"{'n':<5} {'Area S_n':<15} {'Plaquettes':<12} {'S/P² (linear)':<20}")
        lines.append("-" * 70)
        
        for n in sample_shells:
            data = next((r for r in results if r['n'] == n), None)
            if data:
                S = data['area']
                np_count = data['n_plaquettes']
                if 'linear' in data['ratios'] and 'S/P²' in data['ratios']['linear']:
                    ratio = data['ratios']['linear']['S/P²']
                    lines.append(f"{n:<5} {S:<15.6e} {np_count:<12} {ratio:<20.10f}")
                else:
                    lines.append(f"{n:<5} {S:<15.6e} {np_count:<12} {'N/A':<20}")
        
        lines.append("")
        lines.append("="*70)
        lines.append("CONVERGENCE ANALYSIS")
        lines.append("="*70)
        lines.append("")
        
        # Check for convergence in each ratio type
        for phase_model in ['linear', 'quadratic', 'energy_weighted', 'circumference_scaled']:
            lines.append(f"\nPhase Model: {phase_model.upper()}")
            lines.append("-" * 70)
            
            for ratio_type in ['S/P', 'S/P²', 'P²/S', 'P/S']:
                # Get last 10 values
                recent_values = []
                for data in results[-10:]:
                    if phase_model in data['ratios'] and ratio_type in data['ratios'][phase_model]:
                        recent_values.append(data['ratios'][phase_model][ratio_type])
                
                if len(recent_values) >= 5:
                    mean_val = np.mean(recent_values)
                    std_val = np.std(recent_values)
                    trend = "CONVERGING" if std_val / abs(mean_val) < 0.01 else "VARYING"
                    
                    lines.append(f"  {ratio_type:<10} : Mean = {mean_val:>12.6e}, "
                               f"Std = {std_val:>12.6e} [{trend}]")
        
        lines.append("")
        lines.append("="*70)
        lines.append("ALPHA PROXIMITY MATCHES")
        lines.append("="*70)
        lines.append("")
        
        if matches:
            lines.append(f"Found {len(matches)} matches within 10% tolerance:")
            lines.append("-" * 70)
            lines.append(f"{'n':<5} {'Phase':<15} {'Ratio':<10} {'Value':<15} {'Target':<20} {'Error':<10}")
            lines.append("-" * 70)
            
            for match in matches[:20]:  # Show first 20
                lines.append(f"{match['n']:<5} {match['phase_model']:<15} {match['ratio']:<10} "
                           f"{match['value']:<15.6e} {match['target']:<20} {match['rel_error']*100:<10.2f}%")
        else:
            lines.append("NO MATCHES FOUND within 10% tolerance.")
            lines.append("")
            lines.append("This indicates α does NOT appear as a simple geometric ratio")
            lines.append("between electron surface area and photon phase length.")
        
        lines.append("")
        lines.append("="*70)
        lines.append("THEORETICAL INTERPRETATION")
        lines.append("="*70)
        lines.append("")
        lines.append("FINDINGS:")
        lines.append("  1. Paraboloid area S_n scales as ~ n⁴ (quadratic surface)")
        lines.append("  2. Phase length P_n depends on model assumption")
        lines.append("  3. Ratios S/P² test dimensional consistency")
        lines.append("")
        
        # Check specific ratio behaviors
        final_data = results[-1]
        S_final = final_data['area']
        n_final = final_data['n']
        
        lines.append(f"ASYMPTOTIC BEHAVIOR (n = {n_final}):")
        lines.append(f"  Shell area: S_{n_final} = {S_final:.6e}")
        
        for phase_key in ['linear', 'quadratic']:
            if phase_key in final_data['ratios']:
                ratio_dict = final_data['ratios'][phase_key]
                lines.append(f"  Phase model '{phase_key}':")
                for ratio_name, ratio_val in ratio_dict.items():
                    lines.append(f"    {ratio_name} = {ratio_val:.6e}")
        
        lines.append("")
        lines.append("CONCLUSION:")
        if not matches:
            lines.append("  α does NOT emerge as Area/Length² projection ratio.")
            lines.append("  Alternative hypotheses:")
            lines.append("    - α may involve curvature integrals (Gauss-Bonnet)")
            lines.append("    - α may require 4D spacetime geometry (not 3D embedding)")
            lines.append("    - α may be edge weight (coupling strength), not node ratio")
        else:
            lines.append("  Potential α signatures detected!")
            lines.append(f"  Best match: {matches[0]['target']} at n={matches[0]['n']}")
            lines.append("  Further investigation required to confirm physical origin.")
        
        lines.append("")
        lines.append("="*70)
        
        return "\n".join(lines)


def main():
    """Execute electron-photon coupling analysis."""
    print("\n" + "="*70)
    print("GEOMETRIC FIELD THEORY: PHOTON-ELECTRON COUPLING")
    print("="*70)
    print("")
    print("Objective: Find α ≈ 1/137 from projection mismatch")
    print("           between Paraboloid (electron) and U(1) (photon)")
    print("")
    
    # Initialize coupling analyzer (reduced for speed)
    coupling = ElectronPhotonCoupling(max_n=50)
    
    # Run full analysis
    results, report, matches = coupling.run_full_analysis(n_min=1, n_max=50)
    
    # Display report
    print("\n" + report)
    
    # Save to file
    output_file = "light_coupling_report.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nReport saved to: {output_file}")
    
    # Additional detailed data
    if matches:
        print(f"\n⚠ ALPHA SIGNATURES DETECTED: {len(matches)} matches")
        print("See report for details.")
    else:
        print("\n✗ NO ALPHA SIGNATURES: Projection hypothesis fails.")
        print("α does not appear as simple area/length ratio.")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    return results, matches


if __name__ == "__main__":
    try:
        results, matches = main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
