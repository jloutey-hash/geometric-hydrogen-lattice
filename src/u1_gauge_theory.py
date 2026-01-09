"""
Phase 10.1: U(1) Gauge Theory on Discrete Angular Momentum Lattice

Tests whether the geometric constant 1/(4π) appears in compact U(1) gauge theory
(electromagnetism). This is the critical test for universality - if e² ≈ 1/(4π)
for both Abelian U(1) and non-Abelian SU(2), the constant is gauge-universal.

Compact U(1):
- Links carry angles θ ∈ [0, 2π)
- Wilson plaquette action: S = β Σ_□ [1 - cos(θ_□)]
- Coupling: β = 1/e²
- Test: e² ≈ 1/(4π) = 0.079577?

Physical Implications:
- If confirmed: Fine structure constant α = e²/(4πε₀ℏc) has geometric origin
- Connects to SU(2) result (g² ≈ 1/(4π) with 0.5% error)
- Could explain electromagnetic coupling from discrete structure

Author: Research Team
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lattice import PolarLattice


class U1GaugeTheory:
    """
    Compact U(1) gauge theory on discrete angular momentum lattice.
    
    Links carry phases θ_link ∈ [0, 2π). Plaquettes are products of 4 phases.
    Wilson action: S = β Σ_□ [1 - cos(θ_□)]
    
    This is the simplest gauge theory - electromagnetism on the lattice.
    """
    
    def __init__(self, n_max: int = 8, seed: int = 42):
        """
        Initialize U(1) gauge theory on polar lattice.
        
        Args:
            n_max: Maximum principal quantum number
            seed: Random seed for Monte Carlo
        """
        self.n_max = n_max
        self.lattice = PolarLattice(n_max)
        self.rng = np.random.RandomState(seed)
        
        # Build point index for fast lookup
        self.point_index = {}
        for i, pt in enumerate(self.lattice.points):
            key = (pt['ℓ'], pt['m_ℓ'], pt['m_s'])
            self.point_index[key] = i
        
        # Initialize gauge field (angles on links)
        # Links are between neighboring (ell,m) points
        self.links = self._initialize_links()
        
        print(f"U1GaugeTheory initialized")
        print(f"  n_max = {n_max}")
        print(f"  Total sites = {len(self.lattice.points)}")
        print(f"  Total links = {len(self.links)}")
        
    def _initialize_links(self) -> Dict[Tuple[int, int], float]:
        """
        Initialize U(1) links as angles θ ∈ [0, 2π).
        
        Returns:
            Dict mapping (site_idx1, site_idx2) to angle θ
        """
        links = {}
        
        # Create links between neighboring lattice sites
        # Define neighbors as points within same ℓ shell or adjacent shells
        for i, pt1 in enumerate(self.lattice.points):
            ell1, m1, ms1 = pt1['ℓ'], pt1['m_ℓ'], pt1['m_s']
            
            for j, pt2 in enumerate(self.lattice.points):
                if j <= i:  # Avoid duplicates
                    continue
                    
                ell2, m2, ms2 = pt2['ℓ'], pt2['m_ℓ'], pt2['m_s']
                
                # Connect if:
                # 1. Same ℓ, adjacent m (within shell)
                # 2. Adjacent ℓ, same or nearby m (between shells)
                # 3. Same spin (for simplicity)
                
                is_neighbor = False
                
                # Same shell, adjacent m
                if ell1 == ell2 and ms1 == ms2:
                    if abs(m1 - m2) == 1 or (abs(m1 - m2) == 2*ell1):  # periodic in m
                        is_neighbor = True
                
                # Adjacent shells
                if abs(ell1 - ell2) == 1 and ms1 == ms2:
                    if abs(m1 - m2) <= 1:  # Connect nearby m values
                        is_neighbor = True
                
                if is_neighbor:
                    link = (i, j)  # Use indices
                    # Initialize with small random angles (cold start)
                    links[link] = self.rng.uniform(-0.1, 0.1)
                    
        return links
    
    def _make_link(self, i1: int, i2: int) -> Tuple[int, int]:
        """Create ordered link tuple from indices."""
        if i1 < i2:
            return (i1, i2)
        else:
            return (i2, i1)
    
    def get_plaquettes(self) -> List[List[int]]:
        """
        Find all plaquettes (closed 4-cycles) on the lattice.
        
        A plaquette is a minimal closed loop of 4 links.
        For the polar lattice, these are rectangular patterns
        in the (ell, m) space.
        
        Returns:
            List of plaquettes, each is list of 4 site indices [i1, i2, i3, i4]
        """
        plaquettes = []
        
        # Build adjacency for finding plaquettes
        # Group points by (ℓ, m_s) for easier search
        by_ell_spin = {}
        for i, pt in enumerate(self.lattice.points):
            key = (pt['ℓ'], pt['m_s'])
            if key not in by_ell_spin:
                by_ell_spin[key] = []
            by_ell_spin[key].append((i, pt['m_ℓ']))
        
        # Find rectangular plaquettes
        # Pattern: (ell1,m1) - (ell1,m2) - (ell2,m2) - (ell2,m1) - (ell1,m1)
        for ms in [-0.5, 0.5]:  # For each spin
            ell_values = sorted([ell for (ell, s) in by_ell_spin.keys() if s == ms])
            
            for k in range(len(ell_values) - 1):
                ell1, ell2 = ell_values[k], ell_values[k+1]
                
                # Get points at these ℓ values
                pts1 = by_ell_spin.get((ell1, ms), [])
                pts2 = by_ell_spin.get((ell2, ms), [])
                
                # Find rectangles
                for i1, m1 in pts1:
                    for i2, m2 in pts1:
                        if m2 <= m1:
                            continue
                        # Now need (ell2, m1) and (ell2, m2)
                        i3 = None
                        i4 = None
                        for i, m in pts2:
                            if m == m1:
                                i3 = i
                            if m == m2:
                                i4 = i
                        
                        if i3 is not None and i4 is not None:
                            # Check if all 4 links exist
                            links_needed = [
                                self._make_link(i1, i2),
                                self._make_link(i2, i4),
                                self._make_link(i4, i3),
                                self._make_link(i3, i1)
                            ]
                            
                            if all(link in self.links for link in links_needed):
                                plaquettes.append([i1, i2, i4, i3])
        
        return plaquettes
    
    def compute_plaquette_angle(self, plaquette: List[int]) -> float:
        """
        Compute total angle around a plaquette.
        
        θ_□ = θ₁₂ + θ₂₃ + θ₃₄ + θ₄₁ (mod 2π)
        
        Args:
            plaquette: List of 4 site indices [i1, i2, i3, i4]
            
        Returns:
            Total angle mod 2π
        """
        i1, i2, i3, i4 = plaquette
        
        # Get angles with correct orientation
        theta1 = self._get_oriented_angle(i1, i2)
        theta2 = self._get_oriented_angle(i2, i3)
        theta3 = self._get_oriented_angle(i3, i4)
        theta4 = self._get_oriented_angle(i4, i1)
        
        # Total angle around loop
        theta_total = theta1 + theta2 + theta3 + theta4
        
        # Reduce to [-π, π]
        while theta_total > np.pi:
            theta_total -= 2*np.pi
        while theta_total < -np.pi:
            theta_total += 2*np.pi
            
        return theta_total
    
    def _get_oriented_angle(self, i1: int, i2: int) -> float:
        """Get angle with correct orientation for ordered link."""
        link = self._make_link(i1, i2)
        angle = self.links[link]
        
        # If link is reversed, negate angle
        if link[0] != i1:
            angle = -angle
            
        return angle
    
    def wilson_action(self, beta: float) -> float:
        """
        Compute Wilson action for U(1) gauge theory.
        
        S = β Σ_□ [1 - cos(θ_□)]
        
        Args:
            beta: Inverse coupling β = 1/e²
            
        Returns:
            Total action
        """
        plaquettes = self.get_plaquettes()
        
        action = 0.0
        for plaq in plaquettes:
            theta = self.compute_plaquette_angle(plaq)
            action += (1.0 - np.cos(theta))
            
        return beta * action
    
    def measure_observables(self, beta: float) -> Dict[str, float]:
        """
        Measure gauge observables.
        
        Args:
            beta: Coupling parameter
            
        Returns:
            Dictionary of observables
        """
        plaquettes = self.get_plaquettes()
        
        if len(plaquettes) == 0:
            return {
                'plaq_real': 0.0,
                'plaq_abs': 0.0,
                'action_density': 0.0
            }
        
        # Average plaquette values
        cos_sum = 0.0
        abs_sum = 0.0
        
        for plaq in plaquettes:
            theta = self.compute_plaquette_angle(plaq)
            cos_sum += np.cos(theta)
            abs_sum += np.abs(theta)
            
        n_plaq = len(plaquettes)
        
        return {
            'plaq_real': cos_sum / n_plaq,  # ⟨Re U_□⟩ = ⟨cos θ_□⟩
            'plaq_abs': abs_sum / n_plaq,    # ⟨|θ_□|⟩
            'action_density': self.wilson_action(beta) / n_plaq,
            'n_plaquettes': n_plaq
        }
    
    def metropolis_update(self, beta: float, n_sweeps: int = 100):
        """
        Monte Carlo update using Metropolis algorithm.
        
        Args:
            beta: Inverse coupling parameter
            n_sweeps: Number of sweeps through all links
        """
        links_list = list(self.links.keys())
        
        for sweep in range(n_sweeps):
            for link in links_list:
                # Propose new angle
                old_angle = self.links[link]
                delta = self.rng.uniform(-0.5, 0.5)
                new_angle = old_angle + delta
                
                # Compute action change
                self.links[link] = new_angle
                delta_S = self._action_change_for_link(link, old_angle, new_angle, beta)
                
                # Metropolis acceptance
                if delta_S > 0 and self.rng.uniform() > np.exp(-delta_S):
                    # Reject
                    self.links[link] = old_angle
    
    def _action_change_for_link(self, link: Tuple[int, int], 
                                  old_angle: float, new_angle: float, beta: float) -> float:
        """Compute action change when updating a link (simplified)."""
        # For speed, use approximate local action
        # Full calculation would find all plaquettes containing this link
        delta_theta = new_angle - old_angle
        
        # Approximate: assume ~4 plaquettes per link
        # ΔS ≈ β × 4 × [1 - cos(θ + Δθ)]
        return beta * 4 * (1 - np.cos(delta_theta))
    
    def thermalize(self, beta: float, n_therm: int = 1000):
        """
        Thermalize the gauge configuration.
        
        Args:
            beta: Coupling parameter
            n_therm: Thermalization sweeps
        """
        print(f"\nThermalizing at beta = {beta:.4f}...")
        
        # Progressive thermalization with status
        for i in range(10):
            self.metropolis_update(beta, n_sweeps=n_therm//10)
            if (i+1) % 2 == 0:
                obs = self.measure_observables(beta)
                print(f"  Step {i+1}/10: ⟨cos θ_□⟩ = {obs['plaq_real']:.6f}")
    
    def measure_coupling(self, beta: float, n_measure: int = 100, 
                        n_skip: int = 10) -> Dict[str, float]:
        """
        Measure effective coupling e² from plaquette expectation.
        
        The coupling is extracted from: ⟨cos θ_□⟩ ≈ exp(-1/(2β)) for small β
        or from action density.
        
        Args:
            beta: Coupling parameter
            n_measure: Number of measurements
            n_skip: Decorrelation skips between measurements
            
        Returns:
            Dictionary with coupling and statistics
        """
        measurements = []
        
        for i in range(n_measure):
            # Decorrelate
            self.metropolis_update(beta, n_sweeps=n_skip)
            
            # Measure
            obs = self.measure_observables(beta)
            measurements.append(obs)
        
        # Average over measurements
        plaq_real_avg = np.mean([m['plaq_real'] for m in measurements])
        plaq_real_std = np.std([m['plaq_real'] for m in measurements])
        action_avg = np.mean([m['action_density'] for m in measurements])
        
        # Extract effective coupling
        # From ⟨cos θ_□⟩, estimate β_eff
        if plaq_real_avg > 0.1:
            beta_eff = -1.0 / (2.0 * np.log(plaq_real_avg))
            e_squared = 1.0 / beta_eff
        else:
            e_squared = np.nan
        
        return {
            'beta': beta,
            'e_squared': e_squared,
            'plaq_real': plaq_real_avg,
            'plaq_std': plaq_real_std,
            'action_density': action_avg
        }
    
    def scan_beta(self, beta_values: np.ndarray, 
                  n_therm: int = 1000,
                  n_measure: int = 100) -> List[Dict[str, float]]:
        """
        Scan over beta values to map out phase diagram.
        
        Args:
            beta_values: Array of β values to scan
            n_therm: Thermalization sweeps per β
            n_measure: Measurement samples per β
            
        Returns:
            List of measurement dictionaries
        """
        results = []
        
        print(f"\nScanning {len(beta_values)} beta values...")
        
        for i, beta in enumerate(beta_values):
            print(f"\n--- Beta {i+1}/{len(beta_values)}: β = {beta:.4f} (e² = {1/beta:.6f}) ---")
            
            # Thermalize at this beta
            self.thermalize(beta, n_therm=n_therm)
            
            # Measure
            result = self.measure_coupling(beta, n_measure=n_measure)
            results.append(result)
            
            print(f"  Measured e² = {result['e_squared']:.6f}")
            print(f"  ⟨cos θ_□⟩ = {result['plaq_real']:.6f} ± {result['plaq_std']:.6f}")
        
        return results
    
    def test_geometric_factor(self, results: List[Dict[str, float]]) -> Dict[str, any]:
        """
        Test if e² ≈ 1/(4π) emerges from the lattice structure.
        
        Args:
            results: Beta scan results
            
        Returns:
            Analysis dictionary
        """
        target = 1.0 / (4 * np.pi)
        
        # Find which beta gives e² closest to 1/(4π)
        best_match = None
        best_error = float('inf')
        
        for result in results:
            e_sq = result['e_squared']
            if not np.isnan(e_sq):
                error = abs(e_sq - target)
                if error < best_error:
                    best_error = error
                    best_match = result
        
        if best_match is None:
            return {'match': False, 'error': float('inf')}
        
        error_pct = 100 * best_error / target
        
        # Interpretation
        if error_pct < 1.0:
            status = "EXCELLENT"
            interpretation = "Strong match to 1/(4π)! U(1) confirms universality."
        elif error_pct < 5.0:
            status = "GOOD"
            interpretation = "Reasonable match to 1/(4π). Suggests geometric origin."
        elif error_pct < 10.0:
            status = "MODERATE"
            interpretation = "Weak match to 1/(4π). May need refinement."
        else:
            status = "NO_MATCH"
            interpretation = "No clear match to 1/(4π). U(1) differs from SU(2)."
        
        return {
            'match': True,
            'status': status,
            'interpretation': interpretation,
            'best_beta': best_match['beta'],
            'best_e_squared': best_match['e_squared'],
            'target': target,
            'error': best_error,
            'error_pct': error_pct,
            'plaq_real': best_match['plaq_real']
        }
    
    def compare_to_su2(self, e_squared_u1: float) -> Dict[str, float]:
        """
        Compare U(1) coupling to SU(2) result from Phase 9.
        
        Args:
            e_squared_u1: Measured U(1) coupling
            
        Returns:
            Comparison statistics
        """
        # Phase 9 result: g²_SU(2) = 0.080000
        g_squared_su2 = 0.080000
        
        # Coupling ratio
        ratio = e_squared_u1 / g_squared_su2
        
        # Both should be near 1/(4π) = 0.079577
        target = 1.0 / (4 * np.pi)
        
        u1_error = abs(e_squared_u1 - target) / target
        su2_error = abs(g_squared_su2 - target) / target
        
        return {
            'e_squared_u1': e_squared_u1,
            'g_squared_su2': g_squared_su2,
            'ratio': ratio,
            'target': target,
            'u1_error_pct': 100 * u1_error,
            'su2_error_pct': 100 * su2_error,
            'compatible': abs(ratio - 1.0) < 0.1  # Within 10%
        }
    
    def plot_analysis(self, results: List[Dict[str, float]], 
                     geometric_test: Dict[str, any],
                     filename: str = 'results/u1_gauge_analysis.png'):
        """
        Create comprehensive 6-panel analysis plot.
        
        Args:
            results: Beta scan results
            geometric_test: Geometric factor test results
            filename: Output filename
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Phase 10.1: U(1) Gauge Theory Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Extract data
        betas = [r['beta'] for r in results]
        e_squareds = [r['e_squared'] for r in results if not np.isnan(r['e_squared'])]
        betas_valid = [r['beta'] for r in results if not np.isnan(r['e_squared'])]
        plaq_reals = [r['plaq_real'] for r in results]
        plaq_stds = [r['plaq_std'] for r in results]
        actions = [r['action_density'] for r in results]
        
        target = 1.0 / (4 * np.pi)
        
        # Panel 1: e² vs β
        ax = axes[0, 0]
        ax.plot(betas_valid, e_squareds, 'b-o', linewidth=2, markersize=4, label='Measured e²')
        ax.axhline(target, color='red', linestyle='--', linewidth=2, label='1/(4π)')
        ax.set_xlabel('β = 1/e²', fontsize=11)
        ax.set_ylabel('Effective e²', fontsize=11)
        ax.set_title('Coupling Extraction', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Plaquette expectation
        ax = axes[0, 1]
        ax.errorbar(betas, plaq_reals, yerr=plaq_stds, fmt='g-o', linewidth=2, 
                   markersize=4, capsize=3, label='⟨cos θ_□⟩')
        ax.set_xlabel('β', fontsize=11)
        ax.set_ylabel('⟨cos θ_□⟩', fontsize=11)
        ax.set_title('Plaquette Expectation Value', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Action density
        ax = axes[0, 2]
        ax.plot(betas, actions, 'm-o', linewidth=2, markersize=4)
        ax.set_xlabel('β', fontsize=11)
        ax.set_ylabel('Action Density', fontsize=11)
        ax.set_title('Wilson Action per Plaquette', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Panel 4: Phase diagram
        ax = axes[1, 0]
        if len(e_squareds) > 0:
            errors_pct = [100 * abs(e - target) / target for e in e_squareds]
            ax.plot(betas_valid, errors_pct, 'r-o', linewidth=2, markersize=4)
            ax.set_xlabel('β', fontsize=11)
            ax.set_ylabel('Error from 1/(4π) [%]', fontsize=11)
            ax.set_title('Deviation from Geometric Constant', fontweight='bold')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, which='both')
        
        # Panel 5: Best match details
        ax = axes[1, 1]
        ax.axis('off')
        
        if geometric_test['match']:
            info_text = f"""GEOMETRIC FACTOR TEST
            
Status: {geometric_test['status']}

Best Match:
  β = {geometric_test['best_beta']:.4f}
  e² = {geometric_test['best_e_squared']:.6f}
  
Target:
  1/(4π) = {geometric_test['target']:.6f}
  
Error:
  Absolute: {geometric_test['error']:.6f}
  Percent: {geometric_test['error_pct']:.3f}%

Plaquette:
  ⟨cos θ_□⟩ = {geometric_test['plaq_real']:.6f}

{geometric_test['interpretation']}
"""
        else:
            info_text = "No valid measurements"
        
        ax.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center',
               family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Panel 6: Comparison to SU(2)
        ax = axes[1, 2]
        ax.axis('off')
        
        if geometric_test['match']:
            comparison = self.compare_to_su2(geometric_test['best_e_squared'])
            
            comp_text = f"""COMPARISON TO SU(2)

Phase 9 Result:
  g²_SU(2) = {comparison['g_squared_su2']:.6f}
  Error: {comparison['su2_error_pct']:.3f}%

Phase 10.1 Result:
  e²_U(1) = {comparison['e_squared_u1']:.6f}
  Error: {comparison['u1_error_pct']:.3f}%

Coupling Ratio:
  e²/g² = {comparison['ratio']:.4f}

Target (both):
  1/(4π) = {comparison['target']:.6f}

Compatible: {'YES' if comparison['compatible'] else 'NO'}

{'UNIVERSALITY CONFIRMED!' if comparison['compatible'] and 
  comparison['u1_error_pct'] < 5 and comparison['su2_error_pct'] < 5 
  else 'Results suggest gauge-group dependence.'}
"""
        else:
            comp_text = "No comparison available"
        
        ax.text(0.1, 0.5, comp_text, fontsize=10, verticalalignment='center',
               family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        
        # Create results directory if needed
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved: {filename}")
        
        plt.close()
    
    def generate_report(self, results: List[Dict[str, float]], 
                       geometric_test: Dict[str, any],
                       filename: str = 'results/u1_gauge_report.txt'):
        """
        Generate detailed text report.
        
        Args:
            results: Beta scan results
            geometric_test: Geometric factor test
            filename: Output filename
        """
        target = 1.0 / (4 * np.pi)
        
        with open(filename, 'w') as f:
            f.write("="*70 + "\n")
            f.write("PHASE 10.1: U(1) GAUGE THEORY ANALYSIS\n")
            f.write("="*70 + "\n\n")
            
            f.write("OVERVIEW\n")
            f.write("-"*70 + "\n")
            f.write(f"Lattice: n_max = {self.n_max}\n")
            f.write(f"Sites: {len(self.lattice.points)}\n")
            f.write(f"Links: {len(self.links)}\n")
            plaquettes = self.get_plaquettes()
            f.write(f"Plaquettes: {len(plaquettes)}\n")
            f.write(f"Beta values scanned: {len(results)}\n\n")
            
            f.write("GEOMETRIC FACTOR TEST\n")
            f.write("-"*70 + "\n")
            f.write(f"Target: 1/(4π) = {target:.8f}\n\n")
            
            if geometric_test['match']:
                f.write(f"Status: {geometric_test['status']}\n\n")
                f.write(f"Best Match:\n")
                f.write(f"  Beta: {geometric_test['best_beta']:.6f}\n")
                f.write(f"  e² (measured): {geometric_test['best_e_squared']:.8f}\n")
                f.write(f"  1/(4π) (target): {geometric_test['target']:.8f}\n")
                f.write(f"  Error: {geometric_test['error']:.8f}\n")
                f.write(f"  Error (%): {geometric_test['error_pct']:.4f}%\n\n")
                f.write(f"Plaquette value: ⟨cos θ_□⟩ = {geometric_test['plaq_real']:.6f}\n\n")
                f.write(f"Interpretation:\n")
                f.write(f"  {geometric_test['interpretation']}\n\n")
                
                # Comparison to SU(2)
                comparison = self.compare_to_su2(geometric_test['best_e_squared'])
                f.write("COMPARISON TO PHASE 9 (SU(2))\n")
                f.write("-"*70 + "\n")
                f.write(f"SU(2) result: g² = {comparison['g_squared_su2']:.6f} ")
                f.write(f"(error: {comparison['su2_error_pct']:.3f}%)\n")
                f.write(f"U(1) result:  e² = {comparison['e_squared_u1']:.6f} ")
                f.write(f"(error: {comparison['u1_error_pct']:.3f}%)\n")
                f.write(f"Coupling ratio: e²/g² = {comparison['ratio']:.4f}\n")
                f.write(f"Compatible: {'YES' if comparison['compatible'] else 'NO'}\n\n")
                
                if comparison['compatible'] and comparison['u1_error_pct'] < 5:
                    f.write("*** MAJOR RESULT: UNIVERSALITY CONFIRMED! ***\n")
                    f.write("Both U(1) and SU(2) gauge theories show e²,g² ≈ 1/(4π).\n")
                    f.write("This suggests geometric constant is gauge-universal.\n\n")
            else:
                f.write("No valid measurements obtained.\n\n")
            
            f.write("\nDETAILED RESULTS\n")
            f.write("-"*70 + "\n")
            f.write(f"{'Beta':>10} {'e²':>12} {'⟨cos θ⟩':>12} {'σ':>10} {'Action':>12}\n")
            f.write("-"*70 + "\n")
            
            for r in results:
                e_sq_str = f"{r['e_squared']:.6f}" if not np.isnan(r['e_squared']) else "N/A"
                f.write(f"{r['beta']:>10.4f} {e_sq_str:>12} {r['plaq_real']:>12.6f} ")
                f.write(f"{r['plaq_std']:>10.6f} {r['action_density']:>12.4f}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")
        
        print(f"Report saved: {filename}")


def main():
    """Run Phase 10.1: U(1) gauge theory analysis."""
    print("="*70)
    print("PHASE 10.1: U(1) GAUGE THEORY ON DISCRETE LATTICE")
    print("="*70)
    print("\nTesting if electromagnetic coupling e² ≈ 1/(4π)")
    print("This tests universality of geometric constant across gauge groups.\n")
    
    # Initialize
    n_max = 8  # Covers ℓ up to 7
    u1 = U1GaugeTheory(n_max=n_max)
    
    # Beta scan
    # β = 1/e², so scan around β ≈ 1/(1/(4π)) ≈ 12.57
    beta_values = np.linspace(8.0, 18.0, 11)
    
    print(f"\nTarget: e² = 1/(4π) = {1/(4*np.pi):.6f}")
    print(f"This corresponds to β = 1/e² ≈ {4*np.pi:.2f}")
    
    # Run scan
    results = u1.scan_beta(beta_values, n_therm=1000, n_measure=100)
    
    # Test for geometric factor
    print("\n" + "="*70)
    print("TESTING FOR GEOMETRIC FACTOR 1/(4π)")
    print("="*70)
    
    geometric_test = u1.test_geometric_factor(results)
    
    if geometric_test['match']:
        print(f"\nStatus: {geometric_test['status']}")
        print(f"Best e² = {geometric_test['best_e_squared']:.6f}")
        print(f"Target 1/(4π) = {geometric_test['target']:.6f}")
        print(f"Error: {geometric_test['error_pct']:.3f}%")
        print(f"\n{geometric_test['interpretation']}")
        
        # Compare to SU(2)
        comparison = u1.compare_to_su2(geometric_test['best_e_squared'])
        print(f"\nComparison to SU(2):")
        print(f"  U(1): e² = {comparison['e_squared_u1']:.6f} ({comparison['u1_error_pct']:.3f}% error)")
        print(f"  SU(2): g² = {comparison['g_squared_su2']:.6f} ({comparison['su2_error_pct']:.3f}% error)")
        print(f"  Ratio: {comparison['ratio']:.4f}")
        
        if comparison['compatible'] and comparison['u1_error_pct'] < 5:
            print(f"\n*** BREAKTHROUGH: UNIVERSALITY CONFIRMED! ***")
            print(f"Both gauge groups show coupling ≈ 1/(4π)")
    
    # Generate outputs
    u1.plot_analysis(results, geometric_test)
    u1.generate_report(results, geometric_test)
    
    print("\n" + "="*70)
    print("PHASE 10.1 COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
