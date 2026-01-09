"""
Phase 29: SU(2) × U(1) Mixed Link Variables

Goal: Explore whether geometry induces a preferred mixing ratio (toy electroweak angle).

Tasks:
1. Define combined link variables: U_mix = U_SU(2) · exp(iθ w)
2. Introduce tunable mixing parameter w ∈ [0, 1]
3. Compute mixed plaquette observables as function of w
4. Identify whether any plateau or extremum appears

Scientific Value: Lightest test of whether geometry prefers a small U(1) coupling
(analogous to electroweak mixing angle θ_W ≈ 28.7° → sin²θ_W ≈ 0.223)

Author: Research Team  
Date: January 6, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.phase28_u1_wilson_loops import Phase28_U1WilsonLoops


class Phase29_MixedGaugeLinks:
    """
    SU(2) × U(1) mixed gauge theory on discrete lattice.
    
    Tests whether geometry prefers a specific mixing ratio between
    SU(2) and U(1) gauge interactions (toy electroweak model).
    """
    
    def __init__(self, n_max: int = 6):
        """
        Initialize mixed gauge theory.
        
        Args:
            n_max: Maximum principal quantum number
        """
        self.n_max = n_max
        
        # Use Phase 28's U(1) setup
        self.phase28 = Phase28_U1WilsonLoops(n_max=n_max)
        self.u1 = self.phase28.u1
        
        print("="*70)
        print("PHASE 29: SU(2) × U(1) Mixed Link Variables")
        print("="*70)
        print(f"Lattice: n_max = {n_max}")
        print(f"Sites: {len(self.u1.lattice.points)}")
        print(f"Links: {len(self.u1.links)}")
        
        # Initialize SU(2) link variables (identity initially)
        self.su2_links = {}
        for link in self.u1.links:
            # Start with identity SU(2) matrix
            self.su2_links[link] = np.eye(2, dtype=complex)
        
        print("✓ SU(2) links initialized to identity")
    
    def initialize_su2_random(self, amplitude: float = 0.1):
        """
        Initialize SU(2) links with small random SU(2) matrices.
        
        SU(2) matrices parametrized as:
            U = exp(i σ·θ/2)
        where θ = (θ₁, θ₂, θ₃) are small angles.
        
        Args:
            amplitude: Maximum angle for each component
        """
        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        for link in self.u1.links:
            # Random angles
            theta = np.random.uniform(-amplitude, amplitude, 3)
            
            # Generator: σ·θ/2
            generator = 0.5 * (theta[0] * sigma_x + 
                             theta[1] * sigma_y + 
                             theta[2] * sigma_z)
            
            # Exponentiate: U = exp(i generator)
            # For small angles, use series approximation
            U = np.eye(2, dtype=complex) + 1j * generator - 0.5 * generator @ generator
            
            # Unitarize (project back to SU(2))
            U = self._unitarize_su2(U)
            
            self.su2_links[link] = U
        
        print(f"✓ SU(2) links randomized with amplitude {amplitude}")
    
    def _unitarize_su2(self, U: np.ndarray) -> np.ndarray:
        """
        Project matrix onto SU(2) (closest unitary matrix with det=1).
        
        Args:
            U: Approximate SU(2) matrix
            
        Returns:
            Exact SU(2) matrix
        """
        # SVD: U = V Σ W†
        V, S, Wh = np.linalg.svd(U)
        
        # Closest unitary: V W†
        U_unitary = V @ Wh
        
        # Force det = 1
        det = np.linalg.det(U_unitary)
        U_su2 = U_unitary / np.sqrt(det)
        
        return U_su2
    
    def compute_mixed_observable(self, w: float, 
                                 initialization: str = 'random',
                                 u1_amplitude: float = 0.3) -> Dict:
        """
        Compute observables for mixed SU(2) × U(1) gauge theory.
        
        Mixed link: U_mix = U_SU(2) · exp(iθ w)
        where w ∈ [0, 1] is the mixing parameter.
        
        Args:
            w: Mixing weight (0 = pure SU(2), 1 = pure U(1))
            initialization: How to initialize U(1) angles
            u1_amplitude: For random U(1) initialization
            
        Returns:
            Dictionary of observables
        """
        # Initialize U(1) angles
        if initialization == 'random':
            self.phase28.initialize_random_start(amplitude=u1_amplitude)
        else:
            self.phase28.initialize_cold_start()
        
        # Get plaquettes
        plaquettes = self.u1.get_plaquettes()
        
        if len(plaquettes) == 0:
            return {}
        
        # Compute mixed plaquettes
        mixed_traces = []
        su2_only_traces = []
        u1_only_phases = []
        
        for plaq in plaquettes:
            # SU(2) plaquette
            su2_plaq = self._compute_su2_plaquette(plaq)
            su2_trace = np.trace(su2_plaq)
            su2_only_traces.append(su2_trace)
            
            # U(1) plaquette
            u1_phase = self.u1.compute_plaquette_angle(plaq)
            u1_only_phases.append(u1_phase)
            
            # Mixed plaquette: U_SU(2) · exp(iθ w)
            # For trace, we compute Tr[U_SU(2) · exp(iθw) I]
            # = Tr[U_SU(2)] · exp(iθw) (scalar U(1) part)
            u1_factor = np.exp(1j * u1_phase * w)
            mixed_trace = su2_trace * u1_factor
            
            mixed_traces.append(mixed_trace)
        
        mixed_traces = np.array(mixed_traces)
        su2_only_traces = np.array(su2_only_traces)
        u1_only_phases = np.array(u1_only_phases)
        
        # Compute observables
        obs = {
            'w': w,
            'n_plaquettes': len(plaquettes),
            
            # Mixed observables
            'mixed_real': np.mean(np.real(mixed_traces)),
            'mixed_imag': np.mean(np.imag(mixed_traces)),
            'mixed_abs': np.mean(np.abs(mixed_traces)),
            'mixed_var': np.var(np.abs(mixed_traces)),
            
            # SU(2) only
            'su2_real': np.mean(np.real(su2_only_traces)),
            'su2_abs': np.mean(np.abs(su2_only_traces)),
            
            # U(1) only
            'u1_mean': np.mean(u1_only_phases),
            'u1_var': np.var(u1_only_phases),
            
            # Arrays for plotting
            'mixed_trace_array': mixed_traces,
            'su2_trace_array': su2_only_traces,
            'u1_phase_array': u1_only_phases,
        }
        
        return obs
    
    def _compute_su2_plaquette(self, plaquette: List[int]) -> np.ndarray:
        """
        Compute SU(2) plaquette product.
        
        □ = U₁₂ · U₂₃ · U₃₄ · U₄₁
        
        Args:
            plaquette: List of 4 site indices
            
        Returns:
            2×2 SU(2) matrix
        """
        i1, i2, i3, i4 = plaquette
        
        # Get SU(2) matrices with correct orientation
        U12 = self._get_oriented_su2(i1, i2)
        U23 = self._get_oriented_su2(i2, i3)
        U34 = self._get_oriented_su2(i3, i4)
        U41 = self._get_oriented_su2(i4, i1)
        
        # Product
        plaq = U12 @ U23 @ U34 @ U41
        
        return plaq
    
    def _get_oriented_su2(self, i1: int, i2: int) -> np.ndarray:
        """Get SU(2) matrix with correct orientation."""
        link = self.u1._make_link(i1, i2)
        U = self.su2_links[link]
        
        # If link is reversed, take conjugate transpose
        if link[0] != i1:
            U = U.conj().T
            
        return U
    
    def scan_mixing_parameter(self, w_values: np.ndarray = None,
                             initialization: str = 'random',
                             u1_amplitude: float = 0.3,
                             su2_amplitude: float = 0.1) -> List[Dict]:
        """
        Scan mixing parameter w from 0 to 1.
        
        Args:
            w_values: Array of w values to scan
            initialization: U(1) initialization method
            u1_amplitude: U(1) random amplitude
            su2_amplitude: SU(2) random amplitude
            
        Returns:
            List of observable dictionaries for each w
        """
        if w_values is None:
            w_values = np.linspace(0, 1, 21)
        
        print(f"\n✓ Scanning {len(w_values)} mixing parameter values")
        print(f"  U(1) init: {initialization} (amplitude={u1_amplitude})")
        print(f"  SU(2) init: random (amplitude={su2_amplitude})")
        
        # Initialize SU(2) links once
        self.initialize_su2_random(amplitude=su2_amplitude)
        
        results = []
        for i, w in enumerate(w_values):
            obs = self.compute_mixed_observable(w, initialization, u1_amplitude)
            results.append(obs)
            
            if (i+1) % 5 == 0:
                print(f"  Progress: {i+1}/{len(w_values)}")
        
        print(f"✓ Scan complete")
        return results
    
    def plot_mixing_scan(self, results: List[Dict], save_path: str = None):
        """
        Plot observables vs mixing parameter.
        
        Args:
            results: List of observable dictionaries
            save_path: Path to save figure
        """
        w_values = [r['w'] for r in results]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Phase 29: SU(2) × U(1) Mixing Scan', 
                    fontsize=14, fontweight='bold')
        
        # 1. Mixed plaquette real part
        ax = axes[0, 0]
        mixed_real = [r['mixed_real'] for r in results]
        ax.plot(w_values, mixed_real, 'o-', linewidth=2, markersize=6, color='blue')
        ax.set_xlabel('Mixing Parameter w')
        ax.set_ylabel('⟨Re Tr(U_mix)⟩')
        ax.set_title('Mixed Plaquette Expectation (Real Part)')
        ax.grid(alpha=0.3)
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
        
        # Find extrema
        idx_min = np.argmin(mixed_real)
        idx_max = np.argmax(mixed_real)
        ax.plot(w_values[idx_min], mixed_real[idx_min], 'rv', markersize=10, 
               label=f'Min at w={w_values[idx_min]:.2f}')
        ax.plot(w_values[idx_max], mixed_real[idx_max], 'g^', markersize=10,
               label=f'Max at w={w_values[idx_max]:.2f}')
        ax.legend()
        
        # 2. Mixed plaquette magnitude
        ax = axes[0, 1]
        mixed_abs = [r['mixed_abs'] for r in results]
        ax.plot(w_values, mixed_abs, 'o-', linewidth=2, markersize=6, color='green')
        ax.set_xlabel('Mixing Parameter w')
        ax.set_ylabel('⟨|Tr(U_mix)|⟩')
        ax.set_title('Mixed Plaquette Magnitude')
        ax.grid(alpha=0.3)
        
        # 3. Variance (stability measure)
        ax = axes[1, 0]
        mixed_var = [r['mixed_var'] for r in results]
        ax.plot(w_values, mixed_var, 'o-', linewidth=2, markersize=6, color='purple')
        ax.set_xlabel('Mixing Parameter w')
        ax.set_ylabel('Var(|Tr(U_mix)|)')
        ax.set_title('Variance (Stability)')
        ax.grid(alpha=0.3)
        
        # Find minimum variance (most stable configuration)
        idx_min_var = np.argmin(mixed_var)
        ax.plot(w_values[idx_min_var], mixed_var[idx_min_var], 'r*', 
               markersize=15, label=f'Min at w={w_values[idx_min_var]:.2f}')
        ax.legend()
        
        # 4. Electroweak angle comparison
        ax = axes[1, 1]
        ax.axis('off')
        
        # Find extrema and special points
        w_min_var = w_values[idx_min_var]
        w_max_real = w_values[idx_max]
        
        # Physical electroweak angle: sin²θ_W ≈ 0.223
        sin2_theta_w = 0.223
        theta_w_deg = np.arcsin(np.sqrt(sin2_theta_w)) * 180 / np.pi
        
        report_text = f"""
MIXING SCAN RESULTS
{'='*45}

Extrema Found:
  Max Re(Tr):    w = {w_max_real:.3f}
  Min Variance:  w = {w_min_var:.3f}
  
Physical Reference:
  Electroweak angle: θ_W ≈ {theta_w_deg:.1f}°
  sin²(θ_W) ≈ {sin2_theta_w:.3f}
  
Interpretation:
  w = 0: Pure SU(2) gauge theory
  w = 1: Pure U(1) gauge theory
  
"""
        
        if 0.15 < w_min_var < 0.30:
            report_text += f"  ✓ Minimum variance at w ≈ {w_min_var:.2f}\n"
            report_text += f"    Similar to sin²θ_W ≈ {sin2_theta_w:.2f}!\n"
            report_text += f"    Suggests preferred mixing ratio\n"
        else:
            report_text += f"  • Minimum variance at w = {w_min_var:.2f}\n"
            report_text += f"    Different from electroweak angle\n"
        
        ax.text(0.05, 0.95, report_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Saved figure: {save_path}")
            plt.close()
        else:
            plt.show()
    
    def generate_report(self, results: List[Dict]):
        """
        Generate comprehensive text report.
        
        Args:
            results: List of observable dictionaries
        """
        w_values = np.array([r['w'] for r in results])
        mixed_real = np.array([r['mixed_real'] for r in results])
        mixed_abs = np.array([r['mixed_abs'] for r in results])
        mixed_var = np.array([r['mixed_var'] for r in results])
        
        print("\n" + "="*70)
        print("PHASE 29: SU(2) × U(1) MIXING - FINAL REPORT")
        print("="*70)
        
        print("\n" + "-"*70)
        print("SCAN PARAMETERS")
        print("-"*70)
        print(f"Number of w values:  {len(w_values)}")
        print(f"Range:               [{w_values[0]:.2f}, {w_values[-1]:.2f}]")
        print(f"Plaquettes per scan: {results[0]['n_plaquettes']}")
        
        print("\n" + "-"*70)
        print("EXTREMA ANALYSIS")
        print("-"*70)
        
        # Find extrema
        idx_max_real = np.argmax(mixed_real)
        idx_min_real = np.argmin(mixed_real)
        idx_max_abs = np.argmax(mixed_abs)
        idx_min_var = np.argmin(mixed_var)
        
        print(f"Max Re(Tr):          w = {w_values[idx_max_real]:.3f}  "
              f"(value = {mixed_real[idx_max_real]:.4f})")
        print(f"Min Re(Tr):          w = {w_values[idx_min_real]:.3f}  "
              f"(value = {mixed_real[idx_min_real]:.4f})")
        print(f"Max |Tr|:            w = {w_values[idx_max_abs]:.3f}  "
              f"(value = {mixed_abs[idx_max_abs]:.4f})")
        print(f"Min Variance:        w = {w_values[idx_min_var]:.3f}  "
              f"(value = {mixed_var[idx_min_var]:.6f})")
        
        print("\n" + "-"*70)
        print("PHYSICAL INTERPRETATION")
        print("-"*70)
        
        # Electroweak comparison
        sin2_theta_w = 0.223
        theta_w_rad = np.arcsin(np.sqrt(sin2_theta_w))
        theta_w_deg = theta_w_rad * 180 / np.pi
        
        print(f"Electroweak mixing angle:")
        print(f"  θ_W ≈ {theta_w_deg:.1f}°")
        print(f"  sin²(θ_W) ≈ {sin2_theta_w:.3f}")
        
        w_min_var = w_values[idx_min_var]
        
        if 0.15 < w_min_var < 0.30:
            print(f"\n✓ PREFERRED MIXING DETECTED!")
            print(f"  Minimum variance at w ≈ {w_min_var:.3f}")
            print(f"  Comparable to sin²θ_W ≈ {sin2_theta_w:.3f}")
            print(f"  Ratio: {w_min_var/sin2_theta_w:.2f}")
            print(f"\n  Interpretation: Geometry may prefer small U(1) mixing")
            print(f"  similar to electroweak symmetry breaking!")
        elif w_min_var < 0.1:
            print(f"\n• SU(2) Dominated")
            print(f"  Minimum variance at w ≈ {w_min_var:.3f}")
            print(f"  Geometry prefers pure SU(2) structure")
        elif w_min_var > 0.9:
            print(f"\n• U(1) Dominated")
            print(f"  Minimum variance at w ≈ {w_min_var:.3f}")
            print(f"  Geometry prefers pure U(1) structure")
        else:
            print(f"\n• Intermediate Mixing")
            print(f"  Minimum variance at w ≈ {w_min_var:.3f}")
            print(f"  No clear preference for electroweak-like angle")
        
        print("\n" + "="*70)
        print("PHASE 29 COMPLETE ✅")
        print("="*70)
    
    def run_full_analysis(self, n_w: int = 21, 
                         u1_amplitude: float = 0.3,
                         su2_amplitude: float = 0.1,
                         save_dir: str = 'results'):
        """
        Run complete Phase 29 analysis.
        
        Args:
            n_w: Number of w values to scan
            u1_amplitude: U(1) initialization amplitude
            su2_amplitude: SU(2) initialization amplitude
            save_dir: Directory to save outputs
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Scan mixing parameter
        w_values = np.linspace(0, 1, n_w)
        results = self.scan_mixing_parameter(
            w_values=w_values,
            initialization='random',
            u1_amplitude=u1_amplitude,
            su2_amplitude=su2_amplitude
        )
        
        # Generate plot
        print("\nGenerating visualization...")
        plot_path = os.path.join(save_dir, 'phase29_mixing_scan.png')
        self.plot_mixing_scan(results, save_path=plot_path)
        
        # Generate report
        self.generate_report(results)
        
        return results


def main():
    """Run Phase 29 analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 29: SU(2) × U(1) Mixing')
    parser.add_argument('--n_max', type=int, default=6,
                       help='Maximum principal quantum number')
    parser.add_argument('--n_w', type=int, default=21,
                       help='Number of mixing parameters to scan')
    parser.add_argument('--u1_amp', type=float, default=0.3,
                       help='U(1) initialization amplitude')
    parser.add_argument('--su2_amp', type=float, default=0.1,
                       help='SU(2) initialization amplitude')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save outputs')
    
    args = parser.parse_args()
    
    # Run analysis
    phase29 = Phase29_MixedGaugeLinks(n_max=args.n_max)
    phase29.run_full_analysis(
        n_w=args.n_w,
        u1_amplitude=args.u1_amp,
        su2_amplitude=args.su2_amp,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
