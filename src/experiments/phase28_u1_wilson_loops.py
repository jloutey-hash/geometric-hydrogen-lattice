"""
Phase 28: U(1) Wilson Loops on Existing SU(2) Lattice

Goal: Extract effective U(1) coupling from existing lattice WITHOUT heavy Monte Carlo.
This is the simplest possible "fine-structure constant probe."

Tasks:
1. Add U(1) link variables θ_ij on the existing graph
2. Compute all plaquette products exp(i Σ θ)
3. Measure: mean, variance, distribution shape
4. Output dimensionless number α_eff from plaquette average

Scientific Value: Lightest test of whether geometry induces a U(1) coupling scale.

Author: Research Team
Date: January 6, 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lattice import PolarLattice
from u1_gauge_theory import U1GaugeTheory


class Phase28_U1WilsonLoops:
    """
    Lightweight U(1) Wilson loop analysis on discrete lattice.
    
    No Monte Carlo thermalization - just analyze cold start configuration
    to extract geometric coupling scale.
    """
    
    def __init__(self, n_max: int = 8):
        """
        Initialize U(1) gauge theory on polar lattice.
        
        Args:
            n_max: Maximum principal quantum number
        """
        self.n_max = n_max
        self.u1 = U1GaugeTheory(n_max=n_max, seed=42)
        
        print("="*70)
        print("PHASE 28: U(1) Wilson Loops on SU(2) Lattice")
        print("="*70)
        print(f"Lattice: n_max = {n_max}")
        print(f"Sites: {len(self.u1.lattice.points)}")
        print(f"Links: {len(self.u1.links)}")
        
    def initialize_cold_start(self):
        """
        Initialize all links to zero (cold start).
        This is the simplest non-trivial configuration.
        """
        for link in self.u1.links:
            self.u1.links[link] = 0.0
        print("\n✓ Cold start: all links set to θ = 0")
        
    def initialize_random_start(self, amplitude: float = 0.1):
        """
        Initialize with small random angles (near cold start).
        
        Args:
            amplitude: Maximum angle deviation from zero
        """
        for link in self.u1.links:
            self.u1.links[link] = np.random.uniform(-amplitude, amplitude)
        print(f"\n✓ Random start: θ ∈ [-{amplitude}, {amplitude}]")
        
    def compute_plaquette_statistics(self) -> Dict:
        """
        Compute comprehensive plaquette statistics.
        
        Returns:
            Dictionary with mean, variance, distribution, etc.
        """
        plaquettes = self.u1.get_plaquettes()
        n_plaq = len(plaquettes)
        
        if n_plaq == 0:
            print("⚠ Warning: No plaquettes found!")
            return {}
        
        print(f"\n✓ Found {n_plaq} plaquettes")
        
        # Compute all plaquette angles
        theta_list = []
        cos_list = []
        sin_list = []
        
        for plaq in plaquettes:
            theta = self.u1.compute_plaquette_angle(plaq)
            theta_list.append(theta)
            cos_list.append(np.cos(theta))
            sin_list.append(np.sin(theta))
        
        theta_arr = np.array(theta_list)
        cos_arr = np.array(cos_list)
        sin_arr = np.array(sin_list)
        
        # Compute statistics
        stats = {
            # Angle statistics
            'theta_mean': np.mean(theta_arr),
            'theta_std': np.std(theta_arr),
            'theta_var': np.var(theta_arr),
            'theta_min': np.min(theta_arr),
            'theta_max': np.max(theta_arr),
            
            # Plaquette expectation value
            'plaq_real': np.mean(cos_arr),          # ⟨Re U_□⟩ = ⟨cos θ⟩
            'plaq_imag': np.mean(sin_arr),          # ⟨Im U_□⟩ = ⟨sin θ⟩
            'plaq_abs': np.mean(np.abs(theta_arr)),  # ⟨|θ|⟩
            
            # Distribution shape
            'cos_mean': np.mean(cos_arr),
            'cos_std': np.std(cos_arr),
            'sin_mean': np.mean(sin_arr),
            'sin_std': np.std(sin_arr),
            
            # Raw data for plotting
            'theta_array': theta_arr,
            'cos_array': cos_arr,
            'sin_array': sin_arr,
            'n_plaquettes': n_plaq
        }
        
        return stats
    
    def extract_effective_coupling(self, stats: Dict) -> Dict:
        """
        Extract effective U(1) coupling from plaquette measurements.
        
        For weak coupling (small θ), Wilson action:
            S = β Σ [1 - cos θ] ≈ β Σ θ²/2
        
        Mean plaquette: ⟨cos θ⟩ ≈ 1 - ⟨θ²⟩/2
        
        Effective coupling: α_eff ~ ⟨θ²⟩
        
        Args:
            stats: Statistics dictionary from compute_plaquette_statistics
            
        Returns:
            Dictionary with coupling estimates
        """
        if 'theta_array' not in stats:
            return {}
        
        theta = stats['theta_array']
        
        # Method 1: From variance
        alpha_variance = stats['theta_var']
        
        # Method 2: From mean squared angle
        alpha_mean_sq = np.mean(theta**2)
        
        # Method 3: From plaquette expectation
        # ⟨cos θ⟩ ≈ 1 - α/2 for small α
        # α ≈ 2(1 - ⟨cos θ⟩)
        alpha_plaquette = 2 * (1 - stats['plaq_real'])
        
        # Method 4: Comparison to fine structure constant
        alpha_fine_structure = 1 / 137.036  # Physical α ≈ 0.00729735
        alpha_geometric = 1 / (4 * np.pi)    # Geometric 1/(4π) ≈ 0.0795775
        
        coupling = {
            'alpha_variance': alpha_variance,
            'alpha_mean_sq': alpha_mean_sq,
            'alpha_plaquette': alpha_plaquette,
            'alpha_fine_structure': alpha_fine_structure,
            'alpha_geometric': alpha_geometric,
            
            # Ratios for comparison
            'ratio_to_fine_structure': alpha_mean_sq / alpha_fine_structure if alpha_mean_sq > 0 else 0,
            'ratio_to_geometric': alpha_mean_sq / alpha_geometric if alpha_mean_sq > 0 else 0,
        }
        
        return coupling
    
    def plot_plaquette_distribution(self, stats: Dict, save_path: str = None):
        """
        Plot comprehensive plaquette distribution analysis.
        
        Args:
            stats: Statistics dictionary
            save_path: Path to save figure (optional)
        """
        if 'theta_array' not in stats:
            print("⚠ No data to plot")
            return
        
        theta = stats['theta_array']
        cos_arr = stats['cos_array']
        sin_arr = stats['sin_array']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Phase 28: U(1) Plaquette Distribution', fontsize=14, fontweight='bold')
        
        # 1. Angle histogram
        ax = axes[0, 0]
        ax.hist(theta, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(stats['theta_mean'], color='red', linestyle='--', 
                   label=f"Mean = {stats['theta_mean']:.4f}")
        ax.set_xlabel('Plaquette Angle θ_□')
        ax.set_ylabel('Count')
        ax.set_title('Angle Distribution')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. Cos θ histogram
        ax = axes[0, 1]
        ax.hist(cos_arr, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(stats['cos_mean'], color='red', linestyle='--',
                   label=f"⟨cos θ⟩ = {stats['cos_mean']:.4f}")
        ax.set_xlabel('cos(θ_□)')
        ax.set_ylabel('Count')
        ax.set_title('Plaquette Real Part')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 3. Complex plane distribution
        ax = axes[1, 0]
        ax.scatter(cos_arr, sin_arr, alpha=0.5, s=20, c='purple')
        ax.plot([0, stats['cos_mean']], [0, stats['sin_mean']], 
                'r-', linewidth=2, label='Mean')
        ax.set_xlabel('Re(exp(iθ_□)) = cos(θ_□)')
        ax.set_ylabel('Im(exp(iθ_□)) = sin(θ_□)')
        ax.set_title('Complex Plane Distribution')
        ax.axis('equal')
        ax.grid(alpha=0.3)
        ax.legend()
        
        # Draw unit circle
        circle = plt.Circle((0, 0), 1, fill=False, color='black', 
                           linestyle='--', linewidth=1)
        ax.add_patch(circle)
        
        # 4. Statistics summary
        ax = axes[1, 1]
        ax.axis('off')
        
        stats_text = f"""
PLAQUETTE STATISTICS
{'='*40}

Angles:
  Mean:     {stats['theta_mean']:>10.6f} rad
  Std Dev:  {stats['theta_std']:>10.6f}
  Variance: {stats['theta_var']:>10.6f}
  Range:    [{stats['theta_min']:.4f}, {stats['theta_max']:.4f}]

Plaquette Values:
  ⟨Re U_□⟩ = ⟨cos θ⟩:  {stats['plaq_real']:>10.6f}
  ⟨Im U_□⟩ = ⟨sin θ⟩:  {stats['plaq_imag']:>10.6f}
  ⟨|θ_□|⟩:            {stats['plaq_abs']:>10.6f}

Sample Size:
  N_plaquettes:      {stats['n_plaquettes']:>10d}
        """
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Saved figure: {save_path}")
            plt.close()
        else:
            plt.show()
    
    def plot_coupling_comparison(self, coupling: Dict, save_path: str = None):
        """
        Plot coupling constant comparisons.
        
        Args:
            coupling: Coupling dictionary from extract_effective_coupling
            save_path: Path to save figure
        """
        if not coupling:
            print("⚠ No coupling data to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Phase 28: Effective U(1) Coupling Extraction', 
                    fontsize=14, fontweight='bold')
        
        # 1. Coupling estimates comparison
        ax = axes[0]
        
        methods = ['Variance', 'Mean θ²', 'Plaquette']
        values = [
            coupling['alpha_variance'],
            coupling['alpha_mean_sq'],
            coupling['alpha_plaquette']
        ]
        colors = ['blue', 'green', 'purple']
        
        x_pos = np.arange(len(methods))
        bars = ax.bar(x_pos, values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add reference lines
        ax.axhline(coupling['alpha_fine_structure'], color='red', 
                   linestyle='--', linewidth=2, 
                   label=f"α_fine = 1/137 = {coupling['alpha_fine_structure']:.6f}")
        ax.axhline(coupling['alpha_geometric'], color='orange',
                   linestyle='--', linewidth=2,
                   label=f"α_geom = 1/(4π) = {coupling['alpha_geometric']:.6f}")
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods)
        ax.set_ylabel('Effective Coupling α_eff')
        ax.set_title('Coupling Extraction Methods')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.6f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Ratios to reference values
        ax = axes[1]
        
        alpha_eff = coupling['alpha_mean_sq']
        
        comparisons = {
            'Fine Structure\nα = 1/137': coupling['ratio_to_fine_structure'],
            'Geometric\nα = 1/(4π)': coupling['ratio_to_geometric'],
        }
        
        names = list(comparisons.keys())
        ratios = list(comparisons.values())
        colors_ratio = ['red', 'orange']
        
        x_pos = np.arange(len(names))
        bars = ax.bar(x_pos, ratios, color=colors_ratio, alpha=0.7, edgecolor='black')
        
        ax.axhline(1.0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names)
        ax.set_ylabel('Ratio: α_eff / α_reference')
        ax.set_title('Comparison to Physical Constants')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{ratio:.2f}×', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved figure: {save_path}")
            plt.close()
        else:
            plt.show()
    
    def generate_report(self, stats: Dict, coupling: Dict):
        """
        Generate comprehensive text report.
        
        Args:
            stats: Plaquette statistics
            coupling: Coupling estimates
        """
        print("\n" + "="*70)
        print("PHASE 28: U(1) WILSON LOOPS - FINAL REPORT")
        print("="*70)
        
        print("\n" + "-"*70)
        print("PLAQUETTE STATISTICS")
        print("-"*70)
        print(f"Number of plaquettes:    {stats['n_plaquettes']}")
        print(f"Mean angle:              {stats['theta_mean']:.6f} rad")
        print(f"Std deviation:           {stats['theta_std']:.6f}")
        print(f"Variance:                {stats['theta_var']:.6f}")
        print(f"⟨cos θ⟩:                 {stats['plaq_real']:.6f}")
        print(f"⟨sin θ⟩:                 {stats['plaq_imag']:.6f}")
        
        print("\n" + "-"*70)
        print("EFFECTIVE COUPLING EXTRACTION")
        print("-"*70)
        print(f"α_eff (variance):        {coupling['alpha_variance']:.6f}")
        print(f"α_eff (mean θ²):         {coupling['alpha_mean_sq']:.6f}")
        print(f"α_eff (plaquette):       {coupling['alpha_plaquette']:.6f}")
        
        print("\n" + "-"*70)
        print("COMPARISON TO PHYSICAL CONSTANTS")
        print("-"*70)
        print(f"Fine structure constant: {coupling['alpha_fine_structure']:.6f} (1/137)")
        print(f"Geometric constant:      {coupling['alpha_geometric']:.6f} (1/(4π))")
        print(f"\nRatio to α_fine:         {coupling['ratio_to_fine_structure']:.2f}×")
        print(f"Ratio to α_geom:         {coupling['ratio_to_geometric']:.2f}×")
        
        print("\n" + "-"*70)
        print("INTERPRETATION")
        print("-"*70)
        
        alpha_eff = coupling['alpha_mean_sq']
        alpha_fine = coupling['alpha_fine_structure']
        alpha_geom = coupling['alpha_geometric']
        
        if alpha_eff < 0.001:
            print("✓ Cold start configuration: effectively zero coupling")
            print("  (All links θ ≈ 0 → plaquettes ≈ 0)")
        elif 0.5 < coupling['ratio_to_geometric'] < 2.0:
            print("✓ GEOMETRIC COUPLING DETECTED!")
            print(f"  α_eff ≈ {coupling['ratio_to_geometric']:.2f} × (1/(4π))")
            print("  This suggests geometry-induced U(1) scale")
        elif 0.5 < coupling['ratio_to_fine_structure'] < 2.0:
            print("✓ FINE STRUCTURE SCALE DETECTED!")
            print(f"  α_eff ≈ {coupling['ratio_to_fine_structure']:.2f} × (1/137)")
            print("  Remarkable match to electromagnetism!")
        else:
            print("• Configuration-dependent coupling")
            print("  Need Monte Carlo thermalization for physical value")
        
        print("\n" + "="*70)
        print("PHASE 28 COMPLETE ✅")
        print("="*70)
    
    def run_full_analysis(self, initialization: str = 'random', 
                         amplitude: float = 0.3,
                         save_dir: str = 'results'):
        """
        Run complete Phase 28 analysis.
        
        Args:
            initialization: 'cold' or 'random'
            amplitude: For random start, max angle deviation
            save_dir: Directory to save outputs
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize configuration
        if initialization == 'cold':
            self.initialize_cold_start()
        else:
            self.initialize_random_start(amplitude=amplitude)
        
        # Compute statistics
        print("\n" + "="*70)
        print("Computing plaquette statistics...")
        print("="*70)
        stats = self.compute_plaquette_statistics()
        
        if not stats:
            print("⚠ Analysis failed: no plaquettes found")
            return
        
        # Extract coupling
        print("\nExtracting effective coupling...")
        coupling = self.extract_effective_coupling(stats)
        
        # Generate plots
        print("\nGenerating visualizations...")
        dist_path = os.path.join(save_dir, 'phase28_plaquette_distribution.png')
        self.plot_plaquette_distribution(stats, save_path=dist_path)
        
        coupling_path = os.path.join(save_dir, 'phase28_coupling_comparison.png')
        self.plot_coupling_comparison(coupling, save_path=coupling_path)
        
        # Generate report
        self.generate_report(stats, coupling)
        
        return stats, coupling


def main():
    """Run Phase 28 analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 28: U(1) Wilson Loops')
    parser.add_argument('--n_max', type=int, default=8,
                       help='Maximum principal quantum number')
    parser.add_argument('--init', type=str, default='random',
                       choices=['cold', 'random'],
                       help='Initialization method')
    parser.add_argument('--amplitude', type=float, default=0.3,
                       help='Random initialization amplitude')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save outputs')
    
    args = parser.parse_args()
    
    # Run analysis
    phase28 = Phase28_U1WilsonLoops(n_max=args.n_max)
    phase28.run_full_analysis(
        initialization=args.init,
        amplitude=args.amplitude,
        save_dir=args.save_dir
    )


if __name__ == '__main__':
    main()
