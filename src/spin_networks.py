"""
Spin Networks and Loop Quantum Gravity Connection

Our discrete polar lattice IS a spin network with specific topology.
Investigates:
1. Area operators on shells
2. Volume operators
3. Connection to LQG
4. Role of 1/(4pi) in quantum geometry

Key insight: SU(2) representation theory appears naturally.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import factorial
from typing import List, Tuple, Dict
import sys

from lattice import PolarLattice
from angular_momentum import AngularMomentumOperators


class SpinNetworkCalculator:
    """
    Analyze discrete lattice as spin network.
    
    In Loop Quantum Gravity:
    - Nodes: represent spatial regions
    - Links: labeled by SU(2) representations (spins j)
    - Area operator: A = 8*pi*gamma*l_P^2 * sqrt(j(j+1))
    - gamma: Immirzi parameter
    
    Our lattice:
    - Nodes: lattice sites on shells
    - Links: between sites, labeled by ell values
    - Natural SU(2) structure from angular momentum
    """
    
    def __init__(self, ell_max: int):
        """
        Initialize spin network calculator.
        
        Args:
            ell_max: Maximum angular momentum quantum number
        """
        self.ell_max = ell_max
        self.lattice = PolarLattice(ell_max + 1)  # n_max = ell_max + 1
        
        # Physical constants (Planck units)
        self.l_planck = 1.0  # Set to 1 in Planck units
        self.one_over_4pi = 1 / (4 * np.pi)
        
        # Storage
        self.area_eigenvalues = {}
        self.volume_eigenvalues = {}
        self.immirzi_candidates = []
        
        print(f"SpinNetworkCalculator initialized")
        print(f"  ell_max = {ell_max}")
        print(f"  Total nodes = {len(self.lattice.points)}")
        print()
    
    def compute_area_spectrum(self) -> Dict[int, np.ndarray]:
        """
        Compute area operator eigenvalues for each shell.
        
        In LQG: A_j = 8*pi*gamma*l_P^2 * sqrt(j(j+1))
        
        On our lattice:
        - Each shell ell has "area" A_ell
        - Natural quantum number is j = ell/2 or j = ell
        - Test both conventions
        
        Returns:
            areas: Dictionary mapping ell to area values
        """
        print("Computing area spectrum...")
        
        for ell in range(self.ell_max + 1):
            # Convention 1: j = ell (full angular momentum)
            j_full = ell
            
            # Convention 2: j = ell/2 (half to match spin-1/2)
            j_half = ell / 2.0
            
            # LQG formula: A = 8*pi*gamma*l_P^2 * sqrt(j(j+1))
            # For now, set gamma = 1, l_P = 1
            gamma = 1.0
            
            A_full = 8 * np.pi * gamma * self.l_planck**2 * np.sqrt(j_full * (j_full + 1))
            A_half = 8 * np.pi * gamma * self.l_planck**2 * np.sqrt(j_half * (j_half + 1))
            
            # Also test with gamma involving 1/(4pi)
            gamma_geom = self.one_over_4pi
            A_geom = 8 * np.pi * gamma_geom * self.l_planck**2 * np.sqrt(j_full * (j_full + 1))
            
            self.area_eigenvalues[ell] = {
                'j_full': j_full,
                'j_half': j_half,
                'A_full': A_full,
                'A_half': A_half,
                'A_geom': A_geom,
                'r_ell': 1 + 2*ell
            }
        
        print(f"  Computed areas for {len(self.area_eigenvalues)} shells")
        print()
        
        return self.area_eigenvalues
    
    def compute_volume_spectrum(self) -> Dict[int, float]:
        """
        Compute volume operator eigenvalues.
        
        In LQG: V ~ l_P^3 * |epsilon^{ijk} sqrt(J_i(1) * J_j(2) * J_k(3))|
        
        Simplified: V_ell ~ r_ell^3 or involve sqrt(ell(ell+1))
        
        Returns:
            volumes: Dictionary mapping ell to volume
        """
        print("Computing volume spectrum...")
        
        for ell in range(self.ell_max + 1):
            r_ell = 1 + 2*ell
            
            # Classical volume (sphere of radius r_ell)
            V_classical = (4/3) * np.pi * r_ell**3
            
            # Quantum volume (with L^2 = ell(ell+1))
            L_squared = ell * (ell + 1)
            V_quantum = (4/3) * np.pi * r_ell**2 * np.sqrt(L_squared)
            
            # With 1/(4pi) factor
            V_geom = self.one_over_4pi * V_classical
            
            self.volume_eigenvalues[ell] = {
                'V_classical': V_classical,
                'V_quantum': V_quantum,
                'V_geom': V_geom,
                'r_ell': r_ell
            }
        
        print(f"  Computed volumes for {len(self.volume_eigenvalues)} shells")
        print()
        
        return self.volume_eigenvalues
    
    def test_immirzi_parameter(self) -> Dict:
        """
        Test if Immirzi parameter gamma involves 1/(4pi).
        
        The Immirzi parameter fixes the scale of area in LQG.
        Standard value: gamma ~ 0.2375 (from black hole entropy match)
        
        Test: does gamma = C * 1/(4pi) for some constant C?
        """
        print("Testing Immirzi parameter connection...")
        print()
        
        # Standard Immirzi from black hole thermodynamics
        # Requirement: S_BH = A/(4*l_P^2) matches Bekenstein-Hawking
        # This gives: gamma = ln(2) / (pi*sqrt(3)) ≈ 0.2375
        
        gamma_standard = np.log(2) / (np.pi * np.sqrt(3))
        
        # Test ratios
        ratio_to_4pi = gamma_standard / self.one_over_4pi
        ratio_to_1over4pi = gamma_standard * (4 * np.pi)
        
        # Alternative: gamma = 1/(4*pi) gives what?
        gamma_alt = self.one_over_4pi
        
        # Compare area spectra
        ell_test = 10
        j_test = ell_test
        
        A_standard = 8 * np.pi * gamma_standard * np.sqrt(j_test * (j_test + 1))
        A_alt = 8 * np.pi * gamma_alt * np.sqrt(j_test * (j_test + 1))
        A_unity = 8 * np.pi * 1.0 * np.sqrt(j_test * (j_test + 1))
        
        results = {
            'gamma_standard': gamma_standard,
            'gamma_alt': gamma_alt,
            'one_over_4pi': self.one_over_4pi,
            'ratio_to_4pi': ratio_to_4pi,
            'ratio_to_1over4pi': ratio_to_1over4pi,
            'A_standard': A_standard,
            'A_alt': A_alt,
            'A_unity': A_unity,
            'match': abs(gamma_standard - gamma_alt) / gamma_standard
        }
        
        print(f"Standard Immirzi: gamma = {gamma_standard:.6f}")
        print(f"Geometric value: 1/(4pi) = {self.one_over_4pi:.6f}")
        print(f"Ratio: gamma / [1/(4pi)] = {ratio_to_4pi:.6f}")
        print(f"Ratio: gamma * 4pi = {ratio_to_1over4pi:.6f}")
        print()
        print(f"If gamma = 1/(4pi):")
        print(f"  Area at j={j_test}: A = {A_alt:.6f}")
        print(f"  vs standard: A = {A_standard:.6f}")
        print(f"  Match: {results['match']*100:.2f}% difference")
        print()
        
        return results
    
    def analyze_graph_structure(self) -> Dict:
        """
        Analyze the graph/network structure of our lattice.
        
        Properties:
        - Nodes per shell
        - Connectivity
        - Link structure
        """
        print("Analyzing spin network topology...")
        
        # Count nodes per shell
        nodes_per_shell = {}
        for ell in range(self.ell_max + 1):
            n_nodes = 2 * (2*ell + 1)  # From lattice construction
            nodes_per_shell[ell] = n_nodes
        
        total_nodes = sum(nodes_per_shell.values())
        
        # Links between shells (radial)
        radial_links = sum(nodes_per_shell[ell] for ell in range(self.ell_max))
        
        # Links within shells (angular)
        angular_links = sum(nodes_per_shell[ell] for ell in range(self.ell_max + 1))
        
        total_links = radial_links + angular_links
        
        # Average coordination number
        avg_coordination = 2 * total_links / total_nodes  # Each link connects 2 nodes
        
        results = {
            'total_nodes': total_nodes,
            'total_links': total_links,
            'radial_links': radial_links,
            'angular_links': angular_links,
            'avg_coordination': avg_coordination,
            'nodes_per_shell': nodes_per_shell
        }
        
        print(f"  Total nodes: {total_nodes}")
        print(f"  Total links: {total_links}")
        print(f"  Average coordination: {avg_coordination:.2f}")
        print()
        
        return results
    
    def test_geometric_factor_in_areas(self) -> Dict:
        """
        Test if 1/(4pi) appears in area/volume ratios.
        """
        print("Testing for 1/(4pi) in quantum geometry...")
        
        # Compute areas and volumes
        areas = self.compute_area_spectrum()
        volumes = self.compute_volume_spectrum()
        
        # Test: does A_ell / V_ell involve 1/(4pi)?
        ratios = []
        
        for ell in range(1, self.ell_max + 1):
            A = areas[ell]['A_full']
            V = volumes[ell]['V_classical']
            
            if V > 0:
                ratio = A / V
                ratios.append(ratio)
        
        mean_ratio = np.mean(ratios) if ratios else 0
        
        # Also test: normalized area
        # A_ell / (4*pi*r_ell^2) should be ~ geometric factor
        normalized_areas = []
        
        for ell in range(1, self.ell_max + 1):
            A = areas[ell]['A_full']
            r = areas[ell]['r_ell']
            
            A_norm = A / (4 * np.pi * r**2)
            normalized_areas.append(A_norm)
        
        mean_norm = np.mean(normalized_areas) if normalized_areas else 0
        
        # Compare with sqrt(ell(ell+1))/(2*pi*r_ell) → 1/(4pi) pattern from Phase 8
        phase8_ratios = []
        
        for ell in range(1, self.ell_max + 1):
            r = 1 + 2*ell
            alpha = np.sqrt(ell * (ell + 1)) / (2 * np.pi * r)
            phase8_ratios.append(alpha)
        
        mean_phase8 = np.mean(phase8_ratios) if phase8_ratios else 0
        
        results = {
            'mean_A_over_V': mean_ratio,
            'mean_normalized_area': mean_norm,
            'mean_phase8_ratio': mean_phase8,
            'one_over_4pi': self.one_over_4pi,
            'match_phase8': abs(mean_phase8 - self.one_over_4pi) / self.one_over_4pi
        }
        
        print(f"Mean A/V ratio: {mean_ratio:.6f}")
        print(f"Mean normalized area: {mean_norm:.6f}")
        print(f"Mean Phase 8 ratio: {mean_phase8:.6f}")
        print(f"Compare to 1/(4pi): {self.one_over_4pi:.6f}")
        print(f"Phase 8 match: {results['match_phase8']*100:.2f}% error")
        print()
        
        return results
    
    def plot_spin_network_analysis(self, filename: str = 'spin_network_analysis.png'):
        """
        Create comprehensive visualization.
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid for subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Panel 1: Area spectrum
        ax = fig.add_subplot(gs[0, 0])
        
        ells = list(self.area_eigenvalues.keys())
        A_full = [self.area_eigenvalues[ell]['A_full'] for ell in ells]
        A_geom = [self.area_eigenvalues[ell]['A_geom'] for ell in ells]
        
        ax.plot(ells, A_full, 'o-', label='Standard (gamma=1)', linewidth=2)
        ax.plot(ells, A_geom, 's-', label='Geometric (gamma=1/4pi)', linewidth=2)
        ax.set_xlabel('Shell (ell)')
        ax.set_ylabel('Area eigenvalue')
        ax.set_title('Area Spectrum')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Volume spectrum
        ax = fig.add_subplot(gs[0, 1])
        
        V_classical = [self.volume_eigenvalues[ell]['V_classical'] for ell in ells]
        V_quantum = [self.volume_eigenvalues[ell]['V_quantum'] for ell in ells]
        
        ax.plot(ells, V_classical, 'o-', label='Classical', linewidth=2)
        ax.plot(ells, V_quantum, 's-', label='Quantum', linewidth=2)
        ax.set_xlabel('Shell (ell)')
        ax.set_ylabel('Volume')
        ax.set_title('Volume Spectrum')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Area/Volume ratio
        ax = fig.add_subplot(gs[0, 2])
        
        ratios = [self.area_eigenvalues[ell]['A_full'] / self.volume_eigenvalues[ell]['V_classical']
                 for ell in ells if ell > 0]
        
        ax.plot(ells[1:], ratios, 'o-', linewidth=2)
        ax.set_xlabel('Shell (ell)')
        ax.set_ylabel('A / V')
        ax.set_title('Area-to-Volume Ratio')
        ax.grid(True, alpha=0.3)
        
        # Panel 4: Network topology (2D visualization)
        ax = fig.add_subplot(gs[1, 0])
        
        # Plot nodes on shells
        for ell in range(min(8, self.ell_max + 1)):
            r = 1 + 2*ell
            n_points = 2 * (2*ell + 1)
            
            theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            ax.plot(x, y, 'o', markersize=3, alpha=0.7, label=f'ell={ell}' if ell < 3 else '')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Spin Network Topology (2D)')
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Panel 5: Phase 8 geometric ratio
        ax = fig.add_subplot(gs[1, 1])
        
        ratios_phase8 = []
        for ell in range(1, self.ell_max + 1):
            r = 1 + 2*ell
            alpha = np.sqrt(ell * (ell + 1)) / (2 * np.pi * r)
            ratios_phase8.append(alpha)
        
        ax.plot(range(1, self.ell_max + 1), ratios_phase8, 'o-', linewidth=2)
        ax.axhline(self.one_over_4pi, color='red', linestyle='--', 
                  label=f'1/(4pi) = {self.one_over_4pi:.4f}', linewidth=2)
        ax.set_xlabel('ell')
        ax.set_ylabel('sqrt(ell(ell+1))/(2*pi*r_ell)')
        ax.set_title('Phase 8: Geometric Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 6: Immirzi parameter test
        ax = fig.add_subplot(gs[1, 2])
        
        gamma_standard = np.log(2) / (np.pi * np.sqrt(3))
        gamma_values = [gamma_standard, self.one_over_4pi, 1.0]
        labels = ['Standard\n(BH entropy)', '1/(4pi)\n(geometric)', 'Unity']
        colors = ['blue', 'red', 'green']
        
        bars = ax.bar(labels, gamma_values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Immirzi parameter gamma')
        ax.set_title('Immirzi Parameter Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, gamma_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Panel 7: Coordination number distribution
        ax = fig.add_subplot(gs[2, 0])
        
        nodes_per_shell = [2 * (2*ell + 1) for ell in range(self.ell_max + 1)]
        
        ax.bar(range(self.ell_max + 1), nodes_per_shell, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Shell (ell)')
        ax.set_ylabel('Number of nodes')
        ax.set_title('Nodes per Shell: N = 2(2ell+1)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Panel 8: L^2 eigenvalue spectrum
        ax = fig.add_subplot(gs[2, 1])
        
        L_squared = [ell * (ell + 1) for ell in range(self.ell_max + 1)]
        
        ax.plot(range(self.ell_max + 1), L_squared, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('ell')
        ax.set_ylabel('L^2 / hbar^2')
        ax.set_title('Angular Momentum Spectrum')
        ax.grid(True, alpha=0.3)
        
        # Panel 9: Summary comparison
        ax = fig.add_subplot(gs[2, 2])
        
        # Test different geometric factors
        test_names = ['Phase 8\nratio', 'Area\nnorm', 'Immirzi\nmatch']
        
        # Phase 8 ratio match
        mean_phase8 = np.mean(ratios_phase8)
        match1 = abs(mean_phase8 - self.one_over_4pi) / self.one_over_4pi
        
        # Area normalization
        norm_areas = [self.area_eigenvalues[ell]['A_full'] / (4*np.pi*(1+2*ell)**2) 
                     for ell in range(1, self.ell_max + 1)]
        match2 = abs(np.mean(norm_areas) - 2*self.one_over_4pi) / (2*self.one_over_4pi) if norm_areas else 1
        
        # Immirzi match
        match3 = abs(gamma_standard - self.one_over_4pi) / gamma_standard
        
        matches = [match1, match2, match3]
        
        bars = ax.bar(test_names, matches, alpha=0.7, edgecolor='black')
        ax.axhline(0.01, color='green', linestyle='--', alpha=0.5, label='1%')
        ax.axhline(0.05, color='orange', linestyle='--', alpha=0.5, label='5%')
        ax.set_ylabel('Relative error to 1/(4pi)')
        ax.set_title('Search for 1/(4pi) Factor')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Color by quality
        for bar, match in zip(bars, matches):
            if match < 0.01:
                bar.set_color('green')
            elif match < 0.05:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.suptitle('Spin Network and Loop Quantum Gravity Analysis', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        plt.savefig(f'results/{filename}', dpi=150, bbox_inches='tight')
        print(f"Plot saved: results/{filename}")
        print()
        
        return fig
    
    def generate_report(self, filename: str = 'spin_network_report.txt'):
        """Generate comprehensive text report."""
        
        # Compute everything first
        areas = self.compute_area_spectrum()
        volumes = self.compute_volume_spectrum()
        immirzi_results = self.test_immirzi_parameter()
        geom_results = self.test_geometric_factor_in_areas()
        graph_results = self.analyze_graph_structure()
        
        with open(f'results/{filename}', 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("SPIN NETWORK ANALYSIS - DETAILED REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Lattice Parameters:\n")
            f.write(f"  ell_max = {self.ell_max}\n")
            f.write(f"  Total nodes = {graph_results['total_nodes']}\n")
            f.write(f"  Total links = {graph_results['total_links']}\n\n")
            
            f.write(f"Area Spectrum (first 5 shells):\n")
            for ell in range(min(5, self.ell_max + 1)):
                f.write(f"  ell={ell}: A = {areas[ell]['A_full']:.4f}, ")
                f.write(f"r = {areas[ell]['r_ell']:.1f}\n")
            f.write("\n")
            
            f.write(f"Immirzi Parameter:\n")
            f.write(f"  Standard (from BH entropy): {immirzi_results['gamma_standard']:.6f}\n")
            f.write(f"  Geometric (1/4pi): {immirzi_results['gamma_alt']:.6f}\n")
            f.write(f"  Ratio: {immirzi_results['ratio_to_4pi']:.4f}\n")
            f.write(f"  Match: {immirzi_results['match']*100:.2f}% difference\n\n")
            
            f.write(f"Search for 1/(4pi) = {self.one_over_4pi:.8f}:\n")
            f.write(f"  Phase 8 geometric ratio: {geom_results['mean_phase8_ratio']:.6f}\n")
            f.write(f"  Match: {geom_results['match_phase8']*100:.2f}% error\n\n")
            
            f.write("="*70 + "\n")
        
        print(f"Report saved: results/{filename}")


def run_spin_network_investigation(ell_max: int = 15):
    """
    Complete spin network investigation.
    
    Args:
        ell_max: Maximum angular momentum
    """
    print("\n" + "="*70)
    print("PHASE 9.6: SPIN NETWORKS AND LOOP QUANTUM GRAVITY")
    print("="*70)
    print()
    print(f"Analyzing discrete lattice as spin network")
    print(f"Testing connection to LQG and role of 1/(4pi)")
    print()
    
    # Create calculator
    calc = SpinNetworkCalculator(ell_max)
    
    # Run analyses
    areas = calc.compute_area_spectrum()
    volumes = calc.compute_volume_spectrum()
    immirzi_results = calc.test_immirzi_parameter()
    geom_results = calc.test_geometric_factor_in_areas()
    graph_results = calc.analyze_graph_structure()
    
    # Generate visualizations
    calc.plot_spin_network_analysis()
    
    # Generate report
    calc.generate_report()
    
    print("="*70)
    print("INVESTIGATION COMPLETE")
    print("="*70)
    print()
    
    return calc, geom_results


if __name__ == "__main__":
    # Run with default parameters
    calc, results = run_spin_network_investigation(ell_max=15)
    
    print("\nKEY FINDINGS:")
    print(f"Phase 8 ratio match: {results['match_phase8']*100:.2f}% error")
    print(f"Mean geometric ratio: {results['mean_phase8_ratio']:.6f}")
    print(f"Compare to 1/(4pi): {results['one_over_4pi']:.6f}")
    
    if results['match_phase8'] < 0.05:
        print("\n*** STRONG connection to Phase 8 geometric constant!")
    elif results['match_phase8'] < 0.15:
        print("\n** GOOD agreement with 1/(4pi)")
    else:
        print("\n* Moderate agreement")
