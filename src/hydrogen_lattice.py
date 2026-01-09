"""
Hydrogen Atom on Discrete Angular Momentum Lattice

This module solves the hydrogen atom Schrödinger equation using our discrete
SU(2) lattice where angular momentum is exact and the radial coordinate is
discrete: r_ℓ = 1 + 2ℓ.

Key hypothesis: Do energy corrections involve the geometric constant 1/(4π)?

Physical setup:
- Hamiltonian: H = L²/(2mr²) + V(r)
- Potential: V(r) = -e²/(4πε₀r) (Coulomb)
- Discrete radial: r_ℓ = 1 + 2ℓ
- Exact angular momentum: L²|ψ⟩ = ℓ(ℓ+1)ℏ²|ψ⟩

Phase 9.2 - Quick Win Implementation
"""

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import diags
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt

# Physical constants (SI units)
HBAR = 1.054571817e-34  # J·s
M_E = 9.1093837015e-31  # kg (electron mass)
E_CHARGE = 1.602176634e-19  # C
EPSILON_0 = 8.8541878128e-12  # F/m
C = 299792458  # m/s (speed of light)

# Derived constants
A_0 = 4 * np.pi * EPSILON_0 * HBAR**2 / (M_E * E_CHARGE**2)  # Bohr radius
RY = M_E * E_CHARGE**4 / (2 * (4*np.pi*EPSILON_0)**2 * HBAR**2)  # Rydberg energy
ALPHA = E_CHARGE**2 / (4 * np.pi * EPSILON_0 * HBAR * C)  # Fine structure constant

class HydrogenLattice:
    """
    Solve hydrogen atom on discrete SU(2) lattice.
    
    The radial coordinate is discrete: r_ℓ = (1 + 2ℓ) × a_lattice
    where a_lattice is the lattice spacing parameter.
    
    The angular momentum is exact: L² = ℓ(ℓ+1)ℏ²
    """
    
    def __init__(self, ell_max: int = 50, a_lattice: float = 1.0):
        """
        Initialize hydrogen atom solver.
        
        Parameters:
        -----------
        ell_max : int
            Maximum angular momentum quantum number
        a_lattice : float
            Lattice spacing in units of Bohr radius a₀
            Default: 1.0 means r_ℓ in units of a₀
        """
        self.ell_max = ell_max
        self.a_lattice = a_lattice
        
        # Build radial lattice
        self.ell_values = np.arange(0, ell_max + 1)
        self.r_ell = (1 + 2 * self.ell_values) * a_lattice  # Discrete radii
        self.N = len(self.ell_values)
        
        # Angular momentum eigenvalues
        self.L2_ell = self.ell_values * (self.ell_values + 1)
        
        print(f"Hydrogen Lattice initialized:")
        print(f"  ℓ_max = {ell_max}")
        print(f"  N_shells = {self.N}")
        print(f"  r_min = {self.r_ell[0]:.3f} a₀")
        print(f"  r_max = {self.r_ell[-1]:.3f} a₀")
        print(f"  a_lattice = {a_lattice:.3f} a₀")
    
    def coulomb_potential(self) -> np.ndarray:
        """
        Coulomb potential: V(r) = -1/r (in atomic units).
        
        Returns:
        --------
        V : np.ndarray
            Potential energy at each lattice site
        """
        return -1.0 / self.r_ell
    
    def kinetic_energy_diagonal(self) -> np.ndarray:
        """
        Diagonal kinetic energy: T = L²/(2mr²) (in atomic units).
        
        Returns:
        --------
        T : np.ndarray
            Kinetic energy at each lattice site
        """
        # In atomic units: ℏ = m_e = 1, so T = L²/(2r²)
        return self.L2_ell / (2 * self.r_ell**2)
    
    def build_hamiltonian_diagonal(self) -> np.ndarray:
        """
        Build diagonal Hamiltonian (no hopping between shells).
        
        H = T + V = L²/(2r²) - 1/r
        
        This is the simplest approximation: each shell is independent.
        
        Returns:
        --------
        H : np.ndarray (N,)
            Diagonal Hamiltonian
        """
        T = self.kinetic_energy_diagonal()
        V = self.coulomb_potential()
        return T + V
    
    def build_hamiltonian_with_hopping(self, t_hop: Optional[float] = None) -> np.ndarray:
        """
        Build Hamiltonian with radial hopping between adjacent shells.
        
        H = -1/2 ∇² + L²/(2r²) - 1/r
        
        Radial kinetic term approximated by finite differences:
        -1/2 d²/dr² ≈ -1/2 [ψ(r+Δr) - 2ψ(r) + ψ(r-Δr)] / Δr²
        
        Parameters:
        -----------
        t_hop : float, optional
            Hopping amplitude. If None, computed from lattice spacing.
        
        Returns:
        --------
        H : np.ndarray (N, N)
            Full Hamiltonian matrix
        """
        # Estimate hopping from lattice spacing
        if t_hop is None:
            # Δr between shells varies, use average
            dr_avg = np.mean(np.diff(self.r_ell))
            t_hop = 0.5 / dr_avg**2  # From -1/2 d²/dr²
        
        # Start with diagonal terms
        H = np.zeros((self.N, self.N))
        
        # Diagonal: angular momentum + potential + radial kinetic (self-energy)
        for i in range(self.N):
            H[i, i] = self.L2_ell[i] / (2 * self.r_ell[i]**2) + self.coulomb_potential()[i]
            H[i, i] += 2 * t_hop  # Radial kinetic self-energy term
        
        # Off-diagonal: radial hopping
        for i in range(self.N - 1):
            H[i, i+1] = -t_hop
            H[i+1, i] = -t_hop
        
        return H
    
    def solve_diagonal(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve hydrogen atom with diagonal Hamiltonian.
        
        Returns:
        --------
        energies : np.ndarray
            Energy eigenvalues (sorted)
        states : np.ndarray
            Corresponding eigenstates (columns)
        """
        H = self.build_hamiltonian_diagonal()
        
        # For diagonal Hamiltonian, eigenvalues are the diagonal elements
        energies = np.sort(H)
        idx = np.argsort(H)
        
        # Eigenstates are delta functions at each site
        states = np.eye(self.N)[:, idx]
        
        return energies, states
    
    def solve_with_hopping(self, t_hop: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve hydrogen atom with radial hopping.
        
        Parameters:
        -----------
        t_hop : float, optional
            Hopping amplitude. If None, computed from lattice spacing.
        
        Returns:
        --------
        energies : np.ndarray
            Energy eigenvalues (sorted)
        states : np.ndarray
            Corresponding eigenstates (columns)
        """
        H = self.build_hamiltonian_with_hopping(t_hop)
        energies, states = eigh(H)
        return energies, states
    
    def compare_with_continuum(self, n_states: int = 10) -> Dict:
        """
        Compare lattice results with continuum hydrogen atom.
        
        Continuum: E_n = -1/(2n²) (in atomic units)
        
        Parameters:
        -----------
        n_states : int
            Number of states to compare
        
        Returns:
        --------
        results : dict
            Comparison data
        """
        # Solve lattice
        E_lattice_diag, _ = self.solve_diagonal()
        E_lattice_hop, _ = self.solve_with_hopping()
        
        # Continuum energies
        n_values = np.arange(1, n_states + 1)
        E_continuum = -1.0 / (2 * n_values**2)
        
        # Take lowest n_states from lattice
        E_lat_diag = E_lattice_diag[:n_states]
        E_lat_hop = E_lattice_hop[:n_states]
        
        # Compute errors
        error_diag = np.abs(E_lat_diag - E_continuum) / np.abs(E_continuum) * 100
        error_hop = np.abs(E_lat_hop - E_continuum) / np.abs(E_continuum) * 100
        
        return {
            'n': n_values,
            'E_continuum': E_continuum,
            'E_lattice_diagonal': E_lat_diag,
            'E_lattice_hopping': E_lat_hop,
            'error_diagonal': error_diag,
            'error_hopping': error_hop,
            'r_ell': self.r_ell[:n_states],
            'ell': self.ell_values[:n_states]
        }
    
    def find_geometric_factor(self) -> Dict:
        """
        Search for 1/(4π) in energy corrections.
        
        Hypothesis: E_lattice - E_continuum ∝ 1/(4π) × f(n, ℓ)
        
        Returns:
        --------
        analysis : dict
            Geometric factor analysis
        """
        results = self.compare_with_continuum(n_states=min(20, self.N))
        
        # Energy differences
        delta_E_diag = results['E_lattice_diagonal'] - results['E_continuum']
        delta_E_hop = results['E_lattice_hopping'] - results['E_continuum']
        
        # Try to fit: ΔE = A × 1/(4π) × f(n)
        # Expected: f(n) might be 1/n³ or similar
        
        # Test various scaling functions
        n = results['n']
        
        # Model 1: ΔE ~ α₀/(4π·n³)
        scaling_1 = 1 / (4 * np.pi * n**3)
        A_1_diag = np.mean(delta_E_diag / scaling_1) if np.all(scaling_1 != 0) else 0
        A_1_hop = np.mean(delta_E_hop / scaling_1) if np.all(scaling_1 != 0) else 0
        
        # Model 2: ΔE ~ α₀/(4π·n²)
        scaling_2 = 1 / (4 * np.pi * n**2)
        A_2_diag = np.mean(delta_E_diag / scaling_2)
        A_2_hop = np.mean(delta_E_hop / scaling_2)
        
        # Model 3: ΔE ~ α₀·(4π)/(n²)
        scaling_3 = 4 * np.pi / n**2
        A_3_diag = np.mean(delta_E_diag / scaling_3)
        A_3_hop = np.mean(delta_E_hop / scaling_3)
        
        # Residuals for each model
        residual_1_diag = np.std((delta_E_diag - A_1_diag * scaling_1) / np.abs(delta_E_diag))
        residual_2_diag = np.std((delta_E_diag - A_2_diag * scaling_2) / np.abs(delta_E_diag))
        residual_3_diag = np.std((delta_E_diag - A_3_diag * scaling_3) / np.abs(delta_E_diag))
        
        return {
            'n': n,
            'delta_E_diagonal': delta_E_diag,
            'delta_E_hopping': delta_E_hop,
            'models': {
                '1/(4π·n³)': {'A_diag': A_1_diag, 'A_hop': A_1_hop, 'residual': residual_1_diag},
                '1/(4π·n²)': {'A_diag': A_2_diag, 'A_hop': A_2_hop, 'residual': residual_2_diag},
                '4π/n²': {'A_diag': A_3_diag, 'A_hop': A_3_hop, 'residual': residual_3_diag}
            },
            'one_over_4pi': 1 / (4 * np.pi)
        }
    
    def plot_comparison(self, n_states: int = 10, save: bool = True):
        """
        Plot energy level comparison.
        
        Parameters:
        -----------
        n_states : int
            Number of states to plot
        save : bool
            Whether to save figure
        """
        results = self.compare_with_continuum(n_states)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Panel 1: Energy levels
        ax = axes[0, 0]
        ax.plot(results['n'], results['E_continuum'], 'ko-', label='Continuum', linewidth=2)
        ax.plot(results['n'], results['E_lattice_diagonal'], 'bs--', label='Lattice (diagonal)', alpha=0.7)
        ax.plot(results['n'], results['E_lattice_hopping'], 'r^--', label='Lattice (hopping)', alpha=0.7)
        ax.set_xlabel('Principal quantum number n', fontsize=11)
        ax.set_ylabel('Energy (Rydberg)', fontsize=11)
        ax.set_title('Hydrogen Energy Levels', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Percent error
        ax = axes[0, 1]
        ax.semilogy(results['n'], results['error_diagonal'], 'bs-', label='Diagonal')
        ax.semilogy(results['n'], results['error_hopping'], 'r^-', label='With hopping')
        ax.set_xlabel('Principal quantum number n', fontsize=11)
        ax.set_ylabel('Relative error (%)', fontsize=11)
        ax.set_title('Error vs Continuum', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Energy differences
        ax = axes[1, 0]
        delta_E_diag = results['E_lattice_diagonal'] - results['E_continuum']
        delta_E_hop = results['E_lattice_hopping'] - results['E_continuum']
        ax.plot(results['n'], delta_E_diag, 'bs-', label='Diagonal')
        ax.plot(results['n'], delta_E_hop, 'r^-', label='With hopping')
        ax.axhline(1/(4*np.pi), color='green', linestyle='--', linewidth=2, label='1/(4π)', alpha=0.7)
        ax.axhline(-1/(4*np.pi), color='green', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xlabel('Principal quantum number n', fontsize=11)
        ax.set_ylabel('ΔE = E_lattice - E_continuum', fontsize=11)
        ax.set_title('Energy Corrections', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 4: Radial coordinate
        ax = axes[1, 1]
        ax.plot(results['ell'], results['r_ell'], 'ko-', linewidth=2)
        ax.set_xlabel('ℓ quantum number', fontsize=11)
        ax.set_ylabel('r_ℓ (Bohr radii)', fontsize=11)
        ax.set_title('Discrete Radial Lattice: r_ℓ = 1 + 2ℓ', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig('results/hydrogen_lattice_comparison.png', dpi=150, bbox_inches='tight')
            print("Saved: results/hydrogen_lattice_comparison.png")
        
        plt.show()
        return fig
    
    def plot_geometric_factor(self, save: bool = True):
        """
        Plot analysis of geometric factor 1/(4π).
        
        Parameters:
        -----------
        save : bool
            Whether to save figure
        """
        analysis = self.find_geometric_factor()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Panel 1: Energy corrections vs n
        ax = axes[0]
        n = analysis['n']
        delta_E = analysis['delta_E_diagonal']
        
        ax.plot(n, delta_E, 'ko-', linewidth=2, markersize=6, label='ΔE (lattice - continuum)')
        
        # Overlay model predictions
        for model_name, model_data in analysis['models'].items():
            if '4π·n³' in model_name:
                scaling = 1 / (4 * np.pi * n**3)
            elif '4π·n²' in model_name:
                scaling = 1 / (4 * np.pi * n**2)
            else:  # '4π/n²'
                scaling = 4 * np.pi / n**2
            
            prediction = model_data['A_diag'] * scaling
            ax.plot(n, prediction, '--', alpha=0.7, label=f"{model_name} (R={model_data['residual']:.3f})")
        
        ax.axhline(1/(4*np.pi), color='red', linestyle=':', linewidth=2, label='1/(4π) = 0.0796')
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Principal quantum number n', fontsize=11)
        ax.set_ylabel('ΔE (Rydberg)', fontsize=11)
        ax.set_title('Energy Corrections: Search for 1/(4π)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Scaled corrections
        ax = axes[1]
        
        # Scale by different models
        for model_name, model_data in analysis['models'].items():
            if '4π·n³' in model_name:
                scaling = 1 / (4 * np.pi * n**3)
                scaled = delta_E / scaling
            elif '4π·n²' in model_name:
                scaling = 1 / (4 * np.pi * n**2)
                scaled = delta_E / scaling
            else:
                scaling = 4 * np.pi / n**2
                scaled = delta_E / scaling
            
            ax.plot(n, scaled, 'o-', alpha=0.7, label=f"ΔE / ({model_name})")
        
        ax.set_xlabel('Principal quantum number n', fontsize=11)
        ax.set_ylabel('Scaled ΔE', fontsize=11)
        ax.set_title('Scaled Energy Corrections (looking for constant)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        if save:
            plt.savefig('results/hydrogen_geometric_factor.png', dpi=150, bbox_inches='tight')
            print("Saved: results/hydrogen_geometric_factor.png")
        
        plt.show()
        return fig
    
    def generate_report(self, filename: str = 'results/hydrogen_lattice_report.txt'):
        """
        Generate comprehensive text report.
        
        Parameters:
        -----------
        filename : str
            Output filename
        """
        with open(filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("HYDROGEN ATOM ON DISCRETE SU(2) LATTICE - PHASE 9.2\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("LATTICE PARAMETERS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Maximum ℓ: {self.ell_max}\n")
            f.write(f"Number of shells: {self.N}\n")
            f.write(f"Lattice spacing: {self.a_lattice:.3f} a₀\n")
            f.write(f"Radial range: {self.r_ell[0]:.3f} to {self.r_ell[-1]:.3f} a₀\n")
            f.write(f"Discrete radii: r_ℓ = 1 + 2ℓ\n\n")
            
            # Energy comparison
            results = self.compare_with_continuum(n_states=min(15, self.N))
            
            f.write("ENERGY LEVELS COMPARISON\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'n':>3} {'ℓ':>3} {'r_ℓ':>8} {'E_cont':>12} {'E_diag':>12} {'E_hop':>12} {'Err_D(%)':>10} {'Err_H(%)':>10}\n")
            f.write("-" * 80 + "\n")
            
            for i in range(len(results['n'])):
                f.write(f"{results['n'][i]:3d} "
                       f"{results['ell'][i]:3d} "
                       f"{results['r_ell'][i]:8.2f} "
                       f"{results['E_continuum'][i]:12.6f} "
                       f"{results['E_lattice_diagonal'][i]:12.6f} "
                       f"{results['E_lattice_hopping'][i]:12.6f} "
                       f"{results['error_diagonal'][i]:10.3f} "
                       f"{results['error_hopping'][i]:10.3f}\n")
            
            f.write("\n")
            
            # Geometric factor analysis
            analysis = self.find_geometric_factor()
            
            f.write("GEOMETRIC FACTOR ANALYSIS: SEARCH FOR 1/(4π)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Reference value: 1/(4π) = {analysis['one_over_4pi']:.10f}\n\n")
            
            f.write("Testing models for energy corrections:\n")
            f.write("  ΔE = E_lattice - E_continuum = A × scaling(n)\n\n")
            
            for model_name, model_data in analysis['models'].items():
                f.write(f"Model: ΔE ~ A × {model_name}\n")
                f.write(f"  A (diagonal): {model_data['A_diag']:12.6f}\n")
                f.write(f"  A (hopping):  {model_data['A_hop']:12.6f}\n")
                f.write(f"  Residual:     {model_data['residual']:12.6f}\n")
                
                # Check if A is close to 1/(4π) or related
                ratio_to_4pi = model_data['A_diag'] * 4 * np.pi
                f.write(f"  A × 4π =      {ratio_to_4pi:12.6f}\n")
                f.write("\n")
            
            f.write("\n")
            f.write("INTERPRETATION\n")
            f.write("-" * 80 + "\n")
            f.write("The discrete lattice r_ℓ = 1 + 2ℓ creates energy corrections relative\n")
            f.write("to the continuum hydrogen atom. We search for the geometric constant\n")
            f.write("1/(4π) ≈ 0.0796 in these corrections.\n\n")
            
            f.write("If ΔE ∝ 1/(4π), this would connect the geometric discovery from\n")
            f.write("Phase 8 to observable physics (hydrogen spectrum).\n\n")
            
            f.write("Best-fit model determines which scaling gives most constant A.\n")
            f.write("If A ≈ 1/(4π) or A×4π ≈ integer, geometric factor confirmed!\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("END REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"Saved: {filename}")


def main():
    """Main execution: solve hydrogen atom and analyze results."""
    print("=" * 80)
    print("PHASE 9.2: HYDROGEN ATOM ON DISCRETE LATTICE")
    print("=" * 80)
    print()
    
    # Create solver
    hydrogen = HydrogenLattice(ell_max=50, a_lattice=1.0)
    print()
    
    # Compare with continuum
    print("Computing energy levels...")
    results = hydrogen.compare_with_continuum(n_states=15)
    print()
    
    print("Energy Level Summary:")
    print("-" * 80)
    print(f"{'n':>3} {'E_continuum':>12} {'E_lattice':>12} {'Error (%)':>12}")
    print("-" * 80)
    for i in range(min(10, len(results['n']))):
        print(f"{results['n'][i]:3d} "
              f"{results['E_continuum'][i]:12.6f} "
              f"{results['E_lattice_diagonal'][i]:12.6f} "
              f"{results['error_diagonal'][i]:12.3f}")
    print()
    
    # Geometric factor analysis
    print("Analyzing geometric factor 1/(4π)...")
    analysis = hydrogen.find_geometric_factor()
    print()
    
    print("Geometric Factor Analysis:")
    print("-" * 80)
    print(f"Reference: 1/(4π) = {analysis['one_over_4pi']:.10f}")
    print()
    print("Best-fit coefficients:")
    for model_name, model_data in analysis['models'].items():
        print(f"  {model_name:15s}: A = {model_data['A_diag']:10.6f}, "
              f"A×4π = {model_data['A_diag']*4*np.pi:10.6f}, "
              f"Residual = {model_data['residual']:.4f}")
    print()
    
    # Generate visualizations
    print("Generating plots...")
    hydrogen.plot_comparison(n_states=15)
    hydrogen.plot_geometric_factor()
    print()
    
    # Generate report
    print("Generating report...")
    hydrogen.generate_report()
    print()
    
    print("=" * 80)
    print("PHASE 9.2 COMPLETE!")
    print("=" * 80)
    print()
    print("Key findings:")
    print("  • Hydrogen atom solved on discrete lattice")
    print("  • Energy corrections analyzed for 1/(4π) factor")
    print("  • Results saved to results/")
    print()
    print("Next: Review plots and report to assess if 1/(4π) emerges!")


if __name__ == '__main__':
    main()
