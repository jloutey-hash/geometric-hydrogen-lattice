"""
Vacuum Energy Calculation on Discrete Polar Lattice

Tests whether the geometric constant 1/(4π) appears in:
1. Zero-point energy density
2. Mode density ρ(ω)
3. UV cutoff behavior
4. Casimir-like effects between shells

Key hypothesis: Discrete lattice provides natural UV regulator
with cutoff scale related to 1/(4π).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from typing import List, Tuple, Dict
import sys

from lattice import PolarLattice
from angular_momentum import AngularMomentumOperators


class VacuumEnergyCalculator:
    """
    Calculate zero-point (vacuum) energy on discrete polar lattice.
    
    In continuum QFT:
        E_vac = ∫ (1/2)ℏω ρ(ω) dω  → diverges!
    
    On discrete lattice:
        E_vac = Σ_modes (1/2)ℏω_mode  (finite sum)
    
    Test if mode density or cutoff involves 1/(4π).
    """
    
    def __init__(self, ell_max: int, use_units: str = 'natural'):
        """
        Initialize vacuum energy calculator.
        
        Args:
            ell_max: Maximum angular momentum quantum number
            use_units: 'natural' (ℏ=1) or 'physical' (SI units)
        """
        self.ell_max = ell_max
        self.lattice = PolarLattice(ell_max)
        self.use_units = use_units
        
        # Physical constants (SI units)
        self.hbar = 1.054571817e-34 if use_units == 'physical' else 1.0
        self.c = 2.99792458e8 if use_units == 'physical' else 1.0
        
        # Geometric constant
        self.one_over_4pi = 1 / (4 * np.pi)
        
        # Build operators
        self.L_squared = None
        self.modes = None
        self.frequencies = None
        
        # Compute total sites
        total_sites = sum(2 * (2 * ell + 1) for ell in range(ell_max + 1))
        
        print(f"VacuumEnergyCalculator initialized")
        print(f"  ell_max = {ell_max}")
        print(f"  Total sites = {total_sites}")
        print(f"  Units: {use_units}")
        print()
    
    def compute_free_field_modes(self, mass: float = 0.0, 
                                  field_type: str = 'scalar') -> np.ndarray:
        """
        Compute modes for free field on lattice.
        
        For scalar field φ:
            ω² = k² + m²
        
        On discrete lattice, k² → discrete Laplacian eigenvalues.
        
        Args:
            mass: Field mass (in natural units or GeV)
            field_type: 'scalar', 'vector', or 'fermion'
            
        Returns:
            frequencies: Array of mode frequencies
        """
        print(f"Computing {field_type} field modes (m = {mass})...")
        
        # Build L² operator
        ang_mom_ops = AngularMomentumOperators(self.lattice)
        self.L_squared = ang_mom_ops.build_L_squared()
        
        # For free field, radial and angular parts separate
        # Angular: eigenvalues of L² are ℓ(ℓ+1)
        # Radial: discretized radial Laplacian
        
        # Extract unique radii from lattice
        unique_ells = []
        unique_radii = []
        for ell in range(self.ell_max + 1):
            r_ell = 1 + 2 * ell  # From lattice construction
            unique_ells.append(ell)
            unique_radii.append(r_ell)
        
        # Mode frequencies from each (ℓ, r) combination
        frequencies = []
        mode_info = []
        
        for i_shell, (ell, r) in enumerate(zip(unique_ells, unique_radii)):
            # Angular momentum contribution
            k_squared_angular = ell * (ell + 1) / r**2
            
            # Add radial momentum (use shell index as proxy)
            k_squared_radial = (i_shell * np.pi / (self.ell_max + 1))**2
            
            k_squared = k_squared_angular + k_squared_radial
            
            # Dispersion relation
            omega_squared = k_squared + mass**2
            omega = np.sqrt(max(0, omega_squared))
            
            # Degeneracy: each ell contributes 2(2ell+1) lattice points
            # But modes: (2ell+1) for m_l values
            degeneracy = 2 * ell + 1
            
            for _ in range(degeneracy):
                frequencies.append(omega)
                mode_info.append({
                    'ell': ell,
                    'r': r,
                    'k2': k_squared,
                    'omega': omega
                })
        
        self.frequencies = np.array(frequencies)
        self.modes = mode_info
        
        print(f"  Found {len(frequencies)} modes")
        print(f"  w_min = {np.min(self.frequencies):.6f}")
        print(f"  w_max = {np.max(self.frequencies):.6f}")
        print()
        
        return self.frequencies
    
    def zero_point_energy(self) -> float:
        """
        Calculate zero-point energy: E_vac = Σ (1/2)ℏω
        
        Returns:
            E_vac: Total vacuum energy
        """
        if self.frequencies is None:
            raise ValueError("Must call compute_free_field_modes first")
        
        E_vac = 0.5 * self.hbar * np.sum(self.frequencies)
        
        print(f"Zero-point energy:")
        print(f"  E_vac = {E_vac:.6e}")
        print(f"  Per mode: {E_vac / len(self.frequencies):.6e}")
        print()
        
        return E_vac
    
    def mode_density(self, omega_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate density of states ρ(ω).
        
        Test if ρ(ω) ∝ ω² × [geometric factor involving 1/(4π)]
        
        Args:
            omega_bins: Number of frequency bins
            
        Returns:
            omega_centers: Bin centers
            rho: Mode density
        """
        if self.frequencies is None:
            raise ValueError("Must call compute_free_field_modes first")
        
        omega_max = np.max(self.frequencies)
        hist, bin_edges = np.histogram(self.frequencies, bins=omega_bins,
                                       range=(0, omega_max))
        
        omega_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        bin_width = bin_edges[1] - bin_edges[0]
        
        rho = hist / bin_width  # Normalize by bin width
        
        return omega_centers, rho
    
    def test_continuum_comparison(self) -> Dict:
        """
        Compare lattice mode density with continuum prediction.
        
        Continuum (3D): ρ(ω) = ω² / (2π²c³)
        
        Test if lattice gives: ρ(ω) ∝ ω² / (4π × r_cutoff³)
        """
        print("Testing continuum comparison...")
        
        omega_centers, rho_lattice = self.mode_density()
        
        # Continuum prediction (with effective volume)
        r_max = 1 + 2 * self.ell_max  # Maximum radius
        V_eff = (4/3) * np.pi * r_max**3
        rho_continuum = omega_centers**2 * V_eff / (2 * np.pi**2 * self.c**3)
        
        # Test modified continuum with 1/(4π) factor
        rho_modified = omega_centers**2 * V_eff / (4 * np.pi * self.c**3)
        
        # Compare
        chi_squared_standard = np.sum((rho_lattice - rho_continuum)**2)
        chi_squared_modified = np.sum((rho_lattice - rho_modified)**2)
        
        results = {
            'omega': omega_centers,
            'rho_lattice': rho_lattice,
            'rho_continuum': rho_continuum,
            'rho_modified': rho_modified,
            'chi2_standard': chi_squared_standard,
            'chi2_modified': chi_squared_modified,
            'improvement': chi_squared_standard / chi_squared_modified
        }
        
        print(f"  chi2 (standard): {chi_squared_standard:.3e}")
        print(f"  chi2 (with 1/4pi): {chi_squared_modified:.3e}")
        print(f"  Improvement: {results['improvement']:.3f}x")
        print()
        
        return results
    
    def cutoff_scale_analysis(self) -> Dict:
        """
        Analyze UV cutoff provided by lattice.
        
        In continuum, vacuum energy diverges as Λ⁴ where Λ is cutoff.
        On lattice, natural cutoff is Λ_lattice ∼ 1/a where a is spacing.
        
        Test if effective cutoff is Λ_eff = Λ_lattice × f(1/4π)
        """
        print("Analyzing UV cutoff scale...")
        
        # Effective lattice spacing - compute from radii
        radii = [1 + 2*ell for ell in range(self.ell_max + 1)]
        a_eff = np.mean(np.diff(radii))
        
        # Natural cutoff
        Lambda_natural = 1 / a_eff
        
        # Actual maximum frequency
        Lambda_actual = np.max(self.frequencies)
        
        # Test geometric factors
        factor_4pi = Lambda_actual / Lambda_natural
        factor_inv4pi = Lambda_actual * (4 * np.pi)
        
        results = {
            'a_eff': a_eff,
            'Lambda_natural': Lambda_natural,
            'Lambda_actual': Lambda_actual,
            'ratio': Lambda_actual / Lambda_natural,
            'factor_4pi': factor_4pi,
            'match_to_1over4pi': abs(factor_4pi - self.one_over_4pi) / self.one_over_4pi
        }
        
        print(f"  Effective spacing: a = {a_eff:.6f}")
        print(f"  Natural cutoff: Lambda = 1/a = {Lambda_natural:.6f}")
        print(f"  Actual cutoff: w_max = {Lambda_actual:.6f}")
        print(f"  Ratio: w_max/(1/a) = {results['ratio']:.6f}")
        print(f"  Compare to 1/(4pi) = {self.one_over_4pi:.6f}")
        print(f"  Match: {results['match_to_1over4pi']*100:.2f}% error")
        print()
        
        return results
    
    def casimir_effect_shells(self, ell_inner: int, ell_outer: int) -> float:
        """
        Calculate Casimir-like force between two shells.
        
        Vacuum energy difference when field is confined between shells.
        
        Args:
            ell_inner: Inner shell quantum number
            ell_outer: Outer shell quantum number
            
        Returns:
            Force per unit area (or energy per unit volume)
        """
        print(f"Computing Casimir effect between shells ell={ell_inner} and ell={ell_outer}...")
        
        # Find shells
        r_inner = 1 + 2 * ell_inner
        r_outer = 1 + 2 * ell_outer
        
        # Modes between shells
        E_cavity = 0.0
        n_modes = 0
        
        for mode in self.modes:
            if r_inner < mode['r'] < r_outer:
                E_cavity += 0.5 * self.hbar * mode['omega']
                n_modes += 1
        
        # Volume
        V_cavity = (4/3) * np.pi * (r_outer**3 - r_inner**3)
        
        # Energy density
        rho_E = E_cavity / V_cavity if V_cavity > 0 else 0.0
        
        print(f"  r_inner = {r_inner:.3f}, r_outer = {r_outer:.3f}")
        print(f"  Modes in cavity: {n_modes}")
        print(f"  Energy: {E_cavity:.6e}")
        print(f"  Energy density: {rho_E:.6e}")
        print()
        
        return rho_E
    
    def geometric_factor_in_vacuum(self) -> Dict:
        """
        Search for 1/(4π) in various vacuum energy properties.
        
        Tests:
        1. E_vac / N_modes vs 1/(4π)
        2. Cutoff normalization
        3. Energy density scaling
        """
        print("="*60)
        print("SEARCHING FOR 1/(4pi) IN VACUUM ENERGY")
        print("="*60)
        print()
        
        E_vac = self.zero_point_energy()
        N_modes = len(self.frequencies)
        
        # Test 1: Energy per mode
        E_per_mode = E_vac / N_modes
        omega_avg = np.mean(self.frequencies)
        
        ratio1 = E_per_mode / (0.5 * self.hbar * omega_avg)
        match1 = abs(ratio1 - self.one_over_4pi) / self.one_over_4pi
        
        # Test 2: Cutoff scale
        cutoff_results = self.cutoff_scale_analysis()
        match2 = cutoff_results['match_to_1over4pi']
        
        # Test 3: Energy density
        r_max = 1 + 2 * self.ell_max
        V_total = (4/3) * np.pi * r_max**3
        rho_E = E_vac / V_total
        
        # Dimensional analysis: [ρ_E] = Energy/Volume = ω⁴
        omega_typical = np.median(self.frequencies)
        rho_E_scale = omega_typical**4
        
        ratio3 = rho_E / rho_E_scale
        match3 = abs(ratio3 - self.one_over_4pi) / self.one_over_4pi
        
        results = {
            'E_vac': E_vac,
            'N_modes': N_modes,
            'test1_ratio': ratio1,
            'test1_match': match1,
            'test2_match': match2,
            'test3_ratio': ratio3,
            'test3_match': match3,
            'best_match': min(match1, match2, match3)
        }
        
        print(f"Test 1 - Energy per mode normalization:")
        print(f"  Ratio: {ratio1:.6f}")
        print(f"  vs 1/(4pi) = {self.one_over_4pi:.6f}")
        print(f"  Match: {match1*100:.2f}% error")
        print()
        
        print(f"Test 2 - Cutoff scale:")
        print(f"  Match: {match2*100:.2f}% error")
        print()
        
        print(f"Test 3 - Energy density scaling:")
        print(f"  Ratio: {ratio3:.6f}")
        print(f"  vs 1/(4pi) = {self.one_over_4pi:.6f}")
        print(f"  Match: {match3*100:.2f}% error")
        print()
        
        best_test = np.argmin([match1, match2, match3]) + 1
        print(f"BEST MATCH: Test {best_test} with {results['best_match']*100:.2f}% error")
        print()
        
        return results
    
    def plot_vacuum_analysis(self, filename: str = 'vacuum_energy_analysis.png'):
        """
        Create comprehensive visualization of vacuum energy analysis.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Vacuum Energy Analysis on Discrete Lattice', fontsize=14, fontweight='bold')
        
        # Panel 1: Mode frequency distribution
        ax = axes[0, 0]
        ax.hist(self.frequencies, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(self.frequencies), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(self.frequencies):.3f}')
        ax.set_xlabel('Frequency ω')
        ax.set_ylabel('Number of modes')
        ax.set_title('Mode Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Mode density ρ(ω)
        ax = axes[0, 1]
        omega_centers, rho = self.mode_density()
        continuum_results = self.test_continuum_comparison()
        
        ax.plot(omega_centers, rho, 'o-', label='Lattice', markersize=4)
        ax.plot(continuum_results['omega'], continuum_results['rho_continuum'], 
                '--', label='Continuum (standard)', alpha=0.7)
        ax.plot(continuum_results['omega'], continuum_results['rho_modified'], 
                '--', label='Continuum (with 1/4π)', alpha=0.7)
        ax.set_xlabel('Frequency ω')
        ax.set_ylabel('Mode density ρ(ω)')
        ax.set_title('Density of States')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Cumulative vacuum energy
        ax = axes[0, 2]
        sorted_freq = np.sort(self.frequencies)
        cumulative_E = 0.5 * self.hbar * np.cumsum(sorted_freq)
        
        ax.plot(sorted_freq, cumulative_E, linewidth=2)
        ax.axhline(cumulative_E[-1], color='red', linestyle='--', 
                   label=f'Total: {cumulative_E[-1]:.2e}')
        ax.set_xlabel('Frequency ω (sorted)')
        ax.set_ylabel('Cumulative E_vac')
        ax.set_title('Vacuum Energy Accumulation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 4: Energy by shell
        ax = axes[1, 0]
        shell_energies = []
        shell_labels = []
        
        radii = [1 + 2*ell for ell in range(self.ell_max + 1)]
        
        for i_shell in range(min(10, self.ell_max + 1)):
            ell = i_shell
            r = radii[i_shell]
            
            E_shell = sum(0.5 * self.hbar * mode['omega'] 
                         for mode in self.modes if mode['ell'] == ell and abs(mode['r'] - r) < 0.1)
            
            shell_energies.append(E_shell)
            shell_labels.append(f"ℓ={ell}")
        
        ax.bar(range(len(shell_energies)), shell_energies, alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(shell_labels)))
        ax.set_xticklabels(shell_labels, rotation=45)
        ax.set_ylabel('Vacuum Energy')
        ax.set_title('Energy by Shell')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Panel 5: Cutoff analysis
        ax = axes[1, 1]
        cutoff_results = self.cutoff_scale_analysis()
        
        factors = [cutoff_results['ratio'], self.one_over_4pi]
        labels = ['Lattice\ncutoff', '1/(4π)']
        colors = ['blue', 'red']
        
        bars = ax.bar(labels, factors, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Cutoff factor')
        ax.set_title(f"UV Cutoff Scale\nMatch: {cutoff_results['match_to_1over4pi']*100:.1f}% error")
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, factors):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.5f}', ha='center', va='bottom')
        
        # Panel 6: Geometric factor tests
        ax = axes[1, 2]
        geom_results = self.geometric_factor_in_vacuum()
        
        test_names = ['Energy\nper mode', 'Cutoff\nscale', 'Energy\ndensity']
        matches = [geom_results['test1_match'], 
                  geom_results['test2_match'],
                  geom_results['test3_match']]
        
        bars = ax.bar(test_names, matches, alpha=0.7, edgecolor='black')
        ax.axhline(0.01, color='green', linestyle='--', label='1% threshold', alpha=0.5)
        ax.axhline(0.05, color='orange', linestyle='--', label='5% threshold', alpha=0.5)
        ax.set_ylabel('Relative error to 1/(4π)')
        ax.set_title('Search for 1/(4π) Factor')
        ax.set_ylim(0, max(matches) * 1.2)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Color bars by quality
        for bar, match in zip(bars, matches):
            if match < 0.01:
                bar.set_color('green')
            elif match < 0.05:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.tight_layout()
        plt.savefig(f'results/{filename}', dpi=150, bbox_inches='tight')
        print(f"Plot saved: results/{filename}")
        print()
        
        return fig
    
    def generate_report(self, filename: str = 'vacuum_energy_report.txt'):
        """Generate comprehensive text report."""
        with open(f'results/{filename}', 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("VACUUM ENERGY ANALYSIS - DETAILED REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Lattice Parameters:\n")
            f.write(f"  ell_max = {self.ell_max}\n")
            total_sites = sum(2 * (2 * ell + 1) for ell in range(self.ell_max + 1))
            r_max = 1 + 2 * self.ell_max
            f.write(f"  Total sites = {total_sites}\n")
            f.write(f"  r_max = {r_max:.3f}\n\n")
            
            f.write(f"Mode Analysis:\n")
            f.write(f"  Total modes = {len(self.frequencies)}\n")
            f.write(f"  ω_min = {np.min(self.frequencies):.6f}\n")
            f.write(f"  ω_max = {np.max(self.frequencies):.6f}\n")
            f.write(f"  ω_mean = {np.mean(self.frequencies):.6f}\n\n")
            
            E_vac = self.zero_point_energy()
            f.write(f"Vacuum Energy:\n")
            f.write(f"  E_vac = {E_vac:.6e}\n")
            f.write(f"  E_vac / N_modes = {E_vac / len(self.frequencies):.6e}\n\n")
            
            geom_results = self.geometric_factor_in_vacuum()
            f.write(f"Search for 1/(4π) = {self.one_over_4pi:.8f}:\n")
            f.write(f"  Test 1 (energy per mode): {geom_results['test1_match']*100:.2f}% error\n")
            f.write(f"  Test 2 (cutoff scale): {geom_results['test2_match']*100:.2f}% error\n")
            f.write(f"  Test 3 (energy density): {geom_results['test3_match']*100:.2f}% error\n")
            f.write(f"  BEST: {geom_results['best_match']*100:.2f}% error\n\n")
            
            f.write("="*70 + "\n")
        
        print(f"Report saved: results/{filename}")


def run_vacuum_energy_investigation(ell_max: int = 10, mass: float = 0.0):
    """
    Complete vacuum energy investigation.
    
    Args:
        ell_max: Maximum angular momentum
        mass: Field mass (0 for massless)
    """
    print("\n" + "="*70)
    print("PHASE 9.4: VACUUM ENERGY INVESTIGATION")
    print("="*70)
    print()
    print(f"Testing whether 1/(4pi) appears in vacuum energy properties")
    print(f"Using discrete polar lattice as UV regulator")
    print()
    
    # Create calculator
    calc = VacuumEnergyCalculator(ell_max=ell_max)
    
    # Compute modes
    calc.compute_free_field_modes(mass=mass, field_type='scalar')
    
    # Calculate vacuum energy
    E_vac = calc.zero_point_energy()
    
    # Test continuum comparison
    continuum_results = calc.test_continuum_comparison()
    
    # Analyze cutoff
    cutoff_results = calc.cutoff_scale_analysis()
    
    # Search for geometric factor
    geom_results = calc.geometric_factor_in_vacuum()
    
    # Casimir effect
    if ell_max >= 4:
        F_casimir = calc.casimir_effect_shells(2, ell_max - 2)
    
    # Generate visualizations
    calc.plot_vacuum_analysis()
    
    # Generate report
    calc.generate_report()
    
    print("="*70)
    print("INVESTIGATION COMPLETE")
    print("="*70)
    print()
    
    return calc, geom_results


if __name__ == "__main__":
    # Run with default parameters
    calc, results = run_vacuum_energy_investigation(ell_max=15, mass=0.0)
    
    print("\nKEY FINDING:")
    print(f"Best match to 1/(4pi): {results['best_match']*100:.2f}% error")
    
    if results['best_match'] < 0.05:
        print("*** STRONG evidence for 1/(4pi) in vacuum energy!")
    elif results['best_match'] < 0.10:
        print("** GOOD evidence for 1/(4pi) in vacuum energy")
    else:
        print("* MODERATE evidence - further investigation needed")
