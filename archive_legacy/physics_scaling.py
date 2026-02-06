"""
GEOMETRIC BERRY PHASE SCALING ANALYSIS
======================================
Testing the Fundamental Physics Hypothesis:
"Is the Berry Phase the geometric origin of Spin-Orbit Coupling?"

Spin-Orbit Coupling (Fine Structure) scales as: Delta E ~ 1/n^3
We test if Berry Phase follows: theta_n ~ A * n^(-k)

If k ≈ 3: **DISCOVERY** - This is Fine Structure!
If k ≈ 2: Relativistic correction
If k ≈ 1: Coulomb effect

Author: High-Energy Physics Research
Date: February 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
import time
from typing import Dict, List, Tuple
from collections import defaultdict

# Import previous holonomy calculation
from physics_holonomy import PlaquetteHolonomy


class ScalingAnalysis:
    """
    Analyze radial scaling of geometric Berry phase.
    
    Tests if Berry phase scales as 1/n^k where k = 3 (fine structure),
    k = 2 (relativistic), or k = 1 (Coulomb).
    """
    
    def __init__(self):
        """Initialize scaling analyzer."""
        self.holonomy_data = None
        self.shell_data = {}
        self.fit_results = {}
        
    def compute_holonomy_data(self, max_n: int = 20):
        """
        Compute holonomy data using the PlaquetteHolonomy engine.
        
        Parameters:
        -----------
        max_n : int
            Maximum principal quantum number
        """
        print("\n" + "="*80)
        print("COMPUTING HOLONOMY DATA")
        print("="*80)
        print(f"\nGenerating Berry phase measurements for n <= {max_n}...")
        
        # Initialize holonomy calculator
        calc = PlaquetteHolonomy(max_n=max_n)
        
        # Construct plaquettes
        plaquettes = calc.construct_plaquettes()
        
        if len(plaquettes) == 0:
            print("ERROR: No plaquettes found")
            return None
        
        # Initialize spinors
        calc.initialize_spinors()
        
        # Measure holonomies
        holonomy_results = calc.measure_all_holonomies()
        
        self.holonomy_data = holonomy_results
        
        print(f"\n[OK] Computed {len(holonomy_results)} holonomy measurements")
        
        return holonomy_results
    
    def group_by_shell(self) -> Dict[int, List[Dict]]:
        """
        TASK 1: Group plaquettes by principal quantum number n.
        
        Each plaquette has center coordinates (n, l, m).
        We group by the radial shell index (average n of the 4 nodes).
        
        Returns:
        --------
        shell_data : Dict[int, List[Dict]]
            Dictionary mapping n -> list of plaquette measurements
        """
        print("\n" + "="*80)
        print("TASK 1: RADIAL SCALING LAW ANALYSIS")
        print("="*80)
        print("\nGrouping plaquettes by principal quantum number (radial shell)...\n")
        
        if self.holonomy_data is None:
            print("ERROR: No holonomy data available")
            return {}
        
        # Group by shell
        shell_groups = defaultdict(list)
        
        for result in self.holonomy_data:
            nodes = result['nodes']
            
            # Extract n values from the 4 nodes
            n_values = [node[0] for node in nodes]
            
            # Average n (shell center)
            n_avg = np.mean(n_values)
            n_shell = int(np.round(n_avg))
            
            shell_groups[n_shell].append(result)
        
        # Convert to regular dict and sort
        self.shell_data = dict(sorted(shell_groups.items()))
        
        print(f"Shell Distribution:")
        print("-"*80)
        print(f"{'n (shell)':>10} | {'# Plaquettes':>15} | {'Mean Berry (rad)':>18} | {'Std Berry':>12}")
        print("-"*80)
        
        for n, plaquettes in self.shell_data.items():
            berry_phases = [p['berry_phase'] for p in plaquettes]
            mean_berry = np.mean(berry_phases)
            std_berry = np.std(berry_phases)
            
            print(f"{n:10d} | {len(plaquettes):15d} | {mean_berry:18.8f} | {std_berry:12.6f}")
        
        print("-"*80)
        print(f"\nTotal shells analyzed: {len(self.shell_data)}")
        
        return self.shell_data
    
    def extract_shell_statistics(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract average Berry phase per shell with error bars.
        
        Returns:
        --------
        n_values : np.ndarray
            Shell indices
        mean_berry : np.ndarray
            Mean Berry phase per shell
        std_berry : np.ndarray
            Standard deviation per shell
        """
        if not self.shell_data:
            return np.array([]), np.array([]), np.array([])
        
        n_values = []
        mean_berry_list = []
        std_berry_list = []
        
        for n, plaquettes in self.shell_data.items():
            berry_phases = [p['berry_phase'] for p in plaquettes]
            
            n_values.append(n)
            mean_berry_list.append(np.mean(berry_phases))
            std_berry_list.append(np.std(berry_phases))
        
        return np.array(n_values), np.array(mean_berry_list), np.array(std_berry_list)
    
    def fit_power_law(self, n_values: np.ndarray, berry_values: np.ndarray,
                     weights: np.ndarray = None) -> Dict:
        """
        Fit Berry phase to power law: theta(n) = A * n^(-k).
        
        Uses log-log linear regression:
        log(theta) = log(A) - k * log(n)
        
        Parameters:
        -----------
        n_values : np.ndarray
            Shell indices
        berry_values : np.ndarray
            Mean Berry phase per shell
        weights : np.ndarray, optional
            Weights for fitting (1/std)
        
        Returns:
        --------
        fit_result : Dict
            Fit parameters and statistics
        """
        print("\n" + "="*80)
        print("POWER LAW FIT: theta(n) = A * n^(-k)")
        print("="*80)
        print("\nFitting Berry phase scaling...\n")
        
        # Filter out any zeros or negative values
        valid_mask = (n_values > 0) & (berry_values > 0)
        n_fit = n_values[valid_mask]
        berry_fit = berry_values[valid_mask]
        
        if len(n_fit) < 3:
            print("ERROR: Not enough data points for fitting")
            return {}
        
        # Log-log transformation
        log_n = np.log(n_fit)
        log_berry = np.log(berry_fit)
        
        # Linear regression in log-log space
        slope, intercept, r_value, p_value, std_err = linregress(log_n, log_berry)
        
        # Extract power law parameters
        k = -slope  # Exponent (negative because we expect decay)
        A = np.exp(intercept)  # Coefficient
        
        # Compute R^2
        r_squared = r_value ** 2
        
        print("FIT RESULTS:")
        print("-"*80)
        print(f"  Power law:        theta(n) = A * n^(-k)")
        print(f"  Coefficient A:    {A:.10f}")
        print(f"  Exponent k:       {k:.6f}")
        print(f"  R-squared:        {r_squared:.6f}")
        print(f"  Std error:        {std_err:.6f}")
        print(f"  p-value:          {p_value:.6e}")
        
        # Interpretation
        print(f"\n  INTERPRETATION:")
        print("-"*80)
        
        if abs(k - 3.0) < 0.3:
            print(f"  ** k ~ 3.0:  FINE STRUCTURE SCALING (Spin-Orbit) **")
            print(f"     Deviation from k=3: {abs(k - 3.0):.2f}")
            print(f"     This matches the 1/n^3 law for fine structure!")
        elif abs(k - 2.0) < 0.3:
            print(f"  ** k ~ 2.0:  RELATIVISTIC CORRECTION **")
            print(f"     Deviation from k=2: {abs(k - 2.0):.2f}")
        elif abs(k - 1.0) < 0.3:
            print(f"  ** k ~ 1.0:  COULOMB SCALING **")
            print(f"     Deviation from k=1: {abs(k - 1.0):.2f}")
        else:
            print(f"  ** k = {k:.2f}:  NOVEL SCALING LAW **")
            print(f"     Does not match standard atomic physics scaling")
        
        fit_result = {
            'A': A,
            'k': k,
            'r_squared': r_squared,
            'std_err': std_err,
            'p_value': p_value,
            'n_fit': n_fit,
            'berry_fit': berry_fit,
            'log_n': log_n,
            'log_berry': log_berry
        }
        
        self.fit_results = fit_result
        
        return fit_result
    
    def compare_to_alpha(self) -> Dict:
        """
        TASK 2: Extract "Alpha" coefficient and compare to fundamental constants.
        
        Fine structure energy: Delta E ~ alpha^4 / n^3
        If Berry phase is proportional to energy, then:
            theta ~ alpha^2 * f(n)  or  theta ~ alpha^4 * g(n)
        
        Returns:
        --------
        analysis : Dict
            Comparison to alpha powers
        """
        print("\n" + "="*80)
        print("TASK 2: ALPHA COEFFICIENT EXTRACTION")
        print("="*80)
        print("\nComparing coefficient A to powers of fine structure constant...\n")
        
        if not self.fit_results:
            print("ERROR: No fit results available")
            return {}
        
        A = self.fit_results['A']
        k = self.fit_results['k']
        
        # Fundamental constants
        alpha = 1.0 / 137.035999084
        alpha_squared = alpha ** 2
        alpha_cubed = alpha ** 3
        alpha_fourth = alpha ** 4
        
        print("FUNDAMENTAL CONSTANTS:")
        print("-"*80)
        print(f"  alpha (1/137):        {alpha:.12f}")
        print(f"  alpha^2:              {alpha_squared:.12e}")
        print(f"  alpha^3:              {alpha_cubed:.12e}")
        print(f"  alpha^4:              {alpha_fourth:.12e}")
        
        print(f"\n\nFIT COEFFICIENT:")
        print("-"*80)
        print(f"  A (from fit):         {A:.12e}")
        
        # Compute ratios
        ratio_alpha = A / alpha
        ratio_alpha2 = A / alpha_squared
        ratio_alpha3 = A / alpha_cubed
        ratio_alpha4 = A / alpha_fourth
        
        print(f"\n\nRATIO ANALYSIS:")
        print("-"*80)
        print(f"  A / alpha:            {ratio_alpha:.6f}")
        print(f"  A / alpha^2:          {ratio_alpha2:.6f}")
        print(f"  A / alpha^3:          {ratio_alpha3:.6f}")
        print(f"  A / alpha^4:          {ratio_alpha4:.6f}")
        
        # Test which ratio is closest to simple integer or fraction
        print(f"\n\nCOINCIDENCE TESTS:")
        print("-"*80)
        
        tests = [
            ('A / alpha', ratio_alpha, 1.0),
            ('A / alpha^2', ratio_alpha2, 1.0),
            ('A / alpha^3', ratio_alpha3, 1.0),
            ('A / alpha^4', ratio_alpha4, 1.0),
            ('A * 137', A * 137, 1.0),
            ('A * 137^2', A * 137**2, 1.0),
        ]
        
        for name, value, target in tests:
            deviation = abs(value - target) / target * 100
            status = "***MATCH***" if deviation < 20 else ""
            print(f"  {name:20s} = {value:12.6f}  (dev: {deviation:6.2f}%)  {status}")
        
        # Physical interpretation
        print(f"\n\nPHYSICAL INTERPRETATION:")
        print("-"*80)
        
        if k > 2.7 and k < 3.3:
            print(f"  Scaling exponent k ~ 3 confirms FINE STRUCTURE behavior.")
            print(f"  \n  In hydrogen, fine structure energy is:")
            print(f"    Delta E_fs = (alpha^2 * m_e * c^2) * (1/n^3) * f(l,j)")
            print(f"  \n  If Berry phase theta ~ Delta E, then:")
            print(f"    theta(n) = C * alpha^2 * n^(-3)")
            print(f"  \n  Our fit gives:")
            print(f"    theta(n) = {A:.6e} * n^(-{k:.2f})")
            print(f"  \n  Expected coefficient: C ~ alpha^2 = {alpha_squared:.6e}")
            print(f"  Measured coefficient: A = {A:.6e}")
            print(f"  Ratio A / alpha^2 = {ratio_alpha2:.2f}")
            
            if ratio_alpha2 > 0.1 and ratio_alpha2 < 10:
                print(f"  \n  ** CONCLUSION: Geometric Berry phase magnitude is")
                print(f"     consistent with fine structure constant! **")
        
        analysis = {
            'A': A,
            'k': k,
            'alpha': alpha,
            'ratio_to_alpha': ratio_alpha,
            'ratio_to_alpha2': ratio_alpha2,
            'ratio_to_alpha4': ratio_alpha4
        }
        
        return analysis
    
    def generate_report(self, filename: str = "scaling_report.txt"):
        """
        Generate comprehensive scaling analysis report.
        """
        print(f"\n\nGenerating scaling report: {filename}")
        
        with open(filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GEOMETRIC BERRY PHASE SCALING ANALYSIS\n")
            f.write("Testing: Is Berry Phase the origin of Spin-Orbit Coupling?\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Shell data
            if self.shell_data:
                f.write("\n" + "="*80 + "\n")
                f.write("TASK 1: RADIAL SHELL ANALYSIS\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"Number of shells:        {len(self.shell_data)}\n")
                f.write(f"n range:                 [{min(self.shell_data.keys())}, {max(self.shell_data.keys())}]\n\n")
                
                f.write("Shell Statistics:\n")
                f.write("-"*80 + "\n")
                f.write(f"{'n':>5} | {'Count':>8} | {'Mean Berry (rad)':>18} | {'Std':>12}\n")
                f.write("-"*80 + "\n")
                
                for n, plaquettes in self.shell_data.items():
                    berry_phases = [p['berry_phase'] for p in plaquettes]
                    mean_berry = np.mean(berry_phases)
                    std_berry = np.std(berry_phases)
                    f.write(f"{n:5d} | {len(plaquettes):8d} | {mean_berry:18.10f} | {std_berry:12.6f}\n")
                
                f.write("-"*80 + "\n\n")
            
            # Fit results
            if self.fit_results:
                f.write("\n" + "="*80 + "\n")
                f.write("POWER LAW FIT: theta(n) = A * n^(-k)\n")
                f.write("="*80 + "\n\n")
                
                A = self.fit_results['A']
                k = self.fit_results['k']
                r2 = self.fit_results['r_squared']
                
                f.write(f"Fit Parameters:\n")
                f.write(f"  Coefficient A:       {A:.12e}\n")
                f.write(f"  Exponent k:          {k:.8f}\n")
                f.write(f"  R-squared:           {r2:.8f}\n")
                f.write(f"  Std error:           {self.fit_results['std_err']:.6f}\n\n")
                
                f.write(f"Interpretation:\n")
                if abs(k - 3.0) < 0.3:
                    f.write(f"  ** FINE STRUCTURE SCALING (k ~ 3) **\n")
                    f.write(f"  Deviation from k=3:  {abs(k - 3.0):.4f}\n")
                    f.write(f"  This matches the 1/n^3 spin-orbit coupling law!\n\n")
                elif abs(k - 2.0) < 0.3:
                    f.write(f"  ** RELATIVISTIC CORRECTION (k ~ 2) **\n\n")
                elif abs(k - 1.0) < 0.3:
                    f.write(f"  ** COULOMB SCALING (k ~ 1) **\n\n")
                else:
                    f.write(f"  ** NOVEL SCALING (k = {k:.2f}) **\n\n")
            
            # Alpha comparison
            if self.fit_results:
                f.write("\n" + "="*80 + "\n")
                f.write("TASK 2: FINE STRUCTURE CONSTANT CONNECTION\n")
                f.write("="*80 + "\n\n")
                
                alpha = 1.0 / 137.035999084
                A = self.fit_results['A']
                
                f.write(f"Fine structure constant:\n")
                f.write(f"  alpha:               {alpha:.12f}\n")
                f.write(f"  alpha^2:             {alpha**2:.12e}\n")
                f.write(f"  alpha^4:             {alpha**4:.12e}\n\n")
                
                f.write(f"Coefficient ratios:\n")
                f.write(f"  A / alpha:           {A/alpha:.6f}\n")
                f.write(f"  A / alpha^2:         {A/(alpha**2):.6f}\n")
                f.write(f"  A / alpha^4:         {A/(alpha**4):.6f}\n\n")
            
            # Summary
            f.write("\n" + "="*80 + "\n")
            f.write("SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write("Key findings:\n")
            f.write("1. Berry phase grouped by radial shell (principal quantum number)\n")
            f.write("2. Power law scaling fitted to data\n")
            f.write("3. Exponent and coefficient compared to atomic physics predictions\n\n")
            
            if self.fit_results:
                k = self.fit_results['k']
                if abs(k - 3.0) < 0.3:
                    f.write("** MAJOR RESULT: Scaling matches fine structure (1/n^3) **\n")
                    f.write("This suggests the Berry phase is geometrically encoding\n")
                    f.write("the spin-orbit coupling of the hydrogen atom!\n\n")
        
        print(f"[OK] Report saved to {filename}")
    
    def plot_scaling(self, filename: str = "scaling_plot.png"):
        """
        Generate comprehensive scaling visualization.
        """
        if not self.fit_results:
            print("ERROR: No fit results to plot")
            return
        
        n_values, mean_berry, std_berry = self.extract_shell_statistics()
        
        if len(n_values) == 0:
            print("ERROR: No shell data to plot")
            return
        
        # Fit parameters
        A = self.fit_results['A']
        k = self.fit_results['k']
        n_fit = self.fit_results['n_fit']
        berry_fit = self.fit_results['berry_fit']
        
        # Generate fit curve
        n_smooth = np.linspace(n_fit.min(), n_fit.max(), 100)
        berry_smooth = A * n_smooth ** (-k)
        
        # Also generate comparison curves
        alpha = 1.0 / 137.035999084
        berry_k1 = A * n_smooth ** (-1.0)  # Coulomb
        berry_k2 = A * n_smooth ** (-2.0)  # Relativistic
        berry_k3 = A * n_smooth ** (-3.0)  # Fine structure
        
        fig = plt.figure(figsize=(16, 10))
        
        # Plot 1: Log-log plot (main result)
        ax1 = plt.subplot(2, 3, 1)
        ax1.errorbar(n_values, mean_berry, yerr=std_berry, fmt='o', 
                    markersize=8, capsize=5, capthick=2, linewidth=2,
                    color='blue', ecolor='lightblue', label='Data')
        ax1.plot(n_smooth, berry_smooth, 'r-', linewidth=3, 
                label=f'Fit: A*n^(-{k:.2f}), R²={self.fit_results["r_squared"]:.3f}')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Principal Quantum Number n', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Mean Berry Phase (rad)', fontsize=12, fontweight='bold')
        ax1.set_title('Power Law Scaling (Log-Log)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, which='both')
        
        # Plot 2: Linear scale
        ax2 = plt.subplot(2, 3, 2)
        ax2.errorbar(n_values, mean_berry, yerr=std_berry, fmt='o',
                    markersize=8, capsize=5, capthick=2, linewidth=2,
                    color='green', ecolor='lightgreen', label='Data')
        ax2.plot(n_smooth, berry_smooth, 'r-', linewidth=3, label=f'Fit: A*n^(-{k:.2f})')
        ax2.set_xlabel('Principal Quantum Number n', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Mean Berry Phase (rad)', fontsize=12, fontweight='bold')
        ax2.set_title('Power Law Scaling (Linear)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Residuals
        ax3 = plt.subplot(2, 3, 3)
        berry_pred = A * n_values ** (-k)
        residuals = mean_berry - berry_pred
        ax3.errorbar(n_values, residuals, yerr=std_berry, fmt='o',
                    markersize=8, capsize=5, capthick=2, linewidth=2,
                    color='purple', ecolor='lavender')
        ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('Principal Quantum Number n', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Residuals (rad)', fontsize=12, fontweight='bold')
        ax3.set_title('Fit Residuals', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Comparison to different scalings
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(n_values, mean_berry, 'ko', markersize=10, label='Data', zorder=5)
        ax4.plot(n_smooth, berry_smooth, 'r-', linewidth=3, label=f'Best fit (k={k:.2f})', zorder=4)
        ax4.plot(n_smooth, berry_k3, 'b--', linewidth=2, label='k=3 (Fine Structure)', zorder=3)
        ax4.plot(n_smooth, berry_k2, 'g--', linewidth=2, label='k=2 (Relativistic)', zorder=2)
        ax4.plot(n_smooth, berry_k1, 'm--', linewidth=2, label='k=1 (Coulomb)', zorder=1)
        ax4.set_xlabel('Principal Quantum Number n', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Berry Phase (rad)', fontsize=12, fontweight='bold')
        ax4.set_title('Comparison to Standard Scalings', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: n^3 * Berry (should be constant if k=3)
        ax5 = plt.subplot(2, 3, 5)
        scaled_berry = n_values**3 * mean_berry
        scaled_errors = n_values**3 * std_berry
        ax5.errorbar(n_values, scaled_berry, yerr=scaled_errors, fmt='o',
                    markersize=8, capsize=5, capthick=2, linewidth=2,
                    color='orange', ecolor='bisque')
        mean_scaled = np.mean(scaled_berry)
        ax5.axhline(y=mean_scaled, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_scaled:.4f}')
        ax5.set_xlabel('Principal Quantum Number n', fontsize=12, fontweight='bold')
        ax5.set_ylabel('n^3 * Berry Phase', fontsize=12, fontweight='bold')
        ax5.set_title('Fine Structure Test (should be flat)', fontsize=14, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Coefficient comparison
        ax6 = plt.subplot(2, 3, 6)
        x_labels = ['alpha', 'alpha^2', 'Measured\nA', 'alpha^4*10^6']
        values = [alpha, alpha**2, A, alpha**4 * 1e6]
        colors = ['red', 'orange', 'blue', 'green']
        
        bars = ax6.bar(x_labels, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax6.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax6.set_title('Coefficient vs Alpha Powers', fontsize=14, fontweight='bold')
        ax6.set_yscale('log')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height*1.2,
                    f'{val:.2e}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  [OK] Saved: {filename}")
        plt.close()


def main():
    """
    Main execution: Complete scaling analysis.
    """
    print("\n" + "="*80)
    print("GEOMETRIC BERRY PHASE SCALING ANALYSIS")
    print("Testing the Spin-Orbit Coupling Hypothesis")
    print("="*80 + "\n")
    
    # Initialize analyzer
    analyzer = ScalingAnalysis()
    
    # Compute holonomy data
    print("Step 1: Computing holonomy measurements...")
    analyzer.compute_holonomy_data(max_n=20)
    
    # TASK 1: Group by shell and analyze scaling
    print("\n\nStep 2: Analyzing radial scaling...")
    analyzer.group_by_shell()
    
    # Extract statistics
    n_values, mean_berry, std_berry = analyzer.extract_shell_statistics()
    
    if len(n_values) < 3:
        print("\nERROR: Not enough data points for scaling analysis")
        return
    
    # Fit power law
    fit_results = analyzer.fit_power_law(n_values, mean_berry)
    
    # TASK 2: Compare to alpha
    print("\n\nStep 3: Comparing to fine structure constant...")
    alpha_analysis = analyzer.compare_to_alpha()
    
    # Generate outputs
    print("\n\nStep 4: Generating outputs...")
    analyzer.generate_report("scaling_report.txt")
    analyzer.plot_scaling("scaling_plot.png")
    
    print("\n" + "="*80)
    print("SCALING ANALYSIS COMPLETE")
    print("="*80)
    print("\nResults saved to:")
    print("  - scaling_report.txt")
    print("  - scaling_plot.png")
    
    # Final summary
    if fit_results:
        k = fit_results['k']
        print(f"\n\n{'='*80}")
        print("FINAL VERDICT")
        print("="*80)
        print(f"\nMeasured scaling exponent: k = {k:.4f}")
        
        if abs(k - 3.0) < 0.3:
            print("\n** DISCOVERY: Berry Phase follows FINE STRUCTURE scaling (1/n³) **")
            print("   This is the geometric signature of spin-orbit coupling!")
            print("   The lattice curvature encodes quantum mechanical fine structure.")
        elif abs(k - 2.0) < 0.3:
            print("\n** Result: Relativistic correction scaling (1/n²) **")
        elif abs(k - 1.0) < 0.3:
            print("\n** Result: Coulomb scaling (1/n) **")
        else:
            print(f"\n** Result: Novel scaling law (1/n^{k:.2f}) **")
    
    print("\n")


if __name__ == "__main__":
    main()
