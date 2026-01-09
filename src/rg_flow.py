"""
Renormalization Group Flow on Discrete Polar Lattice

Investigates how effective coupling evolves as we integrate out high-energy modes.

Key idea: Start with full lattice (ell_max), then progressively integrate out
outer shells to obtain effective theory at lower ell_max. Track how g^2 changes.

Tests whether beta-function has form: beta(g) ~ g^3/(4*pi)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import sys

from lattice import PolarLattice
from gauge_theory import WilsonGaugeField


class RGFlowCalculator:
    """
    Calculate RG flow by integrating out high-ell shells.
    
    Approach:
    1. Start with lattice at ell_max
    2. Measure effective coupling g^2_eff
    3. Remove highest shell (ell_max -> ell_max - 1)
    4. Remeasure g^2_eff
    5. Compute beta-function from change
    """
    
    def __init__(self, ell_max_initial: int, beta: float):
        """
        Initialize RG flow calculator.
        
        Args:
            ell_max_initial: Starting maximum angular momentum
            beta: Bare gauge coupling parameter (beta = 4/g^2)
        """
        self.ell_max_initial = ell_max_initial
        self.beta = beta
        self.one_over_4pi = 1 / (4 * np.pi)
        
        # Storage for RG trajectory
        self.ell_values = []
        self.g_squared_values = []
        self.scale_values = []
        self.beta_function_values = []
        
        print(f"RGFlowCalculator initialized")
        print(f"  Initial ell_max = {ell_max_initial}")
        print(f"  Beta = {beta:.4f} (bare g^2 = {4/beta:.6f})")
        print()
    
    def measure_effective_coupling(self, ell_max: int, 
                                   n_therm: int = 100,
                                   n_measure: int = 50) -> Tuple[float, float]:
        """
        Measure effective coupling at given cutoff.
        
        Args:
            ell_max: Current cutoff scale
            n_therm: Thermalization sweeps
            n_measure: Measurement sweeps
            
        Returns:
            g_squared_eff: Effective coupling
            g_squared_err: Statistical error
        """
        print(f"  Measuring at ell_max = {ell_max}...")
        
        # Create gauge field at this scale
        gauge = WilsonGaugeField(ell_max, self.beta)
        
        # Thermalize
        for _ in range(n_therm):
            gauge.metropolis_update()
        
        # Measure
        measurements = []
        for _ in range(n_measure):
            gauge.metropolis_update()
            _, g2_eff = gauge.measure_observables()
            measurements.append(g2_eff)
        
        g_squared_eff = np.mean(measurements)
        g_squared_err = np.std(measurements) / np.sqrt(n_measure)
        
        print(f"    g^2_eff = {g_squared_eff:.6f} +/- {g_squared_err:.6f}")
        
        return g_squared_eff, g_squared_err
    
    def run_rg_flow(self, ell_step: int = 1, 
                    n_therm: int = 100,
                    n_measure: int = 50) -> Dict:
        """
        Run complete RG flow from ell_max_initial down to ell_min.
        
        Args:
            ell_step: Step size for integrating out shells
            n_therm: Thermalization sweeps per scale
            n_measure: Measurements per scale
            
        Returns:
            results: Dictionary with RG trajectory
        """
        print("="*70)
        print("RUNNING RG FLOW")
        print("="*70)
        print()
        
        # Determine range
        ell_min = max(2, ell_step)  # Need at least ell=2 for meaningful gauge theory
        
        # Run from high to low energy (ell_max down)
        current_ell = self.ell_max_initial
        
        while current_ell >= ell_min:
            # Measure at this scale
            g2_eff, g2_err = self.measure_effective_coupling(
                current_ell, n_therm, n_measure
            )
            
            # Store
            self.ell_values.append(current_ell)
            self.g_squared_values.append(g2_eff)
            
            # Define scale parameter mu ~ 1/r_ell
            r_ell = 1 + 2 * current_ell
            mu = 1.0 / r_ell
            self.scale_values.append(mu)
            
            # Move to next scale
            current_ell -= ell_step
        
        # Compute beta-function: beta(g) = mu dg/dmu
        self._compute_beta_function()
        
        # Test for 1/(4pi) factor
        results = self._test_geometric_factor()
        
        print()
        print("="*70)
        print("RG FLOW COMPLETE")
        print("="*70)
        print()
        
        return results
    
    def _compute_beta_function(self):
        """
        Compute beta-function from measured trajectory.
        
        beta(g) = mu dg/dmu = d(g)/d(log mu)
        """
        print("\nComputing beta-function...")
        
        g_values = np.sqrt(np.array(self.g_squared_values))
        log_mu = np.log(self.scale_values)
        
        # Numerical derivative
        for i in range(1, len(g_values)):
            dg = g_values[i] - g_values[i-1]
            dlog_mu = log_mu[i] - log_mu[i-1]
            
            if abs(dlog_mu) > 1e-10:
                beta = dg / dlog_mu
                self.beta_function_values.append(beta)
            else:
                self.beta_function_values.append(0.0)
        
        # First point (no derivative available)
        if len(self.beta_function_values) > 0:
            self.beta_function_values.insert(0, self.beta_function_values[0])
        else:
            self.beta_function_values = [0.0] * len(g_values)
        
        print(f"  Beta-function computed at {len(self.beta_function_values)} scales")
        print()
    
    def _test_geometric_factor(self) -> Dict:
        """
        Test if beta-function has form: beta(g) ~ -g^3/(4*pi)
        
        Perturbative prediction for SU(2):
            beta(g) = -b0 * g^3 where b0 = 11/(24*pi^2) for SU(2)
        
        Test if our lattice gives: beta(g) ~ -g^3 * [factor involving 1/(4*pi)]
        """
        print("Testing for 1/(4pi) in beta-function...")
        print()
        
        g_values = np.sqrt(np.array(self.g_squared_values))
        beta_values = np.array(self.beta_function_values)
        
        # Test model: beta(g) = A * g^3
        # Fit coefficient A
        g_cubed = g_values**3
        
        # Avoid division by zero
        valid_idx = np.abs(g_cubed) > 1e-10
        
        if np.sum(valid_idx) > 0:
            A_fit = np.mean(beta_values[valid_idx] / g_cubed[valid_idx])
        else:
            A_fit = 0.0
        
        # Compare with predictions
        # Continuum SU(2): b0 = 11/(24*pi^2)
        b0_continuum = 11 / (24 * np.pi**2)
        
        # Test: A_fit vs -b0 * (4*pi)
        factor_4pi = -A_fit / b0_continuum
        
        # Test: does A_fit involve 1/(4*pi)?
        # If beta ~ -g^3/(4*pi) * C, then A = -C/(4*pi)
        C_extracted = -A_fit * (4 * np.pi)
        
        results = {
            'ell_values': self.ell_values,
            'g_squared': self.g_squared_values,
            'scales': self.scale_values,
            'beta_function': self.beta_function_values,
            'A_fit': A_fit,
            'b0_continuum': b0_continuum,
            'factor_4pi': factor_4pi,
            'C_extracted': C_extracted,
            'match_to_4pi': abs(factor_4pi - 4*np.pi) / (4*np.pi) if factor_4pi != 0 else 1.0
        }
        
        print(f"Beta-function fit: beta(g) = A * g^3")
        print(f"  A_fit = {A_fit:.6f}")
        print(f"  Continuum b0 = {b0_continuum:.6f}")
        print(f"  Factor: A_fit/b0 = {factor_4pi:.6f}")
        print(f"  Compare to 4*pi = {4*np.pi:.6f}")
        print()
        print(f"Alternative form: beta(g) = -g^3 * C/(4*pi)")
        print(f"  C extracted = {C_extracted:.6f}")
        print(f"  Compare to 1/(4*pi) = {self.one_over_4pi:.6f}")
        print()
        
        return results
    
    def plot_rg_flow(self, filename: str = 'rg_flow_analysis.png'):
        """
        Create comprehensive RG flow visualization.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Renormalization Group Flow Analysis', fontsize=14, fontweight='bold')
        
        g_values = np.sqrt(np.array(self.g_squared_values))
        
        # Panel 1: g^2 vs ell_max (cutoff)
        ax = axes[0, 0]
        ax.plot(self.ell_values, self.g_squared_values, 'o-', linewidth=2, markersize=6)
        ax.axhline(self.one_over_4pi, color='red', linestyle='--', 
                   label=f'1/(4pi) = {self.one_over_4pi:.4f}')
        ax.set_xlabel('Cutoff (ell_max)')
        ax.set_ylabel('g^2_eff')
        ax.set_title('Running Coupling vs Cutoff')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 2: g^2 vs log(mu) (scale)
        ax = axes[0, 1]
        log_mu = np.log(self.scale_values)
        ax.plot(log_mu, self.g_squared_values, 'o-', linewidth=2, markersize=6)
        ax.axhline(self.one_over_4pi, color='red', linestyle='--', label='1/(4pi)')
        ax.set_xlabel('log(mu)')
        ax.set_ylabel('g^2(mu)')
        ax.set_title('Running Coupling vs Energy Scale')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Beta-function beta(g) vs g
        ax = axes[0, 2]
        ax.plot(g_values, self.beta_function_values, 'o-', linewidth=2, markersize=6)
        
        # Plot perturbative prediction: beta ~ -g^3 * b0
        g_plot = np.linspace(min(g_values), max(g_values), 100)
        b0_continuum = 11 / (24 * np.pi**2)
        beta_pert = -b0_continuum * g_plot**3
        ax.plot(g_plot, beta_pert, '--', label='Continuum (1-loop)', alpha=0.7)
        
        ax.set_xlabel('g')
        ax.set_ylabel('beta(g)')
        ax.set_title('Beta-function')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', linewidth=0.5)
        
        # Panel 4: beta(g) / g^3 (to see coefficient)
        ax = axes[1, 0]
        g_cubed = g_values**3
        valid = np.abs(g_cubed) > 1e-10
        
        if np.sum(valid) > 0:
            ratio = np.array(self.beta_function_values)[valid] / g_cubed[valid]
            ax.plot(g_values[valid], ratio, 'o-', linewidth=2, markersize=6, label='Lattice')
        
        ax.axhline(-b0_continuum, color='red', linestyle='--', 
                   label=f'Continuum b0 = {-b0_continuum:.6f}')
        ax.set_xlabel('g')
        ax.set_ylabel('beta(g) / g^3')
        ax.set_title('Beta-function Coefficient')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 5: Trajectory in (g^2, ell) plane
        ax = axes[1, 1]
        
        # Color by scale
        scatter = ax.scatter(self.g_squared_values, self.ell_values, 
                            c=log_mu, cmap='viridis', s=100, edgecolors='black')
        
        # Add arrows to show flow direction
        for i in range(len(self.g_squared_values)-1):
            ax.annotate('', xy=(self.g_squared_values[i+1], self.ell_values[i+1]),
                       xytext=(self.g_squared_values[i], self.ell_values[i]),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1))
        
        ax.axvline(self.one_over_4pi, color='red', linestyle='--', alpha=0.5, label='1/(4pi)')
        ax.set_xlabel('g^2_eff')
        ax.set_ylabel('ell_max (cutoff)')
        ax.set_title('RG Trajectory')
        plt.colorbar(scatter, ax=ax, label='log(mu)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 6: Test of 1/(4pi) factor
        ax = axes[1, 2]
        
        # Show g^2 values normalized by 1/(4pi)
        normalized = np.array(self.g_squared_values) / self.one_over_4pi
        
        ax.plot(self.ell_values, normalized, 'o-', linewidth=2, markersize=6)
        ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Perfect match')
        ax.fill_between(self.ell_values, 0.95, 1.05, alpha=0.2, color='green', 
                        label='5% band')
        ax.set_xlabel('ell_max')
        ax.set_ylabel('g^2 / [1/(4pi)]')
        ax.set_title('Match to Geometric Constant')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/{filename}', dpi=150, bbox_inches='tight')
        print(f"Plot saved: results/{filename}")
        print()
        
        return fig
    
    def generate_report(self, results: Dict, filename: str = 'rg_flow_report.txt'):
        """Generate comprehensive text report."""
        with open(f'results/{filename}', 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("RG FLOW ANALYSIS - DETAILED REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Initial Parameters:\n")
            f.write(f"  ell_max_initial = {self.ell_max_initial}\n")
            f.write(f"  Beta = {self.beta:.4f}\n")
            f.write(f"  Bare g^2 = {4/self.beta:.6f}\n\n")
            
            f.write(f"RG Flow:\n")
            f.write(f"  Number of scales = {len(self.ell_values)}\n")
            f.write(f"  ell_max range: {min(self.ell_values)} to {max(self.ell_values)}\n")
            f.write(f"  g^2 range: {min(self.g_squared_values):.6f} to {max(self.g_squared_values):.6f}\n\n")
            
            f.write(f"Beta-function Analysis:\n")
            f.write(f"  Fit: beta(g) = A * g^3\n")
            f.write(f"  A_fit = {results['A_fit']:.6f}\n")
            f.write(f"  Continuum b0 = {results['b0_continuum']:.6f}\n")
            f.write(f"  Ratio: A/b0 = {results['factor_4pi']:.6f}\n")
            f.write(f"  Compare to 4*pi = {4*np.pi:.6f}\n\n")
            
            f.write(f"Search for 1/(4pi) = {self.one_over_4pi:.8f}:\n")
            f.write(f"  Mean g^2: {np.mean(self.g_squared_values):.6f}\n")
            f.write(f"  Std dev: {np.std(self.g_squared_values):.6f}\n")
            
            # Check if g^2 stays near 1/(4pi)
            deviations = [abs(g2 - self.one_over_4pi) / self.one_over_4pi 
                         for g2 in self.g_squared_values]
            mean_dev = np.mean(deviations)
            
            f.write(f"  Mean deviation from 1/(4pi): {mean_dev*100:.2f}%\n\n")
            
            f.write("="*70 + "\n")
        
        print(f"Report saved: results/{filename}")


def run_rg_flow_investigation(ell_max_initial: int = 12, 
                               beta: float = 50.0,
                               ell_step: int = 1):
    """
    Complete RG flow investigation.
    
    Args:
        ell_max_initial: Starting cutoff
        beta: Bare coupling parameter
        ell_step: Step size for flow
    """
    print("\n" + "="*70)
    print("PHASE 9.5: RENORMALIZATION GROUP FLOW")
    print("="*70)
    print()
    print(f"Testing whether beta-function involves 1/(4pi)")
    print(f"Starting from ell_max = {ell_max_initial}, beta = {beta}")
    print()
    
    # Create calculator
    calc = RGFlowCalculator(ell_max_initial, beta)
    
    # Run RG flow
    results = calc.run_rg_flow(ell_step=ell_step, n_therm=100, n_measure=30)
    
    # Generate visualizations
    calc.plot_rg_flow()
    
    # Generate report
    calc.generate_report(results)
    
    print("="*70)
    print("INVESTIGATION COMPLETE")
    print("="*70)
    print()
    
    return calc, results


if __name__ == "__main__":
    # Run with default parameters
    calc, results = run_rg_flow_investigation(ell_max_initial=12, beta=50.0, ell_step=1)
    
    print("\nKEY FINDINGS:")
    print(f"Beta-function coefficient: A = {results['A_fit']:.6f}")
    print(f"Continuum prediction: b0 = {results['b0_continuum']:.6f}")
    
    mean_g2 = np.mean(results['g_squared'])
    one_over_4pi = 1/(4*np.pi)
    match = abs(mean_g2 - one_over_4pi) / one_over_4pi * 100
    
    print(f"\nMean g^2 across scales: {mean_g2:.6f}")
    print(f"Compare to 1/(4pi): {one_over_4pi:.6f}")
    print(f"Match: {match:.2f}% deviation")
    
    if match < 10:
        print("\n*** g^2 remains near 1/(4pi) across scales!")
    elif match < 20:
        print("\n** Moderate stability near 1/(4pi)")
    else:
        print("\n* Significant running observed")
