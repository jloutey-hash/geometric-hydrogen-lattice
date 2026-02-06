"""
SU(3) Impedance-Packing Analysis Module

CRITICAL DISCLAIMER:
====================
This module performs GEOMETRIC and CONTINUUM analysis of SU(3) representations
in spherical coordinates. It explores mathematical relationships between:
- Symplectic capacities (matter manifold geometry)
- Holonomy actions (gauge manifold geometry)
- Packing efficiencies (spherical shell geometry)

This is NOT a derivation of physical QCD coupling constants. The quantities
computed here (Z_eff, C_per_state, etc.) are purely geometric ratios that
characterize the continuum limit of discrete representations. Any numerical
coincidences with physical coupling constants are exploratory observations,
not theoretical predictions.

Purpose: Understand geometric structure of SU(3) representations in the
unified impedance framework, enabling comparison with U(1) hydrogen geometry.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Tuple, Dict, Optional
import warnings


class SU3ImpedanceAnalysis:
    """
    Analyze geometric relationships in SU(3) impedance-packing data.
    
    DISCLAIMER: This class performs continuum/packing exploration only.
    It does not claim to derive physical QCD coupling constants.
    """
    
    def __init__(self, csv_file: str = "su3_impedance_packing_scan.csv"):
        """
        Initialize analyzer by loading and preprocessing data.
        
        Parameters
        ----------
        csv_file : str
            Path to SU(3) impedance-packing scan CSV
        
        Notes
        -----
        This analysis is purely geometric. The impedance values represent
        ratios of symplectic capacities to gauge holonomies in a continuum
        approximation. They are NOT physical QCD coupling constants.
        """
        self.csv_file = csv_file
        self.df_raw = None
        self.df_clean = None
        self.df_derived = None
        
        self._load_and_preprocess()
    
    def _load_and_preprocess(self):
        """
        Load CSV and filter out non-finite impedance values.
        
        Non-finite values (inf, nan) arise from numerical issues in
        symplectic form calculations for certain representations.
        """
        print(f"Loading data from {self.csv_file}...")
        self.df_raw = pd.read_csv(self.csv_file)
        
        print(f"  Total representations: {len(self.df_raw)}")
        
        # Filter out non-finite Z values
        self.df_clean = self.df_raw[np.isfinite(self.df_raw['Z'])].copy()
        
        n_removed = len(self.df_raw) - len(self.df_clean)
        print(f"  Removed {n_removed} rows with non-finite Z values")
        print(f"  Clean dataset: {len(self.df_clean)} representations")
        
        if len(self.df_clean) == 0:
            warnings.warn("No finite Z values found! Check data quality.")
    
    def compute_derived_quantities(self) -> pd.DataFrame:
        """
        Construct derived geometric quantities for each representation.
        
        Computed quantities:
        -------------------
        Z_eff : float
            Effective impedance = Z (from SU(3) symplectic calculation)
            Ratio of gauge holonomy to matter capacity
        
        C_per_state : float
            Matter capacity per state = C_matter / dim
            Measures "capacity density" in representation space
        
        Z_per_state : float
            Impedance per state = Z_eff / dim
            Normalizes impedance by representation dimension
        
        Notes
        -----
        These are GEOMETRIC ratios characterizing continuum properties
        of SU(3) representations on spherical shells. They are NOT
        physical coupling constants.
        
        Returns
        -------
        df_derived : pd.DataFrame
            DataFrame with original + derived columns
        """
        if self.df_clean is None or len(self.df_clean) == 0:
            raise ValueError("No clean data available")
        
        print("\nComputing derived quantities...")
        
        df = self.df_clean.copy()
        
        # Z_eff = Z_impedance (already in CSV)
        df['Z_eff'] = df['Z']
        
        # C_per_state = C_matter / dim
        df['C_per_state'] = df['C_matter'] / df['dim']
        
        # Z_per_state = Z_eff / dim
        df['Z_per_state'] = df['Z_eff'] / df['dim']
        
        self.df_derived = df
        
        print(f"  Added columns: Z_eff, C_per_state, Z_per_state")
        print(f"  Z_eff range: [{df['Z_eff'].min():.4f}, {df['Z_eff'].max():.4f}]")
        print(f"  C_per_state range: [{df['C_per_state'].min():.4f}, {df['C_per_state'].max():.4f}]")
        print(f"  Z_per_state range: [{df['Z_per_state'].min():.4f}, {df['Z_per_state'].max():.4f}]")
        
        return df
    
    def save_derived_data(self, output_file: str = "su3_impedance_derived.csv"):
        """
        Save derived quantities to new CSV file.
        
        Parameters
        ----------
        output_file : str
            Output CSV filename
        
        Notes
        -----
        The output contains GEOMETRIC analysis results for continuum
        exploration. Not physical QCD parameters.
        """
        if self.df_derived is None:
            self.compute_derived_quantities()
        
        self.df_derived.to_csv(output_file, index=False)
        print(f"\nDerived data saved to {output_file}")
        print(f"  Total columns: {len(self.df_derived.columns)}")
        print(f"  Total rows: {len(self.df_derived)}")
    
    def fit_power_law(self, x_col: str = 'C_matter', y_col: str = 'Z_eff',
                      verbose: bool = True) -> Tuple[float, float, float]:
        """
        Fit power law: y ~ x^beta to geometric impedance data.
        
        Parameters
        ----------
        x_col : str
            Column name for x-axis (default: 'C_matter')
        y_col : str
            Column name for y-axis (default: 'Z_eff')
        verbose : bool
            Print fit results
        
        Returns
        -------
        beta : float
            Power law exponent
        r_squared : float
            Coefficient of determination (goodness of fit)
        prefactor : float
            Power law prefactor A in y = A * x^beta
        
        Notes
        -----
        This explores GEOMETRIC scaling relationships in the continuum limit.
        The exponent beta characterizes how impedance scales with capacity
        in representation space. This is NOT a physical QCD calculation.
        
        The power law fit may reveal universal scaling in the geometric
        structure of SU(3) representations, but does not constitute a
        derivation of physical coupling constants.
        """
        if self.df_derived is None:
            self.compute_derived_quantities()
        
        # Extract clean data
        df = self.df_derived.dropna(subset=[x_col, y_col])
        x = df[x_col].values
        y = df[y_col].values
        
        # Remove any remaining non-positive values (can't take log)
        mask = (x > 0) & (y > 0)
        x = x[mask]
        y = y[mask]
        
        if len(x) < 3:
            warnings.warn(f"Insufficient data points ({len(x)}) for power law fit")
            return np.nan, np.nan, np.nan
        
        # Fit in log-space: log(y) = log(A) + beta * log(x)
        log_x = np.log(x)
        log_y = np.log(y)
        
        # Linear regression in log-space
        coeffs = np.polyfit(log_x, log_y, 1)
        beta = coeffs[0]
        log_A = coeffs[1]
        A = np.exp(log_A)
        
        # Compute R²
        y_pred = A * x**beta
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Power Law Fit: {y_col} ~ {x_col}^beta")
            print(f"{'='*80}")
            print(f"  Model: {y_col} = {A:.4e} * {x_col}^{beta:.4f}")
            print(f"  Exponent (beta): {beta:.4f}")
            print(f"  Prefactor (A):   {A:.4e}")
            print(f"  R²:              {r_squared:.4f}")
            print(f"  Data points:     {len(x)}")
            print(f"\n  DISCLAIMER: This is a GEOMETRIC scaling relationship")
            print(f"  in the continuum limit, NOT a physical QCD calculation.")
            print(f"{'='*80}")
        
        return beta, r_squared, A
    
    def plot_analysis(self, output_prefix: str = "su3_analysis",
                     show: bool = False, dpi: int = 150):
        """
        Generate comprehensive analysis plots.
        
        Creates three plots:
        (a) Z_eff vs C_matter (log-log)
        (b) Z_eff vs packing_efficiency_mean (log-log)
        (c) packing_efficiency_mean vs C2
        
        Parameters
        ----------
        output_prefix : str
            Prefix for output filenames
        show : bool
            Display plots interactively
        dpi : int
            Resolution for saved figures
        
        Notes
        -----
        These plots visualize GEOMETRIC relationships in SU(3) representation
        space. They explore continuum scaling and packing efficiency, but do
        NOT represent physical QCD coupling constant calculations.
        """
        if self.df_derived is None:
            self.compute_derived_quantities()
        
        df = self.df_derived
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # (a) Z_eff vs C_matter (log-log)
        ax = axes[0]
        mask = (df['C_matter'] > 0) & (df['Z_eff'] > 0)
        df_plot = df[mask]
        
        ax.scatter(df_plot['C_matter'], df_plot['Z_eff'], 
                  c=df_plot['dim'], cmap='viridis', s=50, alpha=0.7)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Matter Capacity C_matter', fontsize=11)
        ax.set_ylabel('Effective Impedance Z_eff', fontsize=11)
        ax.set_title('(a) Impedance vs Capacity\n(Geometric Scaling)', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add power law fit line
        beta, r2, A = self.fit_power_law('C_matter', 'Z_eff', verbose=False)
        if not np.isnan(beta):
            x_fit = np.linspace(df_plot['C_matter'].min(), 
                               df_plot['C_matter'].max(), 100)
            y_fit = A * x_fit**beta
            ax.plot(x_fit, y_fit, 'r--', linewidth=2, alpha=0.7,
                   label=f'Fit: Z ~ C^{beta:.2f}\nR²={r2:.3f}')
            ax.legend(fontsize=9)
        
        # (b) Z_eff vs packing_efficiency (log-log)
        ax = axes[1]
        mask = (df['packing_efficiency_mean'] > 0) & (df['Z_eff'] > 0)
        df_plot = df[mask]
        
        ax.scatter(df_plot['packing_efficiency_mean'], df_plot['Z_eff'],
                  c=df_plot['C2'], cmap='plasma', s=50, alpha=0.7)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Packing Efficiency', fontsize=11)
        ax.set_ylabel('Effective Impedance Z_eff', fontsize=11)
        ax.set_title('(b) Impedance vs Packing\n(Continuum Exploration)', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Try power law fit
        beta_pack, r2_pack, A_pack = self.fit_power_law('packing_efficiency_mean', 
                                                         'Z_eff', verbose=False)
        if not np.isnan(beta_pack):
            x_fit = np.linspace(df_plot['packing_efficiency_mean'].min(),
                               df_plot['packing_efficiency_mean'].max(), 100)
            y_fit = A_pack * x_fit**beta_pack
            ax.plot(x_fit, y_fit, 'r--', linewidth=2, alpha=0.7,
                   label=f'Fit: Z ~ P^{beta_pack:.2f}\nR²={r2_pack:.3f}')
            ax.legend(fontsize=9)
        
        # (c) Packing efficiency vs C2
        ax = axes[2]
        ax.scatter(df['C2'], df['packing_efficiency_mean'],
                  c=df['dim'], cmap='cool', s=50, alpha=0.7)
        ax.set_xlabel('Casimir C2', fontsize=11)
        ax.set_ylabel('Packing Efficiency', fontsize=11)
        ax.set_title('(c) Packing vs Casimir\n(Shell Geometry)', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        corr = df['C2'].corr(df['packing_efficiency_mean'])
        ax.text(0.05, 0.95, f'ρ = {corr:+.3f}', 
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save figure
        filename = f"{output_prefix}_plots.png"
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"\nPlots saved to {filename}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        # Print disclaimer
        print("\n" + "="*80)
        print("IMPORTANT: These plots show GEOMETRIC relationships in continuum limit.")
        print("They do NOT represent physical QCD coupling constant derivations.")
        print("="*80)


def run_full_analysis(csv_file: str = "su3_impedance_packing_scan.csv",
                      output_csv: str = "su3_impedance_derived.csv",
                      output_plots: str = "su3_analysis"):
    """
    Run complete SU(3) impedance-packing analysis pipeline.
    
    Performs:
    1. Load and filter data
    2. Compute derived quantities (Z_eff, C_per_state, Z_per_state)
    3. Save derived data to CSV
    4. Fit power laws (Z ~ C^beta)
    5. Generate analysis plots
    
    Parameters
    ----------
    csv_file : str
        Input CSV from run_su3_packing_scan.py
    output_csv : str
        Output CSV with derived quantities
    output_plots : str
        Prefix for plot filenames
    
    Notes
    -----
    CRITICAL DISCLAIMER: This function performs GEOMETRIC and CONTINUUM
    analysis of SU(3) representations. It explores mathematical relationships
    between symplectic capacities, gauge holonomies, and packing efficiencies.
    
    This is NOT a derivation of physical QCD coupling constants. The quantities
    are purely geometric ratios characterizing continuum limits of discrete
    representations. Any numerical patterns are exploratory observations, not
    theoretical predictions of physical parameters.
    
    Purpose: Understand geometric structure for comparison with U(1) hydrogen
    in the unified impedance framework.
    
    Returns
    -------
    analyzer : SU3ImpedanceAnalysis
        Analysis object with all results
    """
    print("\n" + "="*80)
    print("SU(3) Impedance-Packing Analysis")
    print("="*80)
    print("\nDISCLAIMER: Geometric continuum exploration ONLY.")
    print("NOT a physical QCD coupling constant calculation.")
    print("="*80 + "\n")
    
    # Initialize analyzer
    analyzer = SU3ImpedanceAnalysis(csv_file)
    
    # Compute derived quantities
    analyzer.compute_derived_quantities()
    
    # Save to CSV
    analyzer.save_derived_data(output_csv)
    
    # Fit power laws
    print("\n" + "="*80)
    print("Power Law Analysis")
    print("="*80)
    
    beta_ZC, r2_ZC, A_ZC = analyzer.fit_power_law('C_matter', 'Z_eff')
    beta_ZP, r2_ZP, A_ZP = analyzer.fit_power_law('packing_efficiency_mean', 'Z_eff')
    
    # Generate plots
    analyzer.plot_analysis(output_plots, show=False)
    
    print("\n" + "="*80)
    print("Analysis Complete")
    print("="*80)
    print(f"Derived data: {output_csv}")
    print(f"Plots: {output_plots}_plots.png")
    print("\nKey geometric scaling relationships:")
    print(f"  Z ~ C^{beta_ZC:.3f} (R²={r2_ZC:.3f})")
    print(f"  Z ~ P^{beta_ZP:.3f} (R²={r2_ZP:.3f})")
    print("="*80 + "\n")
    
    return analyzer


if __name__ == "__main__":
    # Run complete analysis pipeline
    analyzer = run_full_analysis()
