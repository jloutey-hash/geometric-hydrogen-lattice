"""
Comprehensive Geometric Ratio Analysis for Discrete Polar Lattice

This module implements 28+ geometric formulas to search for fundamental constants
emerging from the lattice structure. Based on systematic exploration of:
- Basic Casimir/geometry ratios
- Sphere-based formulas
- Angular momentum density
- Shell-to-shell ratios
- Volume/area scaling
- Complex combinations
- Logarithmic forms
- Parameter searches

Goal: Find dimensionless ratios that converge to fundamental constants
      (α ≈ 1/137, π, e, φ, etc.)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass


# Known physical and mathematical constants for comparison
KNOWN_CONSTANTS = {
    'alpha': 1/137.035999084,
    'pi': np.pi,
    'e': np.e,
    'phi': (1 + np.sqrt(5)) / 2,  # Golden ratio
    '1/pi': 1/np.pi,
    '1/2pi': 1/(2*np.pi),
    '1/4pi': 1/(4*np.pi),
    '2/pi': 2/np.pi,
    'sqrt(2)': np.sqrt(2),
    'sqrt(3)': np.sqrt(3),
    '1/2': 0.5,
    '1/3': 1/3,
    '2/3': 2/3,
}


@dataclass
class FormulaResult:
    """Results for a single geometric formula."""
    name: str
    expression: str
    values: np.ndarray
    ell_values: np.ndarray
    converges: bool
    limit_value: float
    convergence_rate: float
    best_match: Tuple[str, float, float]  # (constant_name, constant_value, error)


class GeometricRatioExplorer:
    """
    Systematic exploration of geometric ratios in discrete polar lattice.
    
    Lattice structure:
    - Ring ℓ radius: r_ℓ = 1 + 2ℓ
    - Ring ℓ points: N_ℓ = 2(2ℓ+1)
    - Casimir invariant: L²_ℓ = ℓ(ℓ+1)
    """
    
    def __init__(self, ell_max: int = 100):
        """
        Initialize explorer.
        
        Parameters
        ----------
        ell_max : int
            Maximum ℓ value to explore
        """
        self.ell_max = ell_max
        self.ell_values = np.arange(1, ell_max + 1)
        
        # Pre-compute basic quantities
        self.r_ell = 1 + 2 * self.ell_values
        self.N_ell = 2 * (2 * self.ell_values + 1)
        self.L2_ell = self.ell_values * (self.ell_values + 1)
        
        # Results storage
        self.results: Dict[str, FormulaResult] = {}
    
    # ============================================================
    # SECTION 1: Basic Ratios Involving Casimir and Geometry
    # ============================================================
    
    def alpha_1(self) -> np.ndarray:
        """α₁ = ℓ(ℓ+1) / r_ℓ²"""
        return self.L2_ell / self.r_ell**2
    
    def alpha_2(self) -> np.ndarray:
        """α₂ = ℓ(ℓ+1) / N_ℓ"""
        return self.L2_ell / self.N_ell
    
    def alpha_3(self) -> np.ndarray:
        """α₃ = ℓ(ℓ+1) / (N_ℓ × r_ℓ)"""
        return self.L2_ell / (self.N_ell * self.r_ell)
    
    def alpha_4(self) -> np.ndarray:
        """α₄ = sqrt(ℓ(ℓ+1)) / r_ℓ"""
        return np.sqrt(self.L2_ell) / self.r_ell
    
    def alpha_5(self) -> np.ndarray:
        """α₅ = ℓ(ℓ+1) / N_ℓ²"""
        return self.L2_ell / self.N_ell**2
    
    # ============================================================
    # SECTION 2: Sphere-Based Formulas
    # ============================================================
    
    def alpha_6(self) -> np.ndarray:
        """α₆ = ℓ(ℓ+1) / [4π × r_ℓ × (2ℓ+1)]"""
        return self.L2_ell / (4 * np.pi * self.r_ell * (2*self.ell_values + 1))
    
    def alpha_7(self) -> np.ndarray:
        """α₇ = ℓ(ℓ+1) / [4π × r_ℓ × sqrt(2ℓ+1)]"""
        return self.L2_ell / (4 * np.pi * self.r_ell * np.sqrt(2*self.ell_values + 1))
    
    def alpha_8(self) -> np.ndarray:
        """α₈ = ℓ(ℓ+1) / [2π × r_ℓ × (2ℓ+1)]"""
        return self.L2_ell / (2 * np.pi * self.r_ell * (2*self.ell_values + 1))
    
    def alpha_9(self) -> np.ndarray:
        """α₉ = sqrt(ℓ(ℓ+1)) / [2π × r_ℓ]"""
        return np.sqrt(self.L2_ell) / (2 * np.pi * self.r_ell)
    
    def alpha_10(self) -> np.ndarray:
        """α₁₀ = ℓ(ℓ+1) / [4π × r_ℓ²]"""
        return self.L2_ell / (4 * np.pi * self.r_ell**2)
    
    # ============================================================
    # SECTION 3: Angular Momentum Density
    # ============================================================
    
    def alpha_11(self) -> np.ndarray:
        """α₁₁ = L_density(ℓ+1) / L_density(ℓ)"""
        L_density = self.L2_ell / (2 * np.pi * self.r_ell)
        # Ratio between adjacent shells
        return L_density[1:] / L_density[:-1]
    
    def alpha_12(self) -> np.ndarray:
        """α₁₂ = L_density(ℓ) × r_ℓ / ℓ"""
        L_density = self.L2_ell / (2 * np.pi * self.r_ell)
        return L_density * self.r_ell / self.ell_values
    
    # ============================================================
    # SECTION 4: Ratios Between Adjacent Shells
    # ============================================================
    
    def alpha_13(self) -> np.ndarray:
        """α₁₃ = [r_{ℓ+1}/r_ℓ] × [ℓ(ℓ+1) / ((ℓ+1)(ℓ+2))]"""
        ell = self.ell_values[:-1]
        r_ratio = self.r_ell[1:] / self.r_ell[:-1]
        L2_ratio = (ell * (ell + 1)) / ((ell + 1) * (ell + 2))
        return r_ratio * L2_ratio
    
    def alpha_14(self) -> np.ndarray:
        """α₁₄ = [N_{ℓ+1}/N_ℓ] × [ℓ(ℓ+1) / ((ℓ+1)(ℓ+2))]"""
        ell = self.ell_values[:-1]
        N_ratio = self.N_ell[1:] / self.N_ell[:-1]
        L2_ratio = (ell * (ell + 1)) / ((ell + 1) * (ell + 2))
        return N_ratio * L2_ratio
    
    def alpha_15(self) -> np.ndarray:
        """α₁₅ = [(ℓ+1)(ℓ+2) - ℓ(ℓ+1)] / [r_{ℓ+1} - r_ℓ]"""
        ell = self.ell_values[:-1]
        L2_diff = (ell + 1) * (ell + 2) - ell * (ell + 1)
        r_diff = self.r_ell[1:] - self.r_ell[:-1]
        return L2_diff / r_diff
    
    def alpha_16(self) -> np.ndarray:
        """α₁₆ = [N_{ℓ+1} - N_ℓ] / ℓ(ℓ+1)"""
        ell = self.ell_values[:-1]
        N_diff = self.N_ell[1:] - self.N_ell[:-1]
        L2 = ell * (ell + 1)
        return N_diff / L2
    
    # ============================================================
    # SECTION 5: Volume/Area Scaling
    # ============================================================
    
    def alpha_17(self) -> np.ndarray:
        """α₁₇ = Ω(ℓ) × r_ℓ² / ℓ(ℓ+1)"""
        Omega = 2 * np.pi / (2 * self.ell_values + 1)
        return Omega * self.r_ell**2 / self.L2_ell
    
    def alpha_18(self) -> np.ndarray:
        """α₁₈ = [Ω(ℓ+1) / Ω(ℓ)] × [r_{ℓ+1} / r_ℓ]"""
        Omega = 2 * np.pi / (2 * self.ell_values + 1)
        Omega_ratio = Omega[1:] / Omega[:-1]
        r_ratio = self.r_ell[1:] / self.r_ell[:-1]
        return Omega_ratio * r_ratio
    
    # ============================================================
    # SECTION 6: Product and Quotient Combinations
    # ============================================================
    
    def alpha_19(self) -> np.ndarray:
        """α₁₉ = [ℓ(ℓ+1)]^(1/3) / r_ℓ"""
        return self.L2_ell**(1/3) / self.r_ell
    
    def alpha_20(self) -> np.ndarray:
        """α₂₀ = [ℓ(ℓ+1)]^(2/3) / r_ℓ"""
        return self.L2_ell**(2/3) / self.r_ell
    
    def alpha_21(self) -> np.ndarray:
        """α₂₁ = ℓ(ℓ+1) / [r_ℓ × sqrt(N_ℓ)]"""
        return self.L2_ell / (self.r_ell * np.sqrt(self.N_ell))
    
    def alpha_22(self) -> np.ndarray:
        """α₂₂ = ℓ(ℓ+1) / [r_ℓ^(3/2) × (2ℓ+1)]"""
        return self.L2_ell / (self.r_ell**(3/2) * (2*self.ell_values + 1))
    
    def alpha_23(self) -> np.ndarray:
        """α₂₃ = sqrt(ℓ(ℓ+1)) / [r_ℓ × (2ℓ+1)]"""
        return np.sqrt(self.L2_ell) / (self.r_ell * (2*self.ell_values + 1))
    
    def alpha_24(self) -> np.ndarray:
        """α₂₄ = ℓ(ℓ+1) / [(2ℓ+1) × sqrt(r_ℓ)]"""
        return self.L2_ell / ((2*self.ell_values + 1) * np.sqrt(self.r_ell))
    
    # ============================================================
    # SECTION 7: Logarithmic and Exponential Forms
    # ============================================================
    
    def alpha_25(self) -> np.ndarray:
        """α₂₅ = log(ℓ(ℓ+1)) / log(r_ℓ)"""
        return np.log(self.L2_ell) / np.log(self.r_ell)
    
    def alpha_26(self) -> np.ndarray:
        """α₂₆ = log(N_ℓ) / log(ℓ(ℓ+1))"""
        return np.log(self.N_ell) / np.log(self.L2_ell)
    
    def alpha_27_scale10(self) -> np.ndarray:
        """α₂₇ = [ℓ(ℓ+1) / r_ℓ] × exp(-ℓ/10)"""
        return (self.L2_ell / self.r_ell) * np.exp(-self.ell_values / 10)
    
    def alpha_27_scale50(self) -> np.ndarray:
        """α₂₇ = [ℓ(ℓ+1) / r_ℓ] × exp(-ℓ/50)"""
        return (self.L2_ell / self.r_ell) * np.exp(-self.ell_values / 50)
    
    def alpha_27_scale137(self) -> np.ndarray:
        """α₂₇ = [ℓ(ℓ+1) / r_ℓ] × exp(-ℓ/137)"""
        return (self.L2_ell / self.r_ell) * np.exp(-self.ell_values / 137)
    
    # ============================================================
    # SECTION 8: Wigner 3-j Symbol Inspired
    # ============================================================
    
    def alpha_28(self) -> np.ndarray:
        """α₂₈ = [r_{2ℓ} / (r_ℓ × r_ℓ)] × [ℓ(ℓ+1) / (2ℓ)(2ℓ+1)]"""
        # Only compute for ℓ where 2ℓ <= ell_max
        valid_ell = self.ell_values[self.ell_values * 2 <= self.ell_max]
        
        r_2ell = 1 + 2 * (2 * valid_ell)
        r_ell = 1 + 2 * valid_ell
        
        r_ratio = r_2ell / (r_ell * r_ell)
        L2_ratio = (valid_ell * (valid_ell + 1)) / ((2*valid_ell) * (2*valid_ell + 1))
        
        return r_ratio * L2_ratio
    
    # ============================================================
    # Analysis Methods
    # ============================================================
    
    def analyze_convergence(self, values: np.ndarray, name: str) -> Tuple[bool, float, float]:
        """
        Determine if sequence converges and estimate limit.
        
        Returns
        -------
        converges : bool
            Whether sequence appears to converge
        limit_value : float
            Estimated limiting value
        convergence_rate : float
            Rate of convergence (if applicable)
        """
        if len(values) < 10:
            return False, np.nan, np.nan
        
        # Use last 20% of values to estimate limit
        n_tail = max(10, len(values) // 5)
        tail_values = values[-n_tail:]
        
        # Check variation in tail
        tail_std = np.std(tail_values)
        tail_mean = np.mean(tail_values)
        
        # Convergence criteria: std < 5% of mean
        if tail_mean != 0:
            relative_std = tail_std / abs(tail_mean)
            converges = relative_std < 0.05
        else:
            converges = tail_std < 0.01
        
        limit_value = tail_mean
        
        # Estimate convergence rate from power law fit
        if converges and len(values) >= 20:
            # Fit |value - limit| ~ ℓ^(-rate)
            deviations = np.abs(values - limit_value)
            # Avoid log(0)
            valid = deviations > 1e-10
            if np.sum(valid) > 10:
                log_ell = np.log(self.ell_values[:len(values)][valid])
                log_dev = np.log(deviations[valid])
                coeffs = np.polyfit(log_ell, log_dev, 1)
                convergence_rate = -coeffs[0]
            else:
                convergence_rate = np.nan
        else:
            convergence_rate = np.nan
        
        return converges, limit_value, convergence_rate
    
    def find_best_match(self, limit_value: float) -> Tuple[str, float, float]:
        """
        Find closest known constant to the limiting value.
        
        Returns
        -------
        name : str
            Name of closest constant
        value : float
            Value of constant
        relative_error : float
            Relative error
        """
        best_name = None
        best_value = None
        best_error = np.inf
        
        for name, value in KNOWN_CONSTANTS.items():
            error = abs(limit_value - value) / abs(value) if value != 0 else abs(limit_value)
            if error < best_error:
                best_error = error
                best_name = name
                best_value = value
        
        return best_name, best_value, best_error
    
    def evaluate_formula(self, name: str, func: Callable, expression: str):
        """Evaluate a formula and store results."""
        values = func()
        
        # Handle variable-length results (for adjacent shell ratios)
        ell_for_values = self.ell_values[:len(values)]
        
        converges, limit_value, conv_rate = self.analyze_convergence(values, name)
        
        if converges:
            best_match = self.find_best_match(limit_value)
        else:
            best_match = (None, np.nan, np.nan)
        
        result = FormulaResult(
            name=name,
            expression=expression,
            values=values,
            ell_values=ell_for_values,
            converges=converges,
            limit_value=limit_value,
            convergence_rate=conv_rate,
            best_match=best_match
        )
        
        self.results[name] = result
    
    def run_all_formulas(self):
        """Evaluate all 28+ formulas."""
        print("Evaluating geometric ratio formulas...")
        
        formulas = [
            ('α₁', self.alpha_1, 'ℓ(ℓ+1) / r_ℓ²'),
            ('α₂', self.alpha_2, 'ℓ(ℓ+1) / N_ℓ'),
            ('α₃', self.alpha_3, 'ℓ(ℓ+1) / (N_ℓ × r_ℓ)'),
            ('α₄', self.alpha_4, 'sqrt(ℓ(ℓ+1)) / r_ℓ'),
            ('α₅', self.alpha_5, 'ℓ(ℓ+1) / N_ℓ²'),
            ('α₆', self.alpha_6, 'ℓ(ℓ+1) / [4π × r_ℓ × (2ℓ+1)]'),
            ('α₇', self.alpha_7, 'ℓ(ℓ+1) / [4π × r_ℓ × sqrt(2ℓ+1)]'),
            ('α₈', self.alpha_8, 'ℓ(ℓ+1) / [2π × r_ℓ × (2ℓ+1)]'),
            ('α₉', self.alpha_9, 'sqrt(ℓ(ℓ+1)) / [2π × r_ℓ]'),
            ('α₁₀', self.alpha_10, 'ℓ(ℓ+1) / [4π × r_ℓ²]'),
            ('α₁₁', self.alpha_11, 'L_density(ℓ+1) / L_density(ℓ)'),
            ('α₁₂', self.alpha_12, 'L_density(ℓ) × r_ℓ / ℓ'),
            ('α₁₃', self.alpha_13, '[r_{ℓ+1}/r_ℓ] × [ℓ(ℓ+1)/((ℓ+1)(ℓ+2))]'),
            ('α₁₄', self.alpha_14, '[N_{ℓ+1}/N_ℓ] × [ℓ(ℓ+1)/((ℓ+1)(ℓ+2))]'),
            ('α₁₅', self.alpha_15, '[(ℓ+1)(ℓ+2) - ℓ(ℓ+1)] / [r_{ℓ+1} - r_ℓ]'),
            ('α₁₆', self.alpha_16, '[N_{ℓ+1} - N_ℓ] / ℓ(ℓ+1)'),
            ('α₁₇', self.alpha_17, 'Ω(ℓ) × r_ℓ² / ℓ(ℓ+1)'),
            ('α₁₈', self.alpha_18, '[Ω(ℓ+1)/Ω(ℓ)] × [r_{ℓ+1}/r_ℓ]'),
            ('α₁₉', self.alpha_19, '[ℓ(ℓ+1)]^(1/3) / r_ℓ'),
            ('α₂₀', self.alpha_20, '[ℓ(ℓ+1)]^(2/3) / r_ℓ'),
            ('α₂₁', self.alpha_21, 'ℓ(ℓ+1) / [r_ℓ × sqrt(N_ℓ)]'),
            ('α₂₂', self.alpha_22, 'ℓ(ℓ+1) / [r_ℓ^(3/2) × (2ℓ+1)]'),
            ('α₂₃', self.alpha_23, 'sqrt(ℓ(ℓ+1)) / [r_ℓ × (2ℓ+1)]'),
            ('α₂₄', self.alpha_24, 'ℓ(ℓ+1) / [(2ℓ+1) × sqrt(r_ℓ)]'),
            ('α₂₅', self.alpha_25, 'log(ℓ(ℓ+1)) / log(r_ℓ)'),
            ('α₂₆', self.alpha_26, 'log(N_ℓ) / log(ℓ(ℓ+1))'),
            ('α₂₇(s=10)', self.alpha_27_scale10, '[ℓ(ℓ+1)/r_ℓ] × exp(-ℓ/10)'),
            ('α₂₇(s=50)', self.alpha_27_scale50, '[ℓ(ℓ+1)/r_ℓ] × exp(-ℓ/50)'),
            ('α₂₇(s=137)', self.alpha_27_scale137, '[ℓ(ℓ+1)/r_ℓ] × exp(-ℓ/137)'),
            ('α₂₈', self.alpha_28, '[r_{2ℓ}/(r_ℓ²)] × [ℓ(ℓ+1)/(2ℓ)(2ℓ+1)]'),
        ]
        
        for name, func, expr in formulas:
            self.evaluate_formula(name, func, expr)
            if self.results[name].converges:
                print(f"  {name}: CONVERGES to {self.results[name].limit_value:.6f}")
        
        print(f"\nEvaluated {len(formulas)} formulas.")
        print(f"Found {sum(r.converges for r in self.results.values())} converging sequences.")
    
    def get_top_candidates(self, n: int = 10) -> List[FormulaResult]:
        """
        Get top n most promising candidates.
        
        Ranked by:
        1. Whether it converges
        2. Match to known constants
        3. Convergence rate (faster is better)
        """
        candidates = list(self.results.values())
        
        def score(result: FormulaResult) -> Tuple[int, float, float]:
            if not result.converges:
                return (0, 1e10, 0)
            
            match_error = result.best_match[2] if result.best_match[0] else 1e10
            conv_rate = result.convergence_rate if not np.isnan(result.convergence_rate) else 0
            
            return (1, match_error, -conv_rate)
        
        candidates.sort(key=score, reverse=False)
        return candidates[:n]
    
    def print_summary_table(self, top_n: int = 10):
        """Print summary table of top candidates."""
        print("\n" + "="*120)
        print("TOP CANDIDATE GEOMETRIC RATIOS")
        print("="*120)
        print(f"{'Rank':<6} {'Formula':<15} {'Limit Value':<15} {'Best Match':<15} {'Error':<12} {'Conv Rate':<12}")
        print("-"*120)
        
        top = self.get_top_candidates(top_n)
        
        for i, result in enumerate(top, 1):
            if result.converges:
                match_name, match_val, error = result.best_match
                error_pct = error * 100
                conv_rate_str = f"{result.convergence_rate:.3f}" if not np.isnan(result.convergence_rate) else "N/A"
                
                print(f"{i:<6} {result.name:<15} {result.limit_value:<15.8f} "
                      f"{match_name:<15} {error_pct:<11.2f}% {conv_rate_str:<12}")
            else:
                print(f"{i:<6} {result.name:<15} {'NO CONVERGENCE':<15} {'':<15} {'':<12} {'':<12}")
        
        print("="*120)
    
    def plot_all_formulas(self, save_path: str = None):
        """Create comprehensive visualization of all formulas."""
        converging = [r for r in self.results.values() if r.converges]
        n_converging = len(converging)
        
        if n_converging == 0:
            print("No converging formulas to plot.")
            return
        
        # Plot converging formulas
        n_cols = min(4, n_converging)
        n_rows = (n_converging + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, result in enumerate(converging):
            ax = axes[idx]
            
            ax.plot(result.ell_values, result.values, 'b-', linewidth=2, label='Formula')
            ax.axhline(result.limit_value, color='r', linestyle='--', linewidth=2, 
                      label=f'Limit = {result.limit_value:.6f}')
            
            if result.best_match[0]:
                match_name, match_val, error = result.best_match
                if error < 0.1:  # Show if within 10%
                    ax.axhline(match_val, color='g', linestyle=':', linewidth=2,
                              label=f'{match_name} = {match_val:.6f}')
            
            ax.set_xlabel('ℓ', fontsize=10, fontweight='bold')
            ax.set_ylabel('Value', fontsize=10, fontweight='bold')
            ax.set_title(f'{result.name}: {result.expression[:40]}...', fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_converging, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot: {save_path}")
        
        return fig
    
    def plot_convergence_rates(self, save_path: str = None):
        """Plot convergence rates for all converging formulas."""
        converging = [(r.name, r.limit_value, r.values, r.ell_values) 
                     for r in self.results.values() if r.converges]
        
        if len(converging) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for name, limit, values, ell_vals in converging:
            deviation = np.abs(values - limit)
            # Avoid log(0)
            valid = deviation > 1e-12
            if np.sum(valid) > 5:
                ax.loglog(ell_vals[valid], deviation[valid], 'o-', label=name, alpha=0.7)
        
        ax.set_xlabel('ℓ', fontsize=12, fontweight='bold')
        ax.set_ylabel('|value - limit|', fontsize=12, fontweight='bold')
        ax.set_title('Convergence Rates (Log-Log Scale)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved convergence plot: {save_path}")
        
        return fig


def main():
    """Run complete geometric ratio analysis."""
    print("="*120)
    print("COMPREHENSIVE GEOMETRIC RATIO ANALYSIS")
    print("Discrete Polar Lattice: r_ℓ = 1 + 2ℓ, N_ℓ = 2(2ℓ+1), L²_ℓ = ℓ(ℓ+1)")
    print("="*120)
    
    # Initialize explorer
    explorer = GeometricRatioExplorer(ell_max=100)
    
    # Run all formulas
    explorer.run_all_formulas()
    
    # Print summary
    explorer.print_summary_table(top_n=10)
    
    # Create plots
    import os
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    print("\nGenerating plots...")
    explorer.plot_all_formulas(os.path.join(results_dir, 'geometric_ratios_overview.png'))
    explorer.plot_convergence_rates(os.path.join(results_dir, 'geometric_ratios_convergence.png'))
    
    # Save detailed results
    results_file = os.path.join(results_dir, 'geometric_ratios_detailed.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("COMPREHENSIVE GEOMETRIC RATIO ANALYSIS\n")
        f.write("="*120 + "\n\n")
        
        for name, result in sorted(explorer.results.items()):
            f.write(f"\n{name}: {result.expression}\n")
            f.write("-"*80 + "\n")
            
            # Sample values
            indices = [0, 1, 4, 9, 19, 49, min(99, len(result.values)-1)]
            indices = [i for i in indices if i < len(result.values)]
            
            f.write("Sample values:\n")
            for i in indices:
                f.write(f"  ℓ = {result.ell_values[i]:3d}: {result.values[i]:.10f}\n")
            
            f.write(f"\nConvergence: {result.converges}\n")
            if result.converges:
                f.write(f"Limit value: {result.limit_value:.10f}\n")
                f.write(f"Convergence rate: {result.convergence_rate:.4f}\n")
                if result.best_match[0]:
                    match_name, match_val, error = result.best_match
                    f.write(f"Best match: {match_name} = {match_val:.10f}\n")
                    f.write(f"Relative error: {error*100:.4f}%\n")
            f.write("\n")
    
    print(f"\nDetailed results saved: {results_file}")
    print("\n" + "="*120)
    print("ANALYSIS COMPLETE")
    print("="*120)


if __name__ == '__main__':
    main()
