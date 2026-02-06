"""
Week 3: SciPy Spherical Harmonics Convergence Analysis

This script validates SU(2) lattice discretization by comparing to SciPy's
scipy.special.sph_harm (gold standard for spherical harmonics).

Key Questions:
1. Do our discrete eigenmodes match continuous spherical harmonics?
2. How does overlap improve with resolution N?
3. What convergence rate α do we observe? (error ~ O(1/N^α))
4. What resolution gives <1% error?

Approach:
- Test multiple quantum numbers (ℓ, m)
- Sweep resolutions: N = 5, 10, 20, 50, 100, 200
- Compute overlap integrals vs SciPy reference
- Fit convergence: overlap = 1 - C/N^α
- Generate publication-quality convergence plots

Author: Week 3 Implementation
Date: 2026-01-14
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm
from scipy.optimize import curve_fit
from scipy.linalg import eigh
from pathlib import Path
import json
from typing import Dict, List, Tuple
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from lattice import PolarLattice
from angular_momentum import AngularMomentumOperators
from operators import LatticeOperators
from quantum_comparison import QuantumComparison


class ScipyConvergenceAnalysis:
    """Compare SU(2) lattice discretization to SciPy spherical harmonics."""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize convergence analysis.
        
        Args:
            verbose: Print progress messages
        """
        self.verbose = verbose
        self.results = {
            'test_cases': [],
            'convergence_fits': {},
            'summary': {}
        }
        
    def compute_scipy_overlap(self, psi_discrete: np.ndarray, 
                             ell: int, m: int,
                             lattice: PolarLattice) -> float:
        """
        Compute overlap between discrete eigenmode and SciPy spherical harmonic.
        
        Overlap = |⟨ψ_discrete | Y_ℓ^m_scipy⟩|²
        
        Args:
            psi_discrete: Discrete eigenmode (normalized)
            ell: Angular momentum quantum number
            m: Magnetic quantum number
            lattice: Lattice for coordinate mapping
            
        Returns:
            Overlap magnitude squared (0 ≤ overlap ≤ 1)
        """
        # Sample SciPy spherical harmonic at lattice points
        operators = LatticeOperators(lattice)
        qc = QuantumComparison(lattice, operators)
        Y_scipy = qc.sample_spherical_harmonic(ell, m)
        
        # Normalize both (should already be normalized)
        psi_norm = psi_discrete / np.linalg.norm(psi_discrete)
        Y_norm = Y_scipy / np.linalg.norm(Y_scipy)
        
        # Compute overlap: |⟨ψ|Y⟩|²
        inner_prod = np.vdot(psi_norm, Y_norm)
        overlap = np.abs(inner_prod)**2
        
        return float(overlap)
    
    def find_eigenmode(self, lattice: PolarLattice, angular: AngularMomentumOperators,
                      ell: int, m: int) -> Tuple[np.ndarray, float]:
        """
        Find discrete eigenmode corresponding to quantum numbers (ℓ, m).
        
        Args:
            lattice: Polar lattice
            angular: Angular momentum operators
            ell: Target angular momentum
            m: Target magnetic quantum number
            
        Returns:
            (eigenmode, eigenvalue) tuple
        """
        # Build L^2 operator
        L2 = angular.build_L_squared()
        Lz = angular.build_Lz()
        
        # Find eigenmodes
        # Use only upper part of spectrum if large
        N = L2.shape[0]
        k = min(50, N-2)  # Number of eigenvalues to compute
        
        try:
            eigenvalues, eigenvectors = eigh(L2.toarray())
        except:
            # Sparse solver for large matrices
            from scipy.sparse.linalg import eigsh
            eigenvalues, eigenvectors = eigsh(L2, k=k, which='SM')
        
        # Find mode with correct quantum numbers
        expected_L2 = ell * (ell + 1)
        expected_Lz = m
        
        best_idx = None
        best_error = float('inf')
        
        for i in range(len(eigenvalues)):
            psi = eigenvectors[:, i]
            
            # Check L^2 eigenvalue
            L2_val = eigenvalues[i]
            
            # Check Lz eigenvalue
            Lz_psi = Lz @ psi
            Lz_val = np.vdot(psi, Lz_psi) / np.vdot(psi, psi)
            
            # Combined error
            L2_error = abs(L2_val - expected_L2)
            Lz_error = abs(Lz_val - expected_Lz)
            total_error = L2_error + 10*Lz_error  # Weight Lz more
            
            if total_error < best_error:
                best_error = total_error
                best_idx = i
        
        if best_idx is None:
            raise ValueError(f"Could not find mode (ℓ={ell}, m={m})")
        
        return eigenvectors[:, best_idx], eigenvalues[best_idx]
    
    def test_single_mode_convergence(self, ell: int, m: int,
                                    resolutions: List[int]) -> Dict:
        """
        Test convergence for a single (ℓ, m) mode across resolutions.
        
        Args:
            ell: Angular momentum quantum number
            m: Magnetic quantum number  
            resolutions: List of N values to test (e.g., [5, 10, 20, 50])
            
        Returns:
            Dictionary with convergence data
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Testing (ℓ={ell}, m={m}) convergence")
            print(f"{'='*70}")
        
        overlaps = []
        errors = []
        
        for N in resolutions:
            if self.verbose:
                print(f"\n  Resolution N = {N}:")
            
            try:
                # Build lattice at this resolution
                lattice = PolarLattice(n_max=N)
                angular = AngularMomentumOperators(lattice)
                
                # Find eigenmode
                psi, eigenval = self.find_eigenmode(lattice, angular, ell, m)
                
                # Compute overlap with SciPy
                overlap = self.compute_scipy_overlap(psi, ell, m, lattice)
                error = 1.0 - overlap
                
                overlaps.append(overlap)
                errors.append(error)
                
                if self.verbose:
                    print(f"    Overlap: {overlap:.6f}")
                    print(f"    Error:   {error:.6e}")
                    print(f"    L² eigenvalue: {eigenval:.4f} (expected: {ell*(ell+1)})")
                
            except Exception as e:
                if self.verbose:
                    print(f"    ❌ Failed at N={N}: {e}")
                overlaps.append(0.0)
                errors.append(1.0)
        
        # Fit convergence rate: error ~ C/N^α
        try:
            alpha, C = self.fit_power_law(resolutions, errors)
            convergence_rate = alpha
        except:
            convergence_rate = None
            C = None
        
        result = {
            'ell': ell,
            'm': m,
            'resolutions': resolutions,
            'overlaps': overlaps,
            'errors': errors,
            'convergence_rate': convergence_rate,
            'prefactor': C
        }
        
        if self.verbose and convergence_rate:
            print(f"\n  ✅ Convergence rate: O(1/N^{convergence_rate:.2f})")
        
        return result
    
    def fit_power_law(self, N_values: List[int], errors: List[float]) -> Tuple[float, float]:
        """
        Fit power law: error = C / N^α
        
        Uses log-log fit: log(error) = log(C) - α*log(N)
        
        Args:
            N_values: Resolution values
            errors: Corresponding errors
            
        Returns:
            (α, C) where error ~ C/N^α
        """
        # Filter out zeros/invalid
        valid = [(N, err) for N, err in zip(N_values, errors) if err > 1e-10]
        if len(valid) < 3:
            raise ValueError("Not enough valid data points for fitting")
        
        N_arr = np.array([x[0] for x in valid])
        err_arr = np.array([x[1] for x in valid])
        
        # Log-log fit
        log_N = np.log(N_arr)
        log_err = np.log(err_arr)
        
        # Linear fit: log(err) = b - α*log(N)
        coeffs = np.polyfit(log_N, log_err, 1)
        alpha = -coeffs[0]  # Slope gives -α
        log_C = coeffs[1]   # Intercept gives log(C)
        C = np.exp(log_C)
        
        return alpha, C
    
    def run_full_analysis(self, test_modes: List[Tuple[int, int]] = None,
                         resolutions: List[int] = None) -> Dict:
        """
        Run complete convergence analysis for multiple modes.
        
        Args:
            test_modes: List of (ℓ, m) tuples to test
                       Default: [(0,0), (1,0), (1,1), (2,0), (2,1), (2,2)]
            resolutions: List of N values to test
                        Default: [5, 10, 20, 50, 100]
                        
        Returns:
            Complete results dictionary
        """
        if test_modes is None:
            test_modes = [
                (0, 0),   # s-wave
                (1, 0),   # p-wave (m=0)
                (1, 1),   # p-wave (m=1)
                (2, 0),   # d-wave (m=0)
                (2, 1),   # d-wave (m=1)
                (2, 2),   # d-wave (m=2)
            ]
        
        if resolutions is None:
            resolutions = [5, 10, 20, 50, 100]
        
        if self.verbose:
            print("\n" + "="*70)
            print("=" + " "*16 + "SciPy CONVERGENCE ANALYSIS" + " "*26 + "=")
            print("=" + " "*68 + "=")
            print("=" + f"  Testing {len(test_modes)} modes at {len(resolutions)} resolutions" + " "*(68-len(f"  Testing {len(test_modes)} modes at {len(resolutions)} resolutions")) + "=")
            print("="*70)
        
        # Test each mode
        for ell, m in test_modes:
            result = self.test_single_mode_convergence(ell, m, resolutions)
            self.results['test_cases'].append(result)
            
            # Store convergence fit
            if result['convergence_rate']:
                key = f"({ell},{m})"
                self.results['convergence_fits'][key] = {
                    'alpha': result['convergence_rate'],
                    'C': result['prefactor']
                }
        
        # Compute summary statistics
        self.compute_summary()
        
        return self.results
    
    def compute_summary(self):
        """Compute summary statistics across all test cases."""
        if not self.results['test_cases']:
            return
        
        # Average convergence rate
        rates = [tc['convergence_rate'] for tc in self.results['test_cases'] 
                if tc['convergence_rate'] is not None]
        
        if rates:
            mean_alpha = np.mean(rates)
            std_alpha = np.std(rates)
        else:
            mean_alpha = None
            std_alpha = None
        
        # Find resolution for <1% error
        resolution_1pct = None
        for tc in self.results['test_cases']:
            for N, err in zip(tc['resolutions'], tc['errors']):
                if err < 0.01:
                    if resolution_1pct is None or N < resolution_1pct:
                        resolution_1pct = N
                    break
        
        self.results['summary'] = {
            'mean_convergence_rate': mean_alpha,
            'std_convergence_rate': std_alpha,
            'resolution_for_1pct_error': resolution_1pct,
            'n_test_cases': len(self.results['test_cases']),
            'n_successful_fits': len(rates)
        }
        
        if self.verbose:
            print("\n" + "="*70)
            print("SUMMARY")
            print("="*70)
            if mean_alpha:
                print(f"Mean convergence rate: α = {mean_alpha:.2f} ± {std_alpha:.2f}")
                print(f"Average scaling: error ~ O(1/N^{mean_alpha:.2f})")
            if resolution_1pct:
                print(f"Resolution for <1% error: N ≥ {resolution_1pct}")
            print(f"Successful fits: {len(rates)}/{len(self.results['test_cases'])}")
    
    def plot_convergence(self, save_path: str = None):
        """
        Generate publication-quality convergence plot.
        
        Args:
            save_path: Path to save figure (default: results/scipy_convergence.png)
        """
        if not self.results['test_cases']:
            print("No results to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Overlap vs Resolution
        for tc in self.results['test_cases']:
            label = f"ℓ={tc['ell']}, m={tc['m']}"
            ax1.plot(tc['resolutions'], tc['overlaps'], 'o-', label=label, markersize=6)
        
        ax1.set_xlabel('Resolution N', fontsize=12)
        ax1.set_ylabel('Overlap with SciPy', fontsize=12)
        ax1.set_title('Convergence to Continuous Spherical Harmonics', fontsize=13, weight='bold')
        ax1.legend(fontsize=9, ncol=2)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='Perfect overlap')
        
        # Plot 2: Error vs Resolution (log-log)
        for tc in self.results['test_cases']:
            if tc['convergence_rate']:
                label = f"ℓ={tc['ell']}, m={tc['m']} (α={tc['convergence_rate']:.1f})"
            else:
                label = f"ℓ={tc['ell']}, m={tc['m']}"
            
            # Plot data
            ax2.loglog(tc['resolutions'], tc['errors'], 'o-', label=label, markersize=6)
            
            # Plot fit line if available
            if tc['convergence_rate'] and tc['prefactor']:
                N_fit = np.logspace(np.log10(min(tc['resolutions'])), 
                                   np.log10(max(tc['resolutions'])), 50)
                err_fit = tc['prefactor'] / N_fit**tc['convergence_rate']
                ax2.loglog(N_fit, err_fit, '--', alpha=0.4, color=ax2.get_lines()[-1].get_color())
        
        ax2.set_xlabel('Resolution N', fontsize=12)
        ax2.set_ylabel('Error (1 - overlap)', fontsize=12)
        ax2.set_title('Convergence Rate Analysis', fontsize=13, weight='bold')
        ax2.legend(fontsize=8, ncol=1)
        ax2.grid(True, alpha=0.3, which='both')
        ax2.axhline(y=0.01, color='r', linestyle=':', alpha=0.5, label='1% error threshold')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = Path('results') / 'scipy_convergence.png'
        else:
            save_path = Path(save_path)
        
        save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if self.verbose:
            print(f"\n✅ Convergence plot saved: {save_path}")
        
        return fig
    
    def save_results(self, output_path: str = None):
        """
        Save results to JSON file.
        
        Args:
            output_path: Path to save JSON (default: results/scipy_convergence.json)
        """
        if output_path is None:
            output_path = Path('results') / 'scipy_convergence.json'
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        if self.verbose:
            print(f"✅ Results saved: {output_path}")


def main():
    """Run Week 3 convergence analysis."""
    print("\n" + "="*70)
    print("WEEK 3: SciPy Spherical Harmonics Convergence Analysis")
    print("="*70)
    print("\nValidating SU(2) lattice discretization against SciPy gold standard")
    print("This analysis demonstrates convergence to continuum limit\n")
    
    # Initialize analyzer
    analyzer = ScipyConvergenceAnalysis(verbose=True)
    
    # Define test cases
    test_modes = [
        (0, 0),   # s-wave (simplest)
        (1, 0),   # p-wave
        (1, 1),   
        (2, 0),   # d-wave
        (2, 1),
        (2, 2),
    ]
    
    # Define resolutions to test
    resolutions = [5, 10, 20, 50, 100]
    
    # Run analysis
    results = analyzer.run_full_analysis(test_modes, resolutions)
    
    # Generate plots
    analyzer.plot_convergence()
    
    # Save results
    analyzer.save_results()
    
    print("\n" + "="*70)
    print("WEEK 3 ANALYSIS COMPLETE!")
    print("="*70)
    print("\nOutputs:")
    print("  - results/scipy_convergence.json  (numerical results)")
    print("  - results/scipy_convergence.png   (convergence plots)")
    print("\nNext step: Add convergence analysis section to SU(2) paper")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
