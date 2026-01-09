"""
Phase 6: Large-ℓ and Continuum Limit Analysis

This module studies the convergence properties of the discrete lattice
operators as ℓ→∞ (continuum limit) and high-n Rydberg-like states.

Key analyses:
1. Convergence of discrete derivatives to continuum ∂/∂θ
2. Eigenvalue convergence for angular momentum operators
3. Scaling behavior of energy levels for large n
4. Rydberg formula validation
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, eigs
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List


class ConvergenceAnalysis:
    """Analyze convergence of discrete operators to continuum limit."""
    
    def __init__(self, lattice, operators):
        """
        Initialize convergence analysis.
        
        Parameters:
        -----------
        lattice : PolarLattice
            The polar lattice
        operators : LatticeOperators
            Lattice operators instance
        """
        self.lattice = lattice
        self.operators = operators
        self.ell_max = lattice.ℓ_max
        
    def test_derivative_convergence(self, m_test=1, ell_values=None):
        """
        Test convergence of discrete angular derivative to continuum.
        
        For a test function cos(m*theta), the angular Laplacian should give -m².
        We test how the discrete version converges as ℓ increases.
        
        Parameters:
        -----------
        m_test : int
            Test mode number (should be << ℓ)
        ell_values : list, optional
            List of ℓ values to test. If None, uses range from 2 to ell_max
            
        Returns:
        --------
        results : dict
            Dictionary with ℓ values, errors, and convergence rate
        """
        if ell_values is None:
            ell_values = list(range(2, self.ell_max + 1))
            
        errors = []
        exact_eigenvalue = -m_test**2
        
        print(f"Testing derivative convergence for test mode m={m_test}")
        print(f"Expected eigenvalue: {exact_eigenvalue}")
        
        for ell in ell_values:
            # Get indices for this ℓ shell
            indices = [i for i, p in enumerate(self.lattice.points) if p['ℓ'] == ell]
            
            if len(indices) == 0:
                continue
                
            N_ell = len(indices)
            
            # Create test function: cos(m * theta) on this ring
            # theta_j = 2*pi*j / N_ell
            test_func = np.zeros(len(self.lattice.points))
            for idx, i in enumerate(indices):
                j = self.lattice.points[i]['j']
                theta = 2 * np.pi * j / N_ell
                test_func[i] = np.cos(m_test * theta)
                
            # Apply discrete Laplacian (only angular part matters on single ring)
            # For convergence test, use angular gradient squared
            laplacian = self.operators.build_full_laplacian()
            result = laplacian @ test_func
            
            # Measure on this ring only
            result_ring = result[indices]
            test_ring = test_func[indices]
            
            # Compute effective eigenvalue: <psi|L|psi> / <psi|psi>
            numerator = np.dot(result_ring, test_ring)
            denominator = np.dot(test_ring, test_ring)
            
            if denominator > 1e-10:
                discrete_eigenvalue = numerator / denominator
                error = np.abs(discrete_eigenvalue - exact_eigenvalue)
                errors.append(error)
            else:
                errors.append(np.nan)
                
        # Fit power law: error = A / ℓ^alpha
        valid_idx = ~np.isnan(errors)
        ell_array = np.array(ell_values)[valid_idx]
        error_array = np.array(errors)[valid_idx]
        
        if len(ell_array) > 3:
            # Log-log fit
            log_ell = np.log(ell_array)
            log_error = np.log(error_array + 1e-16)  # Avoid log(0)
            coeffs = np.polyfit(log_ell, log_error, 1)
            alpha = -coeffs[0]  # Negative slope gives exponent
            A = np.exp(coeffs[1])
        else:
            alpha = np.nan
            A = np.nan
            
        return {
            'ell_values': ell_values,
            'errors': errors,
            'convergence_rate': alpha,
            'amplitude': A,
            'm_test': m_test,
            'exact_eigenvalue': exact_eigenvalue
        }
        
    def analyze_eigenvalue_convergence(self, ell_values=None, n_modes=5):
        """
        Analyze eigenvalue convergence for L² operator as ℓ increases.
        
        For each ℓ, compute lowest eigenvalues and compare to ℓ(ℓ+1).
        
        Parameters:
        -----------
        ell_values : list, optional
            ℓ values to analyze
        n_modes : int
            Number of lowest eigenvalues to compute per ℓ
            
        Returns:
        --------
        results : dict
            Eigenvalues and convergence data for each ℓ
        """
        if ell_values is None:
            ell_values = list(range(1, self.ell_max + 1))
            
        from src.angular_momentum import AngularMomentumOperators
        ang_mom = AngularMomentumOperators(self.lattice)
        L_squared = ang_mom.build_L_squared()
        
        results = {
            'ell_values': ell_values,
            'eigenvalues': {},
            'expected': {},
            'errors': {}
        }
        
        print("Analyzing eigenvalue convergence...")
        
        for ell in ell_values:
            expected = ell * (ell + 1)
            results['expected'][ell] = expected
            
            # Get indices for this ℓ
            indices = [i for i, p in enumerate(self.lattice.points) if p['ℓ'] == ell]
            
            if len(indices) < n_modes:
                continue
                
            # Extract submatrix for this ℓ shell
            L2_sub = L_squared[np.ix_(indices, indices)].toarray()
            
            # Compute eigenvalues
            eigenvalues = np.linalg.eigvalsh(L2_sub)
            eigenvalues = np.sort(eigenvalues)[:n_modes]
            
            results['eigenvalues'][ell] = eigenvalues
            
            # Error: deviation from expected ℓ(ℓ+1)
            # Most eigenvalues should be close to this
            median_eigenvalue = np.median(eigenvalues)
            error = np.abs(median_eigenvalue - expected) / expected
            results['errors'][ell] = error
            
        return results
        

class RydbergAnalysis:
    """Analyze high-n Rydberg-like states and energy scaling."""
    
    def __init__(self):
        """Initialize Rydberg analysis."""
        pass
        
    def analyze_energy_scaling(self, n_max=10):
        """
        Analyze energy level scaling E_n ∝ -1/n².
        
        Creates lattices up to n_max and computes ground state energies
        to test Rydberg formula.
        
        Parameters:
        -----------
        n_max : int
            Maximum n to analyze
            
        Returns:
        --------
        results : dict
            Energy levels, fitted parameters, and scaling analysis
        """
        from src.lattice import PolarLattice
        from src.operators import LatticeOperators
        from src.angular_momentum import AngularMomentumOperators
        
        n_values = list(range(1, n_max + 1))
        ground_energies = []
        shell_energies = {}  # Energy of highest occupied state in each shell
        
        print(f"Analyzing energy scaling for n=1 to n={n_max}")
        
        for n in n_values:
            # Create lattice up to this n
            lattice = PolarLattice(n_max=n)
            operators = LatticeOperators(lattice)
            ang_mom = AngularMomentumOperators(lattice)
            
            # Build Hamiltonian (use L² as proxy for angular kinetic energy)
            H = ang_mom.build_L_squared()
            
            # Compute lowest eigenvalues
            matrix_size = H.shape[0]
            
            if matrix_size <= 10:
                # Matrix is small, use dense solver
                eigenvalues = np.linalg.eigvalsh(H.toarray())
                eigenvalues = np.sort(eigenvalues)
            else:
                # Use sparse solver with appropriate k
                n_states = min(10, matrix_size - 2)  # Must be < N-1 for sparse eigsh
                eigenvalues, _ = eigsh(H, k=n_states, which='SA')
                eigenvalues = np.sort(eigenvalues)
            
            ground_energies.append(eigenvalues[0])
            shell_energies[n] = eigenvalues[-1]  # Highest computed energy
            
        # Fit to Rydberg formula: E_n = -A/n²
        n_array = np.array(n_values[:len(ground_energies)])
        E_array = np.array(ground_energies)
        
        # Since energies are positive (L² eigenvalues), we fit E = A/n² + B
        def rydberg_model(n, A, B):
            return A / n**2 + B
            
        try:
            popt, pcov = curve_fit(rydberg_model, n_array, E_array, p0=[2.0, 0.0])
            A_fit, B_fit = popt
            E_fit = rydberg_model(n_array, A_fit, B_fit)
        except:
            A_fit = np.nan
            B_fit = np.nan
            E_fit = E_array
            
        # Compute energy spacings
        spacings = np.diff(E_array)
        
        return {
            'n_values': n_values[:len(ground_energies)],
            'ground_energies': ground_energies,
            'shell_energies': shell_energies,
            'A_fit': A_fit,
            'B_fit': B_fit,
            'fitted_energies': E_fit,
            'spacings': spacings
        }
        
    def test_spacing_scaling(self, n_max=10):
        """
        Test if energy spacing scales as 1/n³.
        
        For Rydberg formula E_n = -A/n², the spacing is:
        ΔE_n = E_{n+1} - E_n ≈ 2A/n³
        
        Parameters:
        -----------
        n_max : int
            Maximum n to test
            
        Returns:
        --------
        results : dict
            Spacing data and power law fit
        """
        energy_data = self.analyze_energy_scaling(n_max=n_max)
        
        n_values = energy_data['n_values'][:-1]  # Exclude last (no spacing after it)
        spacings = energy_data['spacings']
        
        # Fit to spacing = A / n^alpha
        def spacing_model(n, A, alpha):
            return A / n**alpha
            
        try:
            popt, _ = curve_fit(spacing_model, n_values, np.abs(spacings), 
                               p0=[1.0, 3.0], bounds=([0, 0], [np.inf, 10]))
            A_spacing, alpha = popt
        except:
            A_spacing = np.nan
            alpha = np.nan
            
        return {
            'n_values': n_values,
            'spacings': spacings,
            'A_spacing': A_spacing,
            'alpha': alpha,
            'expected_alpha': 3.0
        }


def visualize_convergence(derivative_results, save_path=None):
    """
    Visualize derivative convergence results.
    
    Parameters:
    -----------
    derivative_results : dict
        Results from test_derivative_convergence()
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ell_values = derivative_results['ell_values']
    errors = derivative_results['errors']
    alpha = derivative_results['convergence_rate']
    
    # Left: Error vs ℓ (log-log)
    axes[0].loglog(ell_values, errors, 'o-', label='Measured error')
    
    if not np.isnan(alpha):
        ell_fit = np.array(ell_values)
        error_fit = derivative_results['amplitude'] / ell_fit**alpha
        axes[0].loglog(ell_fit, error_fit, '--', label=f'Fit: error ~ 1/l^{alpha:.2f}')
        
    axes[0].set_xlabel('l (angular momentum)', fontsize=12)
    axes[0].set_ylabel('|Discrete - Continuum|', fontsize=12)
    axes[0].set_title('Derivative Convergence', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Right: Convergence rate
    axes[1].semilogx(ell_values, errors, 'o-')
    axes[1].set_xlabel('l (angular momentum)', fontsize=12)
    axes[1].set_ylabel('Absolute Error', fontsize=12)
    axes[1].set_title(f'Error Decay (alpha = {alpha:.2f})', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig, axes


def visualize_rydberg_scaling(rydberg_results, save_path=None):
    """
    Visualize Rydberg energy scaling results.
    
    Parameters:
    -----------
    rydberg_results : dict
        Results from analyze_energy_scaling()
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    n_values = rydberg_results['n_values']
    E_measured = rydberg_results['ground_energies']
    E_fit = rydberg_results['fitted_energies']
    A_fit = rydberg_results['A_fit']
    
    # Top-left: E_n vs n
    axes[0, 0].plot(n_values, E_measured, 'o-', label='Measured', markersize=8)
    axes[0, 0].plot(n_values, E_fit, '--', label=f'Fit: E = {A_fit:.3f}/n^2', linewidth=2)
    axes[0, 0].set_xlabel('Principal quantum number n', fontsize=12)
    axes[0, 0].set_ylabel('Ground state energy', fontsize=12)
    axes[0, 0].set_title('Energy Level Scaling', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Top-right: 1/E_n vs n² (should be linear)
    axes[0, 1].plot(np.array(n_values)**2, 1/np.array(E_measured), 'o-')
    axes[0, 1].set_xlabel('n^2', fontsize=12)
    axes[0, 1].set_ylabel('1/E_n', fontsize=12)
    axes[0, 1].set_title('Rydberg Test: 1/E vs n^2', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Bottom-left: Energy spacings
    spacings = rydberg_results['spacings']
    axes[1, 0].plot(n_values[:-1], np.abs(spacings), 'o-')
    axes[1, 0].set_xlabel('n', fontsize=12)
    axes[1, 0].set_ylabel('|E_{n+1} - E_n|', fontsize=12)
    axes[1, 0].set_title('Energy Level Spacings', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Bottom-right: Relative errors
    rel_errors = np.abs((E_measured - E_fit) / E_measured) * 100
    axes[1, 1].plot(n_values, rel_errors, 'o-')
    axes[1, 1].set_xlabel('n', fontsize=12)
    axes[1, 1].set_ylabel('Relative Error (%)', fontsize=12)
    axes[1, 1].set_title('Fit Quality', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig, axes
