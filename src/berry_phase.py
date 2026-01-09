"""
Berry Phase Calculation on Discrete SU(2) Lattice

Computes geometric (Berry) phases for parallel transport of quantum states
around closed loops on the discrete angular momentum lattice.

Key test: Does the accumulated phase involve 4π (giving 1/(4π) normalization)?

Phase 9.3 Implementation
"""

import numpy as np
from scipy.linalg import expm, eigh
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.append('src')
from lattice import PolarLattice
from angular_momentum import AngularMomentumOperators


class BerryPhaseCalculator:
    """
    Calculate Berry phases on discrete angular momentum lattice.
    
    Berry phase: γ = i∮⟨ψ|∇|ψ⟩·dr
    
    For adiabatic transport around a closed loop, this measures
    the geometric phase accumulated by the quantum state.
    """
    
    def __init__(self, ell_max: int = 10):
        """
        Initialize Berry phase calculator.
        
        Parameters:
        -----------
        ell_max : int
            Maximum angular momentum quantum number
        """
        self.ell_max = ell_max
        
        # Build lattice
        self.lattice = PolarLattice(ell_max)
        
        # Compute N per ring
        self.N_per_ring = [2 * (2 * ell + 1) for ell in range(ell_max + 1)]
        
        # Build angular momentum operators
        self.operators = AngularMomentumOperators(self.lattice)
        
        # Get L² operator and compute eigenstates
        L2 = self.operators.build_L_squared()
        self.L2_eigenvalues, self.L2_eigenstates = np.linalg.eigh(L2.toarray())
        
        print(f"Berry Phase Calculator initialized:")
        print(f"  ℓ_max = {ell_max}")
        print(f"  Number of eigenstates: {len(self.L2_eigenvalues)}")
        print(f"  Lattice points per shell ℓ: {self.N_per_ring[:5]}...")
    
    def get_state_on_ring(self, ell: int, state_idx: int) -> np.ndarray:
        """
        Extract quantum state on a specific ℓ-ring.
        
        Parameters:
        -----------
        ell : int
            Ring quantum number
        state_idx : int
            Index of eigenstate
        
        Returns:
        --------
        psi_ring : np.ndarray
            Wavefunction restricted to ring ℓ
        """
        # Get full state
        psi_full = self.L2_eigenstates[:, state_idx]
        
        # Extract ring portion
        ring_start = sum(self.N_per_ring[:ell])
        ring_end = ring_start + self.N_per_ring[ell]
        
        psi_ring = psi_full[ring_start:ring_end]
        
        # Normalize
        norm = np.linalg.norm(psi_ring)
        if norm > 1e-10:
            psi_ring = psi_ring / norm
        
        return psi_ring
    
    def berry_connection_angular(self, ell: int, state_idx: int) -> np.ndarray:
        """
        Compute Berry connection along angular direction on ring ℓ.
        
        A_φ = i⟨ψ|∂/∂φ|ψ⟩
        
        Parameters:
        -----------
        ell : int
            Ring quantum number
        state_idx : int
            Index of eigenstate
        
        Returns:
        --------
        A_phi : np.ndarray
            Berry connection at each point on ring
        """
        psi = self.get_state_on_ring(ell, state_idx)
        N = len(psi)
        
        if N < 2:
            return np.array([0.0])
        
        # Discrete angular derivative: ∂ψ/∂φ ≈ (ψ(φ+Δφ) - ψ(φ))/Δφ
        dphi = 2 * np.pi / N
        
        A_phi = np.zeros(N)
        for j in range(N):
            j_next = (j + 1) % N
            
            # Finite difference derivative
            dpsi_dphi = (psi[j_next] - psi[j]) / dphi
            
            # Berry connection: A = i⟨ψ|∂ψ⟩
            A_phi[j] = np.imag(np.conj(psi[j]) * dpsi_dphi)
        
        return A_phi
    
    def berry_phase_around_ring(self, ell: int, state_idx: int) -> float:
        """
        Compute Berry phase for transport around ring ℓ.
        
        γ = ∮ A_φ dφ
        
        Parameters:
        -----------
        ell : int
            Ring quantum number
        state_idx : int
            Index of eigenstate
        
        Returns:
        --------
        gamma : float
            Berry phase (in radians)
        """
        A_phi = self.berry_connection_angular(ell, state_idx)
        N = len(A_phi)
        
        if N < 2:
            return 0.0
        
        # Integrate around ring
        dphi = 2 * np.pi / N
        gamma = np.sum(A_phi) * dphi
        
        return gamma
    
    def berry_phase_hemisphere(self, state_idx: int, use_north: bool = True) -> float:
        """
        Compute total Berry phase over hemisphere.
        
        Sum Berry phases from all rings in northern or southern hemisphere.
        
        Parameters:
        -----------
        state_idx : int
            Index of eigenstate
        use_north : bool
            If True, use northern hemisphere; else southern
        
        Returns:
        --------
        gamma_total : float
            Total Berry phase over hemisphere
        """
        gamma_total = 0.0
        
        # Integrate over hemisphere
        n_rings = self.ell_max + 1
        
        if use_north:
            # Northern hemisphere: rings 0 to ell_max//2
            rings = range(0, (n_rings + 1) // 2)
        else:
            # Southern hemisphere: rings ell_max//2 to ell_max
            rings = range(n_rings // 2, n_rings)
        
        for ell in rings:
            gamma_ring = self.berry_phase_around_ring(ell, state_idx)
            gamma_total += gamma_ring
        
        return gamma_total
    
    def berry_curvature_at_point(self, ell: int, j: int, state_idx: int) -> float:
        """
        Compute Berry curvature at a lattice point.
        
        F = ∂A_θ/∂φ - ∂A_φ/∂θ (gauge-invariant)
        
        Parameters:
        -----------
        ell : int
            Ring quantum number
        j : int
            Angular index on ring
        state_idx : int
            Index of eigenstate
        
        Returns:
        --------
        F : float
            Berry curvature
        """
        # For now, approximate using angular connection only
        A_phi = self.berry_connection_angular(ell, state_idx)
        
        if len(A_phi) < 2:
            return 0.0
        
        # Estimate curvature from variation of A_φ
        N = len(A_phi)
        j_prev = (j - 1) % N
        j_next = (j + 1) % N
        
        dA_dphi = (A_phi[j_next] - A_phi[j_prev]) / (2 * 2 * np.pi / N)
        
        return dA_dphi
    
    def chern_number(self, state_idx: int) -> float:
        """
        Compute Chern number (topological invariant).
        
        C = (1/2π) ∫∫ F dA
        
        Parameters:
        -----------
        state_idx : int
            Index of eigenstate
        
        Returns:
        --------
        C : float
            Chern number (should be integer for topological states)
        """
        C = 0.0
        
        for ell in range(self.ell_max + 1):
            N = self.N_per_ring[ell]
            
            for j in range(N):
                F = self.berry_curvature_at_point(ell, j, state_idx)
                
                # Area element (approximate)
                dA = (2 * np.pi / N) * (1.0)  # dphi * dr_element
                
                C += F * dA
        
        C = C / (2 * np.pi)
        
        return C
    
    def analyze_all_states(self, n_states: int = 20) -> Dict:
        """
        Analyze Berry phases for multiple eigenstates.
        
        Parameters:
        -----------
        n_states : int
            Number of states to analyze
        
        Returns:
        --------
        analysis : dict
            Berry phase data for all states
        """
        n_states = min(n_states, len(self.L2_eigenvalues))
        
        results = {
            'state_indices': [],
            'L2_values': [],
            'phases_north': [],
            'phases_south': [],
            'phases_total': [],
            'chern_numbers': []
        }
        
        print(f"Analyzing Berry phases for {n_states} states...")
        
        for i in range(n_states):
            # Get L² value
            L2_val = self.L2_eigenvalues[i]
            
            # Compute phases
            gamma_north = self.berry_phase_hemisphere(i, use_north=True)
            gamma_south = self.berry_phase_hemisphere(i, use_north=False)
            gamma_total = gamma_north + gamma_south
            
            # Chern number
            C = self.chern_number(i)
            
            results['state_indices'].append(i)
            results['L2_values'].append(L2_val)
            results['phases_north'].append(gamma_north)
            results['phases_south'].append(gamma_south)
            results['phases_total'].append(gamma_total)
            results['chern_numbers'].append(C)
            
            if (i + 1) % 5 == 0:
                print(f"  Processed {i+1}/{n_states} states...")
        
        # Convert to arrays
        for key in results:
            results[key] = np.array(results[key])
        
        return results
    
    def test_4pi_hypothesis(self, results: Dict) -> Dict:
        """
        Test if Berry phases involve 4π factor.
        
        Hypothesis: Total phase around sphere = n × 2π
        where n relates to 4π through: phase/(2π) = integer
        
        Parameters:
        -----------
        results : dict
            Output from analyze_all_states
        
        Returns:
        --------
        analysis : dict
            Statistical analysis of 4π hypothesis
        """
        phases = results['phases_total']
        
        # Test various scalings
        one_over_4pi = 1 / (4 * np.pi)
        
        # Model 1: phase = n × 2π
        n_values_2pi = phases / (2 * np.pi)
        
        # Model 2: phase = n × π
        n_values_pi = phases / np.pi
        
        # Model 3: phase = n × 4π
        n_values_4pi = phases / (4 * np.pi)
        
        # Check how close to integers
        def closeness_to_integer(values):
            return np.mean(np.abs(values - np.round(values)))
        
        close_2pi = closeness_to_integer(n_values_2pi)
        close_pi = closeness_to_integer(n_values_pi)
        close_4pi = closeness_to_integer(n_values_4pi)
        
        # Determine best model
        models = {
            '2π': close_2pi,
            'π': close_pi,
            '4π': close_4pi
        }
        
        best_model = min(models.keys(), key=lambda k: models[k])
        
        analysis = {
            'one_over_4pi': one_over_4pi,
            'n_values_2pi': n_values_2pi,
            'n_values_pi': n_values_pi,
            'n_values_4pi': n_values_4pi,
            'closeness_2pi': close_2pi,
            'closeness_pi': close_pi,
            'closeness_4pi': close_4pi,
            'best_model': best_model,
            'mean_phase': np.mean(phases),
            'std_phase': np.std(phases)
        }
        
        return analysis
    
    def plot_berry_phases(self, results: Dict, save: bool = True):
        """
        Plot Berry phase analysis results.
        
        Parameters:
        -----------
        results : dict
            Output from analyze_all_states
        save : bool
            Whether to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Panel 1: Phases vs state index
        ax = axes[0, 0]
        ax.plot(results['state_indices'], results['phases_north'], 'bo-', 
                label='Northern hemisphere', alpha=0.7)
        ax.plot(results['state_indices'], results['phases_south'], 'rs-', 
                label='Southern hemisphere', alpha=0.7)
        ax.plot(results['state_indices'], results['phases_total'], 'g^-', 
                label='Total', linewidth=2, markersize=8)
        ax.axhline(2*np.pi, color='purple', linestyle='--', linewidth=2, 
                   label='2π', alpha=0.7)
        ax.axhline(4*np.pi, color='orange', linestyle='--', linewidth=2, 
                   label='4π', alpha=0.7)
        ax.set_xlabel('State index', fontsize=11)
        ax.set_ylabel('Berry phase (radians)', fontsize=11)
        ax.set_title('Berry Phases Around Lattice Loops', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Phase vs L² eigenvalue
        ax = axes[0, 1]
        ax.scatter(results['L2_values'], results['phases_total'], 
                   c=results['state_indices'], cmap='viridis', s=80, alpha=0.7)
        ax.axhline(2*np.pi, color='purple', linestyle='--', linewidth=2, label='2π')
        ax.axhline(4*np.pi, color='orange', linestyle='--', linewidth=2, label='4π')
        ax.set_xlabel('L² eigenvalue', fontsize=11)
        ax.set_ylabel('Total Berry phase (radians)', fontsize=11)
        ax.set_title('Phase vs Angular Momentum', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('State index', fontsize=9)
        
        # Panel 3: Histogram of phases/2π
        ax = axes[1, 0]
        n_2pi = results['phases_total'] / (2 * np.pi)
        ax.hist(n_2pi, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='n=1')
        ax.axvline(2.0, color='red', linestyle='--', linewidth=2, label='n=2')
        ax.set_xlabel('Phase / (2π)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Distribution: Testing Quantization', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Panel 4: Chern numbers
        ax = axes[1, 1]
        ax.plot(results['state_indices'], results['chern_numbers'], 'ko-', 
                linewidth=2, markersize=6)
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax.axhline(1, color='red', linestyle='--', linewidth=2, label='C=1')
        ax.axhline(-1, color='blue', linestyle='--', linewidth=2, label='C=-1')
        ax.set_xlabel('State index', fontsize=11)
        ax.set_ylabel('Chern number', fontsize=11)
        ax.set_title('Topological Invariant (Chern Number)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig('results/berry_phase_analysis.png', dpi=150, bbox_inches='tight')
            print("Saved: results/berry_phase_analysis.png")
        
        plt.show()
        return fig
    
    def generate_report(self, results: Dict, analysis: Dict, 
                       filename: str = 'results/berry_phase_report.txt'):
        """
        Generate comprehensive text report.
        
        Parameters:
        -----------
        results : dict
            Output from analyze_all_states
        analysis : dict
            Output from test_4pi_hypothesis
        filename : str
            Output filename
        """
        with open(filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BERRY PHASE CALCULATION ON DISCRETE SU(2) LATTICE - PHASE 9.3\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("LATTICE PARAMETERS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Maximum ℓ: {self.ell_max}\n")
            f.write(f"Number of states analyzed: {len(results['state_indices'])}\n\n")
            
            f.write("BERRY PHASE RESULTS\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'State':>5} {'L²':>10} {'γ_north':>12} {'γ_south':>12} {'γ_total':>12} {'Chern':>10}\n")
            f.write("-" * 80 + "\n")
            
            for i in range(len(results['state_indices'])):
                f.write(f"{results['state_indices'][i]:5d} "
                       f"{results['L2_values'][i]:10.3f} "
                       f"{results['phases_north'][i]:12.6f} "
                       f"{results['phases_south'][i]:12.6f} "
                       f"{results['phases_total'][i]:12.6f} "
                       f"{results['chern_numbers'][i]:10.6f}\n")
            
            f.write("\n")
            
            f.write("TESTING 4π HYPOTHESIS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Reference: 1/(4π) = {analysis['one_over_4pi']:.10f}\n\n")
            
            f.write("Phase quantization test:\n")
            f.write(f"  Phase/(2π) closeness to integer: {analysis['closeness_2pi']:.6f}\n")
            f.write(f"  Phase/π closeness to integer:    {analysis['closeness_pi']:.6f}\n")
            f.write(f"  Phase/(4π) closeness to integer: {analysis['closeness_4pi']:.6f}\n\n")
            
            f.write(f"Best model: Phases quantized in units of {analysis['best_model']}\n\n")
            
            f.write(f"Mean total phase: {analysis['mean_phase']:.6f} radians\n")
            f.write(f"Std deviation:    {analysis['std_phase']:.6f} radians\n")
            f.write(f"Mean / (2π):      {analysis['mean_phase']/(2*np.pi):.6f}\n")
            f.write(f"Mean / (4π):      {analysis['mean_phase']/(4*np.pi):.6f}\n\n")
            
            f.write("INTERPRETATION\n")
            f.write("-" * 80 + "\n")
            f.write("Berry phase measures geometric phase accumulated by quantum states\n")
            f.write("during parallel transport around closed loops on the lattice.\n\n")
            
            f.write("If phases quantize in units of 4π, this suggests the geometric\n")
            f.write("constant 1/(4π) appears as a normalization factor.\n\n")
            
            f.write("Connection to Phase 8 discovery:\n")
            f.write("  α₉ → 1/(4π) from geometric ratios\n")
            f.write("  Berry phase ~ 4π suggests same geometric origin\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("END REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"Saved: {filename}")


def main():
    """Main execution: Berry phase calculation."""
    print("=" * 80)
    print("PHASE 9.3: BERRY PHASE ON DISCRETE LATTICE")
    print("=" * 80)
    print()
    
    # Create calculator
    berry = BerryPhaseCalculator(ell_max=10)
    print()
    
    # Analyze states
    print("Computing Berry phases...")
    results = berry.analyze_all_states(n_states=20)
    print()
    
    # Test 4π hypothesis
    print("Testing 4π hypothesis...")
    analysis = berry.test_4pi_hypothesis(results)
    print()
    
    print("Berry Phase Summary:")
    print("-" * 80)
    print(f"Mean total phase: {analysis['mean_phase']:.6f} radians")
    print(f"Mean / (2π):      {analysis['mean_phase']/(2*np.pi):.6f}")
    print(f"Mean / (4π):      {analysis['mean_phase']/(4*np.pi):.6f}")
    print()
    print(f"Best quantization: {analysis['best_model']}")
    print(f"Closeness (2π):    {analysis['closeness_2pi']:.6f}")
    print(f"Closeness (π):     {analysis['closeness_pi']:.6f}")
    print(f"Closeness (4π):    {analysis['closeness_4pi']:.6f}")
    print()
    
    # Interpretation
    if analysis['best_model'] == '4π':
        print("✓✓✓ STRONG EVIDENCE: Phases quantize in units of 4π!")
        print("    This suggests 1/(4π) is the natural normalization factor.")
    elif analysis['best_model'] == '2π':
        print("✓✓ MODERATE EVIDENCE: Phases quantize in units of 2π")
        print("   (standard quantum mechanics)")
    else:
        print("✓ Phases quantize in units of π")
    
    print()
    
    # Generate visualizations
    print("Generating plots...")
    berry.plot_berry_phases(results)
    print()
    
    # Generate report
    print("Generating report...")
    berry.generate_report(results, analysis)
    print()
    
    print("=" * 80)
    print("PHASE 9.3 COMPLETE!")
    print("=" * 80)
    print()
    print("Key findings:")
    print("  • Berry phases computed for discrete lattice")
    print("  • Quantization pattern analyzed")
    print(f"  • Best model: {analysis['best_model']} quantization")
    print("  • Results saved to results/")
    print()
    print("Next: Review plots and determine if 4π emerges from geometry!")


if __name__ == '__main__':
    main()
