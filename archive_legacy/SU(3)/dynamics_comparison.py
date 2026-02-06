# Adjoint vs Fundamental Dynamics Comparison
"""
Unified interface for evolving states in different representations and
comparing physical behavior.

Tracks:
- Color charge evolution (I₃(t), Y(t))
- Casimir scaling
- Energy and norm conservation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from typing import Tuple, Dict, Optional
from weight_basis_gellmann import WeightBasisSU3
from adjoint_tensor_product import AdjointSU3


class RepresentationDynamics:
    """
    Unified dynamics engine for different SU(3) representations.
    """
    
    def __init__(self):
        """Initialize representations."""
        self.fund = WeightBasisSU3(1, 0)
        self.antifund = WeightBasisSU3(0, 1)
        self.adj = AdjointSU3()
        
        # Import general rep builder for higher irreps
        try:
            from general_rep_builder import GeneralRepBuilder
            self.rep_builder = GeneralRepBuilder()
            self.has_general_reps = True
        except ImportError:
            self.rep_builder = None
            self.has_general_reps = False
        
        # Build Casimir operators
        self._build_casimirs()
    
    def _build_casimirs(self):
        """Build Casimir operators for each representation."""
        # Fundamental
        ops_fund = [self.fund.T3, self.fund.T8, self.fund.E12, self.fund.E21,
                   self.fund.E23, self.fund.E32, self.fund.E13, self.fund.E31]
        self.C2_fund = sum(T @ T for T in ops_fund)
        
        # Antifundamental
        ops_anti = [self.antifund.T3, self.antifund.T8, self.antifund.E12, self.antifund.E21,
                   self.antifund.E23, self.antifund.E32, self.antifund.E13, self.antifund.E31]
        self.C2_anti = sum(T @ T for T in ops_anti)
        
        # Adjoint (use Hermitian combinations)
        E12_adj = self.adj.E12
        E21_adj = self.adj.E21
        E23_adj = self.adj.E23
        E32_adj = self.adj.E32
        E13_adj = self.adj.E13
        E31_adj = self.adj.E31
        T3_adj = self.adj.T3
        T8_adj = self.adj.T8
        
        lambda1_adj = E12_adj + E21_adj
        lambda2_adj = -1j * (E12_adj - E21_adj)
        lambda3_adj = 2 * T3_adj
        lambda4_adj = E23_adj + E32_adj
        lambda5_adj = -1j * (E23_adj - E32_adj)
        lambda6_adj = E13_adj + E31_adj
        lambda7_adj = -1j * (E13_adj - E31_adj)
        lambda8_adj = 2 * T8_adj
        
        self.C2_adj = (lambda1_adj @ lambda1_adj + lambda2_adj @ lambda2_adj +
                      lambda3_adj @ lambda3_adj + lambda4_adj @ lambda4_adj +
                      lambda5_adj @ lambda5_adj + lambda6_adj @ lambda6_adj +
                      lambda7_adj @ lambda7_adj + lambda8_adj @ lambda8_adj) / 4.0
    
    def evolve_state(self, rep: str, psi0: np.ndarray, H: np.ndarray,
                    t_max: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evolve state under Hamiltonian dynamics.
        
        |ψ(t)⟩ = exp(-iHt) |ψ(0)⟩
        
        Args:
            rep: 'fundamental', 'antifundamental', 'adjoint', '6', '3bar', etc.
            psi0: Initial state
            H: Hamiltonian operator
            t_max: Maximum time
            dt: Time step
            
        Returns:
            times, states: Time array and state evolution
        """
        times = np.arange(0, t_max + dt, dt)
        n_steps = len(times)
        
        # Get dimension
        dim = len(psi0)
        states = np.zeros((n_steps, dim), dtype=complex)
        states[0] = psi0
        
        # Time evolution operator for single step
        U_step = expm(-1j * H * dt)
        
        # Evolve
        psi = psi0.copy()
        for i in range(1, n_steps):
            psi = U_step @ psi
            states[i] = psi
        
        return times, states
    
    def track_color_charges(self, rep: str, states: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Track color charge quantum numbers along trajectory.
        
        Args:
            rep: Representation name ('fundamental', 'adjoint', '6', '8', etc.)
            states: State evolution array (n_steps × dim)
            
        Returns:
            charges: Dictionary with I3, Y, C2 arrays
        """
        n_steps = states.shape[0]
        I3_vals = np.zeros(n_steps)
        Y_vals = np.zeros(n_steps)
        C2_vals = np.zeros(n_steps)
        
        # Get operators for this representation
        if rep == 'fundamental':
            T3 = self.fund.T3
            T8 = self.fund.T8
            C2 = self.C2_fund
        elif rep == 'antifundamental':
            T3 = self.antifund.T3
            T8 = self.antifund.T8
            C2 = self.C2_anti
        elif rep == 'adjoint':
            T3 = self.adj.T3
            T8 = self.adj.T8
            C2 = self.C2_adj
        elif self.has_general_reps:
            # Try to get from general rep builder
            ops = None
            if rep == '6':
                ops = self.rep_builder.get_irrep_operators(2, 0)
            elif rep == '3bar':
                ops = self.rep_builder.get_irrep_operators(0, 1)
            elif rep == '8':
                ops = self.rep_builder.get_irrep_operators(1, 1)
            elif rep == '6bar':
                ops = self.rep_builder.get_irrep_operators(0, 2)
            elif rep == '3':
                ops = self.rep_builder.get_irrep_operators(1, 0)
            
            if ops is not None:
                T3 = ops['T3']
                T8 = ops['T8']
                from irrep_operators import IrrepOperators
                irrep_ops = IrrepOperators()
                C2 = irrep_ops.compute_casimir(ops)
            else:
                raise ValueError(f"Unknown representation: {rep}")
        else:
            raise ValueError(f"Unknown representation: {rep}")
        
        # Track charges
        for i, psi in enumerate(states):
            I3_vals[i] = np.real(psi.conj() @ T3 @ psi)
            Y_vals[i] = 2/np.sqrt(3) * np.real(psi.conj() @ T8 @ psi)
            C2_vals[i] = np.real(psi.conj() @ C2 @ psi)
        
        return {'I3': I3_vals, 'Y': Y_vals, 'C2': C2_vals}
    
    def compare_casimir_scaling(self, verbose: bool = True) -> Dict[str, float]:
        """
        Compare Casimir eigenvalues across representations.
        
        Expected scaling:
        C₂(3) = 4/3  (fundamental)
        C₂(8) = 3    (adjoint)
        Ratio = 9/4 = 2.25
        
        Args:
            verbose: Print results
            
        Returns:
            casimir_values: Dictionary with eigenvalues and ratio
        """
        # Get eigenvalues
        C2_fund_eigs = np.linalg.eigvalsh(self.C2_fund)
        C2_adj_eigs = np.linalg.eigvalsh(self.C2_adj)
        
        # Take unique values (representations are irreducible, should be constant)
        C2_fund_val = np.mean(C2_fund_eigs)
        C2_adj_val = np.mean(C2_adj_eigs)
        
        ratio = C2_adj_val / C2_fund_val if C2_fund_val > 0 else 0
        expected_ratio = 9.0 / 4.0  # Standard normalization
        
        if verbose:
            print("\n" + "="*70)
            print("CASIMIR SCALING COMPARISON")
            print("="*70)
            print(f"\nFundamental (3):")
            print(f"  C₂ eigenvalues: {np.unique(np.round(C2_fund_eigs, 6))}")
            print(f"  Mean: {C2_fund_val:.6f}")
            print(f"\nAdjoint (8):")
            print(f"  C₂ eigenvalues: {np.unique(np.round(C2_adj_eigs, 6))}")
            print(f"  Mean: {C2_adj_val:.6f}")
            print(f"\nScaling:")
            print(f"  C₂(adj)/C₂(fund) = {ratio:.4f}")
            print(f"  Expected = {expected_ratio:.4f}")
            print(f"  Deviation = {abs(ratio - expected_ratio):.4f}")
        
        return {
            'C2_fund': C2_fund_val,
            'C2_adj': C2_adj_val,
            'ratio': ratio,
            'expected_ratio': expected_ratio
        }
    
    def validate_conservation_laws(self, rep: str, psi0: np.ndarray,
                                  H: np.ndarray, t_max: float = 10.0,
                                  dt: float = 0.1, verbose: bool = True) -> Dict[str, float]:
        """
        Validate norm and energy conservation during evolution.
        
        Args:
            rep: Representation
            psi0: Initial state
            H: Hamiltonian
            t_max: Evolution time
            dt: Time step
            verbose: Print results
            
        Returns:
            errors: Dictionary with maximum deviations
        """
        if verbose:
            print("\n" + "="*70)
            print(f"CONSERVATION LAWS TEST: {rep}")
            print("="*70)
        
        # Evolve
        times, states = self.evolve_state(rep, psi0, H, t_max, dt)
        
        # Check norm conservation
        norms = np.array([np.linalg.norm(psi) for psi in states])
        norm_initial = norms[0]
        max_norm_error = np.max(np.abs(norms - norm_initial))
        
        # Check energy conservation
        energies = np.array([np.real(psi.conj() @ H @ psi) for psi in states])
        energy_initial = energies[0]
        max_energy_error = np.max(np.abs(energies - energy_initial))
        
        if verbose:
            print(f"\nInitial norm: {norm_initial:.10f}")
            print(f"Final norm: {norms[-1]:.10f}")
            print(f"Max norm deviation: {max_norm_error:.2e}")
            print(f"\nInitial energy: {energy_initial:.10f}")
            print(f"Final energy: {energies[-1]:.10f}")
            print(f"Max energy deviation: {max_energy_error:.2e}")
            
            if max_norm_error < 1e-10 and max_energy_error < 1e-10:
                print("\n✓ Conservation laws VERIFIED at machine precision!")
            else:
                print("\n⚠ Warning: Conservation may be violated")
        
        return {
            'max_norm_error': max_norm_error,
            'max_energy_error': max_energy_error
        }
    
    def plot_color_trajectory(self, rep: str, I3_vals: np.ndarray,
                             Y_vals: np.ndarray, times: np.ndarray) -> plt.Figure:
        """
        Plot color charge trajectory in (I₃, Y) plane.
        
        Args:
            rep: Representation name
            I3_vals: Isospin trajectory
            Y_vals: Hypercharge trajectory
            times: Time array
            
        Returns:
            fig: Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # I3 vs time
        axes[0].plot(times, I3_vals, 'b-', lw=2)
        axes[0].set_xlabel('Time', fontsize=12)
        axes[0].set_ylabel('I₃', fontsize=12)
        axes[0].set_title(f'{rep}: Isospin Evolution', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        # Y vs time
        axes[1].plot(times, Y_vals, 'r-', lw=2)
        axes[1].set_xlabel('Time', fontsize=12)
        axes[1].set_ylabel('Y', fontsize=12)
        axes[1].set_title(f'{rep}: Hypercharge Evolution', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        # Phase space (I3, Y)
        scatter = axes[2].scatter(I3_vals, Y_vals, c=times, cmap='viridis', s=20, alpha=0.6)
        axes[2].plot(I3_vals[0], Y_vals[0], 'go', markersize=12, label='Start')
        axes[2].plot(I3_vals[-1], Y_vals[-1], 'ro', markersize=12, label='End')
        axes[2].set_xlabel('I₃', fontsize=12)
        axes[2].set_ylabel('Y', fontsize=12)
        axes[2].set_title(f'{rep}: Color Space Trajectory', fontsize=14)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[2], label='Time')
        
        plt.tight_layout()
        return fig


# ============================================================================
# VALIDATION AND DEMONSTRATION
# ============================================================================

def validate_dynamics_comparison():
    """Run full validation suite."""
    print("\n" + "="*70)
    print("REPRESENTATION DYNAMICS VALIDATION")
    print("="*70)
    
    dyn = RepresentationDynamics()
    
    # Test 1: Casimir scaling
    print("\n\nTest 1: Casimir Scaling")
    print("-"*70)
    casimir_results = dyn.compare_casimir_scaling(verbose=True)
    
    # Test 2: Conservation laws (fundamental)
    print("\n\nTest 2: Conservation Laws (Fundamental)")
    print("-"*70)
    psi0_fund = np.array([1, 0, 0], dtype=complex)
    H_fund = 0.5 * dyn.C2_fund
    cons_fund = dyn.validate_conservation_laws('fundamental', psi0_fund, H_fund,
                                              t_max=10.0, dt=0.1, verbose=True)
    
    # Test 3: Conservation laws (adjoint)
    print("\n\nTest 3: Conservation Laws (Adjoint)")
    print("-"*70)
    psi0_adj = np.zeros(8, dtype=complex)
    psi0_adj[3] = 1.0  # Center of weight diagram
    H_adj = 0.5 * dyn.C2_adj
    cons_adj = dyn.validate_conservation_laws('adjoint', psi0_adj, H_adj,
                                             t_max=10.0, dt=0.1, verbose=True)
    
    # Test 4: Color charge tracking
    print("\n\nTest 4: Color Charge Tracking")
    print("-"*70)
    times_fund, states_fund = dyn.evolve_state('fundamental', psi0_fund, H_fund, 10.0, 0.1)
    charges_fund = dyn.track_color_charges('fundamental', states_fund)
    print(f"Tracked {len(times_fund)} time steps")
    print(f"I₃ range: [{charges_fund['I3'].min():.4f}, {charges_fund['I3'].max():.4f}]")
    print(f"Y range: [{charges_fund['Y'].min():.4f}, {charges_fund['Y'].max():.4f}]")
    print(f"C₂ variation: {np.std(charges_fund['C2']):.2e}")
    
    print("\n" + "="*70)
    print("✓ ALL DYNAMICS COMPARISON TESTS COMPLETED")
    print("="*70)
    
    # Summary
    print("\n\nVALIDATION SUMMARY")
    print("="*70)
    print(f"Casimir ratio: {casimir_results['ratio']:.4f} (expected {casimir_results['expected_ratio']:.4f})")
    print(f"Fundamental norm conservation: {cons_fund['max_norm_error']:.2e}")
    print(f"Fundamental energy conservation: {cons_fund['max_energy_error']:.2e}")
    print(f"Adjoint norm conservation: {cons_adj['max_norm_error']:.2e}")
    print(f"Adjoint energy conservation: {cons_adj['max_energy_error']:.2e}")
    
    all_passed = (cons_fund['max_norm_error'] < 1e-10 and
                 cons_fund['max_energy_error'] < 1e-10 and
                 cons_adj['max_norm_error'] < 1e-10 and
                 cons_adj['max_energy_error'] < 1e-10)
    
    if all_passed:
        print("\n✓ All tests PASSED at machine precision!")
    else:
        print("\n⚠ Some tests show deviations")


if __name__ == "__main__":
    validate_dynamics_comparison()
    
    # Generate plots
    print("\n\nGenerating visualization plots...")
    dyn = RepresentationDynamics()
    
    # Fundamental dynamics
    psi0_fund = np.array([1, 0, 0], dtype=complex)
    H_fund = 0.5 * dyn.C2_fund + 0.3 * dyn.fund.T3 + 0.2 * dyn.fund.T8
    times_fund, states_fund = dyn.evolve_state('fundamental', psi0_fund, H_fund, 10.0, 0.05)
    charges_fund = dyn.track_color_charges('fundamental', states_fund)
    
    fig_fund = dyn.plot_color_trajectory('fundamental', charges_fund['I3'],
                                         charges_fund['Y'], times_fund)
    fig_fund.savefig('fundamental_dynamics.png', dpi=150, bbox_inches='tight')
    print("Saved fundamental_dynamics.png")
    
    plt.close('all')
