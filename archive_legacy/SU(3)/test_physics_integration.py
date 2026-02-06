# Test Physics Integration with Higher Irreps
"""
Demonstrate physics modules working with 6, 8, and other irreps.
"""

import numpy as np
import matplotlib.pyplot as plt
from dynamics_comparison import RepresentationDynamics


def test_irrep_dynamics():
    """Test dynamics in different irreps."""
    print("\n" + "="*70)
    print("PHYSICS INTEGRATION: HIGHER IRREP DYNAMICS")
    print("="*70)
    
    dyn = RepresentationDynamics()
    
    if not dyn.has_general_reps:
        print("⚠ General representation builder not available")
        return
    
    # Test 1: Dynamics in 6 (symmetric)
    print("\n\nTest 1: Evolution in 6 (Symmetric Representation)")
    print("-"*70)
    
    # Get operators for 6
    ops_6 = dyn.rep_builder.get_irrep_operators(2, 0)
    
    # Initial state (first basis state)
    psi0_6 = np.zeros(6, dtype=complex)
    psi0_6[0] = 1.0
    
    # Hamiltonian with color charge mixing
    H_6 = 0.5 * dyn.rep_builder.irrep_ops.compute_casimir(ops_6) + \
          0.3 * ops_6['T3'] + 0.2 * ops_6['T8']
    
    # Evolve
    times, states = dyn.evolve_state('6', psi0_6, H_6, t_max=10.0, dt=0.1)
    
    # Track charges
    charges = dyn.track_color_charges('6', states)
    
    print(f"Evolved {len(times)} steps")
    print(f"I₃ range: [{charges['I3'].min():.3f}, {charges['I3'].max():.3f}]")
    print(f"Y range: [{charges['Y'].min():.3f}, {charges['Y'].max():.3f}]")
    print(f"C₂ mean: {np.mean(charges['C2']):.6f}")
    print(f"C₂ variation: {np.std(charges['C2']):.2e}")
    
    # Conservation laws
    norms = np.array([np.linalg.norm(psi) for psi in states])
    energies = np.array([np.real(psi.conj() @ H_6 @ psi) for psi in states])
    
    print(f"\nNorm conservation: max deviation = {np.max(np.abs(norms - 1.0)):.2e}")
    print(f"Energy conservation: max deviation = {np.max(np.abs(energies - energies[0])):.2e}")
    
    # Test 2: Dynamics in 8 (adjoint)
    print("\n\nTest 2: Evolution in 8 (Adjoint Representation)")
    print("-"*70)
    
    ops_8 = dyn.rep_builder.get_irrep_operators(1, 1)
    
    psi0_8 = np.zeros(8, dtype=complex)
    psi0_8[3] = 1.0  # Central state
    
    H_8 = 0.5 * dyn.rep_builder.irrep_ops.compute_casimir(ops_8) + \
          0.3 * ops_8['T3'] + 0.2 * ops_8['T8']
    
    times_8, states_8 = dyn.evolve_state('8', psi0_8, H_8, t_max=10.0, dt=0.1)
    charges_8 = dyn.track_color_charges('8', states_8)
    
    print(f"Evolved {len(times_8)} steps")
    print(f"I₃ range: [{charges_8['I3'].min():.3f}, {charges_8['I3'].max():.3f}]")
    print(f"Y range: [{charges_8['Y'].min():.3f}, {charges_8['Y'].max():.3f}]")
    print(f"C₂ mean: {np.mean(charges_8['C2']):.6f}")
    print(f"C₂ variation: {np.std(charges_8['C2']):.2e}")
    
    norms_8 = np.array([np.linalg.norm(psi) for psi in states_8])
    energies_8 = np.array([np.real(psi.conj() @ H_8 @ psi) for psi in states_8])
    
    print(f"\nNorm conservation: max deviation = {np.max(np.abs(norms_8 - 1.0)):.2e}")
    print(f"Energy conservation: max deviation = {np.max(np.abs(energies_8 - energies_8[0])):.2e}")
    
    # Test 3: Compare Casimir scaling
    print("\n\nTest 3: Casimir Scaling Comparison")
    print("-"*70)
    
    C2_fund = np.mean(charges['C2'])  # Actually testing 6, not fund
    C2_adj = np.mean(charges_8['C2'])
    
    # Expected values
    C2_6_expected = 10/3
    C2_8_expected = 3.0
    
    print(f"C₂(6): {C2_fund:.6f} (expected {C2_6_expected:.6f})")
    print(f"C₂(8): {C2_adj:.6f} (expected {C2_8_expected:.6f})")
    print(f"Ratio C₂(6)/C₂(8): {C2_fund/C2_adj:.4f} (expected {C2_6_expected/C2_8_expected:.4f})")
    
    print("\n" + "="*70)
    print("✓ PHYSICS INTEGRATION TESTS COMPLETED")
    print("="*70)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 6 representation
    axes[0, 0].plot(times, charges['I3'], 'b-', lw=2)
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('I₃')
    axes[0, 0].set_title('6: Isospin Evolution')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(times, charges['Y'], 'r-', lw=2)
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].set_title('6: Hypercharge Evolution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 8 representation
    axes[1, 0].plot(times_8, charges_8['I3'], 'b-', lw=2)
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('I₃')
    axes[1, 0].set_title('8: Isospin Evolution')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(times_8, charges_8['Y'], 'r-', lw=2)
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Y')
    axes[1, 1].set_title('8: Hypercharge Evolution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('higher_irrep_dynamics.png', dpi=150, bbox_inches='tight')
    print("\nSaved higher_irrep_dynamics.png")
    plt.close()


if __name__ == "__main__":
    test_irrep_dynamics()
