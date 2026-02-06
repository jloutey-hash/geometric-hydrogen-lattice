"""
Example usage of operators and Hamiltonians.

Demonstrates Phase 2 capabilities:
- Building adjacency and Laplacian operators
- Solving ring Hamiltonians
- Comparing different potentials
- Visualizing eigenmodes
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib.pyplot as plt
from lattice import PolarLattice
from operators import LatticeOperators


def example_adjacency():
    """Demonstrate adjacency structure."""
    print("=" * 60)
    print("Example 1: Graph Adjacency")
    print("=" * 60)
    
    lattice = PolarLattice(n_max=3)
    ops = LatticeOperators(lattice)
    
    print(f"\nCreated lattice with {len(lattice.points)} points")
    
    # Show neighbors for a specific point
    ℓ, j = 1, 2
    ang_neighbors = ops.get_angular_neighbors(ℓ, j)
    rad_neighbors = ops.get_radial_neighbors(ℓ, j)
    
    print(f"\nNeighbors of point (ℓ={ℓ}, j={j}):")
    print(f"  Angular (same ring): {ang_neighbors}")
    print(f"  Radial (adjacent rings): {rad_neighbors}")
    
    # Degree distribution
    degrees = ops.get_degree_distribution()
    print(f"\nDegree statistics:")
    print(f"  Angular graph: all nodes have degree {int(degrees['angular'][0])}")
    print(f"  Full graph: degrees range from {int(degrees['full'].min())} "
          f"to {int(degrees['full'].max())}")


def example_ring_spectrum():
    """Analyze eigenvalue spectrum of individual rings."""
    print("\n" + "=" * 60)
    print("Example 2: Ring Eigenvalue Spectra")
    print("=" * 60)
    
    lattice = PolarLattice(n_max=4)
    ops = LatticeOperators(lattice)
    
    # Compare spectra for different rings
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, ℓ in enumerate([1, 2, 3, 5]):
        if ℓ > lattice.ℓ_max:
            continue
            
        eigenvalues, eigenvectors = ops.solve_ring_hamiltonian(ℓ)
        
        ax = axes[idx]
        ax.plot(eigenvalues, 'o-', markersize=8, linewidth=2)
        ax.set_xlabel('Mode number', fontsize=11)
        ax.set_ylabel('Energy', fontsize=11)
        ax.set_title(f'ℓ={ℓ} ring (N={len(eigenvalues)} points)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        print(f"\nℓ={ℓ}: {len(eigenvalues)} eigenvalues, "
              f"range [{eigenvalues.min():.3f}, {eigenvalues.max():.3f}]")
    
    plt.tight_layout()
    plt.savefig('example_ring_spectra.png', dpi=150, bbox_inches='tight')
    print("\nSaved: example_ring_spectra.png")


def example_potentials():
    """Compare Hamiltonians with different potentials."""
    print("\n" + "=" * 60)
    print("Example 3: Effect of Different Potentials")
    print("=" * 60)
    
    lattice = PolarLattice(n_max=4)
    ops = LatticeOperators(lattice)
    
    # Get radii for all points
    r_values = np.array([p['r'] for p in lattice.points])
    
    # Define potentials
    potentials = {
        'Free': None,
        'Harmonic': 0.05 * r_values**2,
        'Coulomb': -0.5 / r_values,
        'Square Well': np.where(r_values < 5, -1.0, 0.0)
    }
    
    # Compute low-lying states for each
    n_states = 8
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    print("\nLowest 8 eigenvalues:")
    print(f"  {'Potential':<15}", end='')
    for i in range(n_states):
        print(f" E{i:d}", end='')
    print()
    print("  " + "-" * 75)
    
    colors = ['blue', 'green', 'red', 'purple']
    for (name, V), color in zip(potentials.items(), colors):
        H = ops.build_hamiltonian(potential=V, kinetic_weight=-0.5)
        eigenvalues, _ = ops.solve_hamiltonian(H, n_eig=n_states)
        
        # Plot energy levels
        for i, E in enumerate(eigenvalues):
            ax1.plot([name], [E], 'o', color=color, markersize=8)
        
        # Plot spectrum
        ax2.plot(eigenvalues, 'o-', label=name, color=color, 
                markersize=8, linewidth=2)
        
        # Print values
        print(f"  {name:<15}", end='')
        for E in eigenvalues:
            print(f" {E:7.3f}", end='')
        print()
    
    ax1.set_ylabel('Energy', fontsize=12)
    ax1.set_title('Energy Levels by Potential', fontsize=13)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    ax2.set_xlabel('State number', fontsize=12)
    ax2.set_ylabel('Energy', fontsize=12)
    ax2.set_title('Eigenvalue Spectra', fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('example_potentials.png', dpi=150, bbox_inches='tight')
    print("\nSaved: example_potentials.png")


def example_eigenmode_visualization():
    """Visualize specific eigenmodes."""
    print("\n" + "=" * 60)
    print("Example 4: Eigenmode Visualization")
    print("=" * 60)
    
    lattice = PolarLattice(n_max=4)
    ops = LatticeOperators(lattice)
    
    # Solve for ground state and first few excited states
    H = ops.build_hamiltonian(potential=None, kinetic_weight=-0.5,
                             angular_weight=1.0, radial_weight=0.3)
    
    eigenvalues, eigenvectors = ops.solve_hamiltonian(H, n_eig=6)
    
    print(f"\nComputed {len(eigenvalues)} eigenstates")
    print(f"Ground state energy: E₀ = {eigenvalues[0]:.6f}")
    print(f"First excitation gap: ΔE = {eigenvalues[1] - eigenvalues[0]:.6f}")
    
    # Plot in 2D
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(6):
        ax = axes[i]
        state = eigenvectors[:, i]
        
        # Get positions
        x = [p['x_2d'] for p in lattice.points]
        y = [p['y_2d'] for p in lattice.points]
        
        # Plot
        vmax = np.abs(state).max()
        scatter = ax.scatter(x, y, c=state, cmap='RdBu', s=150,
                           vmin=-vmax, vmax=vmax, edgecolors='black', linewidth=0.5)
        
        ax.set_aspect('equal')
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
        ax.set_title(f'State {i}, E = {eigenvalues[i]:.4f}', fontsize=11)
        plt.colorbar(scatter, ax=ax, label='ψ')
    
    plt.suptitle('Eigenstates of Full Hamiltonian', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig('example_eigenmodes.png', dpi=150, bbox_inches='tight')
    print("\nSaved: example_eigenmodes.png")


def example_ring_modes():
    """Show angular eigenmodes on a single ring."""
    print("\n" + "=" * 60)
    print("Example 5: Angular Eigenmodes on Ring")
    print("=" * 60)
    
    lattice = PolarLattice(n_max=3)
    ops = LatticeOperators(lattice)
    
    ℓ = 2
    eigenvalues, eigenvectors = ops.solve_ring_hamiltonian(ℓ)
    ring_points = lattice.get_ring(ℓ)
    
    print(f"\nAnalyzing ℓ={ℓ} ring with N={len(ring_points)} points")
    
    # Sort by angle for plotting
    angles = np.array([p['θ'] for p in ring_points])
    sort_idx = np.argsort(angles)
    angles_sorted = angles[sort_idx]
    
    # Plot modes
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i in range(min(6, len(eigenvalues))):
        ax = axes[i]
        mode = eigenvectors[sort_idx, i]
        
        ax.plot(angles_sorted, mode, 'o-', markersize=10, linewidth=2.5)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('θ (radians)', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.set_title(f'Mode {i}, E = {eigenvalues[i]:.3f}', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Count nodes (zero crossings)
        nodes = np.sum(np.diff(np.sign(mode)) != 0)
        ax.text(0.05, 0.95, f'{nodes} nodes', transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'Angular Eigenmodes on ℓ={ℓ} Ring', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig('example_ring_modes.png', dpi=150, bbox_inches='tight')
    print("\nSaved: example_ring_modes.png")
    
    print(f"\nFirst 6 eigenvalues: {eigenvalues[:6]}")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 12 + "PHASE 2 OPERATOR EXAMPLES" + " " * 20 + "║")
    print("╚" + "=" * 58 + "╝")
    
    example_adjacency()
    example_ring_spectrum()
    example_ring_modes()
    example_potentials()
    example_eigenmode_visualization()
    
    print("\n" + "=" * 60)
    print("All examples complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - example_ring_spectra.png")
    print("  - example_ring_modes.png")
    print("  - example_potentials.png")
    print("  - example_eigenmodes.png")
    print()


if __name__ == "__main__":
    main()
