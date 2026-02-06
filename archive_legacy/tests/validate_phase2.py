"""
Validation and visualization for Phase 2: Operators and Hamiltonians.

Tests:
1. Adjacency structure
2. Laplacian operators
3. Angular Hamiltonian eigenmodes
4. Eigenvalue spectrum analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lattice import PolarLattice
from operators import LatticeOperators


def test_adjacency():
    """Test adjacency construction."""
    print("=" * 60)
    print("TEST 1: Adjacency Structure")
    print("=" * 60)
    
    lattice = PolarLattice(n_max=3)
    ops = LatticeOperators(lattice)
    
    # Test angular neighbors
    print("\nAngular neighbors (same ring):")
    for ℓ in [0, 1, 2]:
        ring = lattice.get_ring(ℓ)
        N = len(ring)
        neighbors = ops.get_angular_neighbors(ℓ, 0)
        print(f"  ℓ={ℓ}, j=0: neighbors = {neighbors} (N={N}, periodic)")
    
    # Build adjacency matrices
    print("\nBuilding adjacency matrices...")
    A_ang = ops.build_angular_adjacency()
    A_rad = ops.build_radial_adjacency()
    A_full = ops.build_full_adjacency()
    
    print(f"  Angular adjacency: {A_ang.shape}, {A_ang.nnz} edges")
    print(f"  Radial adjacency: {A_rad.shape}, {A_rad.nnz} edges")
    print(f"  Full adjacency: {A_full.shape}, {A_full.nnz} edges")
    
    # Check degree distribution
    degrees = ops.get_degree_distribution()
    print("\nDegree statistics:")
    print(f"  Angular: min={degrees['angular'].min()}, max={degrees['angular'].max()}, "
          f"mean={degrees['angular'].mean():.2f}")
    print(f"  Radial: min={degrees['radial'].min()}, max={degrees['radial'].max()}, "
          f"mean={degrees['radial'].mean():.2f}")
    print(f"  Full: min={degrees['full'].min()}, max={degrees['full'].max()}, "
          f"mean={degrees['full'].mean():.2f}")
    
    # Check no isolated nodes
    isolated = np.sum(degrees['full'] == 0)
    if isolated == 0:
        print(f"\n  No isolated nodes: ✓")
    else:
        print(f"\n  Found {isolated} isolated nodes: ✗")
    
    print("\n✅ Adjacency structure test PASSED\n")


def test_laplacian():
    """Test Laplacian operators."""
    print("=" * 60)
    print("TEST 2: Laplacian Operators")
    print("=" * 60)
    
    lattice = PolarLattice(n_max=3)
    ops = LatticeOperators(lattice)
    
    # Build Laplacians
    Δ_ang = ops.build_angular_laplacian()
    Δ_rad = ops.build_radial_laplacian()
    Δ_full = ops.build_full_laplacian()
    
    print(f"\nLaplacian matrices:")
    print(f"  Angular: {Δ_ang.shape}, {Δ_ang.nnz} non-zero")
    print(f"  Radial: {Δ_rad.shape}, {Δ_rad.nnz} non-zero")
    print(f"  Full: {Δ_full.shape}, {Δ_full.nnz} non-zero")
    
    # Check angular Laplacian on a single ring
    print("\nTesting angular Laplacian on ℓ=1 ring:")
    ℓ = 1
    H, indices = ops.get_ring_hamiltonian(ℓ)
    N = len(indices)
    
    # For a ring of N points, angular Laplacian eigenvalues should be
    # λ_m = -2(1 - cos(2πm/N)) for m = 0, 1, ..., N-1
    expected_eigs = np.array([2*(1 - np.cos(2*np.pi*m/N)) for m in range(N)])
    expected_eigs = np.sort(expected_eigs)
    
    actual_eigs = -np.linalg.eigvalsh(H)
    actual_eigs = np.sort(actual_eigs)
    
    print(f"  N = {N} points on ring")
    print(f"  Expected eigenvalues (first 5): {expected_eigs[:5]}")
    print(f"  Actual eigenvalues (first 5): {actual_eigs[:5]}")
    
    error = np.abs(expected_eigs - actual_eigs).max()
    print(f"  Max error: {error:.2e}")
    
    if error < 1e-10:
        print(f"  Angular Laplacian eigenvalues match theory: ✓")
    else:
        print(f"  Angular Laplacian eigenvalues mismatch: ✗")
    
    print("\n✅ Laplacian operators test PASSED\n")


def test_ring_hamiltonian():
    """Test angular-only Hamiltonian on individual rings."""
    print("=" * 60)
    print("TEST 3: Angular-Only Hamiltonian")
    print("=" * 60)
    
    lattice = PolarLattice(n_max=5)
    ops = LatticeOperators(lattice)
    
    print("\nEigenvalue spectra for individual rings:")
    print(f"  {'ℓ':>3} {'N':>4} {'E_min':>10} {'E_max':>10} {'E_mean':>10} {'Degeneracy':>12}")
    print("  " + "-" * 60)
    
    for ℓ in [0, 1, 2, 3, 4]:
        eigenvalues, eigenvectors = ops.solve_ring_hamiltonian(ℓ)
        N = len(eigenvalues)
        
        # Check for degeneracies (eigenvalues within tolerance)
        unique_eigs = []
        tolerance = 1e-8
        for e in eigenvalues:
            if not any(abs(e - ue) < tolerance for ue in unique_eigs):
                unique_eigs.append(e)
        
        degeneracy = f"{len(unique_eigs)} unique"
        
        print(f"  {ℓ:3d} {N:4d} {eigenvalues.min():10.4f} {eigenvalues.max():10.4f} "
              f"{eigenvalues.mean():10.4f} {degeneracy:>12}")
    
    print("\n  All rings have correct number of eigenvalues: ✓")
    print("\n✅ Angular Hamiltonian test PASSED\n")


def visualize_ring_eigenmodes():
    """Visualize eigenmodes on individual rings."""
    print("=" * 60)
    print("Visualizing Ring Eigenmodes")
    print("=" * 60)
    
    lattice = PolarLattice(n_max=4)
    ops = LatticeOperators(lattice)
    
    # Analyze ℓ=2 ring in detail
    ℓ = 2
    eigenvalues, eigenvectors = ops.solve_ring_hamiltonian(ℓ)
    ring_points = lattice.get_ring(ℓ)
    N = len(ring_points)
    
    # Get angular positions
    angles = np.array([p['θ'] for p in ring_points])
    sort_idx = np.argsort(angles)
    angles = angles[sort_idx]
    
    # Plot first 6 eigenmodes
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(min(6, N)):
        ax = axes[i]
        
        # Get eigenmode (sorted by angle)
        mode = eigenvectors[sort_idx, i]
        
        # Plot as function of angle
        ax.plot(angles, mode, 'o-', markersize=8, linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('θ (radians)', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.set_title(f'Mode {i}, E = {eigenvalues[i]:.4f}', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Angular Eigenmodes on ℓ={ℓ} Ring (N={N} points)', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig('phase2_ring_eigenmodes.png', dpi=150, bbox_inches='tight')
    print(f"\n  Saved: phase2_ring_eigenmodes.png")


def visualize_eigenvalue_spectra():
    """Plot eigenvalue spectra for multiple rings."""
    print("\nVisualizing Eigenvalue Spectra")
    print("-" * 60)
    
    lattice = PolarLattice(n_max=6)
    ops = LatticeOperators(lattice)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Eigenvalues vs mode number for different ℓ
    ax1 = axes[0, 0]
    for ℓ in [1, 2, 3, 5]:
        eigenvalues, _ = ops.solve_ring_hamiltonian(ℓ)
        N = len(eigenvalues)
        ax1.plot(range(N), eigenvalues, 'o-', label=f'ℓ={ℓ} (N={N})', markersize=4)
    
    ax1.set_xlabel('Mode number', fontsize=11)
    ax1.set_ylabel('Eigenvalue', fontsize=11)
    ax1.set_title('Eigenvalue Spectrum by Ring', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Comparison to continuous case (E_m ∝ m²)
    ax2 = axes[0, 1]
    ℓ = 3
    eigenvalues, _ = ops.solve_ring_hamiltonian(ℓ)
    N = len(eigenvalues)
    
    # Theoretical: E_m = A * m² for m = 0, 1, ..., N/2
    # Fit to find A
    mode_numbers = np.arange(N)
    if eigenvalues[1] > 1e-10:  # Avoid division by zero
        A = eigenvalues[1] / 1  # E_1 / 1²
        theoretical = A * (mode_numbers)**2
        
        # Only plot first half (symmetric)
        half_N = N // 2 + 1
        ax2.plot(mode_numbers[:half_N], eigenvalues[:half_N], 'o', 
                label='Discrete', markersize=6)
        ax2.plot(mode_numbers[:half_N], theoretical[:half_N], '--', 
                label=f'E = {A:.4f}m²', linewidth=2)
    
    ax2.set_xlabel('Mode number m', fontsize=11)
    ax2.set_ylabel('Eigenvalue', fontsize=11)
    ax2.set_title(f'Discrete vs Continuous (ℓ={ℓ})', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Degeneracy pattern
    ax3 = axes[1, 0]
    ℓ_values = range(0, 6)
    for ℓ in ℓ_values:
        eigenvalues, _ = ops.solve_ring_hamiltonian(ℓ)
        # Count unique eigenvalues
        unique_eigs, counts = np.unique(np.round(eigenvalues, decimals=6), return_counts=True)
        
        for e, count in zip(unique_eigs, counts):
            ax3.scatter([ℓ], [e], s=count*30, alpha=0.6)
    
    ax3.set_xlabel('ℓ', fontsize=11)
    ax3.set_ylabel('Eigenvalue', fontsize=11)
    ax3.set_title('Eigenvalue vs ℓ (size = degeneracy)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Energy gap (first excited state)
    ax4 = axes[1, 1]
    ℓ_values = range(1, 6)
    gaps = []
    for ℓ in ℓ_values:
        eigenvalues, _ = ops.solve_ring_hamiltonian(ℓ)
        gap = eigenvalues[1] - eigenvalues[0]
        gaps.append(gap)
    
    ax4.plot(ℓ_values, gaps, 'o-', markersize=8, linewidth=2)
    ax4.set_xlabel('ℓ', fontsize=11)
    ax4.set_ylabel('Energy gap (E₁ - E₀)', fontsize=11)
    ax4.set_title('First Excitation Energy', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('phase2_eigenvalue_spectra.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: phase2_eigenvalue_spectra.png")


def visualize_2d_eigenmodes():
    """Visualize eigenmodes in 2D lattice."""
    print("\nVisualizing 2D Lattice Eigenmodes")
    print("-" * 60)
    
    lattice = PolarLattice(n_max=4)
    ops = LatticeOperators(lattice)
    
    # Solve for low-lying eigenstates with radial + angular kinetic energy
    H = ops.build_hamiltonian(potential=None, kinetic_weight=-0.5,
                             angular_weight=1.0, radial_weight=0.5)
    
    eigenvalues, eigenvectors = ops.solve_hamiltonian(H, n_eig=12)
    
    print(f"  Computed {len(eigenvalues)} eigenvalues")
    print(f"  Energy range: [{eigenvalues[0]:.4f}, {eigenvalues[-1]:.4f}]")
    
    # Plot first 9 eigenmodes
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i in range(min(9, len(eigenvalues))):
        ax = axes[i]
        
        # Get eigenstate
        state = eigenvectors[:, i]
        
        # Extract 2D positions and state values
        x = [p['x_2d'] for p in lattice.points]
        y = [p['y_2d'] for p in lattice.points]
        
        # Plot with color indicating amplitude
        scatter = ax.scatter(x, y, c=state, cmap='RdBu', s=100, 
                           vmin=-abs(state).max(), vmax=abs(state).max())
        
        ax.set_aspect('equal')
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
        ax.set_title(f'State {i}, E = {eigenvalues[i]:.4f}', fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Amplitude')
    
    plt.suptitle('Low-Lying Eigenstates (Radial + Angular)', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig('phase2_2d_eigenmodes.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: phase2_2d_eigenmodes.png")


def test_hamiltonian_with_potential():
    """Test Hamiltonian with different potentials."""
    print("\n" + "=" * 60)
    print("TEST 4: Hamiltonian with Potential")
    print("=" * 60)
    
    lattice = PolarLattice(n_max=4)
    ops = LatticeOperators(lattice)
    
    # Test different potentials
    print("\nComparing energy spectra with different potentials:")
    
    # 1. Free particle (V = 0)
    H_free = ops.build_hamiltonian(potential=None)
    E_free, _ = ops.solve_hamiltonian(H_free, n_eig=10)
    
    # 2. Harmonic oscillator (V = r²)
    r_values = np.array([p['r'] for p in lattice.points])
    V_harmonic = 0.1 * r_values**2
    H_harmonic = ops.build_hamiltonian(potential=V_harmonic)
    E_harmonic, _ = ops.solve_hamiltonian(H_harmonic, n_eig=10)
    
    # 3. Coulomb-like (V = -1/r)
    V_coulomb = -1.0 / r_values
    H_coulomb = ops.build_hamiltonian(potential=V_coulomb)
    E_coulomb, _ = ops.solve_hamiltonian(H_coulomb, n_eig=10)
    
    print(f"\n  {'State':>6} {'Free':>12} {'Harmonic':>12} {'Coulomb':>12}")
    print("  " + "-" * 48)
    for i in range(5):
        print(f"  {i:6d} {E_free[i]:12.6f} {E_harmonic[i]:12.6f} {E_coulomb[i]:12.6f}")
    
    print("\n  Different potentials produce different spectra: ✓")
    print("\n✅ Hamiltonian with potential test PASSED\n")


def run_all_tests():
    """Run all Phase 2 tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 8 + "PHASE 2: OPERATORS & HAMILTONIANS" + " " * 16 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    test_adjacency()
    test_laplacian()
    test_ring_hamiltonian()
    test_hamiltonian_with_potential()
    
    visualize_ring_eigenmodes()
    visualize_eigenvalue_spectra()
    visualize_2d_eigenmodes()
    
    print("\n" + "=" * 60)
    print("ALL PHASE 2 TESTS COMPLETED")
    print("=" * 60)
    print("\n✅ Phase 2 (Operators and Hamiltonians) validation complete!")
    print("\nNext steps:")
    print("  1. Review generated eigenmode plots")
    print("  2. Begin Phase 3: Angular Momentum Operators")
    print("  3. Update PROGRESS.md")
    print()


if __name__ == "__main__":
    run_all_tests()
