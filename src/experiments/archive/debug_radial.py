"""
Debug radial discretization - simple 1D test for hydrogen.

For ℓ=0 (s-wave), the radial Schrödinger equation is:
    -d²u/dr² + [-2/r]u = E*u    (using u = r*R)

Or in terms of R:
    -(1/2)d²R/dr² - (1/r)dR/dr + [-1/r]R = E*R

The simplest is to solve for u(r) = r*R(r):
    -(1/2)d²u/dr² - (1/r)u = E*u
    
With boundary conditions: u(0) = 0, u(∞) → 0

Ground state: E₀ = -0.5 Hartree, u₀(r) ∝ r*exp(-r)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh

def build_simple_1d_hydrogen(n_points=100, r_max=30.0):
    """
    Build 1D radial Hamiltonian for hydrogen using u(r) = r*R(r).
    
    The radial equation becomes:
        -(1/2)d²u/dr² - (1/r)u = E*u
    
    With u(0) = 0 boundary condition.
    """
    # Build radial grid INCLUDING r=0
    # For u(r) = r*R(r), we need u(0) = 0 boundary condition
    r_grid = np.linspace(0.0, r_max, n_points)
    dr = r_grid[1] - r_grid[0]
    
    print(f"Grid: {n_points} points, r ∈ [{r_grid[0]:.3f}, {r_grid[-1]:.3f}], dr = {dr:.3f}")
    
    # Build Hamiltonian (skip i=0 since u(0)=0 is enforced)
    # We solve on interior points only: r[1], r[2], ..., r[n-1]
    n_interior = n_points - 1
    H = lil_matrix((n_interior, n_interior), dtype=float)
    r_interior = r_grid[1:]
    
    for i in range(n_interior):
        r = r_interior[i]
        
        # Potential: -1/r (Coulomb)
        V = -1.0 / r
        H[i, i] += V
        
        # Kinetic: -(1/2)d²/dr²
        # At i=0 (r[1]), u(0)=0 is already enforced
        # At i=n_interior-1 (r[-1]), use boundary condition u(r_max) ≈ 0
        if i == 0:
            # First interior point: u(-1) = u(0) = 0 (boundary)
            H[i, i] += 1.0 / dr**2
            H[i, i+1] += -0.5 / dr**2
        elif i == n_interior - 1:
            # Last point: u(n+1) ≈ 0 (far boundary)
            H[i, i-1] += -0.5 / dr**2
            H[i, i] += 1.0 / dr**2
        else:
            # Interior points
            H[i, i-1] += -0.5 / dr**2
            H[i, i] += 1.0 / dr**2
            H[i, i+1] += -0.5 / dr**2
    
    return H.tocsr(), r_interior

def solve_hydrogen_1d(n_points=100, r_max=30.0, n_states=10):
    """Solve 1D radial hydrogen problem."""
    H, r_grid = build_simple_1d_hydrogen(n_points, r_max)
    
    # Solve for lowest states
    try:
        E, psi = eigsh(H, k=n_states, which='SA')
    except:
        print("Warning: eigsh failed, using dense solver")
        E, psi = np.linalg.eigh(H.toarray())
        E = E[:n_states]
        psi = psi[:, :n_states]
    
    # Sort
    idx = np.argsort(E)
    E = E[idx]
    psi = psi[:, idx]
    
    return E, psi, r_grid

def main():
    """Test different grid configurations."""
    print("="*80)
    print("1D RADIAL HYDROGEN - DEBUGGING")
    print("="*80)
    print()
    
    # Test different grid sizes
    for n_points in [50, 100, 200, 500]:
        print(f"\n{'='*80}")
        print(f"Grid: {n_points} points")
        print(f"{'='*80}")
        
        E, psi, r_grid = solve_hydrogen_1d(n_points=n_points, r_max=50.0, n_states=5)
        
        print(f"\nEnergy levels:")
        print(f"{'State':>6} {'E_computed':>15} {'E_theory':>15} {'Error %':>12}")
        print(f"{'-'*50}")
        
        # Theoretical energies
        E_theory = [-0.5, -0.125, -0.125, -0.125, -0.055556]
        n_quantum = [1, 2, 2, 2, 3]
        
        for i in range(len(E)):
            error = abs(E[i] - E_theory[i]) / abs(E_theory[i]) * 100
            print(f"{i:>6} {E[i]:>15.6f} {E_theory[i]:>15.6f} {error:>11.2f}%")
    
    # Plot the best result
    print(f"\n\n{'='*80}")
    print("PLOTTING WAVEFUNCTIONS")
    print(f"{'='*80}")
    
    E, psi, r_grid = solve_hydrogen_1d(n_points=200, r_max=50.0, n_states=3)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot u(r) = r*R(r)
    ax = axes[0]
    for i in range(3):
        u = psi[:, i]
        u = u / np.sqrt(np.trapezoid(u**2, r_grid))  # Normalize
        ax.plot(r_grid, u, label=f'n={i+1}, E={E[i]:.3f}')
    ax.set_xlabel('r (Bohr)')
    ax.set_ylabel('u(r) = r·R(r)')
    ax.set_title('Radial wavefunctions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20)
    
    # Plot probability density
    ax = axes[1]
    for i in range(3):
        u = psi[:, i]
        u = u / np.sqrt(np.trapezoid(u**2, r_grid))
        prob = u**2 / r_grid**2  # |R(r)|² = |u(r)/r|²
        ax.plot(r_grid, prob * r_grid**2 * 4*np.pi, 
                label=f'n={i+1}')
    ax.set_xlabel('r (Bohr)')
    ax.set_ylabel('4πr² |R(r)|²')
    ax.set_title('Radial probability density')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20)
    
    plt.tight_layout()
    plt.savefig('results/debug_radial_hydrogen.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved: results/debug_radial_hydrogen.png")
    
    print(f"\n{'='*80}")
    print("CONCLUSIONS:")
    print(f"{'='*80}")
    print(f"Ground state E₀ = {E[0]:.6f} (theory: -0.5)")
    print(f"Error: {abs(E[0] + 0.5) / 0.5 * 100:.2f}%")
    
    if abs(E[0] + 0.5) / 0.5 < 0.1:  # Within 10%
        print("✓ GOOD AGREEMENT!")
    elif abs(E[0] + 0.5) / 0.5 < 0.5:  # Within 50%
        print("✓ REASONABLE AGREEMENT")
    else:
        print("✗ POOR AGREEMENT - Need to fix discretization")

if __name__ == '__main__':
    main()
