"""
Quick start demo for the PolarLattice.

This script demonstrates basic usage of the lattice implementation.
"""

import sys
import os
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lattice import PolarLattice


def main():
    """Run a simple demonstration of the PolarLattice."""
    print("=" * 60)
    print("POLAR LATTICE DEMO")
    print("=" * 60)
    
    # Create lattice up to n=3
    print("\nCreating lattice with n_max=3...")
    lattice = PolarLattice(n_max=3)
    print(f"  {lattice}")
    
    # Verify degeneracy
    print("\nShell degeneracy:")
    for n in range(1, 4):
        orbitals = lattice.count_orbitals(n)
        states = lattice.count_states(n)
        print(f"  Shell n={n}: {orbitals} orbitals, {states} electron states")
    
    # Show some quantum number mappings
    print("\nQuantum number mapping examples:")
    print("  ℓ=0: ", [lattice.get_quantum_numbers(0, j) for j in range(2)])
    print("  ℓ=1: ", [lattice.get_quantum_numbers(1, j) for j in range(6)])
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 2D lattice
    fig1, ax1 = lattice.plot_2d(n_max=3, color_by='ℓ')
    plt.savefig('demo_2d_lattice.png', dpi=150, bbox_inches='tight')
    print("  Saved: demo_2d_lattice.png")
    
    # 3D spherical lift
    fig2, ax2 = lattice.plot_3d(n_max=3, color_by='m_s')
    plt.savefig('demo_3d_sphere.png', dpi=150, bbox_inches='tight')
    print("  Saved: demo_3d_sphere.png")
    
    print("\n✅ Demo complete!")
    print("\nTo display plots interactively, uncomment plt.show() below:")
    print("# plt.show()")
    
    # Uncomment to show plots interactively
    # plt.show()


if __name__ == "__main__":
    main()
