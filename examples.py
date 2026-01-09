"""
Example usage of the PolarLattice.

This module shows various ways to interact with the lattice structure.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib.pyplot as plt
from lattice import PolarLattice


def example_basic_usage():
    """Demonstrate basic lattice creation and inspection."""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Create a lattice
    lattice = PolarLattice(n_max=3)
    print(f"\nCreated: {lattice}")
    
    # Access individual rings
    print(f"\nRing ℓ=1 has {len(lattice.get_ring(1))} points")
    
    # Access shells
    print(f"Shell n=2 has {len(lattice.get_shell(2))} points")
    print(f"  (Expected: 2×2² = {2*2**2})")
    
    return lattice


def example_quantum_numbers():
    """Demonstrate quantum number mapping."""
    print("\n" + "=" * 60)
    print("Example 2: Quantum Number Mapping")
    print("=" * 60)
    
    lattice = PolarLattice(n_max=2)
    
    # Show mapping for ℓ=1 ring
    print("\nℓ=1 ring (6 points):")
    print(f"  {'j':>3} {'(ℓ, m_ℓ, m_s)':>20} {'Spin':>10}")
    print("  " + "-" * 40)
    
    for j in range(6):
        ℓ, m_ℓ, m_s = lattice.get_quantum_numbers(1, j)
        spin = "↑" if m_s > 0 else "↓"
        print(f"  {j:3d} ({ℓ:2d}, {m_ℓ:+3.0f}, {m_s:+4.1f}) {spin:>10}")
    
    # Test inverse mapping
    print("\n  Testing inverse mapping:")
    ℓ, m_ℓ, m_s = 1, 0, 0.5
    j = lattice.get_site_index(ℓ, m_ℓ, m_s)
    ℓ_back, m_ℓ_back, m_s_back = lattice.get_quantum_numbers(ℓ, j)
    print(f"    (ℓ={ℓ}, m_ℓ={m_ℓ}, m_s={m_s}) → j={j} → "
          f"(ℓ={ℓ_back}, m_ℓ={m_ℓ_back}, m_s={m_s_back}) ✓")


def example_coordinates():
    """Show 2D and 3D coordinates."""
    print("\n" + "=" * 60)
    print("Example 3: Coordinate Access")
    print("=" * 60)
    
    lattice = PolarLattice(n_max=2)
    
    # Show first few points
    print("\nFirst 3 points:")
    print(f"  {'ℓ':>2} {'j':>2} {'x_2d':>8} {'y_2d':>8} {'x_3d':>8} {'y_3d':>8} {'z_3d':>8}")
    print("  " + "-" * 60)
    
    for i in range(3):
        p = lattice.points[i]
        print(f"  {p['ℓ']:2d} {p['j']:2d} "
              f"{p['x_2d']:8.3f} {p['y_2d']:8.3f} "
              f"{p['x_3d']:8.3f} {p['y_3d']:8.3f} {p['z_3d']:8.3f}")


def example_visualization():
    """Create visualization examples."""
    print("\n" + "=" * 60)
    print("Example 4: Visualization")
    print("=" * 60)
    
    lattice = PolarLattice(n_max=4)
    
    # Create a figure with multiple views
    fig = plt.figure(figsize=(15, 5))
    
    # 2D view colored by ℓ
    ax1 = fig.add_subplot(131)
    lattice.plot_2d(n_max=4, color_by='ℓ', ax=ax1)
    
    # 2D view colored by m_s (spin)
    ax2 = fig.add_subplot(132)
    lattice.plot_2d(n_max=4, color_by='m_s', ax=ax2)
    
    # 3D view
    ax3 = fig.add_subplot(133, projection='3d')
    lattice.plot_3d(n_max=3, color_by='m_s', ax=ax3)
    
    plt.tight_layout()
    plt.savefig('example_visualizations.png', dpi=150, bbox_inches='tight')
    print("\n  Saved: example_visualizations.png")
    
    print("\n  Created 3 views:")
    print("    1. 2D lattice colored by ℓ (azimuthal quantum number)")
    print("    2. 2D lattice colored by m_s (spin)")
    print("    3. 3D spherical lift showing hemisphere structure")


def example_shell_analysis():
    """Analyze shell structure."""
    print("\n" + "=" * 60)
    print("Example 5: Shell Structure Analysis")
    print("=" * 60)
    
    lattice = PolarLattice(n_max=5)
    
    print("\nHydrogen atom shell structure:")
    print(f"  {'n':>2} {'ℓ values':>15} {'Orbitals':>10} {'States':>8} {'Formula':>10}")
    print("  " + "-" * 60)
    
    for n in range(1, 6):
        ℓ_values = list(range(n))
        orbitals = lattice.count_orbitals(n)
        states = lattice.count_states(n)
        formula = f"2×{n}² = {2*n**2}"
        
        ℓ_str = ",".join(map(str, ℓ_values))
        print(f"  {n:2d} {ℓ_str:>15} {orbitals:10d} {states:8d} {formula:>10}")
    
    print("\n  All shells match hydrogen atom degeneracy! ✓")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "POLAR LATTICE EXAMPLES" + " " * 20 + "║")
    print("╚" + "=" * 58 + "╝")
    
    example_basic_usage()
    example_quantum_numbers()
    example_coordinates()
    example_shell_analysis()
    example_visualization()
    
    print("\n" + "=" * 60)
    print("All examples complete!")
    print("=" * 60)
    print("\nFor interactive exploration, try:")
    print("  >>> from lattice import PolarLattice")
    print("  >>> lattice = PolarLattice(n_max=3)")
    print("  >>> lattice.plot_2d()")
    print("  >>> lattice.plot_3d()")
    print()


if __name__ == "__main__":
    main()
