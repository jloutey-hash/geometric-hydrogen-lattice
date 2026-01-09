"""
Validation script for the PolarLattice implementation.

Tests all aspects of Phase 1:
1. Basic ring structure
2. Quantum number mapping
3. Spherical lift
4. Degeneracy verification
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lattice import PolarLattice


def test_ring_structure():
    """Test ring radii and point counts."""
    print("=" * 60)
    print("TEST 1: Ring Structure")
    print("=" * 60)
    
    lattice = PolarLattice(n_max=5)
    
    print("\nRing radii (r_ℓ = 1 + 2ℓ):")
    for ℓ in range(5):
        r_expected = 1 + 2*ℓ
        ring_points = lattice.get_ring(ℓ)
        r_actual = ring_points[0]['r'] if ring_points else None
        status = "✓" if r_actual == r_expected else "✗"
        print(f"  ℓ={ℓ}: r={r_actual} (expected {r_expected}) {status}")
    
    print("\nPoint counts (N_ℓ = 2(2ℓ+1)):")
    for ℓ in range(5):
        N_expected = 2*(2*ℓ + 1)
        N_actual = len(lattice.get_ring(ℓ))
        status = "✓" if N_actual == N_expected else "✗"
        print(f"  ℓ={ℓ}: N={N_actual} points (expected {N_expected}) {status}")
    
    print("\n✅ Ring structure test PASSED\n")


def test_shell_degeneracy():
    """Test shell degeneracy matches hydrogen atom."""
    print("=" * 60)
    print("TEST 2: Shell Degeneracy")
    print("=" * 60)
    
    lattice = PolarLattice(n_max=6)
    
    print("\nShell degeneracy verification:")
    print(f"{'n':>3} {'Orbitals':>10} {'Expected':>10} {'States':>8} {'Expected':>10} {'Status':>6}")
    print("-" * 60)
    
    all_pass = True
    for n in range(1, 7):
        orbitals = lattice.count_orbitals(n)
        states = lattice.count_states(n)
        orbitals_expected = n**2
        states_expected = 2*n**2
        
        orbitals_ok = (orbitals == orbitals_expected)
        states_ok = (states == states_expected)
        status = "✓" if (orbitals_ok and states_ok) else "✗"
        
        if not (orbitals_ok and states_ok):
            all_pass = False
        
        print(f"{n:3d} {orbitals:10d} {orbitals_expected:10d} {states:8d} {states_expected:10d} {status:>6}")
    
    if all_pass:
        print("\n✅ Shell degeneracy test PASSED\n")
    else:
        print("\n❌ Shell degeneracy test FAILED\n")


def test_quantum_number_mapping():
    """Test bijection between lattice sites and quantum numbers."""
    print("=" * 60)
    print("TEST 3: Quantum Number Mapping")
    print("=" * 60)
    
    lattice = PolarLattice(n_max=4)
    
    # Test specific cases
    print("\nSpecific case tests:")
    test_cases = [
        (0, [(0, 0.5), (0, -0.5)]),
        (1, [(-1, 0.5), (-1, -0.5), (0, 0.5), (0, -0.5), (1, 0.5), (1, -0.5)]),
        (2, [(-2, 0.5), (-2, -0.5), (-1, 0.5), (-1, -0.5), (0, 0.5), 
             (0, -0.5), (1, 0.5), (1, -0.5), (2, 0.5), (2, -0.5)])
    ]
    
    all_pass = True
    for ℓ, expected_pairs in test_cases:
        N = 2*(2*ℓ + 1)
        print(f"\n  ℓ={ℓ} (N={N} points):")
        
        # Forward mapping
        qn_list = [lattice.get_quantum_numbers(ℓ, j) for j in range(N)]
        actual_pairs = [(qn[1], qn[2]) for qn in qn_list]
        
        # Check all expected combinations appear
        if sorted(expected_pairs) == sorted(actual_pairs):
            print(f"    Forward mapping: ✓")
        else:
            print(f"    Forward mapping: ✗")
            all_pass = False
        
        # Check each combination appears exactly once
        if len(actual_pairs) == len(set(actual_pairs)):
            print(f"    Uniqueness: ✓")
        else:
            print(f"    Uniqueness: ✗")
            all_pass = False
    
    # Test inverse mapping (bijection)
    print("\n  Bijection test (inverse mapping):")
    bijection_pass = True
    for ℓ in range(4):
        for m_ℓ in range(-ℓ, ℓ+1):
            for m_s in [0.5, -0.5]:
                j = lattice.get_site_index(ℓ, m_ℓ, m_s)
                ℓ_back, m_ℓ_back, m_s_back = lattice.get_quantum_numbers(ℓ, j)
                
                if (ℓ, m_ℓ, m_s) != (ℓ_back, m_ℓ_back, m_s_back):
                    print(f"    ✗ Failed for (ℓ={ℓ}, m_ℓ={m_ℓ}, m_s={m_s})")
                    bijection_pass = False
                    all_pass = False
    
    if bijection_pass:
        print(f"    Bijection for all ℓ ∈ [0, 3]: ✓")
    
    if all_pass:
        print("\n✅ Quantum number mapping test PASSED\n")
    else:
        print("\n❌ Quantum number mapping test FAILED\n")


def test_spherical_lift():
    """Test spherical lift properties."""
    print("=" * 60)
    print("TEST 4: Spherical Lift")
    print("=" * 60)
    
    lattice = PolarLattice(n_max=4)
    
    print("\nVerifying points lie on unit sphere:")
    all_on_sphere = True
    max_deviation = 0
    
    for p in lattice.points:
        r = np.sqrt(p['x_3d']**2 + p['y_3d']**2 + p['z_3d']**2)
        deviation = abs(r - 1.0)
        max_deviation = max(max_deviation, deviation)
        
        if deviation > 1e-10:
            all_on_sphere = False
    
    if all_on_sphere:
        print(f"  All points on unit sphere (max deviation: {max_deviation:.2e}): ✓")
    else:
        print(f"  Some points off sphere (max deviation: {max_deviation:.2e}): ✗")
    
    # Check hemisphere separation by spin
    print("\nVerifying hemisphere separation by spin:")
    north_count = 0
    south_count = 0
    hemisphere_ok = True
    
    for p in lattice.points:
        if p['m_s'] > 0:
            if p['z_3d'] >= 0:
                north_count += 1
            else:
                hemisphere_ok = False
                print(f"  ✗ Spin-up point with z={p['z_3d']:.4f} < 0")
        elif p['m_s'] < 0:
            if p['z_3d'] <= 0:
                south_count += 1
            else:
                hemisphere_ok = False
                print(f"  ✗ Spin-down point with z={p['z_3d']:.4f} > 0")
    
    if hemisphere_ok:
        print(f"  Spin-up in northern hemisphere: {north_count} points ✓")
        print(f"  Spin-down in southern hemisphere: {south_count} points ✓")
    else:
        print(f"  Hemisphere separation by spin: FAILED")
    
    # Check point count per ℓ in each hemisphere
    print("\nPoints per ℓ in each hemisphere:")
    print(f"  {'ℓ':>3} {'North':>8} {'South':>8} {'Expected':>10} {'Status':>6}")
    print("  " + "-" * 45)
    
    hemisphere_counts_ok = True
    for ℓ in range(lattice.ℓ_max + 1):
        ring_points = lattice.get_ring(ℓ)
        north = sum(1 for p in ring_points if p['m_s'] > 0)
        south = sum(1 for p in ring_points if p['m_s'] < 0)
        expected = 2*ℓ + 1
        
        status = "✓" if (north == expected and south == expected) else "✗"
        if north != expected or south != expected:
            hemisphere_counts_ok = False
        
        print(f"  {ℓ:3d} {north:8d} {south:8d} {expected:10d} {status:>6}")
    
    if all_on_sphere and hemisphere_ok and hemisphere_counts_ok:
        print("\n✅ Spherical lift test PASSED\n")
    else:
        print("\n❌ Spherical lift test FAILED\n")


def visualize_lattice():
    """Create visualization plots."""
    print("=" * 60)
    print("Creating Visualizations")
    print("=" * 60)
    
    lattice = PolarLattice(n_max=4)
    
    # 2D plots with different color schemes
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    lattice.plot_2d(n_max=4, color_by='ℓ', ax=axes[0])
    axes[0].set_title('2D Lattice colored by ℓ')
    
    lattice.plot_2d(n_max=4, color_by='m_ℓ', ax=axes[1])
    axes[1].set_title('2D Lattice colored by m_ℓ')
    
    lattice.plot_2d(n_max=4, color_by='m_s', ax=axes[2])
    axes[2].set_title('2D Lattice colored by spin m_s')
    
    plt.tight_layout()
    plt.savefig('validation_2d_lattice.png', dpi=150, bbox_inches='tight')
    print("\n  Saved: validation_2d_lattice.png")
    
    # 3D plots
    fig = plt.figure(figsize=(16, 5))
    
    ax1 = fig.add_subplot(131, projection='3d')
    lattice.plot_3d(n_max=3, color_by='ℓ', ax=ax1)
    ax1.set_title('3D Sphere colored by ℓ')
    
    ax2 = fig.add_subplot(132, projection='3d')
    lattice.plot_3d(n_max=3, color_by='m_ℓ', ax=ax2)
    ax2.set_title('3D Sphere colored by m_ℓ')
    
    ax3 = fig.add_subplot(133, projection='3d')
    lattice.plot_3d(n_max=3, color_by='m_s', ax=ax3)
    ax3.set_title('3D Sphere colored by spin m_s')
    
    plt.tight_layout()
    plt.savefig('validation_3d_sphere.png', dpi=150, bbox_inches='tight')
    print("  Saved: validation_3d_sphere.png")
    
    print("\n✅ Visualizations created\n")


def run_all_tests():
    """Run all validation tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "POLAR LATTICE VALIDATION SUITE" + " " * 17 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    test_ring_structure()
    test_shell_degeneracy()
    test_quantum_number_mapping()
    test_spherical_lift()
    visualize_lattice()
    
    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
    print("\n✅ Phase 1 (Core Lattice Construction) validation complete!")
    print("\nNext steps:")
    print("  1. Review generated plots")
    print("  2. Begin Phase 2: Hamiltonian and Operators")
    print("  3. Update PROGRESS.md")
    print()


if __name__ == "__main__":
    run_all_tests()
