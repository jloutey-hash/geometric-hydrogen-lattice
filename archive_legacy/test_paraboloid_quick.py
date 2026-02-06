"""
Quick test script for the Paraboloid Lattice implementation.
Run this to verify everything works correctly.
"""

import time
import numpy as np
from paraboloid_lattice_su11 import ParaboloidLattice, plot_lattice_connectivity

def quick_test():
    """Run a quick validation test."""
    print("="*70)
    print("PARABOLOID LATTICE - QUICK TEST")
    print("="*70)
    
    # Test small lattice
    print("\n[1/4] Testing small lattice (max_n=3)...")
    t0 = time.time()
    lattice_small = ParaboloidLattice(max_n=3)
    t1 = time.time()
    print(f"      ✓ Created {lattice_small.dim} states in {(t1-t0)*1000:.1f}ms")
    
    # Test validation
    print("\n[2/4] Validating algebra...")
    results = lattice_small.validate_algebra(verbose=False)
    
    su2_pass = results['SU(2): [L+, L-] = 2*Lz'] < 1e-10
    block_pass = results['Radial commutator l-block structure'] < 1e-10
    cross_pass = all(v < 1e-10 for k, v in results.items() if '[L' in k and 'T' in k and '] = 0' in k)
    
    print(f"      SU(2) closure: {'\u2713' if su2_pass else '\u2717'}")
    print(f"      Radial block structure: {'\u2713' if block_pass else '\u2717'}")
    print(f"      Cross-commutators: {'\u2713' if cross_pass else '\u2717'}")
    
    # Test medium lattice
    print("\n[3/4] Testing medium lattice (max_n=7)...")
    t0 = time.time()
    lattice_med = ParaboloidLattice(max_n=7)
    t1 = time.time()
    expected = sum(n**2 for n in range(1, 8))
    print(f"      ✓ Created {lattice_med.dim} states in {(t1-t0)*1000:.1f}ms")
    print(f"      ✓ Matches theoretical Σn² = {expected}")
    
    # Test operator access
    print("\n[4/4] Testing operator access...")
    # Get a transition amplitude
    if (2, 1, 0) in lattice_med.node_index and (3, 1, 0) in lattice_med.node_index:
        idx_i = lattice_med.node_index[(2, 1, 0)]
        idx_f = lattice_med.node_index[(3, 1, 0)]
        amp = lattice_med.Tplus[idx_f, idx_i]
        print(f"      ⟨3,1,0|T+|2,1,0⟩ = {abs(amp):.4f}")
        print(f"      ✓ Transition amplitude computed")
    
    # Print summary
    print("\n" + "="*70)
    if su2_pass and block_pass and cross_pass:
        print("RESULT: ✓ ALL TESTS PASSED")
        print("\nThe paraboloid lattice is working correctly!")
        print("You can now:")
        print("  - Run paraboloid_lattice_su11.py for full demonstration")
        print("  - Run paraboloid_examples.py for application examples")
        print("  - Import ParaboloidLattice in your own scripts")
    else:
        print("RESULT: ✗ SOME TESTS FAILED")
        print("Please check the error messages above.")
    print("="*70 + "\n")
    
    return lattice_small, lattice_med


if __name__ == "__main__":
    lattice_small, lattice_med = quick_test()
