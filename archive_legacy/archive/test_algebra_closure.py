"""
Precision Test: SO(4) Algebra Closure for Runge-Lenz Operators

This module verifies that the Runge-Lenz vector operators achieve
MACHINE PRECISION closure of the SO(4) algebra.

Target: All commutator errors < 10^-14

Reference: Biedenharn & Louck, "Angular Momentum in Quantum Physics"
"""

import numpy as np
import sys
from paraboloid_relativistic import RungeLenzLattice


def test_so4_commutators(max_n=3, verbose=True):
    """
    Test all SO(4) commutation relations with machine precision.
    
    The SO(4) algebra for hydrogen:
        [L_i, L_j] = iε_{ijk} L_k        (SU(2) angular momentum)
        [L_i, A_j] = iε_{ijk} A_k        (L rotates A as a vector)
        [A_i, A_j] = -iε_{ijk} L_k       (A algebra, minus sign for bound states)
    
    Parameters:
    -----------
    max_n : int
        Maximum principal quantum number
    verbose : bool
        Print detailed results
    
    Returns:
    --------
    all_errors : dict
        Dictionary of commutator errors
    """
    if verbose:
        print("="*70)
        print("SO(4) ALGEBRA CLOSURE TEST")
        print("="*70)
        print(f"Testing Runge-Lenz operators for n <= {max_n}")
        print(f"Target precision: < 10^-14 (machine epsilon)")
        print()
    
    # Build lattice
    lattice = RungeLenzLattice(max_n=max_n)
    
    # Extract operators as dense arrays for precision testing
    Lx = ((lattice.Lplus + lattice.Lminus) / 2.0).toarray()
    Ly = ((lattice.Lplus - lattice.Lminus) / (2.0j)).toarray()
    Lz = lattice.Lz.toarray()
    
    Ax = lattice.Ax.toarray()
    Ay = lattice.Ay.toarray()
    Az = lattice.Az.toarray()
    
    all_errors = {}
    
    if verbose:
        print("--- Commutation Relations ---\n")
    
    # Test 1: [Lx, Ly] = iLz (SU(2) check - should already be perfect)
    comm_LxLy = Lx @ Ly - Ly @ Lx
    expected_LxLy = 1j * Lz
    error_LxLy = np.linalg.norm(comm_LxLy - expected_LxLy)
    all_errors['[Lx,Ly]-iLz'] = error_LxLy
    
    if verbose:
        print(f"1. [Lx, Ly] - i·Lz")
        print(f"   Error: {error_LxLy:.2e}")
        status = "✓ PASS" if error_LxLy < 1e-14 else "[X] FAIL"
        print(f"   Status: {status}\n")
    
    # Test 2: [Ly, Lz] = iLx
    comm_LyLz = Ly @ Lz - Lz @ Ly
    expected_LyLz = 1j * Lx
    error_LyLz = np.linalg.norm(comm_LyLz - expected_LyLz)
    all_errors['[Ly,Lz]-iLx'] = error_LyLz
    
    if verbose:
        print(f"2. [Ly, Lz] - i·Lx")
        print(f"   Error: {error_LyLz:.2e}")
        status = "✓ PASS" if error_LyLz < 1e-14 else "[X] FAIL"
        print(f"   Status: {status}\n")
    
    # Test 3: [Lz, Lx] = iLy
    comm_LzLx = Lz @ Lx - Lx @ Lz
    expected_LzLx = 1j * Ly
    error_LzLx = np.linalg.norm(comm_LzLx - expected_LzLx)
    all_errors['[Lz,Lx]-iLy'] = error_LzLx
    
    if verbose:
        print(f"3. [Lz, Lx] - i·Ly")
        print(f"   Error: {error_LzLx:.2e}")
        status = "✓ PASS" if error_LzLx < 1e-14 else "[X] FAIL"
        print(f"   Status: {status}\n")
    
    # Test 4: [Lx, Ax] = 0 (L_x commutes with A_x)
    comm_LxAx = Lx @ Ax - Ax @ Lx
    error_LxAx = np.linalg.norm(comm_LxAx)
    all_errors['[Lx,Ax]'] = error_LxAx
    
    if verbose:
        print(f"4. [Lx, Ax] = 0")
        print(f"   Error: {error_LxAx:.2e}")
        status = "✓ PASS" if error_LxAx < 1e-14 else "[X] FAIL"
        print(f"   Status: {status}\n")
    
    # Test 5: [Lx, Ay] = iAz
    comm_LxAy = Lx @ Ay - Ay @ Lx
    expected_LxAy = 1j * Az
    error_LxAy = np.linalg.norm(comm_LxAy - expected_LxAy)
    all_errors['[Lx,Ay]-iAz'] = error_LxAy
    
    if verbose:
        print(f"5. [Lx, Ay] - i·Az")
        print(f"   Error: {error_LxAy:.2e}")
        status = "✓ PASS" if error_LxAy < 1e-14 else "[X] FAIL"
        print(f"   Status: {status}\n")
    
    # Test 6: [Lx, Az] = -iAy
    comm_LxAz = Lx @ Az - Az @ Lx
    expected_LxAz = -1j * Ay
    error_LxAz = np.linalg.norm(comm_LxAz - expected_LxAz)
    all_errors['[Lx,Az]+iAy'] = error_LxAz
    
    if verbose:
        print(f"6. [Lx, Az] + i·Ay")
        print(f"   Error: {error_LxAz:.2e}")
        status = "✓ PASS" if error_LxAz < 1e-14 else "[X] FAIL"
        print(f"   Status: {status}\n")
    
    # Test 7: [Ly, Ax] = -iAz
    comm_LyAx = Ly @ Ax - Ax @ Ly
    expected_LyAx = -1j * Az
    error_LyAx = np.linalg.norm(comm_LyAx - expected_LyAx)
    all_errors['[Ly,Ax]+iAz'] = error_LyAx
    
    if verbose:
        print(f"7. [Ly, Ax] + i·Az")
        print(f"   Error: {error_LyAx:.2e}")
        status = "✓ PASS" if error_LyAx < 1e-14 else "[X] FAIL"
        print(f"   Status: {status}\n")
    
    # Test 8: [Ly, Ay] = 0
    comm_LyAy = Ly @ Ay - Ay @ Ly
    error_LyAy = np.linalg.norm(comm_LyAy)
    all_errors['[Ly,Ay]'] = error_LyAy
    
    if verbose:
        print(f"8. [Ly, Ay] = 0")
        print(f"   Error: {error_LyAy:.2e}")
        status = "✓ PASS" if error_LyAy < 1e-14 else "[X] FAIL"
        print(f"   Status: {status}\n")
    
    # Test 9: [Ly, Az] = iAx
    comm_LyAz = Ly @ Az - Az @ Ly
    expected_LyAz = 1j * Ax
    error_LyAz = np.linalg.norm(comm_LyAz - expected_LyAz)
    all_errors['[Ly,Az]-iAx'] = error_LyAz
    
    if verbose:
        print(f"9. [Ly, Az] - i·Ax")
        print(f"   Error: {error_LyAz:.2e}")
        status = "✓ PASS" if error_LyAz < 1e-14 else "[X] FAIL"
        print(f"   Status: {status}\n")
    
    # Test 10: [Lz, Ax] = iAy (CRITICAL TEST)
    comm_LzAx = Lz @ Ax - Ax @ Lz
    expected_LzAx = 1j * Ay
    error_LzAx = np.linalg.norm(comm_LzAx - expected_LzAx)
    all_errors['[Lz,Ax]-iAy'] = error_LzAx
    
    if verbose:
        print(f"10. [Lz, Ax] - i·Ay")
        print(f"    Error: {error_LzAx:.2e}")
        status = "✓ PASS" if error_LzAx < 1e-14 else "[X] FAIL"
        print(f"    Status: {status}\n")
    
    # Test 11: [Lz, Ay] = -iAx (CRITICAL TEST)
    comm_LzAy = Lz @ Ay - Ay @ Lz
    expected_LzAy = -1j * Ax
    error_LzAy = np.linalg.norm(comm_LzAy - expected_LzAy)
    all_errors['[Lz,Ay]+iAx'] = error_LzAy
    
    if verbose:
        print(f"11. [Lz, Ay] + i·Ax")
        print(f"    Error: {error_LzAy:.2e}")
        status = "✓ PASS" if error_LzAy < 1e-14 else "[X] FAIL"
        print(f"    Status: {status}\n")
    
    # Test 12: [Lz, Az] = 0
    comm_LzAz = Lz @ Az - Az @ Lz
    error_LzAz = np.linalg.norm(comm_LzAz)
    all_errors['[Lz,Az]'] = error_LzAz
    
    if verbose:
        print(f"12. [Lz, Az] = 0")
        print(f"    Error: {error_LzAz:.2e}")
        status = "✓ PASS" if error_LzAz < 1e-14 else "[X] FAIL"
        print(f"    Status: {status}\n")
    
    # Test 13: [Ax, Ay] = -iLz (THE CRITICAL TEST - was failing!)
    comm_AxAy = Ax @ Ay - Ay @ Ax
    expected_AxAy = -1j * Lz
    error_AxAy = np.linalg.norm(comm_AxAy - expected_AxAy)
    all_errors['[Ax,Ay]+iLz'] = error_AxAy
    
    if verbose:
        print(f"13. [Ax, Ay] + i·Lz  **CRITICAL**")
        print(f"    Error: {error_AxAy:.2e}")
        status = "✓ PASS" if error_AxAy < 1e-14 else "[X] FAIL"
        print(f"    Status: {status}\n")
    
    # Test 14: [Ay, Az] = -iLx
    comm_AyAz = Ay @ Az - Az @ Ay
    expected_AyAz = -1j * Lx
    error_AyAz = np.linalg.norm(comm_AyAz - expected_AyAz)
    all_errors['[Ay,Az]+iLx'] = error_AyAz
    
    if verbose:
        print(f"14. [Ay, Az] + i·Lx")
        print(f"    Error: {error_AyAz:.2e}")
        status = "✓ PASS" if error_AyAz < 1e-14 else "[X] FAIL"
        print(f"    Status: {status}\n")
    
    # Test 15: [Az, Ax] = -iLy
    comm_AzAx = Az @ Ax - Ax @ Az
    expected_AzAx = -1j * Ly
    error_AzAx = np.linalg.norm(comm_AzAx - expected_AzAx)
    all_errors['[Az,Ax]+iLy'] = error_AzAx
    
    if verbose:
        print(f"15. [Az, Ax] + i·Ly")
        print(f"    Error: {error_AzAx:.2e}")
        status = "✓ PASS" if error_AzAx < 1e-14 else "[X] FAIL"
        print(f"    Status: {status}\n")
    
    return all_errors


def test_casimir_invariant(max_n=3, verbose=True):
    """
    Test the Casimir invariant: L² + A² = n² - 1
    
    This must be EXACTLY diagonal with eigenvalues n²-1 on each n-shell.
    """
    if verbose:
        print("="*70)
        print("CASIMIR INVARIANT TEST: L² + A² = n² - 1")
        print("="*70)
    
    lattice = RungeLenzLattice(max_n=max_n)
    
    # Compute L²
    Lx = ((lattice.Lplus + lattice.Lminus) / 2.0).toarray()
    Ly = ((lattice.Lplus - lattice.Lminus) / (2.0j)).toarray()
    Lz = lattice.Lz.toarray()
    L2 = Lx @ Lx + Ly @ Ly + Lz @ Lz
    
    # Compute A²
    Ax = lattice.Ax.toarray()
    Ay = lattice.Ay.toarray()
    Az = lattice.Az.toarray()
    A2 = Ax @ Ax + Ay @ Ay + Az @ Az
    
    # Casimir
    Casimir = L2 + A2
    
    # Check diagonality
    off_diagonal = Casimir - np.diag(np.diag(Casimir))
    off_diag_norm = np.linalg.norm(off_diagonal)
    
    if verbose:
        print(f"\nOff-diagonal norm: {off_diag_norm:.2e}")
        if off_diag_norm < 1e-14:
            print("✓ Casimir is diagonal to machine precision\n")
        else:
            print("[X] Casimir has off-diagonal elements\n")
    
    # Check eigenvalues
    diagonal = np.diag(Casimir).real
    
    if verbose:
        print("Diagonal values vs expected (n²-1):\n")
    
    max_error = 0.0
    for i, (n, l, m) in enumerate(lattice.nodes):
        expected = n**2 - 1
        actual = diagonal[i]
        error = abs(actual - expected)
        max_error = max(max_error, error)
        
        if verbose and (error > 1e-14 or i < 5):  # Show first few and errors
            print(f"  Node (n={n}, l={l}, m={m:+d}): "
                  f"Casimir = {actual:.6f}, expected = {expected}, "
                  f"error = {error:.2e}")
    
    if verbose:
        print(f"\nMaximum Casimir error: {max_error:.2e}")
        if max_error < 1e-14:
            print("✓ All Casimir eigenvalues correct to machine precision")
        else:
            print(f"[X] Casimir errors exceed machine precision")
    
    return {'off_diagonal': off_diag_norm, 'max_eigenvalue_error': max_error}


def main():
    """Run comprehensive SO(4) algebra tests."""
    print("\n" + "="*70)
    print("COMPREHENSIVE SO(4) ALGEBRA VERIFICATION")
    print("Biedenharn-Louck Normalized Runge-Lenz Operators")
    print("="*70 + "\n")
    
    # Test commutators
    commutator_errors = test_so4_commutators(max_n=3, verbose=True)
    
    print("\n")
    
    # Test Casimir
    casimir_errors = test_casimir_invariant(max_n=3, verbose=True)
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    max_comm_error = max(commutator_errors.values())
    print(f"\nMaximum commutator error: {max_comm_error:.2e}")
    print(f"Casimir off-diagonal:     {casimir_errors['off_diagonal']:.2e}")
    print(f"Casimir eigenvalue error: {casimir_errors['max_eigenvalue_error']:.2e}")
    
    # Overall verdict
    print("\n" + "-"*70)
    if max_comm_error < 1e-14 and casimir_errors['max_eigenvalue_error'] < 1e-14:
        print("✓✓✓ MACHINE PRECISION ACHIEVED ✓✓✓")
        print("SO(4) algebra closes to numerical epsilon!")
        return 0
    elif max_comm_error < 1e-10:
        print("✓ Excellent precision achieved (< 10^-10)")
        print("Suitable for all practical calculations")
        return 0
    elif max_comm_error < 1e-6:
        print("⚠ Good precision (< 10^-6)")
        print("Further refinement recommended")
        return 1
    else:
        print("[X] PRECISION INSUFFICIENT")
        print("Algebra closure not achieved")
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
