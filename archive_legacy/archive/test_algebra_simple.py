"""Simplified test without Unicode"""
from test_algebra_closure import RungeLenzLattice
import numpy as np

def simple_test():
    lat = RungeLenzLattice(3)
    
    # Build angular momentum from scratch  
    from scipy.sparse import lil_matrix
    dim = len(lat.nodes)
    
    Lx = lil_matrix((dim, dim), dtype=complex)
    Ly = lil_matrix((dim, dim), dtype=complex)
    Lz = lil_matrix((dim, dim), dtype=complex)
    
    for i, (n, l, m) in enumerate(lat.nodes):
        Lz[i, i] = m
        
        # L+ and L- ladder operators
        if m + 1 <= l:
            target = (n, l, m + 1)
            if target in lat.node_index:
                j = lat.node_index[target]
                coeff = np.sqrt(l*(l+1) - m*(m+1))
                Lx[j, i] = coeff / 2
                Ly[j, i] = coeff / (2j)
        
        if m - 1 >= -l:
            target = (n, l, m - 1)
            if target in lat.node_index:
                j = lat.node_index[target]
                coeff = np.sqrt(l*(l+1) - m*(m-1))
                Lx[j, i] = coeff / 2
                Ly[j, i] = -coeff / (2j)
    
    Lx = Lx.tocsr().toarray()
    Ly = Ly.tocsr().toarray()
    Lz = Lz.tocsr().toarray()
    
    Ax = lat.Ax.toarray()
    Ay = lat.Ay.toarray()
    Az = lat.Az.toarray()
    
    print("SO(4) ALGEBRA CLOSURE TEST (ASCII version)")
    print("=" * 60)
    
    tests = [
        ("  1. [Lx, Ly] - iLz", Lx @ Ly - Ly @ Lx, 1j * Lz),
        ("  2. [Ly, Lz] - iLx", Ly @ Lz - Lz @ Ly, 1j * Lx),
        ("  3. [Lz, Lx] - iLy", Lz @ Lx - Lx @ Lz, 1j * Ly),
        ("  4. [Lx, Ax]", Lx @ Ax - Ax @ Lx, np.zeros_like(Lx)),
        ("  5. [Lx, Ay] - iAz", Lx @ Ay - Ay @ Lx, 1j * Az),
        ("  6. [Lx, Az] + iAy", Lx @ Az - Az @ Lx, -1j * Ay),
        ("  7. [Ly, Ax] + iAz", Ly @ Ax - Ax @ Ly, -1j * Az),
        ("  8. [Ly, Ay]", Ly @ Ay - Ay @ Ly, np.zeros_like(Ly)),
        ("  9. [Ly, Az] - iAx", Ly @ Az - Az @ Ly, 1j * Ax),
        (" 10. [Lz, Ax] - iAy", Lz @ Ax - Ax @ Lz, 1j * Ay),
        (" 11. [Lz, Ay] + iAx", Lz @ Ay - Ay @ Lz, -1j * Ax),
        (" 12. [Lz, Az]", Lz @ Az - Az @ Lz, np.zeros_like(Lz)),
        (" 13. [Ax, Ay] - iLz", Ax @ Ay - Ay @ Ax, 1j * Lz),  # Biedenharn-Louck convention
        (" 14. [Ay, Az] - iLx", Ay @ Az - Az @ Ay, 1j * Lx),  # Biedenharn-Louck convention
        (" 15. [Az, Ax] - iLy", Az @ Ax - Ax @ Az, 1j * Ly),  # Biedenharn-Louck convention
    ]
    
    max_error = 0.0
    passed = 0
    
    for name, comm, expected in tests:
        error = np.linalg.norm(comm - expected)
        max_error = max(max_error, error)
        status = "[PASS]" if error < 1e-14 else "[FAIL]"
        print(f"{name}: {error:.2e} {status}")
        if error < 1e-14:
            passed += 1
    
    print("=" * 60)
    print(f"RESULTS: {passed}/15 tests passed")
    print(f"Maximum error: {max_error:.2e}")
    
    if passed == 15:
        print("\n[SUCCESS] MACHINE PRECISION ACHIEVED!")
        return 0
    elif max_error < 1e-10:
        print("\n[GOOD] Sub-10^-10 precision")
        return 1
    else:
        print("\n[FAIL] Precision not achieved")
        return 2

if __name__ == '__main__':
    exit(simple_test())
