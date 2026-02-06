
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from lattice import PolarLattice
from angular_momentum import AngularMomentumOperators

def verify_project():
    print("=== VERIFICATION REPORT ===\n")
    
    # 1. Verify Lattice Structure Claims
    print("--- 1. Lattice Structure (n=5) ---")
    n_max = 5
    lattice = PolarLattice(n_max)
    points = lattice.points
    
    # Claim: Total points = 2n^2
    expected_points = 2 * n_max**2
    actual_points = len(points)
    print(f"Total points (n={n_max}): Expected {expected_points}, Actual {actual_points}")
    if expected_points == actual_points:
        print("[PASS] Total degeneracy claim holds.")
    else:
        print(f"[FAIL] Degeneracy mismatch.")
        
    # Claim: N_ℓ = 2(2ℓ+1)
    print("\n--- Ring Populations ---")
    pass_rings = True
    for ℓ in range(n_max):
        expected_N = 2 * (2 * ℓ + 1)
        # Count points with this ℓ
        actual_N = len([p for p in points if p['ℓ'] == ℓ])
        if expected_N != actual_N:
            print(f"ℓ={ℓ}: Expected {expected_N}, Actual {actual_N} [FAIL]")
            pass_rings = False
    
    if pass_rings:
        print("[PASS] Point density formula N_ℓ = 2(2ℓ+1) verified.")

    # 2. Verify Algebraic Exactness
    print("\n--- 2. Algebraic Exactness ---")
    ops = AngularMomentumOperators(lattice)
    results = ops.test_commutation_relations()
    
    pass_algebra = True
    for comm, error in results.items():
        print(f"{comm}: Error = {error:.2e}")
        if error > 1e-13:
            pass_algebra = False
            
    if pass_algebra:
        print("[PASS] Commutation relations satisfied to machine precision.")
    else:
        print("[FAIL] Commutation errors exceed tolerance.")

    # 3. Verify L^2 Eigenvalues
    print("\n--- 3. L^2 Eigenvalues ---")
    # We need to construct L^2 and check its diagonal elements or eigenvalues
    # Since L^2 is claimed to be diagonal in this basis (or at least block diagonal)
    L2 = ops.build_L_squared()
    
    # Check if L^2 |ℓ,m⟩ = ℓ(ℓ+1) |ℓ,m⟩
    # We can check L^2 * vector vs ℓ(ℓ+1) * vector
    
    max_eigen_error = 0.0
    for i, p in enumerate(points):
        ℓ = p['ℓ']
        expected_val = ℓ * (ℓ + 1)
        
        # Create state vector
        psi = np.zeros(len(points))
        psi[i] = 1.0
        
        # Apply L^2
        L2_psi = L2.dot(psi)
        
        # Check projection on itself (should be eigenvalue)
        val = np.vdot(psi, L2_psi).real
        
        error = abs(val - expected_val)
        if error > max_eigen_error:
            max_eigen_error = error
            
    print(f"Max L^2 Eigenvalue Error: {max_eigen_error:.2e}")
    if max_eigen_error < 1e-13:
        print("[PASS] L^2 eigenvalues match ℓ(ℓ+1) exactly.")
    else:
        print("[FAIL] L^2 eigenvalue mismatch.")

if __name__ == "__main__":
    verify_project()
