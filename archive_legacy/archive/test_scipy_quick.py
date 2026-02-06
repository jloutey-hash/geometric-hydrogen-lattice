"""Quick test of SciPy convergence for Week 3."""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from lattice import PolarLattice
from angular_momentum import AngularMomentumOperators
from operators import LatticeOperators
from quantum_comparison import QuantumComparison
from scipy.linalg import eigh

print("Testing SciPy convergence analysis...")
print("\nTest 1: Create lattice and compute eigenmode")

# Small test: ℓ=1, m=0 at N=10
N = 10
lattice = PolarLattice(n_max=N)
angular = AngularMomentumOperators(lattice)

print(f"  Lattice created: {len(lattice.points)} sites")

# Build L^2 operator
L2 = angular.build_L_squared()
Lz = angular.build_Lz()
print(f"  L² operator: {L2.shape}")

# Compute eigenmodes
print("  Computing eigenmodes...")
eigenvalues, eigenvectors = eigh(L2.toarray())

print(f"  Found {len(eigenvalues)} eigenmodes")
print(f"  First few L² eigenvalues: {eigenvalues[:10]}")

# Find ℓ=1, m=0 mode (L²=2)
expected = 1 * (1 + 1)  # = 2
idx = np.argmin(np.abs(eigenvalues - expected))
psi = eigenvectors[:, idx]

print(f"\n  Mode with L²≈{expected}: eigenvalue = {eigenvalues[idx]:.4f}")

# Compute overlap with SciPy
print("\nTest 2: Compute overlap with SciPy sph_harm")
operators = LatticeOperators(lattice)
qc = QuantumComparison(lattice, operators)
Y_scipy = qc.sample_spherical_harmonic(ell=1, m=0)

print(f"  SciPy Y₁⁰ sampled at {len(Y_scipy)} points")

# Normalize and compute overlap
psi_norm = psi / np.linalg.norm(psi)
Y_norm = Y_scipy / np.linalg.norm(Y_scipy)

overlap = np.abs(np.vdot(psi_norm, Y_norm))**2
print(f"  Overlap: {overlap:.6f}")

if overlap > 0.7:
    print("\n✅ Test passed! Overlap > 0.7")
    print("Week 3 convergence analysis infrastructure is working.")
else:
    print(f"\n⚠️  Low overlap ({overlap:.3f}). May need to tune mode finding.")

print("\nReady to run full scipy_convergence.py analysis.")
