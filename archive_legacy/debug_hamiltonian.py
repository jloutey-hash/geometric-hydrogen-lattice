"""Debug Hamiltonian construction"""
from paraboloid_relativistic import RungeLenzLattice
import numpy as np

lattice = RungeLenzLattice(3)

# Build z operator
z_operator = np.zeros((len(lattice.nodes), len(lattice.nodes)))
for i, (n1, l1, m1) in enumerate(lattice.nodes):
    for j, (n2, l2, m2) in enumerate(lattice.nodes):
        if n1 == n2 and m1 == m2 and abs(l1 - l2) == 1:
            if l2 == l1 - 1:
                z_operator[i, j] = n1**2 * np.sqrt((l1**2 - m1**2) / max(4*l1**2 - 1, 1))
            elif l2 == l1 + 1:
                z_operator[i, j] = n1**2 * np.sqrt(((l1+1)**2 - m1**2) / max(4*(l1+1)**2 - 1, 1))

print("z-operator check:")
print(f"  Non-zero elements: {np.sum(np.abs(z_operator) > 1e-10)}")
print(f"  Max element: {np.max(np.abs(z_operator)):.4f}")

# Build H0
H0 = np.zeros((len(lattice.nodes), len(lattice.nodes)))
for i, (n, l, m) in enumerate(lattice.nodes):
    H0[i, i] = -1.0 / (2 * n**2)

print("\nH0 (diagonal energies in a.u.):")
for i, (n, l, m) in enumerate(lattice.nodes):
    print(f"  State {i} (n={n},l={l},m={m}): E = {H0[i,i]:.6f}")

# Test with a field
F_au = 0.01  # Moderate field in a.u.
H_stark = -F_au * z_operator

print(f"\nStark perturbation (F = {F_au:.4f} a.u.):")
print(f"  Non-zero elements: {np.sum(np.abs(H_stark) > 1e-10)}")
print(f"  Max element: {np.max(np.abs(H_stark)):.6f}")

H_total = H0 + H_stark

# Diagonalize
eigenvalues_0 = np.linalg.eigvalsh(H0)
eigenvalues_F = np.linalg.eigvalsh(H_total)

print("\nEigenvalues comparison:")
print("  F=0:")
for i, E in enumerate(eigenvalues_0):
    print(f"    Level {i}: {E:.8f}")
print(f"\n  F={F_au} a.u.:")
for i, E in enumerate(eigenvalues_F):
    shift = E - eigenvalues_0[i]
    print(f"    Level {i}: {E:.8f}  (shift: {shift:.8f})")

print(f"\nMaximum energy shift: {np.max(np.abs(eigenvalues_F - eigenvalues_0)):.8f} a.u.")
