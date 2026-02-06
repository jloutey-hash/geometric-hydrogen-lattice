"""Debug the z-operator construction"""
from paraboloid_relativistic import RungeLenzLattice
import numpy as np

lattice = RungeLenzLattice(3)

print("Nodes in lattice:")
for i, (n, l, m) in enumerate(lattice.nodes):
    print(f"  {i}: n={n}, l={l}, m={m}")

# Build z operator
z_operator = np.zeros((len(lattice.nodes), len(lattice.nodes)))

for i, (n1, l1, m1) in enumerate(lattice.nodes):
    for j, (n2, l2, m2) in enumerate(lattice.nodes):
        # z couples states within same n, with Δl=±1, Δm=0
        if n1 == n2 and m1 == m2 and abs(l1 - l2) == 1:
            # Approximate dipole matrix element
            if l2 == l1 - 1:  # j has lower l
                z_operator[i, j] = n1**2 * np.sqrt((l1**2 - m1**2) / max(4*l1**2 - 1, 1))
            elif l2 == l1 + 1:  # j has higher l  
                z_operator[i, j] = n1**2 * np.sqrt(((l1+1)**2 - m1**2) / max(4*(l1+1)**2 - 1, 1))

print("\nz-operator matrix (non-zero elements):")
for i in range(len(lattice.nodes)):
    for j in range(len(lattice.nodes)):
        if abs(z_operator[i, j]) > 1e-10:
            print(f"  z[{i},{j}] = {z_operator[i,j]:.4f}  ({lattice.nodes[i]} -> {lattice.nodes[j]})")

print(f"\nTotal non-zero elements: {np.sum(np.abs(z_operator) > 1e-10)}")
print(f"Matrix is symmetric: {np.allclose(z_operator, z_operator.T)}")

# Check Az operator for comparison
Az = lattice.Az.toarray()
print(f"\nAz operator non-zero elements: {np.sum(np.abs(Az) > 1e-10)}")
print("\nAz matrix (non-zero elements):")
for i in range(min(10, len(lattice.nodes))):
    for j in range(min(10, len(lattice.nodes))):
        if abs(Az[i, j]) > 1e-10:
            print(f"  Az[{i},{j}] = {Az[i,j]:.4f}  ({lattice.nodes[i]} -> {lattice.nodes[j]})")
