"""Debug script to check Runge-Lenz matrix elements"""
from test_algebra_closure import RungeLenzLattice
import numpy as np

lat = RungeLenzLattice(3)

print("Node mapping:")
for node, idx in sorted(lat.node_index.items(), key=lambda x: x[1]):
    print(f"  {idx}: n={node[0]}, l={node[1]}, m={node[2]}")

print("\nChecking specific matrix elements:")
print(f"Node (2,0,0) index: {lat.node_index[(2,0,0)]}")
print(f"Node (2,1,0) index: {lat.node_index[(2,1,0)]}")
print(f"Node (2,1,1) index: {lat.node_index[(2,1,1)]}")
print(f"Node (2,1,-1) index: {lat.node_index[(2,1,-1)]}")

Ax = lat.Ax.toarray()
Ay = lat.Ay.toarray()
Az = lat.Az.toarray()

print("\nAx matrix (first 8x8 block):")
print(np.around(Ax[:8,:8], 3))

print("\nAy matrix (first 8x8 block):")
print(np.around(Ay[:8,:8], 3))

print("\nAz matrix (first 8x8 block):")
print(np.around(Az[:8,:8], 3))

# Check the commutator
comm = Ax @ Ay - Ay @ Ax
expected = -1j * lat.Lz.toarray()
error = comm - expected

print(f"\n[Ax, Ay] + iLz error norm: {np.linalg.norm(error):.2e}")
print("\nFirst 8x8 block of error:")
print(np.around(error[:8,:8], 3))
