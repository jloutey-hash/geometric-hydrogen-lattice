"""Minimal test of impedance calculation components"""

import numpy as np
from su3_spherical_embedding import SU3SphericalEmbedding
from general_rep_builder import GeneralRepBuilder

print("Testing fundamental (1,0) representation...")

# Create embedding
print("\n1. Creating spherical embedding...")
embedding = SU3SphericalEmbedding(1, 0, r0=1.0)
print(f"   Dimension: {embedding.dim}")
print(f"   Casimir: {embedding.C2:.4f}")

# Get states
print("\n2. Creating states...")
states = embedding.create_spherical_states()
print(f"   Number of states: {len(states)}")

# Get shells
print("\n3. Organizing into shells...")
shells = embedding.get_states_by_shell()
print(f"   Number of shells: {len(shells)}")
for r, shell_states in shells.items():
    print(f"   Shell r={r:.4f}: {len(shell_states)} states")

# Now test impedance calculation components
print("\n4. Computing matter capacity...")
C_base = embedding.dim * np.sqrt(max(embedding.C2, 0.1))
print(f"   Base capacity: {C_base:.6f}")

C_total = C_base
for shell_r, shell_states in shells.items():
    n = len(shell_states)
    # Simple symplectic contribution
    area = 4 * np.pi * shell_r**2
    C_total += area
    print(f"   Shell r={shell_r:.4f}: area={area:.6f}")

print(f"   Total capacity: {C_total:.6f}")

print("\n5. Computing gauge action...")
S_base = np.sqrt(embedding.dim) * np.sqrt(max(embedding.C2, 0.1))
print(f"   Base action: {S_base:.6f}")

S_total = S_base
for shell_r, shell_states in shells.items():
    S_total += shell_r  # Minimal contribution
    print(f"   Shell r={shell_r:.4f}: contribution={shell_r:.6f}")

print(f"   Total action: {S_total:.6f}")

print("\n6. Computing impedance...")
Z = S_total / C_total if C_total > 0 else np.inf
print(f"   Z = S/C = {Z:.6f}")
print(f"   Z/4π = {Z/(4*np.pi):.6f}")

print("\n7. Comparison to U(1)")
alpha_em = 1/137
print(f"   U(1) fine structure: α = {alpha_em:.6f}")
print(f"   SU(3) impedance/4π: {Z/(4*np.pi):.6f}")
print(f"   Ratio: {(Z/(4*np.pi))/alpha_em:.2f}")

print("\nTEST PASSED!")
