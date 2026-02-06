"""Check [A+, A-] commutator"""
from test_algebra_closure import RungeLenzLattice
import numpy as np

lat = RungeLenzLattice(3)

# Get spherical components (before Cartesian conversion)
# We need to rebuild them from scratch
from scipy.sparse import lil_matrix

n_max = 3
dim = len(lat.nodes)

Aplus = lil_matrix((dim, dim), dtype=complex)
Aminus = lil_matrix((dim, dim), dtype=complex)

for i, (n, l, m) in enumerate(lat.nodes):
    # Transitions: l → l-1
    if l > 0:
        radial = np.sqrt(n**2 - l**2)
        denominator_sqrt = np.sqrt(4 * l**2 - 1)
        
        # A_+: Δm = +1
        target = (n, l - 1, m + 1)
        if target in lat.node_index:
            j = lat.node_index[target]
            numerator = np.sqrt((l - m) * (l - m - 1))
            Aplus[j, i] = -radial * numerator / denominator_sqrt
        
        # A_-: Δm = -1
        target = (n, l - 1, m - 1)
        if target in lat.node_index:
            j = lat.node_index[target]
            numerator = np.sqrt((l + m) * (l + m - 1))
            Aminus[j, i] = radial * numerator / denominator_sqrt
    
    # Transitions: l → l+1
    if l + 1 < n:
        radial = np.sqrt(n**2 - (l + 1)**2)
        denominator_sqrt = np.sqrt(4 * (l + 1)**2 - 1)
        
        # A_+: Δm = +1
        target = (n, l + 1, m + 1)
        if target in lat.node_index:
            j = lat.node_index[target]
            numerator = np.sqrt((l + m + 1) * (l + m + 2))
            Aplus[j, i] = radial * numerator / denominator_sqrt
        
        # A_-: Δm = -1
        target = (n, l + 1, m - 1)
        if target in lat.node_index:
            j = lat.node_index[target]
            numerator = np.sqrt((l - m + 1) * (l - m + 2))
            Aminus[j, i] = -radial * numerator / denominator_sqrt

Aplus = Aplus.tocsr().toarray()
Aminus = Aminus.tocsr().toarray()

# Also need Lz
# Build from scratch using standard formulas
Lz = lil_matrix((dim, dim), dtype=complex)
for i, (n, l, m) in enumerate(lat.nodes):
    Lz[i, i] = m
Lz = Lz.tocsr().toarray()

# Compute commutator
comm = Aplus @ Aminus - Aminus @ Aplus
expected = 2.0 * Lz  # Factor of 2 from SU(2) algebra

error = np.linalg.norm(comm - expected)
print(f"[A+, A-] - 2Lz error: {error:.2e}")

# Also check without the factor of 2
error_alt = np.linalg.norm(comm - Lz)
print(f"[A+, A-] - Lz error: {error_alt:.2e}")

# Print first few diagonal elements
print("\nFirst 8 diagonal elements of [A+, A-]:")
print(np.diag(comm)[:8])
print("\nFirst 8 diagonal elements of 2*Lz:")
print(np.diag(2*Lz)[:8])
print("\nFirst 8 diagonal elements of Lz:")
print(np.diag(Lz)[:8])
