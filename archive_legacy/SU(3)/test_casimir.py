import numpy as np
from weight_basis_gellmann import WeightBasisSU3

fund = WeightBasisSU3(1, 0)  # (1,0) fundamental rep
conj = WeightBasisSU3(0, 1)  # (0,1) conjugate rep

# Build Casimir operator for 3⊗3̄
# Method 1: Using individual representation Casimirs
dim = 3
prod_dim = dim * dim

# Build Casimir for (1,0)
fund_ops = [fund.T3, fund.T8, fund.E12, fund.E21, fund.E23, fund.E32, fund.E13, fund.E31]
C2_fund = sum(T @ T for T in fund_ops)

# Build Casimir for (0,1)
conj_ops = [conj.T3, conj.T8, conj.E12, conj.E21, conj.E23, conj.E32, conj.E13, conj.E31]
C2_conj = sum(T @ T for T in conj_ops)

# Casimir for product: C₂(3⊗3̄) = C₂(fund) ⊗ I + I ⊗ C₂(conj)
C2_prod = np.kron(C2_fund, np.eye(dim)) + np.kron(np.eye(dim), C2_conj)

eigvals = np.linalg.eigvalsh(C2_prod)
print('Casimir eigenvalues (sorted):', np.sort(eigvals)[::-1])
print('Unique values (rounded):', np.unique(np.round(eigvals, 6)))

# Expected: 0 for singlet (1x), 3 for adjoint (8x)
# C2(singlet) = 0, C2(adjoint) = 3
n_singlet = np.sum(np.abs(eigvals) < 0.1)
n_adjoint = np.sum(np.abs(eigvals - 3) < 0.1)
print(f'\nFound {n_singlet} singlet state(s), {n_adjoint} adjoint state(s)')
print(f'Expected: 1 singlet, 8 adjoint (total 9)')
