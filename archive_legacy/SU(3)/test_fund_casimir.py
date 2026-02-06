import numpy as np
from weight_basis_gellmann import WeightBasisSU3

fund = WeightBasisSU3(1, 0)

# Build Casimir operator
ops = [fund.T3, fund.T8, fund.E12, fund.E21, fund.E23, fund.E32, fund.E13, fund.E31]
C2 = sum(T @ T for T in ops)

eigvals = np.linalg.eigvalsh(C2)
print('Casimir eigenvalues for (1,0):', eigvals)
print('Expected: 4/3 = ', 4/3)
