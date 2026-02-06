import numpy as np
from adjoint_tensor_product import AdjointSU3

adj = AdjointSU3()

# Build Casimir operator
ops = [adj.T3, adj.T8, adj.E12, adj.E21, adj.E23, adj.E32, adj.E13, adj.E31]
C2 = sum(T @ T for T in ops)

eigvals = np.linalg.eigvalsh(C2)
print('Casimir eigenvalues for (1,1) adjoint:')
print(eigvals)
print('\nUnique values:', np.unique(np.round(eigvals, 10)))
