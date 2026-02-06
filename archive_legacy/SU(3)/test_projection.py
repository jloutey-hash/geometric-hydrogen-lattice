import numpy as np
from adjoint_tensor_product import AdjointSU3

adj = AdjointSU3()

# Check projection matrix
P = adj.P_adj
print(f"P shape: {P.shape}")
print(f"P†P shape: {(P.conj().T @ P).shape}")

# Check if P is column-orthonormal
PTP = P.conj().T @ P
print(f"\nP†P:")
print(np.round(PTP.real, 3))
print(f"\n||P†P - I||: {np.linalg.norm(PTP - np.eye(8)):.2e}")

# Check E12_prod
E12_prod = adj.E12_prod
print(f"\n||E12_prod - E12_prod†||: {np.linalg.norm(E12_prod - E12_prod.conj().T):.2e}")

# Check projection
E12_adj_manual = P.conj().T @ E12_prod @ P
print(f"\n||E12_adj_manual - E12_adj_manual†||: {np.linalg.norm(E12_adj_manual - E12_adj_manual.conj().T):.2e}")

# Compare with stored E12
print(f"||E12 (stored) - E12_adj_manual||: {np.linalg.norm(adj.E12 - E12_adj_manual):.2e}")
print(f"||E12 (stored) - E12†||: {np.linalg.norm(adj.E12 - adj.E12.conj().T):.2e}")
