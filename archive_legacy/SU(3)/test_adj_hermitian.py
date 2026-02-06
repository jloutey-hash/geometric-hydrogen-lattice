import numpy as np
from adjoint_tensor_product import AdjointSU3

adj = AdjointSU3()

# Check if generators are Hermitian
print("Checking if generators are Hermitian:")
ops = {'T3': adj.T3, 'T8': adj.T8, 'E12': adj.E12, 'E21': adj.E21, 
       'E23': adj.E23, 'E32': adj.E32, 'E13': adj.E13, 'E31': adj.E31}

for name, T in ops.items():
    hermitian_error = np.linalg.norm(T - T.conj().T)
    print(f"  {name}: ||T - T†|| = {hermitian_error:.2e}")

# Build Casimir
C2 = sum(T @ T for T in ops.values())

print(f"\nCasimir Hermiticity: ||C2 - C2†|| = {np.linalg.norm(C2 - C2.conj().T):.2e}")

eigvals = np.linalg.eigvalsh(C2)
print(f"\nEigenvalues: {eigvals}")
print(f"All real: {np.all(np.abs(np.imag(eigvals)) < 1e-10)}")

# Check sum of squares
sum_squares = sum(np.linalg.norm(T, 'fro')**2 for T in ops.values())
trace_C2 = np.trace(C2)
print(f"\nΣ ||Tᵢ||²_F = {sum_squares:.6f}")
print(f"Tr(C₂) = {np.real(trace_C2):.6f}")
