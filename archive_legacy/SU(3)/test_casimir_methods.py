import numpy as np
from weight_basis_gellmann import WeightBasisSU3

fund = WeightBasisSU3(1, 0)
antifund = WeightBasisSU3(0, 1)

# Product space
I3 = np.eye(3, dtype=complex)
def tensor_gen(A, B):
    return np.kron(A, I3) + np.kron(I3, B)

# Build all generators in product space
E12_prod = tensor_gen(fund.E12, antifund.E12)
E21_prod = tensor_gen(fund.E21, antifund.E21)
E23_prod = tensor_gen(fund.E23, antifund.E23)
E32_prod = tensor_gen(fund.E32, antifund.E32)
E13_prod = tensor_gen(fund.E13, antifund.E13)
E31_prod = tensor_gen(fund.E31, antifund.E31)
T3_prod = tensor_gen(fund.T3, antifund.T3)
T8_prod = tensor_gen(fund.T8, antifund.T8)

# Method 1: Gell-Mann combinations
lambda1 = E12_prod + E21_prod
lambda2 = -1j * (E12_prod - E21_prod)
lambda3 = 2 * T3_prod
lambda4 = E23_prod + E32_prod
lambda5 = -1j * (E23_prod - E32_prod)
lambda6 = E13_prod + E31_prod
lambda7 = -1j * (E13_prod - E31_prod)
lambda8 = 2 * T8_prod

C2_method1 = (lambda1 @ lambda1 + lambda2 @ lambda2 + lambda3 @ lambda3 +
              lambda4 @ lambda4 + lambda5 @ lambda5 + lambda6 @ lambda6 +
              lambda7 @ lambda7 + lambda8 @ lambda8) / 4.0

eigvals1 = np.linalg.eigvalsh(C2_method1)
print("Method 1 (Gell-Mann combinations):")
print("  Eigenvalues:", np.round(np.sort(eigvals1), 6))
print("  Unique:", np.unique(np.round(eigvals1, 6)))

# Method 2: Use individual Casimirs from each space
ops_fund = [fund.T3, fund.T8, fund.E12, fund.E21, fund.E23, fund.E32, fund.E13, fund.E31]
C2_fund = sum(T @ T for T in ops_fund)

ops_anti = [antifund.T3, antifund.T8, antifund.E12, antifund.E21, antifund.E23, antifund.E32, antifund.E13, antifund.E31]
C2_anti = sum(T @ T for T in ops_anti)

C2_method2 = np.kron(C2_fund, I3) + np.kron(I3, C2_anti)

eigvals2 = np.linalg.eigvalsh(C2_method2)
print("\nMethod 2 (Sum of individual Casimirs):")
print("  Eigenvalues:", np.round(np.sort(eigvals2), 6))
print("  Unique:", np.unique(np.round(eigvals2, 6)))

# Check what the singlet should have
singlet = np.zeros(9, dtype=complex)
singlet[0] = 1/np.sqrt(3)  # |00>
singlet[4] = 1/np.sqrt(3)  # |11>
singlet[8] = 1/np.sqrt(3)  # |22>

C2_singlet_m1 = C2_method1 @ singlet
C2_singlet_m2 = C2_method2 @ singlet

print("\nSinglet eigenvalue:")
print("  Method 1:", np.vdot(singlet, C2_singlet_m1).real)
print("  Method 2:", np.vdot(singlet, C2_singlet_m2).real)
