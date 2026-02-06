import numpy as np
from weight_basis_gellmann import WeightBasisSU3

fund = WeightBasisSU3(1, 0)
antifund = WeightBasisSU3(0, 1)

print("Fundamental (1,0):")
ops_fund = {'T3': fund.T3, 'T8': fund.T8, 'E12': fund.E12, 'E21': fund.E21}
for name, T in ops_fund.items():
    err = np.linalg.norm(T - T.conj().T)
    print(f"  {name}: ||T - T†|| = {err:.2e}")

print("\nAntifundamental (0,1):")
ops_anti = {'T3': antifund.T3, 'T8': antifund.T8, 'E12': antifund.E12, 'E21': antifund.E21}
for name, T in ops_anti.items():
    err = np.linalg.norm(T - T.conj().T)
    print(f"  {name}: ||T - T†|| = {err:.2e}")

# Check product
I3 = np.eye(3, dtype=complex)
E12_prod = np.kron(fund.E12, I3) + np.kron(I3, antifund.E12)
print(f"\nE12_prod = E12_fund ⊗ I + I ⊗ E12_antifund:")
print(f"  ||E12_prod - E12_prod†|| = {np.linalg.norm(E12_prod - E12_prod.conj().T):.2e}")

# Check individual terms
term1 = np.kron(fund.E12, I3)
term2 = np.kron(I3, antifund.E12)
print(f"\nterm1 = E12_fund ⊗ I:")
print(f"  ||term1 - term1†|| = {np.linalg.norm(term1 - term1.conj().T):.2e}")
print(f"\nterm2 = I ⊗ E12_antifund:")
print(f"  ||term2 - term2†|| = {np.linalg.norm(term2 - term2.conj().T):.2e}")
