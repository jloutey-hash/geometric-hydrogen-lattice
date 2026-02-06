import numpy as np
from weight_basis_gellmann import WeightBasisSU3

fund = WeightBasisSU3(1, 0)

print("E12:")
print(fund.E12)
print("\nE12†:")
print(fund.E12.conj().T)
print("\nE21:")
print(fund.E21)
print("\n||E12† - E21||:", np.linalg.norm(fund.E12.conj().T - fund.E21))
