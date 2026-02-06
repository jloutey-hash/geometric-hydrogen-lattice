import numpy as np
from weight_basis_gellmann import WeightBasisSU3

fund = WeightBasisSU3(1,0)

print("States in weight basis (1,0):")
for i in range(3):
    vec = np.zeros(3, dtype=complex)
    vec[i] = 1
    I3 = (vec.conj() @ fund.T3 @ vec).real
    Y = 2/np.sqrt(3) * (vec.conj() @ fund.T8 @ vec).real
    print(f"  State {i}: I3={I3:6.3f}, Y={Y:6.3f}")

print("\nE12 (should raise I3 by 1, keep Y=1):")
print(fund.E12)
print("\nE23 (should raise I3 by 1/2, raise Y by √3/2):")
print(fund.E23)
print("\nE13 (should raise I3 by 1/2, lower Y by √3/2):")
print(fund.E13)

# Check commutators
print("\n[E12, E23]:")
comm_12_23 = fund.E12 @ fund.E23 - fund.E23 @ fund.E12
print(comm_12_23)

print("\n[E12, E13]:")
comm_12_13 = fund.E12 @ fund.E13 - fund.E13 @ fund.E12
print(comm_12_13)

print("\nE23 expectation (from SU(3) algebra):")
print("[E12, E13] should be proportional to E23")
print("||[E12,E13] - cE23|| for c = ?")
for c in [-1, 0, 1, 1j, -1j]:
    err = np.linalg.norm(comm_12_13 - c*fund.E23)
    print(f"  c = {c:4}: error = {err:.6f}")
