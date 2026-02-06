"""Simple test of SU(3) impedance calculation"""

from su3_impedance import SU3SymplecticImpedance

print("Testing (1,0) fundamental representation...")
calc = SU3SymplecticImpedance(1, 0, verbose=True)
impedance = calc.compute_impedance()

print(f"\nResult:")
print(f"  Z = {impedance.Z_impedance:.6f}")
print(f"  Z/C2 = {impedance.Z_normalized:.6f}")
print(f"  Z/4π = {impedance.Z_dimensionless:.6f}")
