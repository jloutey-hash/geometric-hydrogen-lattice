"""Step-by-step E23 formula calculation for (1,0) fundamental."""

import numpy as np

print("E23 Formula Debug for (1,0) Representation")
print("="*70)

# States for (1,0)
states = [
    (1, 0, 0, 0, 0, 0),  # State 0
    (1, 0, 0, 1, 0, 0),  # State 1
    (1, 0, 0, 1, 0, 1),  # State 2
]

def m_to_l(m13, m23, m33, m12, m22, m11):
    return (m13 + 2, m23 + 1, m33, m12 + 1, m22, m11)

print("\nTransition: State 0 -> State 1")
print("  m-indices: (1,0,0,0,0,0) -> (1,0,0,1,0,0)")
print("  Shift: m12 goes from 0 to 1")

m13, m23, m33, m12, m22, m11 = states[0]
l13, l23, l33, l12, l22, l11 = m_to_l(m13, m23, m33, m12, m22, m11)

print(f"\nInitial state m-indices: ({m13}, {m23}, {m33}, {m12}, {m22}, {m11})")
print(f"Initial state l-indices: ({l13}, {l23}, {l33}, {l12}, {l22}, {l11})")

print("\nTerm 1 formula (m12 -> m12+1):")
print("  sqrt(|(l13-l12-1) * (l23-l12-1) * (l33-l12-1) * (l11-l12) / ((l12-l22) * (l12-l22+1))|)")

numerator = (l13 - l12 - 1) * (l23 - l12 - 1) * (l33 - l12 - 1) * (l11 - l12)
denominator = (l12 - l22) * (l12 - l22 + 1)

print(f"\nSubstituting l-indices:")
print(f"  l13 - l12 - 1 = {l13} - {l12} - 1 = {l13 - l12 - 1}")
print(f"  l23 - l12 - 1 = {l23} - {l12} - 1 = {l23 - l12 - 1}")
print(f"  l33 - l12 - 1 = {l33} - {l12} - 1 = {l33 - l12 - 1}")
print(f"  l11 - l12 = {l11} - {l12} = {l11 - l12}")
print(f"  l12 - l22 = {l12} - {l22} = {l12 - l22}")
print(f"  l12 - l22 + 1 = {l12 - l22 + 1}")

print(f"\nNumerator = {l13 - l12 - 1} * {l23 - l12 - 1} * {l33 - l12 - 1} * {l11 - l12} = {numerator}")
print(f"Denominator = {l12 - l22} * {l12 - l22 + 1} = {denominator}")

if denominator != 0:
    ratio = numerator / denominator
    coeff = np.sqrt(abs(ratio))
    print(f"Ratio = {ratio}")
    print(f"Coefficient = sqrt(|{ratio}|) = {coeff}")
else:
    print("Denominator is zero!")

print("\n" + "="*70)
print("COMPARISON WITH THEORY")
print("="*70)
print("For (1,0) fundamental representation (quark):")
print("  U+ acting on |down> should give |up> with coefficient 1/√2")
print("  This is because [E23, E32] eigenvalue on middle state should be 1.5*T3")
print("  But we're getting coefficient 1.0, which gives wrong commutator")
print("\nKnown correct value: 1/√2 ≈ 0.707")
print(f"v5 computed value: {coeff:.3f}")
print(f"Ratio (computed/correct): {coeff/0.707:.3f}")
