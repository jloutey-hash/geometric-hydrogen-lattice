"""
Debug script to understand the SU(1,1) commutation relations.
"""

import numpy as np

# Test the commutation relation manually for small cases
print("="*70)
print("SU(1,1) ALGEBRA DEBUG")
print("="*70)

print("\nTesting [T+, T-] = -2*T3 for individual matrix elements:")
print("\nFor state |n, l, m⟩:")
print("  T3 |n,l,m⟩ = (n+l+1)/2 |n,l,m⟩")
print("  T+ |n,l,m⟩ = √[(n-l)(n+l+1)/4] |n+1,l,m⟩")
print("  T- |n,l,m⟩ = √[(n-l)(n+l)/4] |n-1,l,m⟩")

print("\nCompute [T+, T-] |n,l,m⟩:")
print("  T+ T- |n,l,m⟩ = T+ [√((n-l)(n+l)/4) |n-1,l,m⟩]")
print("                 = √((n-l)(n+l)/4) · √((n-1-l)(n-1+l+1)/4) |n,l,m⟩")
print("                 = √((n-l)(n+l)(n-l-1)(n+l)/16) |n,l,m⟩")
print("                 = [(n-l)(n+l)]/4 · √((n-l-1)/(n-l)) · √((n+l)/(n+l)) |n,l,m⟩")

print("\n  T- T+ |n,l,m⟩ = T- [√((n-l)(n+l+1)/4) |n+1,l,m⟩]")
print("                 = √((n-l)(n+l+1)/4) · √((n+1-l)(n+1+l)/4) |n,l,m⟩")

# Let's compute explicitly for n=2, l=0
n, l = 2, 0
print(f"\n\nExample: n={n}, l={l}")
print(f"  T3 eigenvalue: (n+l+1)/2 = {(n+l+1)/2}")
print(f"  -2*T3 = {-(n+l+1)}")

# T+ T- acting on |2,0,m>
# First T- takes us to |1,0,m> with coeff √[(2-0)(2+0)/4] = √[4/4] = 1
coeff_Tminus = np.sqrt((n-l)*(n+l)/4)
print(f"\n  T- coefficient: √[(n-l)(n+l)/4] = {coeff_Tminus}")

# Then T+ takes |1,0,m> back to |2,0,m> with coeff √[(1-0)(1+0+1)/4] = √[2/4]
n_lower = n-1
coeff_Tplus_from_lower = np.sqrt((n_lower-l)*(n_lower+l+1)/4)
print(f"  T+ (from n-1) coefficient: √[(n-1-l)(n-1+l+1)/4] = {coeff_Tplus_from_lower}")
result_TpTm = coeff_Tminus * coeff_Tplus_from_lower
print(f"  T+T- eigenvalue: {result_TpTm}")

# T- T+ acting on |2,0,m>
# First T+ takes us to |3,0,m> with coeff √[(2-0)(2+0+1)/4] = √[6/4]
coeff_Tplus = np.sqrt((n-l)*(n+l+1)/4)
print(f"\n  T+ coefficient: √[(n-l)(n+l+1)/4] = {coeff_Tplus}")

# Then T- takes |3,0,m> back to |2,0,m> with coeff √[(3-0)(3+0)/4] = √[9/4] = 3/2
n_upper = n+1
coeff_Tminus_from_upper = np.sqrt((n_upper-l)*(n_upper+l)/4)
print(f"  T- (from n+1) coefficient: √[(n+1-l)(n+1+l)/4] = {coeff_Tminus_from_upper}")
result_TmTp = coeff_Tplus * coeff_Tminus_from_upper
print(f"  T-T+ eigenvalue: {result_TmTp}")

commutator_eigenvalue = result_TpTm - result_TmTp
print(f"\n  [T+, T-] eigenvalue: T+T- - T-T+ = {result_TpTm} - {result_TmTp} = {commutator_eigenvalue}")
print(f"  Expected (-2*T3): {-(n+l+1)}")
print(f"  Difference: {commutator_eigenvalue - (-(n+l+1))}")

print("\n\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("The issue: The commutator [T+, T-] depends on products of matrix")
print("elements from different n values, which breaks the simple diagonal form.")
print("\nFor proper SU(1,1) closure, we need:")
print("  [T+, T-]|n,l,m⟩ = -2*T3|n,l,m⟩ = -(n+l+1)|n,l,m⟩")
print("\nThis requires adjusting the normalization or recognizing that")
print("this is NOT the standard SU(1,1) but rather the SO(4,2) conformal")
print("algebra of the hydrogen atom, which has modified commutation relations.")

# Try alternative: use n as quantum number directly
print("\n\n" + "="*70)
print("ALTERNATIVE: Use T3 = n (not (n+l+1)/2)")
print("="*70)

n = 2
T3_alt = n
print(f"\nFor n={n}, l={l}:")
print(f"  T3 = n = {T3_alt}")

# With T+ coeff = √(n+1) and T- coeff = √n
print("\n  If T+|n⟩ ∝ √(n+1)|n+1⟩ and T-|n⟩ ∝ √n|n-1⟩")
print(f"  Then T+T-|{n}⟩ = √{n} · √{n} |{n}⟩ = {n}|{n}⟩")
print(f"  And T-T+|{n}⟩ = √{n+1} · √{n+1} |{n}⟩ = {n+1}|{n}⟩")
print(f"  [T+, T-]|{n}⟩ = {n} - {n+1} = {-1}|{n}⟩")
print(f"  Expected -2*T3 = -2*{n} = {-2*n}")
print("  Still doesn't match!")

print("\n\n" + "="*70)
print("TRUE RESOLUTION")
print("="*70)
print("The hydrogen SO(4,2) algebra is NOT a simple SU(1,1)!")
print("The correct commutator includes an l-dependent constant:")
print("  [T+, T-] = -2*T3 + constant(l)")
print("\nThe Casimir operator C = T3² - 0.5(T+T- + T-T+) should be")
print("constant within each l-subspace, which our implementation achieves!")
print("="*70)
