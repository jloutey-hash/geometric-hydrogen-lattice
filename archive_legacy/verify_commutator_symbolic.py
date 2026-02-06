"""
Symbolic Verification: Radial Commutator [T+, T-] for Hydrogen

This script uses SymPy to symbolically compute the commutator [T+, T-]
using the Biedenharn-Louck normalization factors implemented in our code.

The goal is to determine the exact polynomial form of C(l) such that:
    [T+, T-] = -2*T3 + C(l)

Author: Josh Loutey
Date: February 2026
"""

import sympy as sp
from sympy import sqrt, simplify, expand, symbols, Matrix, latex

# Define symbolic variables
n, l = symbols('n l', positive=True, integer=True)

# Define the operators as matrices acting on basis states |n, l, m>
# We work in the subspace where m is fixed (since [T+, T-] doesn't affect m)

print("="*80)
print("SYMBOLIC VERIFICATION: Radial Commutator [T+, T-]")
print("="*80)
print()

# The key is to compute the action on a generic state |n, l, m>
# 
# T+ |n> = sqrt((n-l)(n+l+1)/4) |n+1>
# T- |n> = sqrt((n-l)(n+l)/4) |n-1>
# T3 |n> = (n+l+1)/2 |n>
#
# Note: In our code, we use factors WITHOUT the 1/4 division in the definition,
# but the sqrt includes it. Let me verify from the code...

print("Matrix elements (from paraboloid_lattice_su11.py):")
print("  T+ |n, l, m> = sqrt[(n-l)(n+l+1)/4] |n+1, l, m>")
print("  T- |n, l, m> = sqrt[(n-l)(n+l)/4] |n-1, l, m>")
print("  T3 |n, l, m> = (n+l+1)/2 |n, l, m>")
print()

# Define the matrix elements as functions
def T_plus_coeff(n_val, l_val):
    """Coefficient for T+ acting on |n, l>"""
    return sqrt((n_val - l_val) * (n_val + l_val + 1) / 4)

def T_minus_coeff(n_val, l_val):
    """Coefficient for T- acting on |n, l>"""
    return sqrt((n_val - l_val) * (n_val + l_val) / 4)

def T3_eigenvalue(n_val, l_val):
    """Eigenvalue of T3 on |n, l>"""
    return (n_val + l_val + 1) / 2

# Compute [T+, T-] |n> = (T+ T- - T- T+) |n>
# Key insight: Work with EIGENVALUES, not matrix elements
# Since T+T- returns to the same state |n>, we get an eigenvalue

print("Computing [T+, T-] eigenvalue on |n, l>:")
print()
print("Method: Compute the products of matrix elements (eigenvalues)")
print()

# T+T- eigenvalue:
# T- takes |n> -> |n-1> with coefficient sqrt[(n-l)(n+l)/4]
# T+ takes |n-1> -> |n> with coefficient sqrt[(n-1-l)(n-1+l+1)/4] = sqrt[(n-l-1)(n+l)/4]
# Product: [(n-l)(n+l)/4] * [(n-l-1)(n+l)/4] = (n-l)(n+l)(n-l-1)(n+l)/16

print("T+T- eigenvalue:")
print("  T- |n> = sqrt[(n-l)(n+l)/4] |n-1>")
print("  T+ |n-1> = sqrt[(n-l-1)(n+l)/4] |n>")
print("  Combined eigenvalue: [(n-l)(n+l)] * [(n-l-1)(n+l)] / 16")

T_plus_T_minus_eigenval = (n-l)*(n+l) * (n-l-1)*(n+l) / 16
T_plus_T_minus_simplified = expand(T_plus_T_minus_eigenval)
print(f"  = {T_plus_T_minus_simplified}")
print()

# T-T+ eigenvalue:
# T+ takes |n> -> |n+1> with coefficient sqrt[(n-l)(n+l+1)/4]
# T- takes |n+1> -> |n> with coefficient sqrt[(n+1-l)(n+1+l)/4] = sqrt[(n-l+1)(n+l+1)/4]
# Product: [(n-l)(n+l+1)/4] * [(n-l+1)(n+l+1)/4] = (n-l)(n+l+1)(n-l+1)(n+l+1)/16

print("T-T+ eigenvalue:")
print("  T+ |n> = sqrt[(n-l)(n+l+1)/4] |n+1>")
print("  T- |n+1> = sqrt[(n-l+1)(n+l+1)/4] |n>")
print("  Combined eigenvalue: [(n-l)(n+l+1)] * [(n-l+1)(n+l+1)] / 16")

T_minus_T_plus_eigenval = (n-l)*(n+l+1) * (n-l+1)*(n+l+1) / 16
T_minus_T_plus_simplified = expand(T_minus_T_plus_eigenval)
print(f"  = {T_minus_T_plus_simplified}")
print()

# Commutator eigenvalue
print("[T+, T-] eigenvalue:")
commutator_eigenval = T_plus_T_minus_eigenval - T_minus_T_plus_eigenval
commutator_simplified = expand(commutator_eigenval)
print(f"  = {T_plus_T_minus_simplified}")
print(f"    - {T_minus_T_plus_simplified}")
print(f"  = {commutator_simplified}")
print()

# Now compare to -2*T3
print("Comparing to -2*T3 + C(l):")
minus_2_T3 = -2 * T3_eigenvalue(n, l)
minus_2_T3_expanded = expand(minus_2_T3)
print(f"  -2*T3 = -2*(n+l+1)/2 = {minus_2_T3_expanded}")
print()

# Solve for C(l)
print("Solving for C(l):")
print(f"  [T+, T-] = -2*T3 + C(l)")
print(f"  {commutator_simplified} = {minus_2_T3_expanded} + C(l)")
print()

C_l = commutator_simplified - minus_2_T3_expanded
C_l_expanded = expand(C_l)
print(f"  C(l) = {C_l_expanded}")
print()

# Verify C(l) is indeed independent of n
from sympy import diff
dC_dn = diff(C_l_expanded, n)
dC_dn_simplified = simplify(dC_dn)
print(f"Verification: ∂C/∂n = {dC_dn_simplified}")
if dC_dn_simplified == 0:
    print("  ✓ C(l) is independent of n (as required)")
else:
    print(f"  ✗ WARNING: C(l) depends on n!")
print()

# Express in standard form
print("="*80)
print("FINAL RESULT:")
print("="*80)
print()
print(f"[T+, T-] = -2*T3 + C(l)")
print()
print(f"where C(l) = {C_l_expanded}")
print()

# Try to factor or simplify further
C_l_factored = sp.factor(C_l_expanded)
print(f"Factored form: C(l) = {C_l_factored}")
print()

# LaTeX output for the paper
print("="*80)
print("LATEX OUTPUT FOR PAPER:")
print("="*80)
print()
print("Commutator equation:")
print(f"  [T_+, T_-] = {latex(commutator_simplified)}")
print()
print("C(l) term:")
print(f"  C(l) = {latex(C_l_expanded)}")
print()

# Try to factor or simplify further
C_l_factored = sp.factor(C_l_expanded)
if C_l_factored != C_l_expanded:
    print("Factored form:")
    print(f"  C(l) = {latex(C_l_factored)}")
    print()

# Numerical verification for a few values
print("="*80)
print("NUMERICAL CHECK:")
print("="*80)
print()
print("Testing C(l) for specific (n, l) values:")
print()
test_cases = [(2, 0), (2, 1), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (4, 3)]
for n_val, l_val in test_cases:
    C_numerical = C_l_expanded.subs([(n, n_val), (l, l_val)])
    print(f"  n={n_val}, l={l_val}: C(l) = {C_numerical} = {float(C_numerical):.6f}")
print()

# Compare to what we had in the paper: (l^2 + l + 1)/2
print("="*80)
print("Comparing to paper's claim: C(l) = (l² + l + 1)/2")
print("="*80)
paper_claim = (l**2 + l + 1) / 2
print(f"  Our C(l) = {C_l_expanded}")
print(f"  Paper claim = {paper_claim}")
print()
print("Testing if they match for various l values:")
for l_val in [0, 1, 2, 3, 4]:
    our_val = C_l_expanded.subs(l, l_val)
    paper_val = paper_claim.subs(l, l_val)
    match = "✓" if simplify(our_val - paper_val) == 0 else "✗"
    print(f"  l={l_val}: Our={our_val}, Paper={paper_val}, Match={match}")
print()

if simplify(C_l_expanded - paper_claim) == 0:
    print("  ✓✓✓ Paper's formula is CORRECT!")
else:
    difference = simplify(C_l_expanded - paper_claim)
    print(f"  ✗✗✗ Paper's formula is INCORRECT!")
    print(f"  Difference = {difference}")
    print()
    print("  The CORRECT formula should be:")
    print(f"    C(l) = {C_l_expanded}")
    print(f"    LaTeX: {latex(C_l_expanded)}")

print()
print("="*80)
print("Verification complete.")
print("="*80)
