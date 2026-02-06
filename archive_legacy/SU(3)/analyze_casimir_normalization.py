"""
Analyze Casimir eigenvalues to determine correct normalization factor.
"""

import numpy as np
from operators_v8 import SU3OperatorsV8

np.set_printoptions(precision=6, suppress=True)

print("Casimir Eigenvalue Analysis\n")
print("="*70)

representations = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1)]

results = []

for p, q in representations:
    ops = SU3OperatorsV8(p, q)
    
    # Compute Casimir
    C2 = (ops.E12 @ ops.E21 + ops.E21 @ ops.E12 +
          ops.E23 @ ops.E32 + ops.E32 @ ops.E23 +
          ops.E13 @ ops.E31 + ops.E31 @ ops.E13 +
          ops.T3 @ ops.T3 + ops.T8 @ ops.T8)
    
    # Get eigenvalues
    casimir_eigenvalues = np.diag(C2).real
    mean_eigenvalue = np.mean(casimir_eigenvalues)
    std_eigenvalue = np.std(casimir_eigenvalues)
    
    # Theoretical value
    expected = (p**2 + q**2 + 3*p + 3*q + p*q) / 3.0
    
    # Ratio
    if expected > 1e-10:
        ratio = mean_eigenvalue / expected
    else:
        ratio = np.nan
    
    results.append({
        'p': p,
        'q': q,
        'dim': ops.dim,
        'mean': mean_eigenvalue,
        'std': std_eigenvalue,
        'expected': expected,
        'ratio': ratio,
        'eigenvalues': casimir_eigenvalues
    })
    
    print(f"({p},{q}) dim={ops.dim:2d}: mean={mean_eigenvalue:7.4f}, std={std_eigenvalue:.4f}, expected={expected:7.4f}, ratio={ratio:.4f}")
    if ops.dim <= 8:
        print(f"  Eigenvalues: {casimir_eigenvalues}")

# Analyze the ratio pattern
print("\n" + "="*70)
print("Ratio Analysis (mean / expected):")
print("="*70)

ratios = [r['ratio'] for r in results if not np.isnan(r['ratio'])]
print(f"Ratios: {ratios}")
print(f"Mean ratio: {np.mean(ratios):.6f}")
print(f"Std of ratios: {np.std(ratios):.6f}")

# The ratio should tell us the normalization factor
# If all ladder operators are scaled by factor α, then C2 scales by α²
# So if ratio = C2_actual / C2_expected = α², then α = sqrt(ratio)

mean_ratio = np.mean(ratios)
normalization_factor = np.sqrt(mean_ratio)

print(f"\nNormalization factor α = sqrt(mean_ratio) = {normalization_factor:.6f}")
print(f"This means ladder operators should be scaled by 1/α = {1/normalization_factor:.6f}")

# Check individual ladder operator contributions
print("\n" + "="*70)
print("Detailed (1,0) Analysis:")
print("="*70)

ops = SU3OperatorsV8(1, 0)

print("\nIndividual contributions to Casimir:")
contrib_E12 = ops.E12 @ ops.E21 + ops.E21 @ ops.E12
contrib_E23 = ops.E23 @ ops.E32 + ops.E32 @ ops.E23
contrib_E13 = ops.E13 @ ops.E31 + ops.E31 @ ops.E13
contrib_T3 = ops.T3 @ ops.T3
contrib_T8 = ops.T8 @ ops.T8

print(f"E12@E21 + E21@E12 diagonal: {np.diag(contrib_E12).real}")
print(f"E23@E32 + E32@E23 diagonal: {np.diag(contrib_E23).real}")
print(f"E13@E31 + E31@E13 diagonal: {np.diag(contrib_E13).real}")
print(f"T3@T3 diagonal: {np.diag(contrib_T3).real}")
print(f"T8@T8 diagonal: {np.diag(contrib_T8).real}")

total = contrib_E12 + contrib_E23 + contrib_E13 + contrib_T3 + contrib_T8
print(f"\nTotal Casimir diagonal: {np.diag(total).real}")
print(f"Expected (all equal to): {(1**2 + 0**2 + 3*1 + 3*0 + 1*0) / 3.0}")

# Check if the std of eigenvalues correlates with anything
print("\n" + "="*70)
print("Eigenvalue Spread Analysis:")
print("="*70)

for r in results:
    if r['expected'] > 1e-10:
        relative_std = r['std'] / r['expected']
        print(f"({r['p']},{r['q']}): std/expected = {relative_std:.4f}")
