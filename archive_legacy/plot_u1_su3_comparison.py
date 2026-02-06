"""
Enhanced U(1) vs SU(3) comparison with visualization.

Generates comparison plot showing:
1. U(1) hydrogen Z value (horizontal line)
2. SU(3) fundamental reps (pure: p=0 or q=0)
3. SU(3) mixed reps (both p>0 and q>0)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unified_impedance_comparison import compare_u1_su3_geometric

# Compare with comprehensive set of SU(3) reps
pure_reps = [(1,0), (0,1), (2,0), (0,2), (3,0), (0,3), (4,0), (0,4)]
mixed_reps = [(1,1), (1,2), (2,1), (1,3), (3,1), (2,2)]

all_reps = pure_reps + mixed_reps

# Get comparison data
df = compare_u1_su3_geometric(n_hydrogen=5, su3_reps=all_reps, verbose=False)

# Separate by gauge group
df_u1 = df[df['gauge_group'] == 'U(1)']
df_su3 = df[df['gauge_group'] == 'SU(3)']

# Classify SU(3) reps
df_su3['rep_type'] = df_su3['system'].apply(
    lambda s: 'pure' if '(0,' in s or ',0)' in s else 'mixed'
)

df_pure = df_su3[df_su3['rep_type'] == 'pure']
df_mixed = df_su3[df_su3['rep_type'] == 'mixed']

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: Z values comparison
ax1 = axes[0]

# U(1) reference line
Z_u1 = df_u1['Z'].iloc[0]
ax1.axhline(Z_u1, color='red', linestyle='--', linewidth=2, 
            label=f'U(1) H(n=5): Z={Z_u1:.2f}', alpha=0.7)

# SU(3) pure reps
if len(df_pure) > 0:
    ax1.scatter(df_pure['dim'], df_pure['Z'], s=100, alpha=0.7, 
                label=f'SU(3) pure (n={len(df_pure)})', marker='o', color='blue')
    for idx, row in df_pure.iterrows():
        ax1.annotate(row['system'].replace('SU(3) ', ''), 
                    (row['dim'], row['Z']), fontsize=8, 
                    xytext=(5, 5), textcoords='offset points', alpha=0.6)

# SU(3) mixed reps
if len(df_mixed) > 0:
    ax1.scatter(df_mixed['dim'], df_mixed['Z'], s=150, alpha=0.7,
                label=f'SU(3) mixed (n={len(df_mixed)})', marker='s', color='green')
    for idx, row in df_mixed.iterrows():
        ax1.annotate(row['system'].replace('SU(3) ', ''), 
                    (row['dim'], row['Z']), fontsize=8,
                    xytext=(5, 5), textcoords='offset points', alpha=0.6)

ax1.set_xlabel('Dimension', fontsize=12)
ax1.set_ylabel('Impedance Z', fontsize=12)
ax1.set_title('U(1) vs SU(3) Impedance Comparison', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# Panel 2: Z_per_state vs dimension
ax2 = axes[1]

# Compute Z_per_state for SU(3)
df_su3['Z_per_state'] = df_su3['Z'] / df_su3['dim']

df_pure = df_su3[df_su3['rep_type'] == 'pure']
df_mixed = df_su3[df_su3['rep_type'] == 'mixed']

# U(1) reference (Z_per_state = Z since dim=1)
ax2.axhline(Z_u1, color='red', linestyle='--', linewidth=2,
            label=f'U(1) H(n=5): Z/dim={Z_u1:.2f}', alpha=0.7)

# SU(3) reps
if len(df_pure) > 0:
    ax2.scatter(df_pure['dim'], df_pure['Z_per_state'], s=100, alpha=0.7,
                label='SU(3) pure', marker='o', color='blue')

if len(df_mixed) > 0:
    ax2.scatter(df_mixed['dim'], df_mixed['Z_per_state'], s=150, alpha=0.7,
                label='SU(3) mixed', marker='s', color='green')

# Fit power law for SU(3): Z_per_state ~ dim^(-α)
if len(df_su3) > 3:
    log_dim = np.log(df_su3['dim'])
    log_z_per = np.log(df_su3['Z_per_state'])
    coeffs = np.polyfit(log_dim, log_z_per, 1)
    alpha = -coeffs[0]
    A = np.exp(coeffs[1])
    
    dim_fit = np.linspace(df_su3['dim'].min(), df_su3['dim'].max(), 100)
    z_per_fit = A * dim_fit**(-alpha)
    
    ax2.plot(dim_fit, z_per_fit, 'k-', alpha=0.5, linewidth=1.5,
            label=f'Fit: Z/dim ∝ dim^(-{alpha:.2f})')

ax2.set_xlabel('Dimension', fontsize=12)
ax2.set_ylabel('Z per state', fontsize=12)
ax2.set_title('Normalized Impedance (Z/dim)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('u1_su3_comparison_plots.png', dpi=150, bbox_inches='tight')
print("Comparison plot saved to u1_su3_comparison_plots.png")

# Print summary statistics
print("\n" + "="*80)
print("U(1) vs SU(3) COMPARISON SUMMARY")
print("="*80)

print(f"\nU(1) Hydrogen (n=5):")
print(f"  Z_U1 = {Z_u1:.4f}")

print(f"\nSU(3) Pure Representations (p=0 or q=0):")
print(f"  Count: {len(df_pure)}")
print(f"  Z range: [{df_pure['Z'].min():.6f}, {df_pure['Z'].max():.6f}]")
print(f"  Z mean: {df_pure['Z'].mean():.6f}")
if len(df_pure) > 0:
    print(f"  Ratio Z_U1/Z_SU3(pure): {Z_u1 / df_pure['Z'].mean():.1f}")

print(f"\nSU(3) Mixed Representations (p>0 and q>0):")
print(f"  Count: {len(df_mixed)}")
print(f"  Z range: [{df_mixed['Z'].min():.6f}, {df_mixed['Z'].max():.6f}]")
print(f"  Z mean: {df_mixed['Z'].mean():.6f}")
if len(df_mixed) > 0:
    print(f"  Ratio Z_U1/Z_SU3(mixed): {Z_u1 / df_mixed['Z'].mean():.1f}")

print(f"\nKey Finding:")
print(f"  Mixed reps have Z ~ {df_mixed['Z'].mean() / df_pure['Z'].mean():.1f}× higher than pure reps")
print(f"  Suggests enhanced geometric complexity in non-symmetric color configurations")

print("\n" + "="*80)
print("DISCLAIMER: Geometric exploration ONLY. NOT physical QCD coupling.")
print("="*80)

plt.show()
