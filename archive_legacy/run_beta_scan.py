"""
Beta scan for Wilson gauge fields - Phase 9.1
Test hypothesis: g² = C × 1/(4π) across different β values
"""
import sys
sys.path.append('src')
from gauge_theory import WilsonGaugeField
import numpy as np
import matplotlib.pyplot as plt

print('='*80)
print('PHASE 9.1: GAUGE COUPLING β-SCAN')
print('='*80)
print()
print('Testing hypothesis: g² = C × 1/(4π)')
print()

# Beta values to scan
# β = 4/g² for SU(2), so we scan around β where g² ≈ 1/(4π)
# Expected: β ≈ 4 / (1/(4π)) ≈ 50.27
beta_values = [20, 30, 40, 50, 60, 80, 100]
one_over_4pi = 1 / (4 * np.pi)

print(f'Reference: 1/(4π) = {one_over_4pi:.6f}')
print(f'Expected β for g² = 1/(4π): β ≈ {4/one_over_4pi:.2f}')
print()
print(f'Scanning {len(beta_values)} β values...')
print('='*80)
print()

results = []

for i, beta in enumerate(beta_values):
    print(f'\n[{i+1}/{len(beta_values)}] β = {beta:.1f}')
    print('-'*80)
    
    # Create gauge field
    gauge = WilsonGaugeField(ell_max=2, beta=beta)
    print()
    
    # Initial state
    avg_plaq_init = gauge.average_plaquette()
    print(f'Initial: ⟨U⟩ = {avg_plaq_init:.6f}')
    
    # Thermalize (shorter for scan)
    print('Thermalizing (100 sweeps)...')
    gauge.thermalize(n_sweeps=100)
    
    # Measure
    print('Measuring (10 samples)...')
    data = gauge.measure_observables(n_measurements=10, n_sweeps_between=5)
    
    # Store results
    results.append({
        'beta': beta,
        'g2_bare': data['g2_bare'],
        'g2_eff_mean': data['g2_mean'],
        'g2_eff_std': data['g2_std'],
        'avg_plaq_mean': data['avg_plaq_mean'],
        'avg_plaq_std': data['avg_plaq_std']
    })
    
    # Print
    ratio_bare = data['g2_bare'] / one_over_4pi
    ratio_eff = data['g2_mean'] / one_over_4pi
    
    print(f'\nResults:')
    print(f'  g²_bare = {data["g2_bare"]:.6f}')
    print(f'  g²_eff  = {data["g2_mean"]:.6f} ± {data["g2_std"]:.6f}')
    print(f'  Ratio (bare):     {ratio_bare:.4f}')
    print(f'  Ratio (effective): {ratio_eff:.4f}')
    
    if 0.9 < ratio_eff < 1.1:
        print('  ✓✓✓ EXCELLENT match to 1/(4π)!')
    elif 0.5 < ratio_eff < 2.0:
        print('  ✓✓ GOOD match to 1/(4π)')
    else:
        print(f'  ✗ Ratio = {ratio_eff:.2f} (not close)')

print()
print('='*80)
print('SCAN COMPLETE!')
print('='*80)
print()

# Analysis
print('SUMMARY TABLE:')
print('-'*80)
print(f'{"β":>8} {"g²_bare":>10} {"g²_eff":>10} {"Ratio_bare":>12} {"Ratio_eff":>12}')
print('-'*80)

for r in results:
    ratio_bare = r['g2_bare'] / one_over_4pi
    ratio_eff = r['g2_eff_mean'] / one_over_4pi
    print(f'{r["beta"]:8.1f} {r["g2_bare"]:10.6f} {r["g2_eff_mean"]:10.6f} '
          f'{ratio_bare:12.4f} {ratio_eff:12.4f}')

print('-'*80)
print()

# Find best match
best_idx = np.argmin([abs(r['g2_eff_mean'] - one_over_4pi) for r in results])
best = results[best_idx]
print(f'Best match: β = {best["beta"]:.1f}')
print(f'  g²_eff = {best["g2_eff_mean"]:.6f}')
print(f'  1/(4π) = {one_over_4pi:.6f}')
print(f'  Difference: {abs(best["g2_eff_mean"] - one_over_4pi):.6f} ({abs(best["g2_eff_mean"] - one_over_4pi)/one_over_4pi * 100:.2f}%)')
print()

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: g² vs β
ax = axes[0]
betas = [r['beta'] for r in results]
g2_bare = [r['g2_bare'] for r in results]
g2_eff = [r['g2_eff_mean'] for r in results]
g2_eff_err = [r['g2_eff_std'] for r in results]

ax.plot(betas, g2_bare, 'bo-', label='g² bare = 4/β', linewidth=2, markersize=8)
ax.errorbar(betas, g2_eff, yerr=g2_eff_err, fmt='rs-', label='g² effective (measured)', 
            linewidth=2, markersize=8, capsize=5)
ax.axhline(one_over_4pi, color='green', linestyle='--', linewidth=2, 
           label=f'1/(4π) = {one_over_4pi:.6f}')
ax.set_xlabel('β', fontsize=12)
ax.set_ylabel('g²', fontsize=12)
ax.set_title('Gauge Coupling vs β', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 2: Ratio to 1/(4π)
ax = axes[1]
ratio_bare = [r['g2_bare'] / one_over_4pi for r in results]
ratio_eff = [r['g2_eff_mean'] / one_over_4pi for r in results]

ax.plot(betas, ratio_bare, 'bo-', label='Bare', linewidth=2, markersize=8)
ax.plot(betas, ratio_eff, 'rs-', label='Effective', linewidth=2, markersize=8)
ax.axhline(1.0, color='green', linestyle='--', linewidth=2, label='Perfect match')
ax.fill_between(betas, 0.9, 1.1, alpha=0.2, color='green', label='±10% band')
ax.set_xlabel('β', fontsize=12)
ax.set_ylabel('g² / [1/(4π)]', fontsize=12)
ax.set_title('Ratio to Geometric Constant', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/gauge_beta_scan.png', dpi=150, bbox_inches='tight')
print('Saved: results/gauge_beta_scan.png')
plt.show()

print()
print('='*80)
print('CONCLUSION:')
print('='*80)

# Check how many are within 10% of 1/(4π)
within_10_percent = sum(1 for r in results if 0.9 < r['g2_eff_mean']/one_over_4pi < 1.1)

print(f'{within_10_percent}/{len(results)} β values give g²_eff within 10% of 1/(4π)')
print()

if within_10_percent >= len(results) // 2:
    print('✓✓✓ STRONG EVIDENCE: g² consistently matches 1/(4π) across β values!')
    print()
    print('This suggests the geometric constant 1/(4π) plays a fundamental')
    print('role in the gauge coupling of our discrete SU(2) lattice.')
    print()
    print('**POTENTIAL BREAKTHROUGH**: Physical coupling from pure geometry!')
elif within_10_percent >= 2:
    print('✓✓ MODERATE EVIDENCE: Some β values show g² ≈ 1/(4π)')
    print()
    print('Further investigation needed, but promising initial results.')
else:
    print('✗ WEAK EVIDENCE: g² does not consistently match 1/(4π)')
    print()
    print('The hypothesis may need refinement or the connection is more subtle.')

print('='*80)
