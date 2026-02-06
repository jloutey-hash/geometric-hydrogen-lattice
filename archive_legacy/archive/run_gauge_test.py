"""Quick gauge field test for Phase 9.1"""
import sys
sys.path.append('src')
from gauge_theory import WilsonGaugeField

print('='*60)
print('PHASE 9.1: QUICK GAUGE FIELD TEST')
print('='*60)
print()

# Create small gauge field
gauge = WilsonGaugeField(ell_max=2, beta=50.0)
print()

# Initial state
print('Initial state (cold start):')
print(f'  Average plaquette: {gauge.average_plaquette():.6f}')
print(f'  g² effective: {gauge.effective_coupling():.6f}')
print()

# Thermalize
print('Running 200 thermalization sweeps...')
gauge.thermalize(n_sweeps=200)
print()

# Measure
print('Measuring observables (20 samples)...')
data = gauge.measure_observables(n_measurements=20, n_sweeps_between=5)
print()

# Results
print('='*60)
print('RESULTS')
print('='*60)
print(f'g² bare:      {data["g2_bare"]:.6f}')
print(f'g² effective: {data["g2_mean"]:.6f} ± {data["g2_std"]:.6f}')
print(f'1/(4π):       {data["one_over_4pi"]:.6f}')
print(f'Ratio g²/[1/(4π)]: {data["g2_mean"]/data["one_over_4pi"]:.6f}')
print('='*60)
print()

# Interpretation
ratio = data["g2_mean"]/data["one_over_4pi"]
if 0.5 < ratio < 2.0:
    print('✓✓✓ STRONG EVIDENCE: g² and 1/(4π) are same order of magnitude!')
    print(f'    They differ by only a factor of {ratio:.2f}')
elif 0.1 < ratio < 10:
    print('✓✓ SUGGESTIVE: g² and 1/(4π) are close')
    print(f'   Ratio = {ratio:.2f}')
else:
    print('✗ No clear connection between g² and 1/(4π)')
    print(f'  Ratio = {ratio:.2f}')
print()
print('Phase 9.1 quick test complete!')
