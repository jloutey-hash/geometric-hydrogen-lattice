"""
Quick test to verify SU(3) impedance fix works for higher-dimensional reps.
"""
import sys
sys.path.insert(0, 'SU(3)')
import warnings
warnings.filterwarnings('ignore')

from su3_impedance import SU3SymplecticImpedance

# Test representations
reps = [(1,0), (0,1), (2,0), (0,2), (1,1), (3,0), (0,3), (2,1), (1,2)]

print("="*80)
print("SU(3) Impedance Fix Verification")
print("="*80)
print(f"{'(p,q)':<10} {'dim':<6} {'C2':<10} {'C_matter':<15} {'Z':<15}")
print("-"*80)

results = []
for p, q in reps:
    calc = SU3SymplecticImpedance(p, q, verbose=False)
    result = calc.compute_impedance()
    
    results.append({
        'p': p,
        'q': q,
        'dim': calc.dim,
        'C2': calc.C2,
        'C_matter': result.C_matter,
        'Z': result.Z_impedance
    })
    
    status = "✓" if result.C_matter > 0 and result.Z_impedance < 1e10 else "✗"
    print(f"({p},{q}){'':<7} {calc.dim:<6} {calc.C2:<10.3f} {result.C_matter:<15.3f} {result.Z_impedance:<15.6f} {status}")

print("-"*80)

# Count finite values
finite_count = sum(1 for r in results if r['C_matter'] > 0 and r['Z'] < 1e10)
print(f"\nFinite Z values: {finite_count}/{len(results)}")

if finite_count == len(results):
    print("✓ SUCCESS: All representations have finite impedance!")
else:
    print(f"✗ ISSUE: {len(results) - finite_count} representations still have non-finite Z")
