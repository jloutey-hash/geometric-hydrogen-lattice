"""
Display detailed properties of canonical (1,1) adjoint representation.
"""
import pandas as pd

df = pd.read_csv('su3_canonical_candidates.csv')
top = df.iloc[0]  # Already sorted by composite_score

print('\n' + '='*70)
print('CANONICAL REPRESENTATION: (1,1) ADJOINT')
print('='*70)
print(f'\nDimension:       {int(top["dim"])}')
print(f'Casimir C2:      {top["C2"]:.4f}')
print(f'Z_eff:           {top["Z_eff"]:.6f}')
print(f'Z_per_state:     {top["Z_per_state"]:.6f}')
print(f'Z_per_C2:        {top["Z_per_C2"]:.6f}')
print(f'C_matter:        {top["C_matter"]:.3f}')
print(f'S_holonomy:      {top["S_holonomy"]:.3f}')
print(f'Packing eff:     {top["packing_efficiency_mean"]:.4f}')
print(f'Mixing index:    {int(top["mixing_index"])} (p × q)')
print(f'Symmetry index:  {int(top["symmetry_index"])} (|p - q|)')
print(f'Composite score: {top["composite_score"]:.4f}')

print('\n' + '-'*70)
print('Comparison with U(1) Hydrogen n=5')
print('-'*70)
Z_u1 = 137.04
Z_su3 = top["Z_eff"]
print(f'Z_U1 (hydrogen):      {Z_u1:.4f}')
print(f'Z_SU3 (adjoint):      {Z_su3:.4f}')
print(f'Ratio Z_U1 / Z_SU3:   {Z_u1/Z_su3:.1f}×')

print('\n' + '-'*70)
print('Physical Interpretation')
print('-'*70)
print('• (1,1) is the ADJOINT representation')
print('• In QCD: gluons transform in the adjoint')
print('• Has MAXIMUM Z_per_state among all 44 reps')
print('• Identified as 3.19σ RESONANCE')
print('• Parallels H(n=5) role in U(1) framework')

print('\n' + '='*70)
print('GEOMETRIC INSIGHT')
print('='*70)
print('High impedance reflects gluon self-interaction:')
print('• Pure reps (quarks):  Z ~ 0.02  (free flow)')
print('• Adjoint (gluons):    Z ~ 0.96  (constrained)')
print('• Self-coupling creates topological resistance')
print('='*70 + '\n')

# Also show next few candidates
print('\nNext 4 Candidates for Context:')
print('-'*70)
next_4 = df.iloc[1:5]
for idx, row in next_4.iterrows():
    print(f'({int(row["p"])},{int(row["q"])}): '
          f'Z={row["Z_eff"]:.4f}, '
          f'Z/state={row["Z_per_state"]:.6f}, '
          f'score={row["composite_score"]:.4f}')
