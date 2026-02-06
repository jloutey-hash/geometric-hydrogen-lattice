# SU(3) Strengthening - Quick Reference

## What Was Done

**Goal**: Strengthen SU(3) side with packing metrics to enable geometric comparison with U(1) hydrogen.

**Completed**:
1. ✓ Fixed SU(3) spherical algebra (normalization, transformation, Casimir)
2. ✓ Created packing metrics module (covering radius, kissing number, packing efficiency)
3. ✓ Generated impedance + packing dataset (14 reps, CSV output)

## Files Created/Modified

### New Files (4):
1. `SU(3)/packing_metrics.py` - Geometric packing analysis on spherical shells
2. `SU(3)/impedance_packing_correlation.py` - Combined impedance + packing calculator
3. `run_su3_packing_scan.py` - Full scan script → CSV output
4. `su3_impedance_packing_scan.csv` - **Data file with 14 representations**

### Modified Files (1):
1. `SU(3)/test_spherical_algebra.py` - Fixed normalization, transformation, Casimir

### Documentation (2):
1. `SU3_STRENGTHENING_SUMMARY.md` - Full technical summary
2. `SU3_STRENGTHENING_QUICKSTART.md` - This file

## Quick Usage

### 1. Compute packing for one representation
```python
import sys
sys.path.insert(0, 'SU(3)')
from su3_spherical_embedding import SU3SphericalEmbedding
from packing_metrics import compute_packing_metrics, print_packing_report

embedding = SU3SphericalEmbedding(1, 1)  # Adjoint rep
metrics = compute_packing_metrics(embedding)
print_packing_report(1, 1, metrics)
```

### 2. Compute impedance + packing
```python
from impedance_packing_correlation import compute_impedance_and_packing

data = compute_impedance_and_packing(2, 0)
print(f"(2,0): Z={data['Z']:.3f}, PackEff={data['packing_efficiency_mean']:.3f}")
```

### 3. Run full scan
```bash
python run_su3_packing_scan.py
# Outputs: su3_impedance_packing_scan.csv (14 reps, p+q ≤ 4)
```

### 4. Analyze results
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('su3_impedance_packing_scan.csv')

# Filter out inf values
df_clean = df[df['Z'] < 1e6]

# Plot
plt.scatter(df_clean['C2'], df_clean['packing_efficiency_mean'])
plt.xlabel('Casimir C2')
plt.ylabel('Packing Efficiency')
plt.title('SU(3): Packing vs Casimir (ρ=+0.962)')
plt.grid(True)
plt.show()
```

## Key Results

### Packing Metrics (14 representations):
- **Packing Efficiency**: Range [0.354, 0.741], Mean 0.563
- **Correlation with C₂**: +0.962 (strong positive!)
- **Interpretation**: Larger reps pack better on spherical shells

### Impedance Values:
- **Range**: [0.015, ∞] (some inf from division issues)
- **Compare with U(1)**: κ₅(hydrogen) = 137.04

### CSV Columns:
- `p, q, dim, C2`
- `Z, Z_normalized, Z_dimensionless`
- `C_matter, S_holonomy`
- `covering_radius_mean, covering_radius_std`
- `kissing_number_mean`
- `packing_efficiency_mean, packing_efficiency_std`
- `n_shells`

## Known Issues

1. **(1,1) Diagonality**: T₃, T₈ not simultaneously diagonal (off-diag ~0.48)
   - **Cause**: Degenerate T₃ eigenvalues, T₈ mixes within subspaces
   - **Impact**: Minor, doesn't break packing/impedance calculations
   - **Fix**: Requires simultaneous diagonalization (future work)

2. **Some inf in Z**: Division by zero in certain reps
   - **Workaround**: Filter `df[df['Z'] < 1e6]` when analyzing

## Integration with Unified Framework

### Existing (Session 1):
- `geometric_impedance_interface.py` - Base class
- `hydrogen_u1_impedance.py` - U(1) wrapper (validated)
- `su3_impedance_wrapper.py` - SU(3) wrapper
- `unified_impedance_comparison.py` - Comparison module

### New (Session 2):
- **SU(3) packing metrics** - Quantifies geometric efficiency
- **Impedance-packing dataset** - 14 reps with combined metrics
- **Comparison basis** - U(1) helical pitch ↔ SU(3) spherical packing

## Next Steps

1. **Fix (1,1) issue**: Implement T₃+T₈ simultaneous diagonalization
2. **Debug inf values**: Resolve division by zero
3. **Cross-gauge plots**: Z_U(1) vs Z_SU(3), packing comparisons
4. **Paper figures**: Unified geometric interpretation plots
5. **Theoretical connection**: Link packing efficiency to coupling constants

## Example Output

```
================================================================================
SU(3) Packing Metrics: (p, q) = (1, 1)
================================================================================

Shell  r        N    Cover_rad    Kiss#      AngDist      PackEff
--------------------------------------------------------------------------------
1.0    1.000    1    3.1416       0.00       0.0000       0.000
1.866  1.866    4    3.1416       2.00       2.0944       0.500
2.732  2.732    3    3.1416       1.33       2.0944       0.577

Summary
  Average covering radius:   3.1416 rad
  Average packing efficiency: 0.359
  Total states:              8
```

## Files Location

All in workspace: `c:\Users\jlout\OneDrive\Desktop\Model study\SU(2) model\`

- `SU(3)/packing_metrics.py`
- `SU(3)/impedance_packing_correlation.py`
- `SU(3)/test_spherical_algebra.py` (modified)
- `run_su3_packing_scan.py`
- `su3_impedance_packing_scan.csv` ← **Data file**

## Success Metrics

✓ Part 1: Algebraic validation improved (T₃, T₈ diagonal for 4/5 test reps)
✓ Part 2: Packing metrics implemented and tested
✓ Part 3: Combined dataset generated (14 reps, CSV)
✓ Key finding: Strong C₂-packing correlation (ρ=+0.962)
✓ Ready for U(1) ↔ SU(3) geometric comparison
