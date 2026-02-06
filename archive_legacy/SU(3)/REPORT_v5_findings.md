# SU(3) v5 Analysis: Root Cause Identified

## Executive Summary
All three specifications (07, 08, 09) contain **systematically incorrect E23 formulas**. The algebraic closure approach (v5) exposes the issue clearly: the formulas produce E23 coefficients that are too large by factors of √2 or 2, causing cascade failures in T3, T8, and Casimir.

## Key Finding: Normalization Mismatch

### Test Case: (1,0) Fundamental Representation
**States (GT patterns):**
- State 0: (1,0,0,0,0,0) → I₃=0, Y=-2/3
- State 1: (1,0,0,1,0,0) → I₃=-1/2, Y=1/3  
- State 2: (1,0,0,1,0,1) → I₃=1/2, Y=1/3

**E23 Transition: State 0 → State 1**
- Spec 09 formula gives: **coefficient = 1.0**
- Correct value should be: **coefficient = 1/√2 ≈ 0.707**
- Error factor: **√2 ≈ 1.414**

### Consequence Chain

1. **E23 coefficient wrong** → [E23, E32] diagonal wrong
2. **[E23, E32] = diag(-1, 1, 0)** but should be diag(-0.5, 1.5, -1)
3. **Algebraically-derived T3 = diag(0, -1, 1)** but theory requires diag(0, -0.5, 0.5)
4. **T3 off by factor of 2** → all commutators involving T3 fail
5. **Casimir wrong** → eigenvalue 4.22 instead of 1.33 (factor of 3.2)

## Validation Results Summary

| Irrep | Max Comm Error | Casimir Error | Status |
|-------|----------------|---------------|---------|
| (0,0) | 0.000e+00 | 0.000e+00 | ✅ PERFECT |
| (1,0) | 3.000e+00 | 2.889e+00 | ❌ FAIL |
| (1,1) | 3.674e+00 | 5.801e+00 | ❌ FAIL |
| (2,1) | 5.196e+00 | 9.327e+00 | ❌ FAIL |

**Target:** < 1e-13

## What Works in v5

✅ **Algebraic closure concept is sound:**
- [T3, T8] = 0 (exact)
- [E12, E21] - 2*T3 (exact)
- [E12, E23] = E13 (exact by construction)
- Hermiticity (exact)

✅ **I-spin operators perfect:**
- E12 formula is correct
- Achieves machine precision ~1e-16

## Root Problem

**All three specs have wrong E23 formulas:**
- Spec 07 (m-index): produces ~3.0 error
- Spec 08 (l-index shifted): produces ~8.0 error (worse!)
- Spec 09 (l-index algebraic closure): produces ~1.5-5.0 error

The formulas compute coefficients that are systematically too large. For (1,0) fundamental:
- Formula: `sqrt(|(-2)/2|) = 1.0`
- Correct: `1/√2 = 0.707`

## Recommendation

Need **verified Biedenharn-Louck formulas from authoritative source**:
1. Original 1981 paper: Biedenharn & Louck, "Angular Momentum in Quantum Physics"
2. Reference implementation: SymPy's quantum mechanics module
3. Textbook formulas: Georgi "Lie Algebras in Particle Physics" or similar

The GT framework is proven correct (I-spin at machine precision). Issue is solely getting correct E23 coefficients.

## Files Created
- `operators_v5.py`: Algebraic closure implementation
- `validate_v5.py`: Comprehensive testing across 8 irreps
- `debug_v5_fundamental.py`: Detailed (1,0) analysis
- `compare_t3_t8_theory_vs_algebraic.py`: Exposes factor-of-2 error in T3
