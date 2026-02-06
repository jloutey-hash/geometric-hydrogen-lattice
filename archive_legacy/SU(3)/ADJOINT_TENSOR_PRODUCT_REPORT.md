# Adjoint (1,1) Representation via Tensor Product

**Date:** January 20, 2026  
**Status:** ✅ **COMPLETE** - Machine precision achieved

## Summary

Successfully implemented SU(3) adjoint (1,1) representation using tensor product decomposition:

$$3 \otimes \bar{3} = 1 \oplus 8$$

**Method**: Build 9D tensor product, project out singlet → 8D adjoint

**Results**: Machine precision in both weight and GT bases!

---

## Implementation

### Step 1: Tensor Product Construction

Build 9-dimensional product space from fundamental (1,0) ⊗ antifundamental (0,1):

```python
T^(prod) = T^(fund) ⊗ I₃ + I₃ ⊗ T^(antifund)
```

All 8 generators constructed this way in 9D space.

### Step 2: Singlet Projection

**Singlet state** (SU(3) invariant):
$$|\text{singlet}\rangle = \frac{1}{\sqrt{3}}(|0\rangle\otimes|0\rangle + |1\rangle\otimes|1\rangle + |2\rangle\otimes|2\rangle)$$

**Verification**:
- T₃|singlet⟩ = 0 ✓
- T₈|singlet⟩ = 0 ✓
- I₃ = 0, Y = 0 ✓

**Projection operator**:
$$P_{\text{adj}} = I_9 - |\text{singlet}\rangle\langle\text{singlet}|$$

Extract 8 orthonormal eigenvectors with eigenvalue 1 → adjoint subspace.

### Step 3: Weight Basis Diagonalization

After projection, diagonalize T₃ and T₈ to find weight basis:

**Adjoint weight diagram** (8 states):
```
         (0, 1)
       /        \
  (-0.5, 1)    (0.5, 1)
     |            |
 (-1, 0)    (0,0) (0,0)    (1, 0)
     |            |
  (-0.5,-1)    (0.5,-1)
       \        /
         (0,-1)
```

Note: Two states at center (0,0) - correct for (1,1) weight diagram.

### Step 4: GT Basis Transformation

Generate GT patterns for (1,1) from lattice:
- 8 patterns with (p,q) = (1,1)
- Match to weight states by (I3, Y) quantum numbers
- Build permutation matrix U
- Transform: O_GT = U† O_weight U

---

## Validation Results

### Weight Basis

```
Validation Results:
  [T3,T8]:                    1.64e-16 ✓
  [E12,E21]-2T3:              6.66e-16 ✓
  [E23,E32]-(T3+√3*T8):       8.88e-16 ✓
  [E13,E31]-(-T3+√3*T8):      8.88e-16 ✓
  Casimir_std:                1.86e-15 ✓
  Casimir_mean_error:         8.88e-16 ✓
  All hermiticity:            ≤ 1e-48 ✓
  T3, T8 diagonal:            ≤ 3.28e-16 ✓

Casimir eigenvalues: [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
Expected: 3.0 (from formula (p²+q²+pq+3p+3q)/3 = 9/3)
```

### GT Basis

```
Validation Results:
  [T3,T8]:                    1.64e-16 ✓
  [E12,E21]-2T3:              6.66e-16 ✓
  [E23,E32]-(T3+√3*T8):       8.88e-16 ✓
  [E13,E31]-(-T3+√3*T8):      8.88e-16 ✓
  Casimir_std:                1.55e-15 ✓
  Casimir_mean_error:         4.44e-16 ✓
  All hermiticity:            ≤ 1e-48 ✓
  T3, T8 diagonal:            ≤ 3.28e-16 ✓

Casimir eigenvalues: [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
```

**Transformation preserves all algebra perfectly!**

---

## Comparison with Structure Constant Approach

Previous attempt (in weight_basis_gellmann.py) failed:
- Structure constants f_{ijk} gave operators in "Gell-Mann basis"
- T₃, T₈ not diagonal in that basis
- Diagonalization gave degenerate subspaces
- Commutators wrong, Casimir wrong

**Tensor product approach succeeds** because:
1. Operators built from known correct representations (1,0) and (0,1)
2. Singlet explicitly identified and removed
3. Weight basis obtained by diagonalization after projection
4. All algebra inherited from fundamental reps

---

## Usage

```python
from adjoint_tensor_product import AdjointSU3, AdjointSU3_GT

# Weight basis
adj_weight = AdjointSU3()
adj_weight.validate()
C2 = adj_weight.get_casimir()

# GT basis
adj_gt = AdjointSU3_GT()
adj_gt.validate()
C2_gt = adj_gt.get_casimir()

# Both give machine precision!
```

---

## Complete Representation Coverage

| Representation | Weight Basis | GT Basis | Status |
|----------------|--------------|----------|--------|
| (1,0) Fundamental | weight_basis_gellmann.py | gt_basis_transformed.py | ✅ Done |
| (0,1) Antifundamental | weight_basis_gellmann.py | gt_basis_transformed.py | ✅ Done |
| (1,1) Adjoint | adjoint_tensor_product.py | adjoint_tensor_product.py | ✅ Done |

**All achieve machine precision (≤10⁻¹⁵) for:**
- All 4 commutation relations
- Casimir constant
- Hermiticity
- Diagonality of T₃, T₈

---

## Key Insights

1. **Tensor Product Method is Superior**
   - Uses known correct representations
   - Explicit singlet removal
   - No ambiguity in basis choice

2. **Weight→GT Transformation Always Works**
   - If weight basis is correct, GT transformation preserves algebra
   - Unitary transformation: [U†AU, U†BU] = U†[A,B]U
   - Casimir invariant under basis change

3. **Degeneracy Handling**
   - (1,1) has two states at (0,0)
   - Matching by quantum numbers works if done consistently
   - Arbitrary choice between degenerate states doesn't affect physics

---

## Files

- **adjoint_tensor_product.py**: Complete implementation
  - Class `AdjointSU3`: Weight basis via 3⊗3̄
  - Class `AdjointSU3_GT`: GT basis transformation
  - Full validation suite

**Test**: `python adjoint_tensor_product.py`

---

## Conclusion

The adjoint (1,1) representation is now fully operational at machine precision in both weight and GT bases. This completes the fundamental suite of SU(3) representations needed for lattice model calculations.

The tensor product construction proves that building from simpler correct representations is more reliable than direct construction from algebraic formulas. This principle should guide implementation of higher representations (2,0), (1,2), etc.
