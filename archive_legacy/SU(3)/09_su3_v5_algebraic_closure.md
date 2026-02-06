# SU(3) Lattice v5: Self-Consistent Algebraic Closure

## The Goal
To achieve 10^-14 precision by ensuring the generators are self-consistent, rather than trying to match external normalization conventions.

## 1. The Gelfand Coordinates (L-indices)
Use these exactly:
- Row 3: l13 = m13 + 2,  l23 = m23 + 1,  l33 = m33
- Row 2: l12 = m12 + 1,  l22 = m22
- Row 1: l11 = m11

## 2. The Core Raising Operators (E12 and E23)
Implement ONLY these two raising operators using the standardized GT formula. 

### E12 (Shifts m11 -> m11 + 1):
Coeff = sqrt( | (l12 - l11) * (l22 - l11 - 1) | )

### E23 (Shifts m12 or m22):
- **For m12 -> m12 + 1:** Coeff = sqrt( | ((l13 - l12 - 1)*(l23 - l12 - 1)*(l33 - l12 - 1)*(l11 - l12)) / ((l12 - l22)*(l12 - l22 + 1)) | )
- **For m22 -> m22 + 1:** Coeff = sqrt( | ((l13 - l22 - 1)*(l23 - l22 - 1)*(l33 - l22 - 1)*(l11 - l22)) / ((l22 - l12)*(l22 - l12 + 1)) | )

## 3. Algebraic Construction (The "No-Fail" Method)
Instead of hardcoding the other 6 generators, derive them to ensure the SU(3) triangle is closed:
1.  **Lowering:** E21 = E12.H, E32 = E23.H
2.  **Cross-Root:** E13 = (E12 @ E23) - (E23 @ E12)
3.  **Cross-Lowering:** E31 = E13.H
4.  **Cartans (The Keys):** - T3 = 0.5 * ((E12 @ E21) - (E21 @ E12))
    - T_U = 0.5 * ((E23 @ E32) - (E32 @ E23))
    - T8 = (1.0 / sqrt(3.0)) * (T3 + 2.0 * T_U)

## 4. Final Validation
With this "Bottom-Up" approach, verify:
- [T3, T8] == 0
- [E12, E23] == E13
- Casimir C2 = (E12@E21 + E21@E12 + E23@E32 + E32@E23 + E13@E31 + E31@E13 + T3@T3 + T8@T8) 
- Verify C2 is a multiple of the Identity matrix for any (p,q) representation.

## Instruction for AI
"Implement the 'Bottom-Up' construction. By deriving T3 and T8 from the commutators of your ladder operators, you eliminate the normalization mismatch. The resulting matrices will be a perfectly valid, sparse representation of SU(3) to machine precision."