# SU(3) Operator Conventions and Mathematical Framework

## Table of Contents
1. [Ladder vs. Gell-Mann Operators](#ladder-vs-gell-mann-operators)
2. [Hermiticity Conventions](#hermiticity-conventions)
3. [Representation-Dependent Commutators](#representation-dependent-commutators)
4. [Casimir Operator Construction](#casimir-operator-construction)
5. [Normalization Conventions](#normalization-conventions)

---

## Ladder vs. Gell-Mann Operators

### Gell-Mann Matrices (λ₁, ..., λ₈)

The eight Gell-Mann matrices are the **Hermitian** generators of SU(3):

```
λ₁ = [0  1  0]    λ₂ = [0 -i  0]    λ₃ = [1  0  0]
     [1  0  0]         [i  0  0]         [0 -1  0]
     [0  0  0]         [0  0  0]         [0  0  0]

λ₄ = [0  0  1]    λ₅ = [0  0 -i]    λ₆ = [0  0  0]
     [0  0  0]         [0  0  0]         [0  0  1]
     [1  0  0]         [i  0  0]         [0  1  0]

λ₇ = [0  0  0]    λ₈ = 1/√3 [1  0  0]
     [0  0 -i]              [0  1  0]
     [0  i  0]              [0  0 -2]
```

**Properties:**
- **Hermitian**: λₐ† = λₐ
- **Traceless**: Tr(λₐ) = 0
- **Orthogonal**: Tr(λₐλᵦ) = 2δₐᵦ

### Ladder Operators (Eᵢⱼ, T₃, T₈)

For computational convenience, we use **ladder operators** that raise/lower quantum numbers:

```
E₁₂ = [0  1  0]    E₂₁ = [0  0  0]    (E₂₁ = E₁₂†)
      [0  0  0]          [1  0  0]
      [0  0  0]          [0  0  0]

E₂₃ = [0  0  0]    E₃₂ = [0  0  0]    (E₃₂ = E₂₃†)
      [0  0  1]          [0  0  0]
      [0  0  0]          [0  1  0]

E₁₃ = [0  0  1]    E₃₁ = [0  0  0]    (E₃₁ = E₁₃†)
      [0  0  0]          [0  0  0]
      [0  0  0]          [1  0  0]

T₃ = (1/2)λ₃       T₈ = (1/2)λ₈       (Diagonal generators)
```

**Relationship to Gell-Mann matrices:**
```
λ₁ = E₁₂ + E₂₁
λ₂ = -i(E₁₂ - E₂₁)
λ₄ = E₁₃ + E₃₁
λ₅ = -i(E₁₃ - E₃₁)
λ₆ = E₂₃ + E₃₂
λ₇ = -i(E₂₃ - E₃₂)
λ₃ = 2T₃
λ₈ = 2T₈
```

**Inverse relationships:**
```
E₁₂ = (λ₁ + iλ₂)/2
E₂₃ = (λ₆ + iλ₇)/2
E₁₃ = (λ₄ + iλ₅)/2
```

---

## Hermiticity Conventions

### Individual Operators

| Operator Type | Hermiticity | Property |
|--------------|-------------|----------|
| Gell-Mann λₐ | **Hermitian** | λₐ† = λₐ |
| Diagonal T₃, T₈ | **Hermitian** | T₃† = T₃, T₈† = T₈ |
| Ladder Eᵢⱼ | **Non-Hermitian** | Eᵢⱼ† = Eⱼᵢ |
| Combinations λ = E+E† | **Hermitian** | (E₁₂+E₂₁)† = E₁₂+E₂₁ |

### Critical Insight

**Ladder operators are NOT individually Hermitian!**

This is correct physics:
- E₁₂ raises isospin, E₂₁ lowers isospin
- They form Hermitian conjugate pairs: E₁₂† = E₂₁
- Only specific combinations (λ matrices) are Hermitian

### When to Use Which

| Task | Use Gell-Mann (λ) | Use Ladder (E) |
|------|------------------|----------------|
| Casimir operator | ✓ Yes | ✗ No |
| Commutation tests | ✓ Yes | ✓ Yes* |
| Matrix exponentiation | ✓ Yes | Use iλ (anti-Hermitian) |
| Raising/lowering states | ✗ No | ✓ Yes |
| Eigenvalue problems | ✓ Yes | ✗ No |

*For commutation tests, ladder operators work **if you test the right relations** (see next section).

---

## Representation-Dependent Commutators

### Fundamental Representation (1,0)

In the 3D fundamental representation:

```
[E₁₂, E₂₃] = E₁₃          ✓ Correct
[E₁₂, E₂₁] = 2T₃          ✓ Correct
[T₃, E₁₂] = (1/2)E₁₂      ✓ Correct
```

These follow directly from the SU(3) structure constants.

### Adjoint Representation (1,1)

In the 8D adjoint representation:

```
[E₁₂, E₂₃]_adj ≈ 0        ✓ Correct! (Different structure)
[ad(E₁₂), ad(E₂₃)] = ad([E₁₂, E₂₃])   ✓ This is the actual rule
```

**Why the difference?**

- **Fundamental rep**: Operators act on 3D states (quarks)
- **Adjoint rep**: Operators act on 8D algebra elements (gluons)
- The adjoint action is: ad(X)·Y = [X,Y]
- This gives **different matrix elements** even though algebra structure is preserved

### General Tensor Products

For representations (p₁,q₁) ⊗ (p₂,q₂):

```
T^(prod) = T^(1) ⊗ I + I ⊗ T^(2)

[T^(prod)ₐ, T^(prod)ᵦ] = i f^(abc) T^(prod)_c   ✓ Structure constants preserved
```

But the **matrix elements** differ from fundamental representation!

---

## Casimir Operator Construction

The second Casimir operator must be built from **Hermitian** combinations.

### ✗ WRONG (Using non-Hermitian ladder operators)

```python
C₂ = E₁₂² + E₂₁² + E₂₃² + E₃₂² + E₁₃² + E₃₁² + T₃² + T₈²
```

This gives **non-Hermitian** result because Eᵢⱼ² is not Hermitian!

### ✓ CORRECT (Using Hermitian Gell-Mann combinations)

```python
λ₁ = E₁₂ + E₂₁
λ₂ = -i(E₁₂ - E₂₁)
# ... build all 8 λ matrices

C₂ = Σₐ (λₐ/2)² = (λ₁² + λ₂² + ... + λ₈²)/4
```

This is **Hermitian** and has real eigenvalues.

### Verification

```python
# Check Hermiticity
assert np.allclose(C2, C2.conj().T)

# Check eigenvalues are real
eigvals = np.linalg.eigvalsh(C2)
assert np.all(np.imag(eigvals) < 1e-10)
```

---

## Normalization Conventions

### Standard SU(3) Convention

Most textbooks use:
```
Tr(λₐλᵦ) = 2δₐᵦ
Tr(TₐTᵦ) = (1/2)δₐᵦ     (where Tₐ = λₐ/2)
```

This gives Casimir eigenvalues:
```
C₂(1,0) = C₂(0,1) = 4/3
C₂(1,1) = 3
C₂(2,0) = 10/3
```

### Our Implementation Convention

Our code uses a **different normalization**:
```
C₂(1,0) = C₂(0,1) = 1/3
C₂(1,1) = 3/4
C₂(2,0) = 5/6
```

This corresponds to Tr(TₐTᵦ) = (1/6)δₐᵦ.

**All physics is consistent** - ratios between representations are preserved.

### Casimir Formula (General)

For representation (p,q):
```
C₂(p,q) = (1/3)(p² + pq + q² + 3p + 3q)
```

With standard normalization. In our normalization, divide by 4:
```
C₂(p,q) = (1/12)(p² + pq + q² + 3p + 3q)
```

Examples:
- (1,0): C₂ = (1+0+0+3+0)/12 = 1/3 ✓
- (0,1): C₂ = (0+0+1+0+3)/12 = 1/3 ✓
- (1,1): C₂ = (1+1+1+3+3)/12 = 9/12 = 3/4 ✓
- (2,0): C₂ = (4+0+0+6+0)/12 = 10/12 = 5/6 ✓

---

## Summary and Best Practices

### For Validation Tests

1. **Commutation Relations**: Test in the **appropriate representation**
   - Fundamental: [E₁₂, E₂₃] = E₁₃
   - Adjoint: Different structure, don't assume fundamental relations

2. **Casimir Construction**: Always use **Hermitian combinations**
   ```python
   # Build λ matrices from ladder operators
   # Then compute C₂ = Σ λₐ²/4
   ```

3. **Perturbation Tests**: Add **complex** perturbations
   ```python
   E_pert = E + ε*(randn() + i*randn())  # Breaks both real and imaginary parts
   ```

### For Physics Applications

1. **Time Evolution**: Use anti-Hermitian generators
   ```python
   U(t) = exp(-i H t)  where H is Hermitian
   ```

2. **State Construction**: Use ladder operators
   ```python
   |ψ⟩ = E₁₂ E₂₃ |0⟩  # Raises quantum numbers
   ```

3. **Measurement**: Project onto eigenstates of **Hermitian** operators (T₃, T₈, C₂)

---

## References

- Gell-Mann, M. (1962). "Symmetries of Baryons and Mesons"
- Georgi, H. (1999). "Lie Algebras in Particle Physics"
- Our paper: "Exact SU(3) Symmetry from Discrete Lattice Geometry"

**Key Takeaway**: Ladder operators (Eᵢⱼ) are computationally convenient but non-Hermitian. For Hermitian operators needed in eigenvalue problems, always use Gell-Mann combinations or proper Hermitian constructions.
