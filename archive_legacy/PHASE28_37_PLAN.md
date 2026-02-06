# Phases 28-37: Lightweight Investigation Suite

**Date:** January 6, 2026  
**Status:** Planning Phase  
**Previous Work:** Phases 1-27 complete (SU(2) lattice through Standard Model)  
**New Direction:** Lightweight computational experiments for deeper theoretical insights  

---

## Context

Phases 1-27 built a complete framework from angular momentum through quark confinement and Standard Model physics. These new phases are **lightweight computational experiments** designed to:

1. Extract new physics insights with minimal compute cost
2. Explore connections to fine-structure constant and electroweak physics
3. Validate mathematical completeness and scaling behaviors
4. Test geometric origins of fundamental constants

**Computation Philosophy:** All phases designed for **laptop hardware**, runtime < 1 minute each (except Monte Carlo phases which may take ~10 minutes).

---

## Phase Enumeration

### Phase 28: U(1) Wilson Loops on Existing SU(2) Lattice
**Status:** Not Started  
**Compute:** < 10 seconds  
**Dependencies:** Existing SU(2) lattice from Phases 1-21  

**Goal:** Extract effective U(1) coupling without heavy Monte Carlo

**Tasks:**
- Add U(1) link variables θ_ij on existing graph edges
- Compute all plaquette products exp(i Σ θ)
- Measure: mean, variance, distribution shape
- Output dimensionless coupling α_eff from plaquette average

**Scientific Value:** Simplest possible "fine-structure constant probe"

---

### Phase 29: SU(2) × U(1) Mixed Link Variables
**Status:** Not Started  
**Compute:** < 30 seconds  
**Dependencies:** Phase 28 (U(1) implementation)  

**Goal:** Explore geometry-induced mixing ratio (toy electroweak angle)

**Tasks:**
- Define combined links: U_mix = U_SU(2) · exp(iθ w)
- Introduce tunable mixing parameter w ∈ [0, 1]
- Compute mixed plaquette observables vs w
- Search for plateau/extremum indicating preferred mixing

**Scientific Value:** Lightest test of whether geometry prefers small U(1) coupling (analogous to electroweak mixing angle θ_W ≈ 28.7°)

---

### Phase 30: SU(3) Flavor Weight Diagram Generator
**Status:** Not Started  
**Compute:** < 5 seconds (pure combinatorics)  
**Dependencies:** None (independent module)  

**Goal:** Build SU(3) flavor lattice to explore geometric multiplet analogies

**Tasks:**
- Implement Gell-Mann matrices (λ₁, ..., λ₈)
- Generate weight diagrams for:
  - Fundamental (3): triangular
  - Adjoint (8): hexagonal + center
  - Decuplet (10): tetrahedral
- Plot in (I₃, Y) plane (isospin vs hypercharge)
- Compare degeneracy patterns to SU(2) rings

**Scientific Value:** Cheap way to test if concentric-ring insight generalizes to QCD multiplets

---

### Phase 31: Discrete Higgs Scalar on SU(2) Lattice
**Status:** Not Started  
**Compute:** < 1 minute (gradient descent)  
**Dependencies:** Existing SU(2) lattice  

**Goal:** Add scalar field to explore symmetry breaking

**Tasks:**
- Define scalar φ_i on each lattice site
- Add potential V(φ) = -μ²|φ|² + λ|φ|⁴ (Mexican hat)
- Couple to SU(2): H = Σ |φ_i - U_ij φ_j|²
- Compute lowest-energy config via gradient descent
- Measure: ⟨φ⟩, symmetry breaking pattern

**Scientific Value:** Simplest Higgs-like experiment without 4D lattice (complements Phase 26's full implementation)

---

### Phase 32: Radial-Only Hydrogen Solver Optimization
**Status:** Not Started  
**Compute:** < 10 seconds  
**Dependencies:** Existing radial solver from Phase 2  

**Goal:** Improve radial solver accuracy 10× without increasing grid size

**Tasks:**
- Implement Numerov method: fourth-order finite difference
- Add adaptive radial grid: dense near origin, sparse at large r
- Compare methods:
  - Current: uniform second-order
  - Numerov: uniform fourth-order
  - Adaptive: second-order variable spacing
- Output error vs analytic E_n = -13.6/n² eV

**Scientific Value:** 10× accuracy boost for zero compute cost (algorithmic improvement)

---

### Phase 33: SU(2) Representation Completeness Tests
**Status:** ✅ ALREADY COMPLETE (Phase 21)  
**Action:** Document connection to new phases  

**Previous Results (Phase 21):**
- Wigner D-matrices generated for j = 0, 1/2, 1, ..., 10
- All unitarity tests pass: ||D†D - I|| < 10⁻¹⁵
- Tensor products verified: j₁ ⊗ j₂ = |j₁-j₂| ⊕ ... ⊕ (j₁+j₂)
- Peter-Weyl completeness demonstrated

**New Connection:**
- Phase 21 results validate mathematical foundation for Phases 28-37
- Completeness guarantees any U(1) ⊂ SU(2) extraction is well-defined
- Provides baseline for comparing SU(2), U(1), SU(3) structures

---

### Phase 34: High-ℓ Scaling Study
**Status:** Not Started  
**Compute:** < 1 second (pure data analysis)  
**Dependencies:** Existing α_ℓ data from Phases 8-9  

**Goal:** Extract deeper structure behind 1/(4π) convergence

**Tasks:**
- Fit existing α_ℓ data to:
  - 1/ℓ (leading order)
  - 1/ℓ² (next-to-leading)
  - 1/(ℓ+½) (Langer correction)
  - 1/(ℓ(ℓ+1)) (quantum correction)
- Compare χ² for each model
- Extrapolate α∞ with error bars
- Search for subleading constants

**Scientific Value:** Pure analysis phase—might reveal geometric constants beyond 1/(4π)

---

### Phase 35: SU(2) Heat Kernel / Diffusion Operator
**Status:** Not Started  
**Compute:** < 30 seconds  
**Dependencies:** Existing Laplacian from Phase 2  

**Goal:** Build discrete heat kernel to probe spectral geometry

**Tasks:**
- Construct discrete Laplacian Δ on SU(2) lattice
- Compute heat kernel: K(t) = exp(-tΔ)
- Measure trace: Z(t) = Tr[exp(-tΔ)]
- Compare to analytic SU(2) heat kernel asymptotic expansion:
  - Z(t) ~ (4πt)^(-3/2) [1 + a₁t + a₂t² + ...]
- Extract geometric coefficients from early-time expansion

**Scientific Value:** Heat kernels encode curvature—geometric constants "hide" in expansion coefficients

---

### Phase 36: Minimal RG Flow Experiment
**Status:** Not Started  
**Compute:** < 30 seconds  
**Dependencies:** Phase 28 (U(1) implementation)  

**Goal:** See if effective U(1) coupling runs with scale

**Tasks:**
- Compute U(1) plaquette averages for three scales:
  - ℓ_max = 3 (short distance)
  - ℓ_max = 5 (intermediate)
  - ℓ_max = 7 (long distance)
- Extract effective coupling α(ℓ_max) from each
- Fit running: α(ℓ) = α₀ + β log(ℓ/ℓ₀)
- Test detectability: Is β ≠ 0 at 2σ significance?

**Scientific Value:** Lightest-weight renormalization experiment possible—no Monte Carlo needed!

---

### Phase 37: SU(2) → S³ Sampling Experiment
**Status:** Not Started  
**Compute:** < 10 seconds  
**Dependencies:** Existing Hopf fibration from Phase 19  

**Goal:** Test whether lattice approximates uniform S³ sampling

**Tasks:**
- Map each lattice site to S³ via Hopf lift:
  - (ℓ, m, m_s) → (ψ, θ, φ) ∈ S³
- Compute geometric statistics:
  - Nearest-neighbor distances on S³
  - Distribution of chord lengths
  - Spherical cap counts (Voronoi-like cells)
- Compare to:
  - Uniform random sampling on S³
  - Fibonacci sphere (optimal uniform)
  
**Scientific Value:** Probes geometric origin of 1/(4π) = dim(SU(2))/vol(S³) directly

---

## Implementation Strategy

### Parallel Execution Groups

**Group A (Independent):** Can run simultaneously
- Phase 28: U(1) Wilson loops
- Phase 30: SU(3) weight diagrams  
- Phase 32: Numerov solver
- Phase 34: High-ℓ scaling (data analysis)

**Group B (Depends on Phase 28):**
- Phase 29: SU(2) × U(1) mixing
- Phase 36: RG flow

**Group C (Independent, pure geometry):**
- Phase 35: Heat kernel
- Phase 37: S³ sampling

**Group D (Documentation):**
- Phase 33: Already complete, just document

### Code Organization

```
src/experiments/
├── phase28_u1_wilson_loops.py          # U(1) plaquettes
├── phase29_su2_u1_mixing.py            # Mixed gauge groups
├── phase30_su3_flavor_weights.py       # SU(3) combinatorics
├── phase31_discrete_higgs.py           # Scalar field + gradient descent
├── phase32_numerov_solver.py           # Improved radial solver
├── phase33_completeness_review.py      # Documentation + Phase 21 links
├── phase34_high_ell_scaling.py         # Data fitting
├── phase35_heat_kernel.py              # Spectral geometry
├── phase36_rg_flow.py                  # Running coupling
└── phase37_s3_sampling.py              # Uniformity test
```

### Validation Criteria

Each phase produces:
1. **Quick output:** Runtime confirmation < 1 minute
2. **Key metric:** Single number or plot answering "why this matters"
3. **Comparison:** Analytic result, null hypothesis, or previous phase
4. **Visualization:** 1-2 publication-ready figures
5. **Report:** ~50-line summary with interpretation

---

## Expected Scientific Outcomes

### Tier 1 (High confidence)
- Phase 28: Extract α_U(1) ~ 1/137 or different scale
- Phase 30: Confirm/refute ring structure in SU(3)
- Phase 32: 10× accuracy improvement (guaranteed)
- Phase 34: Identify subleading terms in α_ℓ
- Phase 37: Quantify S³ uniformity vs Fibonacci sphere

### Tier 2 (Exploratory)
- Phase 29: Might find preferred mixing angle or null result
- Phase 35: Heat kernel coefficients may match theory or diverge
- Phase 36: Running may be too weak to detect at these scales

### Tier 3 (Learning regardless of outcome)
- Phase 31: Either breaks symmetry or doesn't—both are interesting
- All phases teach something even if "nothing happens"

---

## Timeline Estimate

**Total:** ~4-6 hours of implementation + testing

- Phase 28: 30 min (U(1) links straightforward)
- Phase 29: 20 min (builds on 28)
- Phase 30: 30 min (SU(3) matrices + plotting)
- Phase 31: 45 min (gradient descent implementation)
- Phase 32: 40 min (Numerov method + adaptive grid)
- Phase 33: 10 min (documentation only)
- Phase 34: 20 min (curve fitting)
- Phase 35: 45 min (heat kernel exponentiation)
- Phase 36: 25 min (reuse Phase 28 code)
- Phase 37: 30 min (S³ geometry + statistics)

---

## Success Metrics

**Minimum Success:**
- All 10 phases produce output without errors
- Each generates at least one interpretable plot/number
- Runtime < 1 minute per phase (verified)

**Target Success:**
- Extract α_eff from U(1) plaquettes (Phase 28)
- Identify best-fit scaling law for α_ℓ (Phase 34)
- Improve radial solver accuracy 10× (Phase 32)
- Generate publication-ready SU(3) weight diagrams (Phase 30)

**Stretch Goals:**
- Discover preferred electroweak-like mixing angle (Phase 29)
- Detect running coupling (Phase 36)
- Match heat kernel expansion to theory (Phase 35)
- Prove S³ sampling is near-optimal (Phase 37)

---

## Next Steps

1. **Phase 28** (U(1) Wilson loops) — Start here, foundational for phases 29, 36
2. **Phase 30** (SU(3) weights) — Independent, quick win
3. **Phase 32** (Numerov) — High-confidence success
4. **Phase 34** (High-ℓ scaling) — Pure analysis, fast

Then proceed through remaining phases based on results.

---

## Notes

- **Phase 33 already done:** Phase 21 completed this work, just need to document connection
- **Low compute risk:** All phases designed to complete on laptop
- **High learning value:** Even null results teach us about the structure
- **Modular:** Can execute any phase independently (except 29, 36 depend on 28)

Ready to begin implementation with Phase 28.
