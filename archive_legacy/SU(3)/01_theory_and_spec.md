# SU(3) Lattice Discretization: Theoretical Specification

## Goal
To implement a "Graph Laplacian" construction for SU(3) symmetry, analogous to the SU(2) polar lattice method. We aim to prove that exact SU(3) commutation relations and Casimir eigenvalues can be preserved on a discrete, finite lattice if the geometry is chosen correctly.

## The Physical Problem
The previous SU(2) lattice failed for SU(3) because it used 1D rings. SU(3) is a Rank-2 group, meaning its states (weights) map to a 2D plane, not a 1D line.
* **SU(2) Geometry:** States $m$ sit on a line (ring).
* **SU(3) Geometry:** States $(I_3, Y)$ sit on a hexagonal/triangular grid.

## The Hypothesis
If we construct a lattice where points correspond exactly to the **Weight Diagrams** of SU(3) irreducible representations (irreps), a sparse matrix operator constructed from nearest-neighbor hopping on this grid will commute with the SU(3) generators and yield exact Casimir eigenvalues.

## Definitions
1.  **Quantum Numbers:** We define states by Dynkin labels $(p, q)$ for the representation, and internal weight labels.
2.  **Lattice Geometry:** A triangular grid.
    * Basis vectors: $\alpha_1 = (1, 0)$, $\alpha_2 = (-1/2, \sqrt{3}/2)$.
3.  **Ladder Operators:** SU(3) has 3 sets of raising/lowering operators (the root system):
    * $I_\pm$ (Isospin): Moves left/right.
    * $U_\pm$ (U-spin): Moves diagonal (top-left/bottom-right).
    * $V_\pm$ (V-spin): Moves diagonal (top-right/bottom-left).

## Success Criteria
1.  **Algebra:** The computed sparse matrices must satisfy the Gell-Mann commutation relations $[T_a, T_b] = i f_{abc} T_c$ to machine precision.
2.  **Eigenvalues:** The Quadratic Casimir operator $C_2 = \sum T_a^2$ must be diagonal with eigenvalue:
    $$C_2(p,q) = \frac{1}{3} (p^2 + q^2 + 3p + 3q + pq)$$