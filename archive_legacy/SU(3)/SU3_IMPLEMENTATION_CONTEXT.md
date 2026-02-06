# SU(3) Exact Discretization: Implementation Context & Task

## 1. Project Overview
The objective is to extend a successful **SU(2) discretization method** to **SU(3)**. The original SU(2) implementation achieves **machine precision** ($10^{-14}$) commutation relations and exact eigenvalues by mapping quantum states to discrete lattice coordinates and constructing operators via specialized graph Laplacians. The goal for SU(3) is to replicate this "algebraic exactness" using a Gelfand-Tsetlin basis.

## 2. Current Status
The project has transitioned to a **Gelfand-Tsetlin (GT) basis** implementation to handle weight multiplicities correctly.

### ✅ Completed & Working
* **GT Pattern Generation:** Correctly generates the full basis for any $(p,q)$ representation.
    * (1,1) Adjoint: 8 states.
    * (2,1) Representation: 15 states.
    * (2,2) Representation: 27 states.
* **Cartan Subalgebra:** $T_3$ and $T_8$ (diagonal operators) are exact and hermitian.
* **Isospin (I-spin) Operators:** $E_{12}$ and $E_{21}$ achieve machine precision accuracy.
    * **Formula:** $E_{12} = \sqrt{(m_{12}-m_{11})(m_{11}-m_{22}+1)}$.
    * **Commutator:** $[E_{12}, E_{21}] = 2T_3$ (Error: $\sim 10^{-16}$).

### ❌ Current Bottleneck
* **U-spin and V-spin Operators:** Current implementations for $E_{23}$ (and adjoint $E_{32}$) and $E_{13}$ (and $E_{31}$) produce large errors ($\sim 1.5$ to $8.0$). These prevent the system from reaching the target precision of $<10^{-13}$.

## 3. The Gelfand-Tsetlin Basis
The basis states are labeled by GT patterns:
$$
\begin{pmatrix} 
m_{13} & m_{23} & m_{33} \\ 
& m_{12} & m_{22} \\ 
& & m_{11} 
\end{pmatrix}
$$
For a representation defined by $(p,q)$:
* $m_{13} = p + q$
* $m_{23} = q$
* $m_{33} = 0$
* Between-row constraints: $m_{i,j} \ge m_{i,j-1} \ge m_{i+1,j}$

## 4. The Task for the AI
The primary task is to implement the **exact matrix element formulas** for the raising operator $E_{23}$ in the GT basis.

### Requirements:
1.  **Formula Verification:** Identify and implement the correct Biedenharn-Louck or Gelfand-Tsetlin coefficients for $E_{23}$. Note that $E_{23}$ acts on the $m_{12}$ and $m_{22}$ indices.
2.  **Phase Convention:** Ensure a consistent phase convention (usually Condon-Shortley) to maintain the hermiticity: $E_{32} = E_{23}^\dagger$.
3.  **Target Accuracy:** Commutators must satisfy the SU(3) algebra to machine precision:
    * $[E_{23}, E_{32}] = - \frac{3}{2}T_3 + \frac{\sqrt{3}}{2}T_8$ (or equivalent normalization).
    * $[E_{12}, E_{23}] = E_{13}$.
    * Target Error: $<10^{-13}$.
4.  **Refinement:** Discard previous "shifted $l$-index" attempts which produced $\sim 8.0$ error. Focus on the standard matrix elements $\langle (m)' | E_{23} | (m) \rangle$.

## 5. Reference Files
* `lattice.py`: Contains the working GT pattern generator and state indexing.
* `operators.py`: Contains the current (failing) implementation of $E_{23}$ and $E_{13}$.
* `validate.py`: The testing suite used to measure commutator errors and Casimir eigenvalues.