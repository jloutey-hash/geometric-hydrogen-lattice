# SU(3) Lattice v2: Gelfand-Tsetlin Implementation

## The Problem
The previous attempt failed because it ignored state multiplicity at the center of weight diagrams. We must move from (I3, Y) coordinates to Gelfand-Tsetlin (GT) patterns to ensure every state is uniquely represented.

## Task
1.  **GT Pattern Generator:** Implement a function that generates all valid GT patterns for a given $(p,q)$ representation.
    - Top row $(m_{13}, m_{23}, m_{33})$ is defined by $(p,q)$: $m_{13}=p+q, m_{23}=q, m_{33}=0$.
    - Constraints: $m_{13} \ge m_{12} \ge m_{23} \ge m_{22} \ge m_{33}$ and $m_{12} \ge m_{11} \ge m_{22}$.
2.  **Matrix Elements (The Physics):** Use the **Biedenharn-Louck** (or Shertzer) formulas for SU(3) ladder operators acting on GT patterns. These formulas provide the *exact* coefficients for $I_\pm, U_\pm, V_\pm$.
3.  **Mapping to Grid:** After calculating the operators on the GT basis, map each pattern back to $(I_3, Y)$ coordinates for visualization:
    - $I_3 = m_{11} - \frac{1}{2}(m_{12} + m_{22})$
    - $Y = (m_{12} + m_{22}) - \frac{2}{3}(m_{13} + m_{23} + m_{33})$
4.  **Verification:**
    - The dimension of the matrix for $(p,q)$ must be $\frac{1}{2}(p+1)(q+1)(p+q+2)$.
    - The Commutator Error MUST drop to $< 10^{-13}$.

## Code Prompt
"Rewrite `lattice.py` and `operators.py` to use Gelfand-Tsetlin patterns as the basis for the SU(3) representation. Ensure you handle the case where multiple GT patterns map to the same (I3, Y) coordinate by assigning them distinct indices in the sparse matrix."