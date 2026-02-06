# Step 2: Operator Construction (`operators.py`)

## Task
Construct the 8 generators of SU(3) (the Gell-Mann matrices $\lambda_1...\lambda_8$) as sparse matrices acting on the `SU3Lattice`.

## The "Graph Laplacian" Logic
In the SU(2) paper, matrix elements were $\sqrt{l(l+1) - m(m+1)}$.
For SU(3), we must use the shifting operators $E_{\alpha}$.

## Implementation Details

1.  **Diagonal Operators (Cartan Subalgebra):**
    * $T_3$ (Isospin $z$): Diagonal matrix with values $I_3$.
    * $T_8$ (Hypercharge): Diagonal matrix with values $\frac{\sqrt{3}}{2} Y$.

2.  **Ladder Operators (The Edges):**
    * Construct $I_\pm, U_\pm, V_\pm$ by finding neighbors in the lattice.
    * **Selection Rules:**
        * $I_+$ connects state $|i_3, y\rangle \to |i_3+1, y\rangle$.
        * $U_+$ connects state $|i_3, y\rangle \to |i_3-0.5, y+1\rangle$.
        * $V_+$ connects state $|i_3, y\rangle \to |i_3+0.5, y+1\rangle$.
    * **Coefficients:** You must implement the generic Gelfand-Tsetlin patterns or the specific ladder coefficients for SU(3) weights. *This is the critical step.* If the geometric "hopping" is correct, the coefficient depends on the distance from the edge of the weight diagram.

3.  **The Casimir:**
    * Construct the quadratic Casimir operator:
        $$C_2 = T_3^2 + T_8^2 + \frac{1}{2} \{I_+, I_-\} + \frac{1}{2} \{U_+, U_-\} + \frac{1}{2} \{V_+, V_-\}$$
    * (Note: Verify the normalization factor based on the specific definition of generators used).

## Code Prompt
"Implement `SU3Operators` class. It should ingest the lattice. It needs methods `build_T3`, `build_T8`, `build_Iplus`, `build_Uplus`, `build_Vplus`. Finally, build `build_C2`."