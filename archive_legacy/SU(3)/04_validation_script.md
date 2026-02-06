# Step 3: Validation (`validate.py`)

## Task
Verify if the discrete construction preserves the algebra exactness.

## Tests

1.  **Commutation Test:**
    * Check $[I_+, I_-] = 2T_3$.
    * Check $[I_+, V_+] = 0$ (or appropriate root sum).
    * Check the Jacobi identity $[A, [B, C]] + [B, [C, A]] + [C, [A, B]] = 0$.
    * **Passing Criteria:** Max absolute difference < $10^{-13}$.

2.  **Eigenvalue Test:**
    * Diagonalize the sparse matrix $C_2$.
    * Compare eigenvalues to the theoretical prediction: $C_{theory} = \frac{1}{3}(p^2 + q^2 + 3p + 3q + pq)$.
    * **Passing Criteria:** Relative error < $10^{-12}$.

3.  **Visualization:**
    * Plot the lattice points.
    * Color code them by their computed $C_2$ expectation value (should be uniform for each shell $(p,q)$).

## Code Prompt
"Write a script that builds the lattice for `max_p=2, max_q=2`, constructs the operators, computes the commutators, and validates the eigenvalues against theory."