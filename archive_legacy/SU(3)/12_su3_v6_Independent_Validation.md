You are acting as an independent QA engineer. Your task is to verify the correctness of an SU(3) representation framework that was constructed by another AI. You must NOT assume any of the previous implementation is correct. You must independently re-derive, re-check, and validate every component.

Your responsibilities:

============================================================
1. Load the following files from the workspace:
   - weight_basis_gellmann.py
   - gt_basis_transformed.py
   - adjoint_tensor_product.py
   - lattice.py
   - Any other supporting modules in the directory

Do NOT modify these files unless a defect is proven.
============================================================

============================================================
2. Perform a full SU(3) algebra validation for each representation:
   - (1,0) fundamental
   - (0,1) antifundamental
   - (1,1) adjoint

For each representation, construct all 8 generators and verify:

  A. Commutators:
     - [T3, T8] = 0
     - [E12, E21] = 2*T3
     - [E23, E32] = T3 + sqrt(3)*T8
     - [E13, E31] = -T3 + sqrt(3)*T8

  B. Hermiticity:
     - T3† = T3
     - T8† = T8
     - E21 = E12†
     - E32 = E23†
     - E31 = E13†

  C. Casimir operator:
     - C2 = sum(T_a @ T_a) over all 8 generators
     - All eigenvalues must be identical within numerical precision
     - Compare against theoretical value:
         C2(p,q) = (p^2 + q^2 + pq + 3p + 3q) / 3

  D. Diagonality:
     - T3 and T8 must be diagonal in both weight and GT bases

  E. Basis transformation:
     - Verify O_GT = U† O_weight U for all generators
     - Confirm U is unitary: U†U = I

Record maximum absolute error for each test.
============================================================

============================================================
3. Perform structural validation:
   - Confirm representation dimensions match (p,q) formula
   - Confirm GT patterns match expected count and constraints
   - Confirm weight states match expected (I3, Y) values
   - Confirm degeneracies (e.g., two (0,0) states in adjoint)
============================================================

============================================================
4. Perform numerical stability checks:
   - Condition number of U
   - Norm preservation under transformation
   - Floating point drift analysis
============================================================

============================================================
5. Produce a final QA report containing:
   - A table of all commutator errors
   - A table of all Casimir eigenvalues and deviations
   - A table of hermiticity deviations
   - A table of diagonality deviations
   - A table verifying U†U = I
   - Pass/fail summary for each representation
   - Any defects found, with reproduction steps
   - If everything passes: explicit statement of full verification
============================================================

Important:
- You must run all tests independently.
- You must not rely on comments or claims inside the provided files.
- You must treat this as a black-box verification of a third-party system.
- If any test fails, you must isolate the failure and provide a minimal reproduction.

Begin by loading the files and printing a summary of the representations detected.