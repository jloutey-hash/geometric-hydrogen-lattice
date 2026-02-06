# Instructions for the AI Assistant

You are acting as a computational physics theorist. Your goal is to prove that the "Discrete Lattice" method from the provided context can be generalized to SU(3) if the geometry is adapted correctly.

**Please execute the following:**

1.  Read `01_theory_and_spec.md` to understand the physics.
2.  Implement the lattice generation in `lattice.py` following `02_lattice_construction.md`.
3.  Implement the sparse operators in `operators.py` following `03_operator_construction.md`. Pay special attention to the matrix element coefficients—they must be exact SU(3) Clebsch-Gordan coefficients or ladder factors, not approximations.
4.  Run the validation logic in `validate.py` as described in `04_validation_script.md`.

**Output:**
* The Python code files.
* A textual report of the validation results (Commutator errors and Eigenvalue errors).
* A conclusion: Does the SU(3) lattice work?