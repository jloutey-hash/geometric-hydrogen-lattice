You are implementing SU(3) operators in two bases: the standard weight basis and the Gelfand–Tsetlin (GT) basis.

Your task is to build the SU(3) representation matrices in the weight basis first, where the ladder operator formulas are standard and unambiguous, and then transform them into the GT basis using a numerically constructed unitary transformation.

Follow these steps:

1. Implement a function that generates all weight states (I3, Y) for a given (p,q) irrep, including multiplicities.

2. Implement the SU(3) generators in the weight basis:
   - T3 and T8 diagonal with standard eigenvalues.
   - E12, E23, E13 raising operators using standard weight-shift rules and known SU(3) ladder coefficients.
   - Lowering operators are Hermitian conjugates.

3. Assemble the full set of 8 generators in the weight basis and verify:
   - [T3, T8] = 0
   - [E12, E21] = 2*T3
   - [E23, E32] = T3 + sqrt(3)*T8
   - [E13, E31] = -T3 + sqrt(3)*T8
   - Casimir C2 is proportional to the identity.

4. Construct the GT basis:
   - Generate all GT patterns for the same (p,q) irrep.
   - Compute T3_GT and T8_GT directly from GT formulas (they are diagonal).

5. Build the unitary transformation U from weight basis to GT basis:
   - For each GT pattern, compute its (I3, Y) values.
   - Match each GT state to the corresponding weight-basis eigenvector.
   - Construct U so that U† T3_weight U = T3_GT and U† T8_weight U = T8_GT.

6. Transform all operators:
   - T3_GT = U† T3_weight U
   - T8_GT = U† T8_weight U
   - E12_GT = U† E12_weight U
   - E23_GT = U† E23_weight U
   - E13_GT = U† E13_weight U
   - and their adjoints.

7. Validate in the GT basis:
   - All commutators match SU(3) exactly to machine precision.
   - Casimir is constant across the irrep.
   - Hermiticity and adjoint relations hold.

Output:
- Python files implementing weight-basis operators, GT-basis operators, and the basis transformation.
- A validation report showing commutator errors and Casimir eigenvalue errors.