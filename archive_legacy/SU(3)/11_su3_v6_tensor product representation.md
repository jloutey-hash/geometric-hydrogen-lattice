You are extending the SU(3) representation engine.

Your task is to construct the adjoint (1,1) representation using the tensor product
of the fundamental (1,0) and antifundamental (0,1) representations, and then
transform it into the Gelfand–Tsetlin (GT) basis.

Follow these steps:

1. Import the existing WeightBasisSU3 implementation for (1,0) and (0,1).

2. Build the tensor product representation:
   - Define the product space H = H_fund ⊗ H_antifund (dimension 9).
   - Construct all generators in the product basis:
       T_a^(prod) = T_a^(fund) ⊗ I - I ⊗ T_a^(antifund)^T
     (the minus sign and transpose implement the conjugate action).

3. Extract the adjoint subspace:
   - Compute the identity operator in the product space.
   - Project out the singlet component (trace part).
   - The remaining 8-dimensional subspace is the adjoint irrep.
   - Construct the projection operator P_adj and compute:
       T_a^(adj) = P_adj T_a^(prod) P_adj

4. Validate the adjoint representation in the weight basis:
   - Check all SU(3) commutators to machine precision.
   - Compute the Casimir and verify it is constant.
   - Verify hermiticity and adjoint relations.

5. Transform the adjoint representation to the GT basis:
   - Generate GT patterns for (1,1).
   - Compute (I3, Y) for each GT state.
   - Match GT states to weight-basis states by (I3, Y).
   - Build the unitary transformation U.
   - Transform all operators: O_GT = U† O_weight O.

6. Validate the adjoint representation in the GT basis:
   - All commutators must match SU(3) exactly.
   - Casimir must be constant.
   - T3 and T8 must be diagonal.
   - All hermiticity tests must pass.

Output:
- Python file implementing the adjoint representation in both bases.
- Validation report showing machine-precision results.