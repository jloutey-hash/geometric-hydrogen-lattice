Task:
Using the existing SU(3) Ziggurat representation engine (weight basis → GT basis → 3D lattice), implement the following physics‑driven validation modules. Each module must include numerical tests, visualizations where appropriate, and machine‑precision verification.

Module 1 — Two‑Hop Commutator Geometry Test
1. 	Pick a representative lattice site in each irrep (1,0), (0,1), and (1,1).
2. 	Compute the two-hop sequences:
• 	
• 	
3. 	Compare the resulting state vectors.
4. 	Verify that their difference matches the action of  to machine precision.
5. 	Output commutator error norms.

Module 2 — Wilson Loop Computation
1. 	Treat each ladder operator as a directed link with amplitude.
2. 	Construct minimal closed loops:
• 	Triangle in the weight plane
• 	Hexagon loop
• 	Vertical loop involving z‑layers
3. 	Compute Wilson loops .
4. 	Verify gauge invariance and compare loops across irreps.

Module 3 — Adjoint Dynamics Test
1. 	Build the adjoint Hamiltonian .
2. 	Evolve random states under .
3. 	Verify that all states have identical energy (Casimir eigenvalue).
4. 	Plot probability flow across the 3‑layer Ziggurat.

Module 4 — Tensor Product Fusion
1. 	Construct product spaces:
• 	
• 	
2. 	Compute the Casimir operator in the product space.
3. 	Identify irreps by clustering eigenvalues.
4. 	Verify dimensions and weight multiplicities.
5. 	Output projection matrices for each irrep.

Module 5 — Geometric Casimir Flow
1. 	Apply the Casimir operator to a localized state.
2. 	Track probability distribution after repeated applications.
3. 	Visualize flow on the 3D Ziggurat.
4. 	Confirm that flow respects SU(3) symmetry.

Module 6 — Symmetry‑Breaking Perturbations
1. 	Introduce controlled geometric distortions (e.g., shift z‑coordinates).
2. 	Recompute commutators and Casimir eigenvalues.
3. 	Plot error vs. distortion strength.
4. 	Identify thresholds where SU(3) symmetry breaks.

Deliverables:
• 	Numerical results for each module
• 	Error tables
• 	Visualizations (2D and 3D)
• 	A summary report explaining which geometric features are essential for exact SU(3) symmetry