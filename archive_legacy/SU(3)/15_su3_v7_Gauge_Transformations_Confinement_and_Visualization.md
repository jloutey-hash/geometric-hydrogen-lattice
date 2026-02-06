Task:
Extend the SU(3) Ziggurat physics framework with the following modules. Maintain the existing coding style: explicit functions, modular design, test‑driven development, and machine‑precision validation. All new code must integrate cleanly with the existing physics_applications.py and validation suite.

Module 1 — Correct SU(3) Gauge Transformation Engine
Implement a robust method for generating SU(3) group elements and applying gauge transformations to states and operators.
Requirements:
- Use Hermitian Gell‑Mann generators λₐ as the basis for exponentiation.
- Implement su3_group_element(theta_vector) returning exp(i Σ θₐ λₐ).
- Implement gauge_transform_operator(O, g) returning g O g†.
- Implement gauge_transform_state(psi, g) returning g psi.
- Add validation tests:
- g g† = I to machine precision
- gauge invariance of Casimir
- gauge covariance of ladder operators

Module 2 — Quark–Antiquark Potential and Confinement Diagnostics
Build a full confinement analysis suite using the Ziggurat geometry.
Requirements:
- Implement a function to compute Wilson loops of arbitrary rectangular size.
- Extract potential V(R) from Wilson loop scaling.
- Fit V(R) to V(R) = σ R + c and extract string tension σ.
- Implement flux‑tube visualization: probability density along the path between quark and antiquark.
- Add tests verifying:
- area‑law scaling for large loops
- σ > 0
- symmetry under loop orientation

Module 3 — Adjoint vs Fundamental Dynamics Comparison
Create a unified interface for evolving states in both representations and comparing physical behavior.
Requirements:
- Implement evolve_state(rep, psi0, H, t_max, dt) for rep ∈ {fundamental, adjoint}.
- Track color charge trajectories (I₃(t), Y(t)).
- Compare Casimir scaling: C₂(adj) / C₂(fund) ≈ 9.
- Add tests verifying:
- norm conservation
- energy conservation
- correct Casimir scaling

Module 4 — Ziggurat Geometry Visualization Tools
Add plotting utilities to visualize the 3D lattice and operator actions.
Requirements:
- 3D scatter plot of all lattice sites (x, y, z).
- Directed edges showing hopping amplitudes for a chosen operator.
- Animation of state evolution under Hamiltonian dynamics.
- Visualization of flux tubes from Module 2.
- Add tests ensuring:
- coordinate consistency
- correct mapping between GT patterns and lattice sites

Module 5 — Higher Representation Builder (General (p,q))
Generalize the tensor‑product projection method to arbitrary irreps.
Requirements:
- Implement a function to construct (p,q) via repeated tensor products of 3 and 3̄.
- Identify irreps using Casimir eigenvalues.
- Extract GT patterns and build the corresponding Ziggurat geometry.
- Add tests verifying:
- correct dimension formula
- correct weight multiplicities
- correct Casimir eigenvalue

Module 6 — Physics Notebooks
Create Jupyter notebooks demonstrating the new physics capabilities.
Requirements:
- Quark color precession
- Gluon dynamics
- Confinement potential extraction
- Flux‑tube visualization
- Gauge transformation demonstrations

Deliverables:
- Updated physics_applications.py
- New gauge_transformations.py
- New visualization_tools.py
- New higher_representations.py
- Updated validation suite
- Jupyter notebooks for all physics modules
Standards:
- Machine‑precision validation
- Clear documentation
- Modular, test‑driven code
- No silent failures
- Deterministic behavior
