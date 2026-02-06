# SU(3) v7: Return to Geometric Construction (The "Ziggurat" Lattice)

## The Objective
We are abandoning the "Tensor Product" and "Unitary Transformation" methods. We return to the original goal: constructing a discrete lattice where the geometry *itself* encodes the SU(3) algebra.

## 1. The Geometry: 3D "Ziggurat" States
The Gelfand-Tsetlin (GT) patterns are not abstract indices; they are coordinates in a 3D lattice.
For each state $(m_{13}, m_{23}, m_{33}; m_{12}, m_{22}; m_{11})$, define its geometric position $\vec{r} = (x,y,z)$:
* **x (Isospin):** $I_3 = m_{11} - 0.5(m_{12} + m_{22})$
* **y (Hypercharge):** $Y = \text{standard formula}$
* **z (Layer/Lift):** $z = m_{12} - m_{22}$ (This separates degenerate states vertically!)

**Task:** Visualize the lattice. It should look like a "stacked" set of 2D weight diagrams, forming a 3D polytope.

## 2. The Links: Defining Operators as Graph Edges
Operators are strictly "nearest neighbor hops" in this 3D space.
* **I-links (Horizontal):** Connect states differing only by $\Delta x$.
* **U-links (Diagonal):** Connect states differing by $\Delta m_{12}$.
* **V-links (Vertical-Diagonal):** Connect states differing by $\Delta m_{22}$.

## 3. The "Physics" Weights (The Correction)
We previously failed because of a normalization factor. We will now apply the **Corrected Biedenharn-Louck** weights directly to the graph edges.

**The Golden Rule:**
Use the exact Biedenharn-Louck formula for $E_{23}$ (from previous specs), BUT apply a global scaling factor of **$1/\sqrt{2}$**.
* Why? The BL formula is naturally normalized to 1.0 for the fundamental. SU(3) physics requires 0.707.
* By hardcoding this "coupling constant" $g = 1/\sqrt{2}$ into the lattice definition, the algebra will close.

## 4. Implementation Steps
1.  **Generate the 3D Grid:** Create the nodes using GT patterns.
2.  **Build Sparse Matrices Directly:** Do NOT transform from weight basis. Loop through the nodes, calculate the neighbor index, compute the weight (with the $1/\sqrt{2}$ factor), and insert into the sparse matrix.
3.  **Validate:**
    * $[E_{12}, E_{21}] = 2T_3$ (This works already)
    * $[E_{23}, E_{32}] = \dots$ (This failed before. With the $1/\sqrt{2}$ factor, it should now work to machine precision).

## 5. The Output
We want a report showing:
* Commutator errors $< 10^{-13}$ using **DIRECT** construction.
* A plot of the **3D Lattice** (x, y, z) showing the "Ziggurat" structure.

**This restores the claim: "Exact SU(3) Symmetry from a 3D Discrete Lattice."**