# Step 1: Lattice Construction (`lattice.py`)

## Task
Create a Python class `SU3Lattice` that generates the geometric points for SU(3) representations.

## Requirements

1.  **Input:** `max_p`, `max_q` (defines which representations to include).
2.  **Structure:**
    * Iterate through representations $(p, q)$ (analogous to 'shells' in the atom).
    * For each $(p, q)$, generate all valid weight states.
    * Note: Unlike SU(2), SU(3) weights can have multiplicity (multiple states at the same geometric point).
    * *Simplification for V1:* Stick to the "boundary" representations $(N, 0)$ and $(0, N)$ where multiplicity is always 1, OR handle multiplicity by creating distinct graph nodes for degenerate states.

3.  **Coordinate Mapping:**
    * Map each state to 2D Cartesian coordinates $(x, y)$ for visualization.
    * $x = I_3$ (Isospin component)
    * $y = Y$ (Hypercharge)

4.  **Output Data Structure:**
    * A list of dictionaries, where each dict is a state:
        `{'p': p, 'q': q, 'i3': val, 'y': val, 'index': unique_id}`
    * A lookup table `get_index(p, q, i3, y)` to find state indices quickly.

## Code Prompt
"Write a Python class `SU3Lattice` that generates the states for the $(p,q)$ irreducible representations of SU(3). Use the weight diagram algorithm to populate the $I_3$ and $Y$ values for a given $p, q$."