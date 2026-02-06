# SU(3) Lattice v3: Exact Biedenharn-Louck Coefficients

## The Situation
The GT pattern generation is 100% correct (dimensions match). The Isospin (I-spin) algebra is perfect. We now need to fix the U-spin and V-spin operators using the exact matrix elements for SU(3) ladder operators in the GT basis.

## The Matrix Elements
For a GT pattern with rows $M_3$ (top), $M_2$ (middle), $M_1$ (bottom), the raising operators $E_{ik}$ are defined as follows:

### 1. Isospin (I+) - Already correct, but use this for consistency:
$I_+$ increases $m_{11}$ by 1.
Matrix Element: $\sqrt{-(m_{11}-m_{12}+1)(m_{11}-m_{22})}$

### 2. The "Upper" Raising Operator (E23 / U-spin-like):
This operator increases $m_{12}$ or $m_{22}$.
In the GT basis, $E_{23}$ is a sum of two terms (one shifting $m_{12}$, one shifting $m_{22}$):

**Term 1 (Shift $m_{12} \to m_{12} + 1$):**
Coeff = $\sqrt{ \frac{(m_{12}-m_{13})(m_{12}-m_{23}+1)(m_{12}-m_{33}+2)(m_{12}-m_{11}+1)}{(m_{12}-m_{22}+1)(m_{12}-m_{22}+2)} }$

**Term 2 (Shift $m_{22} \to m_{22} + 1$):**
Coeff = $\sqrt{ \frac{(m_{22}-m_{13}-1)(m_{22}-m_{23})(m_{22}-m_{33}+1)(m_{22}-m_{11})}{(m_{22}-m_{12}-1)(m_{22}-m_{12})} }$

## Implementation Strategy
1. **Consistency:** Ensure $T_3$ and $T_8$ are defined exactly as:
   - $T_3 = m_{11} - \frac{1}{2}(m_{12} + m_{22})$
   - $T_8 = \frac{\sqrt{3}}{2} [ (m_{12} + m_{22}) - \frac{2}{3}(m_{13} + m_{23} + m_{33}) ]$
2. **Adjoint Check:** Once you have the raising operators ($E_{12}, E_{23}, E_{13}$), the lowering operators MUST be their conjugate transposes ($E_{ji} = E_{ij}^\dagger$). Do not calculate lowering operators with a separate formula; this ensures hermiticity by construction.
3. **Commutator Target:** Once these coefficients are implemented, the error for $[U+, U-]$ and $[V+, V-]$ should drop from ~5.0 to < $10^{-13}$.

## Code Prompt
"Update `operators_v2.py` using the Biedenharn-Louck coefficients provided. Use the $E_{ij}$ notation to build the 8 generators. Validate that the Quadratic Casimir $C_2$ is now a perfect identity matrix (scaled by the eigenvalue) for any given (p,q) representation."