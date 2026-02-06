"""
Manually verify the correct weight diagram for (2,1)
"""
import numpy as np

p, q = 2, 1

print(f"Weight diagram for ({p}, {q}) representation")
print(f"Dimension: {(p+1)*(q+1)*(p+q+2)//2} = 15")

# Highest weight
i3_max = (p - q) / 2.0  # 0.5
y_max = (p + q) / 3.0   # 1.0

print(f"\nHighest weight: I3={i3_max}, Y={y_max}")

# Simple roots in (I3, Y) coordinates
alpha1 = np.array([1.0, 0.0])          # I-spin lowering
alpha2 = np.array([-0.5, np.sqrt(3)/2.0])  # U-spin lowering

print(f"\nSimple roots:")
print(f"  alpha1 = {alpha1}")
print(f"  alpha2 = {alpha2}")

# Generate all weights by applying lowering operators
print(f"\nAll weights (m1, m2 gives how many times we subtracted each root):")
weights = []
for m1 in range(p + 1):
    for m2 in range(q + 1):
        i3 = i3_max - m1 * alpha1[0] - m2 * alpha2[0]
        y = y_max - m1 * alpha1[1] - m2 * alpha2[1]
        
        # Round for display
        i3_r = round(i3 * 2) / 2.0
        y_r = round(y * 6) / 6.0
        
        weights.append((m1, m2, i3_r, y_r))
        print(f"  m1={m1}, m2={m2}: I3={i3_r:.2f}, Y={y_r:.2f}")

# Check for degeneracies
weight_coords = [(i3, y) for _, _, i3, y in weights]
unique = set(weight_coords)
print(f"\nUnique (I3, Y) positions: {len(unique)}")
print(f"Total states from (m1, m2) iteration: {len(weights)}")

# This is the issue - we're only getting (p+1)*(q+1) = 6 states
# but we should get 15!

# The correct approach is to use the Weyl chamber recursion
# Let me generate the full weight diagram properly

print("\n" + "="*60)
print("Correct approach: Full weight diagram with multiplicities")
print("="*60)

# For (p,q), the weight multiplicity can be computed using Freudenthal formula
# But for simple cases, we can enumerate manually

# The (2,1) irrep should contain these weights (with multiplicities):
theoretical_weights = [
    (0.5, 1.0, 1),      # Highest weight
    (1.5, 0.17, 1),
    (-0.5, 1.0, 1),
    (0.5, 0.17, 2),     # Multiplicity 2!
    (-1.5, 1.0, 1),
    (-0.5, 0.17, 2),    # Multiplicity 2!
    (-1.5, 0.17, 1),
    (0.5, -0.67, 1),
    (-0.5, -0.67, 2),   # Multiplicity 2!
    (-1.5, -0.67, 1),
    (0.5, -1.5, 1),
    (-0.5, -1.5, 1),    # Lowest weights
]

print("\nExpected weights with multiplicities:")
total = 0
for i3, y, mult in theoretical_weights:
    print(f"  I3={i3:5.2f}, Y={y:6.2f}, multiplicity={mult}")
    total += mult
print(f"\nTotal dimension: {total}")

# Hmm, that's not right either. Let me recalculate.
# For (2,1): dimension = (2+1)*(1+1)*(2+1+2)/2 = 3*2*5/2 = 15

print("\n" + "="*60)
print("Using proper SU(3) weight diagram construction")
print("="*60)

# For (p, q) = (2, 1), we need to properly generate weights
# The issue is that naive (m1, m2) iteration doesn't capture internal multiplicities

# Proper method: Use Dynkin labels and apply Weyl group reflections
# OR: Use the fact that (2,1) ⊗ (0,0) - but we need the Clebsch-Gordan structure

# For now, let's identify the issue: simple (m1, m2) loop gives only outer shell
# We need to include weights that appear from applying BOTH raising and lowering

# A simpler method: for (p,q), generate by considering the weight string
# between highest and lowest weight, including all intermediate weights

# Actually, the cleanest is to use the crystal basis/Young tableaux method
# But that's complex. Let me use a different approach:

# The dimension formula is: dim(p,q) = (p+1)(q+1)(p+q+2)/2
# For (2,1): (3)(2)(5)/2 = 15

# The weights must span from highest to lowest
# In the hexagonal pattern, for (2,1):

print("\nManual enumeration of (2,1) weight diagram:")
manual_weights = []

# Start from highest and work down by subtracting roots
highest = np.array([i3_max, y_max])

# Apply all valid combinations of simple root subtractions
# But also need to handle multiplicities properly

# For (2,1), the multiplicity pattern follows from
# tensor product decompositions or Littelmann paths

# Let me just list the 15 weights explicitly for (2,1):
# This requires consulting SU(3) tables

print("(Requires explicit SU(3) weight tables - implementation incomplete)")
