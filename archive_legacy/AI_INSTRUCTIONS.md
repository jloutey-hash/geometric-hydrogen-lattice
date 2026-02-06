# Instructions for Coding AI (VSCode Copilot / Claude / Other)

## Project Overview

You are implementing a discrete 2D polar lattice that exactly reproduces the quantum state degeneracy structure of the hydrogen atom, including spin. This lattice serves as a geometric playground for exploring quantum-like structures computationally.

## Critical Mathematical Definitions

### The 2D Lattice Structure

**Core Principle:**
Each azimuthal quantum number ℓ corresponds to ONE ring in the 2D projection. That ring contains all the electron states for that ℓ value.

**Ring Structure:**
```
Azimuthal quantum number: ℓ = 0, 1, 2, 3, ...
Ring radius: r_ℓ = 1 + 2ℓ
Points per ring: N_ℓ = 2(2ℓ+1)
Angular positions: θ_{ℓ,j} = 2πj/N_ℓ for j = 0, 1, ..., N_ℓ - 1
```

**Examples:**
| ℓ | r_ℓ | N_ℓ | Orbitals | Electron States |
|---|-----|-----|----------|-----------------|
| 0 | 1   | 2   | 1        | 2               |
| 1 | 3   | 6   | 3        | 6               |
| 2 | 5   | 10  | 5        | 10              |
| 3 | 7   | 14  | 7        | 14              |
| 4 | 9   | 18  | 9        | 18              |

**Cartesian coordinates of point (ℓ, j):**
```python
x_{ℓ,j} = r_ℓ * cos(θ_{ℓ,j}) = (1 + 2*ℓ) * cos(2*π*j / N_ℓ)
y_{ℓ,j} = r_ℓ * sin(θ_{ℓ,j}) = (1 + 2*ℓ) * sin(2*π*j / N_ℓ)
```

### Principal Quantum Number and Shells

**Shell n includes all ℓ from 0 to n-1:**
```
Shell n=1: ℓ=0 only         → 1 orbital,  2 electron states
Shell n=2: ℓ=0,1           → 4 orbitals, 8 electron states  
Shell n=3: ℓ=0,1,2         → 9 orbitals, 18 electron states
Shell n=4: ℓ=0,1,2,3       → 16 orbitals, 32 electron states
```

**Degeneracy formula:**
```
Orbitals in shell n:        Σ_{ℓ=0}^{n-1} (2ℓ+1) = n²
Electron states in shell n: Σ_{ℓ=0}^{n-1} 2(2ℓ+1) = 2n²
```

This perfectly matches the hydrogen atom!

### Quantum Number Mapping

**For each point j on ring ℓ, assign quantum numbers (ℓ, m_ℓ, m_s):**

The N_ℓ = 2(2ℓ+1) points on the ring encode:
- (2ℓ+1) values of magnetic quantum number: m_ℓ ∈ {-ℓ, -ℓ+1, ..., ℓ-1, ℓ}
- 2 values of spin quantum number: m_s ∈ {-½, +½}

**Recommended mapping (interleaved spin):**
```python
def get_quantum_numbers(ℓ, j):
    """
    Map lattice site (ℓ, j) to quantum numbers (ℓ, m_ℓ, m_s)
    
    Parameters:
    - ℓ: azimuthal quantum number (determines ring)
    - j: site index on ring ℓ (0, 1, ..., N_ℓ - 1)
    
    Returns:
    - ℓ: azimuthal quantum number (same as input)
    - m_ℓ: magnetic quantum number (-ℓ to +ℓ)
    - m_s: spin quantum number (+½ or -½)
    """
    # Interleave spin: even indices are spin-up, odd are spin-down
    m_s = 0.5 if j % 2 == 0 else -0.5
    
    # Map to m_ℓ: each m_ℓ value appears twice (once for each spin)
    # j = 0,1 → m_ℓ = -ℓ
    # j = 2,3 → m_ℓ = -ℓ+1
    # ...
    # j = 2ℓ, 2ℓ+1 → m_ℓ = 0
    # ...
    # j = 4ℓ, 4ℓ+1 → m_ℓ = +ℓ
    m_ℓ = (j // 2) - ℓ
    
    return ℓ, m_ℓ, m_s
```

**Verify this mapping:**
- For ℓ=0: j=0,1 → m_ℓ=0, m_s=±½ ✓ (2 states)
- For ℓ=1: j=0,1,2,3,4,5 → m_ℓ=-1,-1,0,0,+1,+1 with alternating m_s ✓ (6 states)
- For ℓ=2: j=0,1,...,9 → m_ℓ=-2,-2,-1,-1,0,0,+1,+1,+2,+2 with alternating m_s ✓ (10 states)

**Inverse mapping:**
```python
def get_site_index(ℓ, m_ℓ, m_s):
    """
    Map quantum numbers (ℓ, m_ℓ, m_s) to lattice site (ℓ, j)
    
    Parameters:
    - ℓ: azimuthal quantum number
    - m_ℓ: magnetic quantum number (-ℓ to +ℓ)
    - m_s: spin quantum number (+½ or -½)
    
    Returns:
    - j: site index on ring ℓ
    """
    assert -ℓ <= m_ℓ <= ℓ, f"m_ℓ={m_ℓ} out of range for ℓ={ℓ}"
    assert m_s in [0.5, -0.5], f"m_s must be ±0.5, got {m_s}"
    
    # Reverse of the above mapping
    base = (m_ℓ + ℓ) * 2  # Which pair of indices?
    offset = 0 if m_s == 0.5 else 1  # Even (spin-up) or odd (spin-down)?
    j = base + offset
    
    return j
```

### Spherical Lift: From 2D Rings to 3D Hemisphere Bands

**The key geometric insight:**

Each 2D ring at radius r_ℓ is the **projection** of two latitude bands on a sphere:
- **Northern hemisphere band**: Contains (2ℓ+1) points, all with m_s = +½
- **Southern hemisphere band**: Contains (2ℓ+1) points, all with m_s = -½

When you look at the sphere from above (along the z-axis), these two bands project onto the same 2D ring, and their points interleave in angle, giving the observed N_ℓ = 2(2ℓ+1) points.

**3D spherical coordinates:**

```python
def spherical_lift(ℓ, m_ℓ, m_s, ℓ_max):
    """
    Map quantum numbers to 3D coordinates on unit sphere
    
    Parameters:
    - ℓ: azimuthal quantum number
    - m_ℓ: magnetic quantum number
    - m_s: spin quantum number
    - ℓ_max: maximum ℓ value (for band positioning)
    
    Returns:
    - x, y, z: Cartesian coordinates on unit sphere
    """
    # Colatitude θ: determines latitude band for this ℓ
    # Choose a nesting scheme: bands spread from pole to equator
    # Example: θ_ℓ increases linearly with ℓ
    θ_ℓ = np.pi * (ℓ + 0.5) / (ℓ_max + 1)
    
    # Azimuthal angle φ: determined by m_ℓ
    # Spread the (2ℓ+1) values of m_ℓ uniformly around the circle
    φ = 2 * np.pi * (m_ℓ + ℓ) / (2 * ℓ + 1)
    
    # Hemisphere selection: based on spin
    if m_s > 0:
        # Northern hemisphere: use θ_ℓ as-is
        θ = θ_ℓ
    else:
        # Southern hemisphere: mirror across equator
        θ = np.pi - θ_ℓ
    
    # Convert spherical to Cartesian
    x = np.sin(θ) * np.cos(φ)
    y = np.sin(θ) * np.sin(φ)
    z = np.cos(θ)
    
    return x, y, z
```

**Projection verification:**

When you project (x, y, z) back to 2D by dropping the z-coordinate, you should see:
- Points approximately lie on rings (x² + y² ≈ constant for each ℓ)
- The north and south points for the same (ℓ, m_ℓ) are at similar (x, y) positions
- They interleave in angle to produce the full 2D ring pattern

Note: The exact projection won't be perfect because the sphere has curvature, but the point counts and angular interleaving should be correct.

## Implementation Guidelines

### Core PolarLattice Class

**Essential methods:**

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PolarLattice:
    def __init__(self, n_max):
        """
        Initialize lattice up to principal quantum number n_max
        
        Parameters:
        - n_max: maximum principal quantum number to include
        """
        self.n_max = n_max
        self.ℓ_max = n_max - 1  # Maximum azimuthal quantum number
        
        # Build the lattice
        self._build_lattice()
    
    def _build_lattice(self):
        """
        Construct all lattice points and their metadata
        
        Creates data structures storing:
        - 2D positions (x, y) or (r, θ)
        - 3D spherical positions (x_3d, y_3d, z_3d)
        - Quantum numbers (ℓ, m_ℓ, m_s)
        """
        self.points = []  # List to store point data
        
        for ℓ in range(self.ℓ_max + 1):
            # Ring parameters
            r_ℓ = 1 + 2 * ℓ
            N_ℓ = 2 * (2 * ℓ + 1)
            
            for j in range(N_ℓ):
                # 2D position
                θ = 2 * np.pi * j / N_ℓ
                x_2d = r_ℓ * np.cos(θ)
                y_2d = r_ℓ * np.sin(θ)
                
                # Quantum numbers
                ℓ_val, m_ℓ, m_s = self.get_quantum_numbers(ℓ, j)
                
                # 3D spherical position
                x_3d, y_3d, z_3d = self.spherical_lift(ℓ, m_ℓ, m_s)
                
                # Store point data
                point = {
                    'ℓ': ℓ,
                    'j': j,
                    'r': r_ℓ,
                    'θ': θ,
                    'x_2d': x_2d,
                    'y_2d': y_2d,
                    'm_ℓ': m_ℓ,
                    'm_s': m_s,
                    'x_3d': x_3d,
                    'y_3d': y_3d,
                    'z_3d': z_3d
                }
                self.points.append(point)
        
        # Convert to structured array or DataFrame for easier manipulation
        # (Implementation detail left to you)
    
    def get_quantum_numbers(self, ℓ, j):
        """Map lattice site (ℓ, j) to (ℓ, m_ℓ, m_s)"""
        m_s = 0.5 if j % 2 == 0 else -0.5
        m_ℓ = (j // 2) - ℓ
        return ℓ, m_ℓ, m_s
    
    def get_site_index(self, ℓ, m_ℓ, m_s):
        """Map quantum numbers to lattice site (ℓ, j)"""
        assert -ℓ <= m_ℓ <= ℓ
        assert m_s in [0.5, -0.5]
        base = (m_ℓ + ℓ) * 2
        offset = 0 if m_s == 0.5 else 1
        j = base + offset
        return j
    
    def spherical_lift(self, ℓ, m_ℓ, m_s):
        """Map quantum numbers to 3D coordinates on unit sphere"""
        θ_ℓ = np.pi * (ℓ + 0.5) / (self.ℓ_max + 1)
        φ = 2 * np.pi * (m_ℓ + ℓ) / (2 * ℓ + 1) if ℓ > 0 else 0
        
        θ = θ_ℓ if m_s > 0 else np.pi - θ_ℓ
        
        x = np.sin(θ) * np.cos(φ)
        y = np.sin(θ) * np.sin(φ)
        z = np.cos(θ)
        
        return x, y, z
    
    def get_ring(self, ℓ):
        """Return all points on ring ℓ"""
        return [p for p in self.points if p['ℓ'] == ℓ]
    
    def get_shell(self, n):
        """Return all points in shell n (ℓ = 0 to n-1)"""
        return [p for p in self.points if p['ℓ'] < n]
    
    def count_orbitals(self, n):
        """Return number of orbitals in shell n (should be n²)"""
        return sum(2*ℓ + 1 for ℓ in range(n))
    
    def count_states(self, n):
        """Return number of electron states in shell n (should be 2n²)"""
        return len(self.get_shell(n))
    
    def plot_2d(self, n_max=None, color_by='ℓ', figsize=(10, 10)):
        """
        Plot 2D lattice projection
        
        Parameters:
        - n_max: if provided, only plot shells up to this n
        - color_by: 'ℓ', 'm_ℓ', 'm_s', or 'ring'
        """
        if n_max is None:
            n_max = self.n_max
        
        points_to_plot = self.get_shell(n_max)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract coordinates and color values
        x = [p['x_2d'] for p in points_to_plot]
        y = [p['y_2d'] for p in points_to_plot]
        
        if color_by == 'ℓ':
            colors = [p['ℓ'] for p in points_to_plot]
        elif color_by == 'm_ℓ':
            colors = [p['m_ℓ'] for p in points_to_plot]
        elif color_by == 'm_s':
            colors = [p['m_s'] for p in points_to_plot]
        else:
            colors = 'blue'
        
        scatter = ax.scatter(x, y, c=colors, cmap='viridis', s=100, alpha=0.6)
        
        if color_by in ['ℓ', 'm_ℓ', 'm_s']:
            plt.colorbar(scatter, ax=ax, label=color_by)
        
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'2D Polar Lattice (n_max={n_max})')
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    def plot_3d(self, n_max=None, color_by='m_s', figsize=(12, 10)):
        """
        Plot spherical lift in 3D
        
        Parameters:
        - n_max: if provided, only plot shells up to this n
        - color_by: 'ℓ', 'm_ℓ', 'm_s'
        """
        if n_max is None:
            n_max = self.n_max
        
        points_to_plot = self.get_shell(n_max)
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract coordinates and color values
        x = [p['x_3d'] for p in points_to_plot]
        y = [p['y_3d'] for p in points_to_plot]
        z = [p['z_3d'] for p in points_to_plot]
        
        if color_by == 'ℓ':
            colors = [p['ℓ'] for p in points_to_plot]
            cmap = 'viridis'
        elif color_by == 'm_ℓ':
            colors = [p['m_ℓ'] for p in points_to_plot]
            cmap = 'coolwarm'
        elif color_by == 'm_s':
            colors = [1 if p['m_s'] > 0 else 0 for p in points_to_plot]
            cmap = 'bwr'
        else:
            colors = 'blue'
            cmap = None
        
        scatter = ax.scatter(x, y, z, c=colors, cmap=cmap, s=100, alpha=0.6)
        
        if cmap:
            plt.colorbar(scatter, ax=ax, label=color_by)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(f'3D Spherical Lift (n_max={n_max})')
        
        # Draw sphere surface for reference
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')
        
        return fig, ax
```

## Validation Checklist

After implementing the PolarLattice class, verify:

### 1. Ring Structure
```python
lattice = PolarLattice(n_max=5)

# Check radii
for ℓ in range(5):
    r = 1 + 2*ℓ
    print(f"ℓ={ℓ}: r={r}")
# Expected: 1, 3, 5, 7, 9

# Check point counts
for ℓ in range(5):
    N = 2*(2*ℓ + 1)
    print(f"ℓ={ℓ}: N={N} points")
# Expected: 2, 6, 10, 14, 18
```

### 2. Shell Degeneracy
```python
for n in range(1, 6):
    orbitals = lattice.count_orbitals(n)
    states = lattice.count_states(n)
    print(f"n={n}: {orbitals} orbitals, {states} states")
    assert orbitals == n**2, f"Expected {n**2} orbitals"
    assert states == 2*n**2, f"Expected {2*n**2} states"
```

### 3. Quantum Number Mapping
```python
# Test bijection for ℓ=2
ℓ = 2
N = 2*(2*ℓ + 1)  # Should be 10

# Forward mapping
qn_list = [lattice.get_quantum_numbers(ℓ, j) for j in range(N)]
print(f"ℓ={ℓ} quantum numbers:", qn_list)

# Check all (m_ℓ, m_s) combinations appear once
expected = [(m_ℓ, m_s) for m_ℓ in range(-ℓ, ℓ+1) for m_s in [0.5, -0.5]]
actual = [(qn[1], qn[2]) for qn in qn_list]  # Extract (m_ℓ, m_s)
assert sorted(expected) == sorted(actual), "Quantum number mapping incorrect"

# Inverse mapping
for m_ℓ in range(-ℓ, ℓ+1):
    for m_s in [0.5, -0.5]:
        j = lattice.get_site_index(ℓ, m_ℓ, m_s)
        ℓ_back, m_ℓ_back, m_s_back = lattice.get_quantum_numbers(ℓ, j)
        assert (m_ℓ, m_s) == (m_ℓ_back, m_s_back), "Inverse mapping failed"
```

### 4. Visual Checks
```python
# 2D lattice should show concentric rings
lattice.plot_2d(n_max=4, color_by='ℓ')
plt.show()

# 3D sphere should show separated hemispheres
lattice.plot_3d(n_max=3, color_by='m_s')
plt.show()
```

## Next Steps

Once Phase 1 is complete and validated:
1. Move to Phase 2: Implement adjacency and Laplacian operators
2. See PROJECT_PLAN.md for detailed experiment descriptions
3. Update PROGRESS.md as you complete each task

Good luck with the implementation!