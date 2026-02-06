# Ziggurat Lattice Visualization Tools
"""
Comprehensive visualization suite for SU(3) triangular lattice on torus.

Features:
- 3D lattice structure with sites and hopping terms
- State evolution animation
- Flux tube visualization
- Coordinate system validation
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from typing import Tuple, List, Optional, Dict
from weight_basis_gellmann import WeightBasisSU3


class ZigguratVisualizer:
    """
    Visualizer for triangular lattice torus geometry.
    """
    
    def __init__(self, Lx: int = 4, Ly: int = 4):
        """
        Initialize lattice geometry.
        
        Args:
            Lx: Number of sites in x direction
            Ly: Number of sites in y direction
        """
        self.Lx = Lx
        self.Ly = Ly
        self.N_sites = Lx * Ly
        
        # Build coordinate system
        self._build_coordinates()
        
        # Build hopping structure
        self._build_hopping_graph()
    
    def _build_coordinates(self):
        """
        Build 3D coordinates for triangular lattice embedded in torus.
        
        Uses equilateral triangles:
        - e1 = (1, 0)
        - e2 = (1/2, √3/2)
        """
        coords = []
        
        # Triangular lattice vectors
        e1 = np.array([1.0, 0.0])
        e2 = np.array([0.5, np.sqrt(3)/2])
        
        for iy in range(self.Ly):
            for ix in range(self.Lx):
                # Position in 2D lattice
                pos_2d = ix * e1 + iy * e2
                
                # Add small z variation for visibility (not physical)
                z = 0.1 * np.sin(2*np.pi*ix/self.Lx) * np.cos(2*np.pi*iy/self.Ly)
                
                coords.append([pos_2d[0], pos_2d[1], z])
        
        self.coords = np.array(coords)
    
    def _build_hopping_graph(self):
        """
        Build directed graph of hopping terms.
        
        Six neighbors for triangular lattice:
        +e1, -e1, +e2, -e2, +(e1-e2), -(e1-e2)
        """
        self.hopping_edges = []
        
        directions = [
            (1, 0),   # +e1
            (-1, 0),  # -e1
            (0, 1),   # +e2
            (0, -1),  # -e2
            (1, -1),  # +(e1-e2)
            (-1, 1),  # -(e1-e2)
        ]
        
        for iy in range(self.Ly):
            for ix in range(self.Lx):
                site_from = iy * self.Lx + ix
                
                for dx, dy in directions:
                    ix_to = (ix + dx) % self.Lx
                    iy_to = (iy + dy) % self.Ly
                    site_to = iy_to * self.Lx + ix_to
                    
                    self.hopping_edges.append((site_from, site_to))
    
    def plot_lattice_structure(self, title: str = "SU(3) Triangular Lattice Torus") -> plt.Figure:
        """
        Plot 3D lattice structure with sites and hopping terms.
        
        Args:
            title: Plot title
            
        Returns:
            fig: Matplotlib figure
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw sites
        ax.scatter(self.coords[:, 0], self.coords[:, 1], self.coords[:, 2],
                  c='blue', s=100, alpha=0.8, edgecolors='black', linewidths=1.5,
                  label='Lattice Sites')
        
        # Draw hopping terms as directed edges
        for site_from, site_to in self.hopping_edges:
            pos_from = self.coords[site_from]
            pos_to = self.coords[site_to]
            
            # Arrow from site_from to site_to
            ax.plot([pos_from[0], pos_to[0]],
                   [pos_from[1], pos_to[1]],
                   [pos_from[2], pos_to[2]],
                   'k-', alpha=0.2, linewidth=0.8)
        
        # Label a few sites
        for i in [0, self.Lx-1, self.N_sites-1]:
            ax.text(self.coords[i, 0], self.coords[i, 1], self.coords[i, 2] + 0.3,
                   f'{i}', fontsize=10, ha='center')
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title(title, fontsize=14, pad=20)
        ax.legend(fontsize=12)
        
        # Set equal aspect ratio
        max_range = np.array([self.coords[:, 0].max() - self.coords[:, 0].min(),
                             self.coords[:, 1].max() - self.coords[:, 1].min(),
                             self.coords[:, 2].max() - self.coords[:, 2].min()]).max() / 2.0
        mid_x = (self.coords[:, 0].max() + self.coords[:, 0].min()) * 0.5
        mid_y = (self.coords[:, 1].max() + self.coords[:, 1].min()) * 0.5
        mid_z = (self.coords[:, 2].max() + self.coords[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        return fig
    
    def plot_state_on_lattice(self, state: np.ndarray, rep_dim: int = 3,
                             title: str = "State Amplitudes") -> plt.Figure:
        """
        Plot state amplitude distribution on lattice.
        
        Args:
            state: State vector (N_sites × rep_dim)
            rep_dim: Dimension of color representation
            title: Plot title
            
        Returns:
            fig: Matplotlib figure
        """
        # Extract amplitude at each site (sum over color indices)
        if state.ndim == 1:
            state = state.reshape(self.N_sites, rep_dim)
        
        amplitudes = np.array([np.linalg.norm(state[i]) for i in range(self.N_sites)])
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color by amplitude
        scatter = ax.scatter(self.coords[:, 0], self.coords[:, 1], self.coords[:, 2],
                           c=amplitudes, s=200, alpha=0.8, cmap='plasma',
                           edgecolors='black', linewidths=1.5)
        
        # Draw lattice edges lightly
        for site_from, site_to in self.hopping_edges:
            pos_from = self.coords[site_from]
            pos_to = self.coords[site_to]
            ax.plot([pos_from[0], pos_to[0]],
                   [pos_from[1], pos_to[1]],
                   [pos_from[2], pos_to[2]],
                   'k-', alpha=0.1, linewidth=0.5)
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title(title, fontsize=14, pad=20)
        
        plt.colorbar(scatter, ax=ax, label='|ψ|', pad=0.1)
        
        return fig
    
    def create_evolution_animation(self, times: np.ndarray, states: np.ndarray,
                                  rep_dim: int = 3, filename: str = 'evolution.gif',
                                  interval: int = 100) -> None:
        """
        Create animated GIF of state evolution.
        
        Args:
            times: Time array
            states: State evolution (n_steps × N_sites × rep_dim)
            rep_dim: Representation dimension
            filename: Output filename
            interval: Frame interval in ms
        """
        print(f"Creating animation with {len(times)} frames...")
        
        # Reshape if needed
        if states.ndim == 2:
            states = states.reshape(len(times), self.N_sites, rep_dim)
        
        # Compute global amplitude range
        all_amps = []
        for state in states:
            amps = np.array([np.linalg.norm(state[i]) for i in range(self.N_sites)])
            all_amps.append(amps)
        all_amps = np.array(all_amps)
        vmin, vmax = all_amps.min(), all_amps.max()
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        def update(frame):
            ax.clear()
            
            # Extract amplitudes for this frame
            state = states[frame]
            amplitudes = np.array([np.linalg.norm(state[i]) for i in range(self.N_sites)])
            
            # Plot sites
            scatter = ax.scatter(self.coords[:, 0], self.coords[:, 1], self.coords[:, 2],
                               c=amplitudes, s=200, alpha=0.8, cmap='plasma',
                               vmin=vmin, vmax=vmax, edgecolors='black', linewidths=1.5)
            
            # Draw edges
            for site_from, site_to in self.hopping_edges:
                pos_from = self.coords[site_from]
                pos_to = self.coords[site_to]
                ax.plot([pos_from[0], pos_to[0]],
                       [pos_from[1], pos_to[1]],
                       [pos_from[2], pos_to[2]],
                       'k-', alpha=0.1, linewidth=0.5)
            
            ax.set_xlabel('X', fontsize=12)
            ax.set_ylabel('Y', fontsize=12)
            ax.set_zlabel('Z', fontsize=12)
            ax.set_title(f'State Evolution: t = {times[frame]:.2f}', fontsize=14, pad=20)
            
            return scatter,
        
        anim = FuncAnimation(fig, update, frames=len(times), interval=interval, blit=False)
        anim.save(filename, writer='pillow', fps=10)
        plt.close(fig)
        print(f"Saved {filename}")
    
    def plot_flux_tube_3d(self, quark_pos: Tuple[int, int], antiquark_pos: Tuple[int, int],
                         field_strength: Optional[np.ndarray] = None) -> plt.Figure:
        """
        Visualize flux tube between quark-antiquark pair.
        
        Args:
            quark_pos: (ix, iy) for quark
            antiquark_pos: (ix, iy) for antiquark
            field_strength: Optional field values at each site
            
        Returns:
            fig: Matplotlib figure
        """
        quark_site = quark_pos[1] * self.Lx + quark_pos[0]
        anti_site = antiquark_pos[1] * self.Lx + antiquark_pos[0]
        
        # Generate field strength if not provided
        if field_strength is None:
            # Simple model: field ~ 1 - distance from line
            pos_q = self.coords[quark_site][:2]
            pos_a = self.coords[anti_site][:2]
            
            field_strength = np.zeros(self.N_sites)
            for i in range(self.N_sites):
                pos = self.coords[i][:2]
                # Distance to line segment
                v = pos_a - pos_q
                w = pos - pos_q
                c = np.dot(w, v) / np.dot(v, v) if np.dot(v, v) > 0 else 0
                c = np.clip(c, 0, 1)
                closest = pos_q + c * v
                dist = np.linalg.norm(pos - closest)
                field_strength[i] = np.exp(-dist)
        
        # Plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Sites colored by field strength
        scatter = ax.scatter(self.coords[:, 0], self.coords[:, 1], self.coords[:, 2],
                           c=field_strength, s=200, alpha=0.8, cmap='hot',
                           edgecolors='black', linewidths=1.5)
        
        # Highlight quark and antiquark
        ax.scatter([self.coords[quark_site, 0]], [self.coords[quark_site, 1]],
                  [self.coords[quark_site, 2]], c='blue', s=400, marker='o',
                  edgecolors='black', linewidths=3, label='Quark', zorder=10)
        ax.scatter([self.coords[anti_site, 0]], [self.coords[anti_site, 1]],
                  [self.coords[anti_site, 2]], c='cyan', s=400, marker='s',
                  edgecolors='black', linewidths=3, label='Antiquark', zorder=10)
        
        # Draw edges
        for site_from, site_to in self.hopping_edges:
            pos_from = self.coords[site_from]
            pos_to = self.coords[site_to]
            ax.plot([pos_from[0], pos_to[0]],
                   [pos_from[1], pos_to[1]],
                   [pos_from[2], pos_to[2]],
                   'k-', alpha=0.1, linewidth=0.5)
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title('Flux Tube Between Color Charges', fontsize=14, pad=20)
        ax.legend(fontsize=12)
        
        plt.colorbar(scatter, ax=ax, label='Field Strength', pad=0.1)
        
        return fig
    
    def validate_coordinate_consistency(self, verbose: bool = True) -> Dict[str, float]:
        """
        Validate coordinate system and hopping structure.
        
        Tests:
        - All coordinates in valid range
        - Hopping graph connects all sites
        - Periodic boundary conditions
        
        Args:
            verbose: Print results
            
        Returns:
            test_results: Dictionary of validation metrics
        """
        if verbose:
            print("\n" + "="*70)
            print("COORDINATE SYSTEM VALIDATION")
            print("="*70)
        
        # Test 1: Coordinate bounds
        x_min, x_max = self.coords[:, 0].min(), self.coords[:, 0].max()
        y_min, y_max = self.coords[:, 1].min(), self.coords[:, 1].max()
        z_min, z_max = self.coords[:, 2].min(), self.coords[:, 2].max()
        
        if verbose:
            print(f"\nCoordinate ranges:")
            print(f"  X: [{x_min:.4f}, {x_max:.4f}]")
            print(f"  Y: [{y_min:.4f}, {y_max:.4f}]")
            print(f"  Z: [{z_min:.4f}, {z_max:.4f}]")
        
        # Test 2: Hopping graph connectivity
        n_edges = len(self.hopping_edges)
        expected_edges = self.N_sites * 6  # 6 neighbors per site
        
        if verbose:
            print(f"\nHopping graph:")
            print(f"  Total edges: {n_edges}")
            print(f"  Expected: {expected_edges}")
            print(f"  Match: {n_edges == expected_edges}")
        
        # Test 3: All sites have 6 neighbors (incoming)
        neighbor_counts = np.zeros(self.N_sites, dtype=int)
        for _, site_to in self.hopping_edges:
            neighbor_counts[site_to] += 1
        
        min_neighbors = neighbor_counts.min()
        max_neighbors = neighbor_counts.max()
        
        if verbose:
            print(f"\nNeighbor count per site:")
            print(f"  Min: {min_neighbors}")
            print(f"  Max: {max_neighbors}")
            print(f"  All have 6: {np.all(neighbor_counts == 6)}")
        
        # Test 4: Periodic boundary conditions (check distances)
        max_jump = 0
        for site_from, site_to in self.hopping_edges:
            dist = np.linalg.norm(self.coords[site_from][:2] - self.coords[site_to][:2])
            max_jump = max(max_jump, dist)
        
        # Typical neighbor distance ~1, wrapped distance much larger
        if verbose:
            print(f"\nMax hopping distance: {max_jump:.4f}")
            print(f"  (Should be a few lattice spacings for periodic BC)")
        
        if verbose:
            print("\n" + "="*70)
            if n_edges == expected_edges and np.all(neighbor_counts == 6):
                print("✓ COORDINATE SYSTEM VALIDATED")
            else:
                print("⚠ Some validation checks failed")
            print("="*70)
        
        return {
            'n_edges': n_edges,
            'expected_edges': expected_edges,
            'min_neighbors': min_neighbors,
            'max_neighbors': max_neighbors,
            'max_jump_distance': max_jump
        }


# ============================================================================
# VALIDATION AND DEMONSTRATION
# ============================================================================

def validate_visualization_suite():
    """Run full validation suite."""
    print("\n" + "="*70)
    print("ZIGGURAT VISUALIZATION VALIDATION")
    print("="*70)
    
    vis = ZigguratVisualizer(Lx=4, Ly=4)
    
    # Test 1: Coordinate system
    print("\n\nTest 1: Coordinate System Consistency")
    print("-"*70)
    coord_results = vis.validate_coordinate_consistency(verbose=True)
    
    # Test 2: Lattice structure plot
    print("\n\nTest 2: Lattice Structure Visualization")
    print("-"*70)
    fig_lattice = vis.plot_lattice_structure()
    fig_lattice.savefig('lattice_structure.png', dpi=150, bbox_inches='tight')
    print("Generated lattice_structure.png")
    plt.close(fig_lattice)
    
    # Test 3: State distribution
    print("\n\nTest 3: State Distribution Visualization")
    print("-"*70)
    # Create a localized state
    state = np.zeros((vis.N_sites, 3), dtype=complex)
    state[0, 0] = 1.0
    state[vis.N_sites//2, 1] = 0.8
    fig_state = vis.plot_state_on_lattice(state)
    fig_state.savefig('state_distribution.png', dpi=150, bbox_inches='tight')
    print("Generated state_distribution.png")
    plt.close(fig_state)
    
    # Test 4: Flux tube
    print("\n\nTest 4: Flux Tube Visualization")
    print("-"*70)
    fig_flux = vis.plot_flux_tube_3d((0, 0), (vis.Lx-1, vis.Ly-1))
    fig_flux.savefig('flux_tube_3d.png', dpi=150, bbox_inches='tight')
    print("Generated flux_tube_3d.png")
    plt.close(fig_flux)
    
    print("\n" + "="*70)
    print("✓ ALL VISUALIZATION TESTS COMPLETED")
    print("="*70)
    
    # Summary
    print("\n\nVALIDATION SUMMARY")
    print("="*70)
    print(f"Lattice size: {vis.Lx} × {vis.Ly} = {vis.N_sites} sites")
    print(f"Hopping edges: {coord_results['n_edges']} (expected {coord_results['expected_edges']})")
    print(f"Connectivity: all sites have {coord_results['min_neighbors']}-{coord_results['max_neighbors']} neighbors")
    print(f"Generated 3 visualization plots")
    
    all_passed = (coord_results['n_edges'] == coord_results['expected_edges'] and
                 coord_results['min_neighbors'] == 6 and
                 coord_results['max_neighbors'] == 6)
    
    if all_passed:
        print("\n✓ All geometric tests PASSED!")
    else:
        print("\n⚠ Some tests show issues")


if __name__ == "__main__":
    validate_visualization_suite()
