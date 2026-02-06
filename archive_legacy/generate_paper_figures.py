"""
Publication-Quality Figure Generation for "The Geometric Atom" Paper

This script generates three publication-ready figures:
1. 3D Paraboloid Architecture with color-coded shells
2. 2D Projection showing Balmer series transition path
3. Spy plot demonstrating operator sparsity

Outputs: PDF/PNG/SVG vector graphics for LaTeX inclusion

Author: J. Louthan
Date: February 2026
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, lil_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches

# Set publication style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.dpi'] = 150


# =============================================================================
# ParaboloidLattice Class (Embedded for Standalone Execution)
# =============================================================================

class ParaboloidLattice:
    """
    Discrete Paraboloid Lattice implementing SO(4,2) conformal algebra.
    Maps quantum states |n,l,m⟩ to 3D paraboloid surface coordinates.
    """
    
    def __init__(self, max_n: int):
        self.max_n = max_n
        self.nodes = []
        self.node_index = {}
        self.coordinates = []
        
        self._construct_nodes()
        self._compute_coordinates()
        
        self.dim = len(self.nodes)
        
        # Operators
        self.Lz = None
        self.Lplus = None
        self.Lminus = None
        self.T3 = None
        self.Tplus = None
        self.Tminus = None
        
        self._build_operators()
    
    def _construct_nodes(self):
        """Generate all valid (n, l, m) quantum numbers."""
        idx = 0
        for n in range(1, self.max_n + 1):
            for l in range(n):
                for m in range(-l, l + 1):
                    node = (n, l, m)
                    self.nodes.append(node)
                    self.node_index[node] = idx
                    idx += 1
    
    def _compute_coordinates(self):
        """Map quantum numbers to 3D Euclidean coordinates."""
        for n, l, m in self.nodes:
            r = n**2
            z = -1.0 / (n**2)
            
            if n > 1:
                theta = np.pi * l / (n - 1)
            else:
                theta = 0.0
            
            if l > 0:
                phi = 2 * np.pi * (m + l) / (2 * l + 1)
            else:
                phi = 0.0
            
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            
            self.coordinates.append((x, y, z))
        
        self.coordinates = np.array(self.coordinates)
    
    def _build_operators(self):
        """Construct sparse matrix operators."""
        # Lz (diagonal)
        Lz_data = [m for n, l, m in self.nodes]
        self.Lz = sp.diags(Lz_data, 0, shape=(self.dim, self.dim),
                          dtype=np.complex128, format='csr')
        
        # L+ (raising)
        Lplus_mat = lil_matrix((self.dim, self.dim), dtype=np.complex128)
        for n, l, m in self.nodes:
            if m < l:
                idx_from = self.node_index[(n, l, m)]
                idx_to = self.node_index[(n, l, m + 1)]
                coeff = np.sqrt((l - m) * (l + m + 1))
                Lplus_mat[idx_to, idx_from] = coeff
        self.Lplus = Lplus_mat.tocsr()
        
        # L- (lowering)
        Lminus_mat = lil_matrix((self.dim, self.dim), dtype=np.complex128)
        for n, l, m in self.nodes:
            if m > -l:
                idx_from = self.node_index[(n, l, m)]
                idx_to = self.node_index[(n, l, m - 1)]
                coeff = np.sqrt((l + m) * (l - m + 1))
                Lminus_mat[idx_to, idx_from] = coeff
        self.Lminus = Lminus_mat.tocsr()
        
        # T3 (diagonal)
        T3_data = [(n + l + 1) / 2.0 for n, l, m in self.nodes]
        self.T3 = sp.diags(T3_data, 0, shape=(self.dim, self.dim),
                          dtype=np.complex128, format='csr')
        
        # T+ (radial raising)
        Tplus_mat = lil_matrix((self.dim, self.dim), dtype=np.complex128)
        for n, l, m in self.nodes:
            if n < self.max_n:
                target = (n + 1, l, m)
                if target in self.node_index:
                    idx_from = self.node_index[(n, l, m)]
                    idx_to = self.node_index[target]
                    coeff = np.sqrt((n - l) * (n + l + 1) / 4.0)
                    Tplus_mat[idx_to, idx_from] = coeff
        self.Tplus = Tplus_mat.tocsr()
        
        # T- (radial lowering)
        Tminus_mat = lil_matrix((self.dim, self.dim), dtype=np.complex128)
        for n, l, m in self.nodes:
            if n > 1:
                target = (n - 1, l, m)
                if target in self.node_index:
                    idx_from = self.node_index[(n, l, m)]
                    idx_to = self.node_index[target]
                    coeff = np.sqrt((n - l) * (n + l) / 4.0)
                    Tminus_mat[idx_to, idx_from] = coeff
        self.Tminus = Tminus_mat.tocsr()


# =============================================================================
# Figure 1: 3D Paraboloid Architecture
# =============================================================================

def generate_figure1_paraboloid_3d(max_n=5, save_formats=['pdf', 'png']):
    """
    Generate 3D visualization of the paraboloid lattice.
    
    Features:
    - Nodes colored by principal quantum number n
    - Angular connections (L±) in grey
    - Radial connections (T±) in red
    """
    print("Generating Figure 1: 3D Paraboloid Architecture...")
    
    lattice = ParaboloidLattice(max_n=max_n)
    coords = lattice.coordinates
    nodes = lattice.nodes
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract n values for coloring
    n_values = np.array([n for n, l, m in nodes])
    
    # Color map for shells
    cmap = plt.colormaps.get_cmap('viridis')
    norm = Normalize(vmin=1, vmax=max_n)
    colors = cmap(norm(n_values))
    
    # Plot nodes
    scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                        c=n_values, cmap='viridis', s=60, alpha=0.8,
                        edgecolors='black', linewidth=0.5, depthshade=True)
    
    # Draw angular connections (L±) - grey rings
    print("  Drawing angular connections...")
    angular_connections = []
    for idx, (n, l, m) in enumerate(nodes):
        if m < l:
            target = (n, l, m + 1)
            if target in lattice.node_index:
                idx_to = lattice.node_index[target]
                angular_connections.append((idx, idx_to))
    
    # Sample to avoid overcrowding
    max_angular = min(200, len(angular_connections))
    if len(angular_connections) > max_angular:
        indices = np.random.choice(len(angular_connections), max_angular, replace=False)
        angular_sample = [angular_connections[i] for i in indices]
    else:
        angular_sample = angular_connections
    
    for i, j in angular_sample:
        ax.plot([coords[i, 0], coords[j, 0]],
               [coords[i, 1], coords[j, 1]],
               [coords[i, 2], coords[j, 2]],
               'grey', alpha=0.15, linewidth=0.6, zorder=1)
    
    # Draw radial connections (T±) - red ladders
    print("  Drawing radial connections...")
    radial_connections = []
    for idx, (n, l, m) in enumerate(nodes):
        if n < max_n:
            target = (n + 1, l, m)
            if target in lattice.node_index:
                idx_to = lattice.node_index[target]
                radial_connections.append((idx, idx_to))
    
    for i, j in radial_connections:
        ax.plot([coords[i, 0], coords[j, 0]],
               [coords[i, 1], coords[j, 1]],
               [coords[i, 2], coords[j, 2]],
               'red', alpha=0.5, linewidth=1.2, zorder=2)
    
    # Styling
    ax.set_xlabel(r'$x$ (a.u.)', fontsize=11, labelpad=8)
    ax.set_ylabel(r'$y$ (a.u.)', fontsize=11, labelpad=8)
    ax.set_zlabel(r'$z = -1/n^2$ (Energy)', fontsize=11, labelpad=8)
    ax.set_title(r'The Geometric Atom: Paraboloid Lattice Structure',
                fontsize=13, fontweight='bold', pad=15)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7, aspect=15)
    cbar.set_label(r'Principal Quantum Number $n$', fontsize=10)
    
    # View angle
    ax.view_init(elev=25, azim=45)
    
    # Legend
    grey_line = plt.Line2D([0], [0], color='grey', linewidth=2, alpha=0.5)
    red_line = plt.Line2D([0], [0], color='red', linewidth=2, alpha=0.7)
    ax.legend([grey_line, red_line],
             [r'Angular ($L_\pm$)', r'Radial ($T_\pm$)'],
             loc='upper left', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save
    for fmt in save_formats:
        filename = f'figure1_paraboloid_3d.{fmt}'
        plt.savefig(filename, format=fmt, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filename}")
    
    plt.close()


# =============================================================================
# Figure 2: 2D Projection with Balmer Series Path
# =============================================================================

def generate_figure2_transition_path(max_n=6, save_formats=['pdf', 'png']):
    """
    Generate 2D side-view projection showing a transition pathway.
    Highlights the Balmer series: n=3→2 decay channel.
    """
    print("\nGenerating Figure 2: 2D Projection with Transition Path...")
    
    lattice = ParaboloidLattice(max_n=max_n)
    coords = lattice.coordinates
    nodes = lattice.nodes
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # === Left Panel: Side View (r-z projection) ===
    
    # Compute radial distance from z-axis
    r_radial = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
    z_coords = coords[:, 2]
    
    n_values = np.array([n for n, l, m in nodes])
    
    # Plot all nodes
    scatter1 = ax1.scatter(r_radial, z_coords, c=n_values, cmap='viridis',
                          s=40, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Highlight Balmer series path: 3p → 2p (example with l=1, m=0)
    if (3, 1, 0) in lattice.node_index and (2, 1, 0) in lattice.node_index:
        idx_3p = lattice.node_index[(3, 1, 0)]
        idx_2p = lattice.node_index[(2, 1, 0)]
        
        # Draw transition arrow
        ax1.annotate('',
                    xy=(r_radial[idx_2p], z_coords[idx_2p]),
                    xytext=(r_radial[idx_3p], z_coords[idx_3p]),
                    arrowprops=dict(arrowstyle='->', lw=2.5, color='red'))
        
        # Label states
        ax1.plot(r_radial[idx_3p], z_coords[idx_3p], 'ro', markersize=10,
                label=r'$|3,1,0\rangle$ (initial)', zorder=5)
        ax1.plot(r_radial[idx_2p], z_coords[idx_2p], 'bs', markersize=10,
                label=r'$|2,1,0\rangle$ (final)', zorder=5)
    
    # Draw some radial connections
    for idx, (n, l, m) in enumerate(nodes):
        if n < max_n and (n+1, l, m) in lattice.node_index:
            idx_to = lattice.node_index[(n+1, l, m)]
            ax1.plot([r_radial[idx], r_radial[idx_to]],
                    [z_coords[idx], z_coords[idx_to]],
                    'grey', alpha=0.15, linewidth=0.5, zorder=1)
    
    ax1.set_xlabel(r'Radial Distance $r = \sqrt{x^2 + y^2}$ (a.u.)', fontsize=11)
    ax1.set_ylabel(r'Energy Coordinate $z = -1/n^2$', fontsize=11)
    ax1.set_title(r'Side View: Balmer Series $3p \to 2p$ Transition',
                 fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='lower right')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    cbar1 = plt.colorbar(scatter1, ax=ax1, pad=0.02, shrink=0.8)
    cbar1.set_label(r'$n$', fontsize=10)
    
    # === Right Panel: Top View (x-y projection) ===
    
    scatter2 = ax2.scatter(coords[:, 0], coords[:, 1], c=n_values, cmap='viridis',
                          s=40, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Draw angular connections for visualization
    for idx, (n, l, m) in enumerate(nodes):
        if m < l and (n, l, m+1) in lattice.node_index:
            idx_to = lattice.node_index[(n, l, m+1)]
            ax2.plot([coords[idx, 0], coords[idx_to, 0]],
                    [coords[idx, 1], coords[idx_to, 1]],
                    'grey', alpha=0.2, linewidth=0.5, zorder=1)
    
    # Highlight n=2 shell (Balmer final state)
    n2_indices = [i for i, (n, l, m) in enumerate(nodes) if n == 2]
    if len(n2_indices) > 0:
        ax2.scatter(coords[n2_indices, 0], coords[n2_indices, 1],
                   s=120, facecolors='none', edgecolors='red',
                   linewidth=2.5, label=r'$n=2$ shell', zorder=4)
    
    ax2.set_xlabel(r'$x$ (a.u.)', fontsize=11)
    ax2.set_ylabel(r'$y$ (a.u.)', fontsize=11)
    ax2.set_title(r'Top View: $SO(4)$ Angular Structure',
                 fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.set_aspect('equal')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    cbar2 = plt.colorbar(scatter2, ax=ax2, pad=0.02, shrink=0.8)
    cbar2.set_label(r'$n$', fontsize=10)
    
    plt.suptitle(r'Transition Pathways on the Paraboloid Lattice',
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    for fmt in save_formats:
        filename = f'figure2_transition_path.{fmt}'
        plt.savefig(filename, format=fmt, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filename}")
    
    plt.close()


# =============================================================================
# Figure 3: Sparsity Spy Plot
# =============================================================================

def generate_figure3_sparsity(max_n=7, save_formats=['pdf', 'png']):
    """
    Generate spy plots showing the sparsity structure of operators.
    Demonstrates computational efficiency.
    """
    print("\nGenerating Figure 3: Operator Sparsity Structure...")
    
    lattice = ParaboloidLattice(max_n=max_n)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    
    # === Top Left: T+ (Radial Raising) ===
    ax1 = axes[0, 0]
    ax1.spy(lattice.Tplus, markersize=2, color='red', alpha=0.8)
    nnz_Tp = lattice.Tplus.nnz
    density_Tp = 100 * nnz_Tp / (lattice.dim**2)
    ax1.set_title(r'$T_+$ (Radial Raising Operator)' + f'\n{nnz_Tp} non-zero / {lattice.dim}$^2$ = {density_Tp:.2f}%',
                 fontsize=11, fontweight='bold')
    ax1.set_xlabel('Column Index', fontsize=10)
    ax1.set_ylabel('Row Index', fontsize=10)
    
    # === Top Right: L+ (Angular Raising) ===
    ax2 = axes[0, 1]
    ax2.spy(lattice.Lplus, markersize=2, color='blue', alpha=0.8)
    nnz_Lp = lattice.Lplus.nnz
    density_Lp = 100 * nnz_Lp / (lattice.dim**2)
    ax2.set_title(r'$L_+$ (Angular Raising Operator)' + f'\n{nnz_Lp} non-zero / {lattice.dim}$^2$ = {density_Lp:.2f}%',
                 fontsize=11, fontweight='bold')
    ax2.set_xlabel('Column Index', fontsize=10)
    ax2.set_ylabel('Row Index', fontsize=10)
    
    # === Bottom Left: L² (Angular Momentum Casimir) ===
    Lz_squared = lattice.Lz @ lattice.Lz
    L_anticomm = lattice.Lplus @ lattice.Lminus + lattice.Lminus @ lattice.Lplus
    L_squared = Lz_squared + 0.5 * L_anticomm
    
    ax3 = axes[1, 0]
    ax3.spy(L_squared, markersize=2, color='green', alpha=0.8)
    nnz_L2 = L_squared.nnz
    density_L2 = 100 * nnz_L2 / (lattice.dim**2)
    ax3.set_title(r'$L^2$ (Angular Momentum Casimir)' + f'\n{nnz_L2} non-zero / {lattice.dim}$^2$ = {density_L2:.2f}%',
                 fontsize=11, fontweight='bold')
    ax3.set_xlabel('Column Index', fontsize=10)
    ax3.set_ylabel('Row Index', fontsize=10)
    
    # === Bottom Right: Combined Hamiltonian Approximation ===
    # H ≈ T3 + L² (simplified, for visualization)
    H_approx = lattice.T3 + 0.1 * L_squared
    
    ax4 = axes[1, 1]
    ax4.spy(H_approx, markersize=2, color='purple', alpha=0.8)
    nnz_H = H_approx.nnz
    density_H = 100 * nnz_H / (lattice.dim**2)
    ax4.set_title(r'$H_{\rm approx} = T_3 + 0.1 L^2$' + f'\n{nnz_H} non-zero / {lattice.dim}$^2$ = {density_H:.2f}%',
                 fontsize=11, fontweight='bold')
    ax4.set_xlabel('Column Index', fontsize=10)
    ax4.set_ylabel('Row Index', fontsize=10)
    
    # Add text box with statistics
    textstr = f'Lattice Statistics:\n'
    textstr += f'  max_n = {max_n}\n'
    textstr += f'  Total states = {lattice.dim}\n'
    textstr += f'  Matrix size = {lattice.dim}×{lattice.dim}\n'
    textstr += f'  Dense elements = {lattice.dim**2:,}\n'
    textstr += f'  Sparse storage = ~{nnz_Tp + nnz_Lp} elements\n'
    textstr += f'  Compression = {100*(1 - (nnz_Tp+nnz_Lp)/(lattice.dim**2)):.1f}%'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    fig.text(0.98, 0.02, textstr, fontsize=9, verticalalignment='bottom',
            horizontalalignment='right', bbox=props, family='monospace')
    
    plt.suptitle(r'Sparse Matrix Structure: Computational Efficiency of the Lattice',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.05, 1, 0.99])
    
    # Save
    for fmt in save_formats:
        filename = f'figure3_sparsity.{fmt}'
        plt.savefig(filename, format=fmt, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filename}")
    
    plt.close()


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Generate all publication figures."""
    print("="*70)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print("For: 'The Geometric Atom: A Discrete Conformal Paraboloid'")
    print("="*70 + "\n")
    
    # Generate figures with appropriate max_n for clarity
    generate_figure1_paraboloid_3d(max_n=5, save_formats=['pdf', 'png'])
    generate_figure2_transition_path(max_n=6, save_formats=['pdf', 'png'])
    generate_figure3_sparsity(max_n=7, save_formats=['pdf', 'png'])
    
    print("\n" + "="*70)
    print("FIGURE GENERATION COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  figure1_paraboloid_3d.pdf / .png")
    print("  figure2_transition_path.pdf / .png")
    print("  figure3_sparsity.pdf / .png")
    print("\nThese can be included in LaTeX using:")
    print("  \\includegraphics[width=0.8\\linewidth]{figure1_paraboloid_3d.pdf}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
