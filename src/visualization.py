"""
Phase 7: Visualization and Interpretation Module

This module provides comprehensive visualization tools for exploring and comparing
lattice eigenstates with quantum mechanical predictions. It includes:
1. 2D/3D lattice plots with quantum number color-coding
2. Interactive comparison dashboards
3. Time evolution animations
4. Probability density and transition strength visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D


class LatticeVisualizer:
    """Visualization tools for exploring polar lattice and eigenstates."""
    
    def __init__(self, lattice):
        """
        Initialize visualizer with a PolarLattice instance.
        
        Parameters:
        -----------
        lattice : PolarLattice
            The polar lattice to visualize
        """
        self.lattice = lattice
        self.N = len(lattice.points)
        
    def plot_lattice_2d(self, color_by='shell', figsize=(10, 10), save_path=None):
        """
        Create 2D plot of lattice points with color-coding.
        
        Parameters:
        -----------
        color_by : str
            Color scheme: 'shell' (by n), 'hemisphere' (north/south), 
            'angular' (by theta), or 'phi' (by azimuthal angle)
        figsize : tuple
            Figure size (width, height)
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        fig, ax : matplotlib figure and axis objects
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Extract coordinates from lattice points
        x = np.array([p['x_3d'] for p in self.lattice.points])
        y = np.array([p['y_3d'] for p in self.lattice.points])
        z = np.array([p['z_3d'] for p in self.lattice.points])
        ell_values = np.array([p['ℓ'] for p in self.lattice.points])
        m_s_values = np.array([p['m_s'] for p in self.lattice.points])
        
        # For 2D projection, use x-y of 3D sphere
        
        # Determine colors
        if color_by == 'shell':
            colors = ell_values
            cmap = 'tab20'
            label = 'ℓ value'
        elif color_by == 'hemisphere':
            colors = (m_s_values > 0).astype(int)
            cmap = 'coolwarm'
            label = 'Spin (up/down)'
        elif color_by == 'angular':
            # Use z-coordinate as angular indicator
            colors = np.arccos(np.clip(z, -1, 1))
            cmap = 'viridis'
            label = 'Theta (rad)'
        elif color_by == 'phi':
            # Compute azimuthal angle
            colors = np.arctan2(y, x)
            cmap = 'hsv'
            label = 'Phi (rad)'
        else:
            colors = np.ones(self.N)
            cmap = 'Blues'
            label = 'Uniform'
            
        scatter = ax.scatter(x, y, c=colors, cmap=cmap, s=50, alpha=0.7, edgecolors='black', linewidths=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, label=label)
        
        # Add unit circle
        circle = Circle((0, 0), 1.0, fill=False, color='gray', linestyle='--', linewidth=2)
        ax.add_patch(circle)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(f'Polar Lattice 2D Projection (n_max={self.lattice.n_max}, N={self.N})', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig, ax
        
    def plot_lattice_3d(self, color_by='shell', figsize=(12, 10), save_path=None):
        """
        Create 3D plot of lattice points on unit sphere.
        
        Parameters:
        -----------
        color_by : str
            Color scheme (same options as plot_lattice_2d)
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        fig, ax : matplotlib figure and axis objects
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract Cartesian coordinates
        x = np.array([p['x_3d'] for p in self.lattice.points])
        y = np.array([p['y_3d'] for p in self.lattice.points])
        z = np.array([p['z_3d'] for p in self.lattice.points])
        ell_values = np.array([p['ℓ'] for p in self.lattice.points])
        m_s_values = np.array([p['m_s'] for p in self.lattice.points])
        
        # Determine colors
        if color_by == 'shell':
            colors = ell_values
            cmap = 'tab20'
            label = 'ℓ value'
        elif color_by == 'hemisphere':
            colors = (m_s_values > 0).astype(int)
            cmap = 'coolwarm'
            label = 'Spin (up/down)'
        elif color_by == 'angular':
            colors = np.arccos(np.clip(z, -1, 1))
            cmap = 'viridis'
            label = 'Theta (rad)'
        elif color_by == 'phi':
            colors = np.arctan2(y, x)
            cmap = 'hsv'
            label = 'Phi (rad)'
        else:
            colors = np.ones(self.N)
            cmap = 'Blues'
            label = 'Uniform'
            
        scatter = ax.scatter(x, y, z, c=colors, cmap=cmap, s=50, alpha=0.8, edgecolors='black', linewidths=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, label=label, pad=0.1, shrink=0.8)
        
        # Add wireframe sphere for reference
        u = np.linspace(0, 2*np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        sphere_x = np.outer(np.cos(u), np.sin(v))
        sphere_y = np.outer(np.sin(u), np.sin(v))
        sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color='gray', alpha=0.1, linewidth=0.5)
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title(f'Polar Lattice 3D View (n_max={self.lattice.n_max}, N={self.N})', fontsize=14)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig, ax
        
    def plot_eigenstate(self, state_vector, title='Eigenstate', figsize=(12, 5), save_path=None):
        """
        Visualize a state vector as probability density on the lattice.
        
        Parameters:
        -----------
        state_vector : np.ndarray
            State vector (N,) or (N, 1)
        title : str
            Plot title
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        fig, axes : matplotlib figure and axis objects
        """
        state_vector = state_vector.flatten()
        probability = np.abs(state_vector)**2
        phase = np.angle(state_vector)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Extract coordinates for plotting
        x = np.array([p['x_3d'] for p in self.lattice.points])
        y = np.array([p['y_3d'] for p in self.lattice.points])
        
        # Plot 1: Probability density
        scatter1 = axes[0].scatter(x, y, c=probability, cmap='hot', s=100, 
                                   alpha=0.8, edgecolors='black', linewidths=0.5)
        plt.colorbar(scatter1, ax=axes[0], label='|psi|^2')
        circle1 = Circle((0, 0), 1.0, fill=False, color='gray', linestyle='--', linewidth=2)
        axes[0].add_patch(circle1)
        axes[0].set_xlim(-1.2, 1.2)
        axes[0].set_ylim(-1.2, 1.2)
        axes[0].set_aspect('equal')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].set_title(f'{title}: Probability Density')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Phase
        scatter2 = axes[1].scatter(x, y, c=phase, cmap='hsv', s=100, vmin=-np.pi, vmax=np.pi,
                                   alpha=0.8, edgecolors='black', linewidths=0.5)
        plt.colorbar(scatter2, ax=axes[1], label='Phase (rad)')
        circle2 = Circle((0, 0), 1.0, fill=False, color='gray', linestyle='--', linewidth=2)
        axes[1].add_patch(circle2)
        axes[1].set_xlim(-1.2, 1.2)
        axes[1].set_ylim(-1.2, 1.2)
        axes[1].set_aspect('equal')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        axes[1].set_title(f'{title}: Phase')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig, axes
        
    def animate_time_evolution(self, initial_state, hamiltonian, t_max=10.0, n_frames=100, 
                               figsize=(8, 8), save_path=None):
        """
        Create animation of quantum state time evolution.
        
        Parameters:
        -----------
        initial_state : np.ndarray
            Initial state vector
        hamiltonian : scipy.sparse matrix or np.ndarray
            Hamiltonian operator
        t_max : float
            Maximum time for evolution
        n_frames : int
            Number of animation frames
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save animation (e.g., 'evolution.mp4' or 'evolution.gif')
            
        Returns:
        --------
        anim : matplotlib.animation.Animation object
        """
        from scipy.linalg import expm
        from scipy.sparse import issparse
        
        # Convert sparse to dense for expm
        if issparse(hamiltonian):
            H = hamiltonian.toarray()
        else:
            H = hamiltonian
            
        initial_state = initial_state.flatten()
        
        # Generate time points
        times = np.linspace(0, t_max, n_frames)
        
        # Setup figure
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.array([p['x_3d'] for p in self.lattice.points])
        y = np.array([p['y_3d'] for p in self.lattice.points])
        
        # Initial plot
        scatter = ax.scatter(x, y, c=np.abs(initial_state)**2, cmap='hot', 
                            s=100, vmin=0, vmax=np.max(np.abs(initial_state)**2),
                            alpha=0.8, edgecolors='black', linewidths=0.5)
        plt.colorbar(scatter, ax=ax, label='|psi(t)|^2')
        
        circle = Circle((0, 0), 1.0, fill=False, color='gray', linestyle='--', linewidth=2)
        ax.add_patch(circle)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        title_text = ax.set_title(f'Time Evolution: t = 0.00')
        ax.grid(True, alpha=0.3)
        
        def update(frame):
            t = times[frame]
            # Time evolution: psi(t) = exp(-i H t) psi(0)
            U = expm(-1j * H * t)
            state_t = U @ initial_state
            probability = np.abs(state_t)**2
            
            scatter.set_array(probability)
            title_text.set_text(f'Time Evolution: t = {t:.2f}')
            return scatter, title_text
        
        anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)
        
        if save_path:
            if save_path.endswith('.gif'):
                anim.save(save_path, writer='pillow', fps=20)
            elif save_path.endswith('.mp4'):
                anim.save(save_path, writer='ffmpeg', fps=20)
                
        return anim


class ComparisonDashboard:
    """Tools for side-by-side comparison of lattice results with quantum mechanics."""
    
    def __init__(self, lattice, quantum_comparison):
        """
        Initialize dashboard.
        
        Parameters:
        -----------
        lattice : PolarLattice
            The polar lattice
        quantum_comparison : QuantumComparison
            Quantum comparison object from Phase 4
        """
        self.lattice = lattice
        self.qc = quantum_comparison
        self.visualizer = LatticeVisualizer(lattice)
        
    def compare_eigenstates(self, lattice_state, ell, m, figsize=(16, 6), save_path=None):
        """
        Compare lattice eigenstate with spherical harmonic Y_l^m.
        
        Parameters:
        -----------
        lattice_state : np.ndarray
            Lattice eigenstate vector
        ell : int
            Angular momentum quantum number
        m : int
            Magnetic quantum number
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        fig, axes : matplotlib figure and axes
        overlap : float
            Overlap <Y_l^m | psi_lattice>
        """
        # Sample spherical harmonic on lattice
        ylm_sampled = self.qc.sample_spherical_harmonic(ell, m)
        
        # Compute overlap
        overlap = np.abs(np.vdot(ylm_sampled, lattice_state))**2
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        x = np.array([p['x_3d'] for p in self.lattice.points])
        y = np.array([p['y_3d'] for p in self.lattice.points])
        
        # Row 1: Lattice state
        # Probability
        prob_lattice = np.abs(lattice_state.flatten())**2
        scatter1 = axes[0, 0].scatter(x, y, c=prob_lattice, cmap='hot', s=80, alpha=0.8)
        plt.colorbar(scatter1, ax=axes[0, 0])
        axes[0, 0].set_title('Lattice: |psi|^2')
        axes[0, 0].set_aspect('equal')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Real part
        real_lattice = np.real(lattice_state.flatten())
        scatter2 = axes[0, 1].scatter(x, y, c=real_lattice, cmap='RdBu', s=80, alpha=0.8)
        plt.colorbar(scatter2, ax=axes[0, 1])
        axes[0, 1].set_title('Lattice: Re(psi)')
        axes[0, 1].set_aspect('equal')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Phase
        phase_lattice = np.angle(lattice_state.flatten())
        scatter3 = axes[0, 2].scatter(x, y, c=phase_lattice, cmap='hsv', s=80, alpha=0.8, vmin=-np.pi, vmax=np.pi)
        plt.colorbar(scatter3, ax=axes[0, 2])
        axes[0, 2].set_title('Lattice: Phase')
        axes[0, 2].set_aspect('equal')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Row 2: Quantum state Y_l^m
        # Probability
        prob_ylm = np.abs(ylm_sampled)**2
        scatter4 = axes[1, 0].scatter(x, y, c=prob_ylm, cmap='hot', s=80, alpha=0.8)
        plt.colorbar(scatter4, ax=axes[1, 0])
        axes[1, 0].set_title(f'Quantum: |Y_{ell}^{m}|^2')
        axes[1, 0].set_aspect('equal')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Real part
        real_ylm = np.real(ylm_sampled)
        scatter5 = axes[1, 1].scatter(x, y, c=real_ylm, cmap='RdBu', s=80, alpha=0.8)
        plt.colorbar(scatter5, ax=axes[1, 1])
        axes[1, 1].set_title(f'Quantum: Re(Y_{ell}^{m})')
        axes[1, 1].set_aspect('equal')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Phase
        phase_ylm = np.angle(ylm_sampled)
        scatter6 = axes[1, 2].scatter(x, y, c=phase_ylm, cmap='hsv', s=80, alpha=0.8, vmin=-np.pi, vmax=np.pi)
        plt.colorbar(scatter6, ax=axes[1, 2])
        axes[1, 2].set_title(f'Quantum: Phase')
        axes[1, 2].set_aspect('equal')
        axes[1, 2].grid(True, alpha=0.3)
        
        fig.suptitle(f'Lattice vs Quantum: l={ell}, m={m} (Overlap = {overlap:.4f})', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig, axes, overlap
        
    def compare_energy_levels(self, lattice_energies, hydrogen_energies, n_max=5, figsize=(12, 6), save_path=None):
        """
        Side-by-side energy level comparison.
        
        Parameters:
        -----------
        lattice_energies : np.ndarray
            Lattice eigenvalues (sorted)
        hydrogen_energies : np.ndarray
            Hydrogen eigenvalues for comparison
        n_max : int
            Maximum principal quantum number to display
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        fig, axes : matplotlib figure and axes
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Left: Energy level diagram
        n_states = min(len(lattice_energies), len(hydrogen_energies))
        indices = np.arange(n_states)
        
        axes[0].plot(indices, lattice_energies[:n_states], 'o-', label='Lattice', markersize=4)
        axes[0].plot(indices, hydrogen_energies[:n_states], 's-', label='Hydrogen', markersize=4, alpha=0.7)
        axes[0].set_xlabel('State Index')
        axes[0].set_ylabel('Energy')
        axes[0].set_title('Energy Levels Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Right: Error analysis
        errors = lattice_energies[:n_states] - hydrogen_energies[:n_states]
        relative_errors = np.abs(errors / hydrogen_energies[:n_states]) * 100
        
        axes[1].plot(indices, relative_errors, 'o-', color='red', markersize=4)
        axes[1].set_xlabel('State Index')
        axes[1].set_ylabel('Relative Error (%)')
        axes[1].set_title('Energy Level Error')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig, axes
        
    def selection_rules_heatmap(self, dipole_matrix, ell_values, m_values, figsize=(14, 5), save_path=None):
        """
        Create heatmap showing dipole transition strengths and selection rules.
        
        Parameters:
        -----------
        dipole_matrix : np.ndarray
            Dipole transition matrix element magnitudes
        ell_values : list
            Angular momentum values for states
        m_values : list
            Magnetic quantum numbers for states
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
            
        Returns:
        --------
        fig, axes : matplotlib figure and axes
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Dipole matrices for x, y, z components (assuming 3D array or separate matrices)
        if dipole_matrix.ndim == 3:
            matrices = [dipole_matrix[:, :, i] for i in range(3)]
            labels = ['D_x', 'D_y', 'D_z']
        else:
            matrices = [dipole_matrix]
            labels = ['Dipole']
            
        for idx, (mat, label) in enumerate(zip(matrices[:3], labels)):
            im = axes[idx].imshow(np.abs(mat), cmap='hot', aspect='auto', interpolation='nearest')
            plt.colorbar(im, ax=axes[idx], label='|<i|' + label + '|j>|')
            axes[idx].set_xlabel('Final State')
            axes[idx].set_ylabel('Initial State')
            axes[idx].set_title(f'{label} Transitions')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig, axes


class DocumentationGenerator:
    """Generate summary reports and documentation of findings."""
    
    def __init__(self, project_name='State Space Model'):
        """
        Initialize documentation generator.
        
        Parameters:
        -----------
        project_name : str
            Name of the project
        """
        self.project_name = project_name
        self.findings = []
        
    def add_finding(self, phase, description, metrics, status='SUCCESS'):
        """
        Add a finding to the documentation.
        
        Parameters:
        -----------
        phase : str
            Phase identifier (e.g., 'Phase 4.1')
        description : str
            Description of the experiment/test
        metrics : dict
            Dictionary of measured metrics
        status : str
            Status: 'SUCCESS', 'PARTIAL', 'FAIL'
        """
        self.findings.append({
            'phase': phase,
            'description': description,
            'metrics': metrics,
            'status': status
        })
        
    def generate_summary_report(self, output_path='FINDINGS_SUMMARY.md'):
        """
        Generate markdown summary report of all findings.
        
        Parameters:
        -----------
        output_path : str
            Path to save the report
            
        Returns:
        --------
        report : str
            The generated report text
        """
        report_lines = [
            f"# {self.project_name}: Summary of Findings\n",
            f"Generated: {np.datetime64('today')}\n",
            "---\n\n"
        ]
        
        # Group by phase
        phases = {}
        for finding in self.findings:
            phase = finding['phase']
            if phase not in phases:
                phases[phase] = []
            phases[phase].append(finding)
            
        # Write findings by phase
        for phase in sorted(phases.keys()):
            report_lines.append(f"## {phase}\n\n")
            
            for finding in phases[phase]:
                status_emoji = {'SUCCESS': 'Y', 'PARTIAL': '~', 'FAIL': 'X'}[finding['status']]
                report_lines.append(f"### [{status_emoji}] {finding['description']}\n\n")
                
                if finding['metrics']:
                    report_lines.append("**Metrics:**\n")
                    for key, value in finding['metrics'].items():
                        if isinstance(value, float):
                            report_lines.append(f"- {key}: {value:.6f}\n")
                        else:
                            report_lines.append(f"- {key}: {value}\n")
                    report_lines.append("\n")
                    
        # Add summary statistics
        report_lines.append("---\n\n## Summary Statistics\n\n")
        total = len(self.findings)
        success = sum(1 for f in self.findings if f['status'] == 'SUCCESS')
        partial = sum(1 for f in self.findings if f['status'] == 'PARTIAL')
        fail = sum(1 for f in self.findings if f['status'] == 'FAIL')
        
        report_lines.append(f"- Total Experiments: {total}\n")
        report_lines.append(f"- Successful: {success} ({100*success/total:.1f}%)\n")
        report_lines.append(f"- Partial Success: {partial} ({100*partial/total:.1f}%)\n")
        report_lines.append(f"- Failed: {fail} ({100*fail/total:.1f}%)\n")
        
        report = ''.join(report_lines)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
            
        return report
        
    def generate_technical_summary(self, phases_data, output_path='TECHNICAL_SUMMARY.md'):
        """
        Generate detailed technical summary with equations and results.
        
        Parameters:
        -----------
        phases_data : dict
            Dictionary containing data from all phases
        output_path : str
            Path to save summary
            
        Returns:
        --------
        summary : str
            The generated technical summary
        """
        lines = [
            f"# {self.project_name}: Technical Summary\n\n",
            "## Abstract\n\n",
            "This document summarizes the implementation and validation of a discrete polar lattice\n",
            "model for quantum angular momentum on the sphere S^2. The model approximates continuous\n",
            "quantum mechanics using a finite lattice of points with discrete operators.\n\n",
            "---\n\n"
        ]
        
        # Add phase summaries
        for phase_name, phase_info in phases_data.items():
            lines.append(f"## {phase_name}\n\n")
            lines.append(f"**Objective:** {phase_info.get('objective', 'N/A')}\n\n")
            
            if 'methods' in phase_info:
                lines.append("**Methods:**\n")
                for method in phase_info['methods']:
                    lines.append(f"- {method}\n")
                lines.append("\n")
                
            if 'results' in phase_info:
                lines.append("**Key Results:**\n")
                for result in phase_info['results']:
                    lines.append(f"- {result}\n")
                lines.append("\n")
                
            if 'conclusions' in phase_info:
                lines.append("**Conclusions:**\n")
                for conclusion in phase_info['conclusions']:
                    lines.append(f"- {conclusion}\n")
                lines.append("\n")
                
            lines.append("---\n\n")
            
        summary = ''.join(lines)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)
            
        return summary
