"""
Core lattice construction module for quantum-geometric 2D polar lattice.

This module implements a discrete 2D polar lattice that exactly reproduces
the quantum state degeneracy structure of the hydrogen atom, including spin.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class PolarLattice:
    """
    A discrete 2D polar lattice that reproduces hydrogen atom degeneracy structure.
    
    Each azimuthal quantum number ℓ corresponds to one ring with:
    - Radius: r_ℓ = 1 + 2ℓ
    - Points: N_ℓ = 2(2ℓ+1) (encoding 2ℓ+1 orbitals × 2 spins)
    
    Principal quantum number n includes all ℓ from 0 to n-1, giving:
    - n² orbitals per shell
    - 2n² electron states per shell
    """
    
    def __init__(self, n_max):
        """
        Initialize lattice up to principal quantum number n_max.
        
        Parameters
        ----------
        n_max : int
            Maximum principal quantum number to include (n=1,2,3,...)
        """
        self.n_max = n_max
        self.ℓ_max = n_max - 1  # Maximum azimuthal quantum number
        
        # Build the lattice
        self._build_lattice()
    
    def _build_lattice(self):
        """
        Construct all lattice points and their metadata.
        
        Creates data structures storing:
        - 2D positions (x, y) and polar (r, θ)
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
        
        # Convert to numpy structured array for efficient access
        self.points_array = np.array(
            [(p['ℓ'], p['j'], p['r'], p['θ'], p['x_2d'], p['y_2d'], 
              p['m_ℓ'], p['m_s'], p['x_3d'], p['y_3d'], p['z_3d']) 
             for p in self.points],
            dtype=[('ℓ', int), ('j', int), ('r', float), ('θ', float),
                   ('x_2d', float), ('y_2d', float), ('m_ℓ', float), ('m_s', float),
                   ('x_3d', float), ('y_3d', float), ('z_3d', float)]
        )
    
    def get_quantum_numbers(self, ℓ, j):
        """
        Map lattice site (ℓ, j) to quantum numbers (ℓ, m_ℓ, m_s).
        
        Uses interleaved spin scheme:
        - Even j → m_s = +½ (spin up)
        - Odd j → m_s = -½ (spin down)
        - m_ℓ ranges from -ℓ to +ℓ, each appearing twice
        
        Parameters
        ----------
        ℓ : int
            Azimuthal quantum number (determines ring)
        j : int
            Site index on ring ℓ (0, 1, ..., N_ℓ - 1)
        
        Returns
        -------
        tuple
            (ℓ, m_ℓ, m_s) where m_ℓ ∈ [-ℓ, ℓ] and m_s ∈ {-½, +½}
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
    
    def get_site_index(self, ℓ, m_ℓ, m_s):
        """
        Map quantum numbers (ℓ, m_ℓ, m_s) to lattice site index j.
        
        Inverse of get_quantum_numbers.
        
        Parameters
        ----------
        ℓ : int
            Azimuthal quantum number
        m_ℓ : int or float
            Magnetic quantum number (-ℓ to +ℓ)
        m_s : float
            Spin quantum number (+½ or -½)
        
        Returns
        -------
        int
            Site index j on ring ℓ
        """
        assert -ℓ <= m_ℓ <= ℓ, f"m_ℓ={m_ℓ} out of range for ℓ={ℓ}"
        assert m_s in [0.5, -0.5], f"m_s must be ±0.5, got {m_s}"
        
        # Reverse of the get_quantum_numbers mapping
        base = int((m_ℓ + ℓ) * 2)  # Which pair of indices?
        offset = 0 if m_s == 0.5 else 1  # Even (spin-up) or odd (spin-down)?
        j = base + offset
        
        return j
    
    def spherical_lift(self, ℓ, m_ℓ, m_s):
        """
        Map quantum numbers to 3D coordinates on unit sphere.
        
        Each 2D ring lifts to two latitude bands (north/south hemispheres).
        - North band: m_s = +½
        - South band: m_s = -½
        
        Parameters
        ----------
        ℓ : int
            Azimuthal quantum number
        m_ℓ : int or float
            Magnetic quantum number
        m_s : float
            Spin quantum number
        
        Returns
        -------
        tuple
            (x, y, z) Cartesian coordinates on unit sphere
        """
        # Colatitude θ: determines latitude band for this ℓ
        # Bands spread from pole (0) to equator (π/2)
        if self.ℓ_max == 0:
            θ_ℓ = np.pi / 4  # Special case for single ring
        else:
            θ_ℓ = (np.pi / 2) * (ℓ + 0.5) / (self.ℓ_max + 1)
        
        # Azimuthal angle φ: determined by m_ℓ
        # Spread the (2ℓ+1) values of m_ℓ uniformly around the circle
        if ℓ == 0:
            φ = 0  # ℓ=0 has only one m_ℓ value
        else:
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
    
    def get_ring(self, ℓ):
        """
        Return all points on ring ℓ.
        
        Parameters
        ----------
        ℓ : int
            Azimuthal quantum number
        
        Returns
        -------
        list
            List of point dictionaries for ring ℓ
        """
        return [p for p in self.points if p['ℓ'] == ℓ]
    
    def get_shell(self, n):
        """
        Return all points in shell n (ℓ = 0 to n-1).
        
        Parameters
        ----------
        n : int
            Principal quantum number
        
        Returns
        -------
        list
            List of point dictionaries in shell n
        """
        return [p for p in self.points if p['ℓ'] < n]
    
    def count_orbitals(self, n):
        """
        Return number of orbitals in shell n.
        
        Should be n² for hydrogen atom.
        
        Parameters
        ----------
        n : int
            Principal quantum number
        
        Returns
        -------
        int
            Number of orbitals (without counting spin)
        """
        return sum(2*ℓ + 1 for ℓ in range(n))
    
    def count_states(self, n):
        """
        Return number of electron states in shell n.
        
        Should be 2n² for hydrogen atom (including spin).
        
        Parameters
        ----------
        n : int
            Principal quantum number
        
        Returns
        -------
        int
            Number of electron states (including spin)
        """
        return len(self.get_shell(n))
    
    def plot_2d(self, n_max=None, color_by='ℓ', figsize=(10, 10), ax=None):
        """
        Plot 2D lattice projection.
        
        Parameters
        ----------
        n_max : int, optional
            If provided, only plot shells up to this n
        color_by : str, default='ℓ'
            Color scheme: 'ℓ', 'm_ℓ', 'm_s', or 'ring'
        figsize : tuple, default=(10, 10)
            Figure size
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure
        
        Returns
        -------
        tuple
            (fig, ax) matplotlib figure and axes
        """
        if n_max is None:
            n_max = self.n_max
        
        points_to_plot = self.get_shell(n_max)
        
        # Create figure if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Extract coordinates and color values
        x = [p['x_2d'] for p in points_to_plot]
        y = [p['y_2d'] for p in points_to_plot]
        
        if color_by == 'ℓ':
            colors = [p['ℓ'] for p in points_to_plot]
            label = 'ℓ'
        elif color_by == 'm_ℓ':
            colors = [p['m_ℓ'] for p in points_to_plot]
            label = 'm_ℓ'
        elif color_by == 'm_s':
            colors = [p['m_s'] for p in points_to_plot]
            label = 'm_s'
        else:
            colors = 'blue'
            label = None
        
        scatter = ax.scatter(x, y, c=colors, cmap='viridis', s=100, alpha=0.6)
        
        if label:
            plt.colorbar(scatter, ax=ax, label=label)
        
        ax.set_aspect('equal')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(f'2D Polar Lattice (n_max={n_max})', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    def plot_3d(self, n_max=None, color_by='m_s', figsize=(12, 10), ax=None):
        """
        Plot spherical lift in 3D.
        
        Parameters
        ----------
        n_max : int, optional
            If provided, only plot shells up to this n
        color_by : str, default='m_s'
            Color scheme: 'ℓ', 'm_ℓ', 'm_s'
        figsize : tuple, default=(12, 10)
            Figure size
        ax : matplotlib.axes.Axes3D, optional
            3D axes to plot on. If None, creates new figure
        
        Returns
        -------
        tuple
            (fig, ax) matplotlib figure and 3D axes
        """
        if n_max is None:
            n_max = self.n_max
        
        points_to_plot = self.get_shell(n_max)
        
        # Create figure if not provided
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure
        
        # Extract coordinates and color values
        x = [p['x_3d'] for p in points_to_plot]
        y = [p['y_3d'] for p in points_to_plot]
        z = [p['z_3d'] for p in points_to_plot]
        
        if color_by == 'ℓ':
            colors = [p['ℓ'] for p in points_to_plot]
            cmap = 'viridis'
            label = 'ℓ'
        elif color_by == 'm_ℓ':
            colors = [p['m_ℓ'] for p in points_to_plot]
            cmap = 'coolwarm'
            label = 'm_ℓ'
        elif color_by == 'm_s':
            colors = [1 if p['m_s'] > 0 else 0 for p in points_to_plot]
            cmap = 'bwr'
            label = 'm_s'
        else:
            colors = 'blue'
            cmap = None
            label = None
        
        scatter = ax.scatter(x, y, z, c=colors, cmap=cmap, s=100, alpha=0.6)
        
        if cmap:
            plt.colorbar(scatter, ax=ax, label=label, shrink=0.5)
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_zlabel('z', fontsize=12)
        ax.set_title(f'3D Spherical Lift (n_max={n_max})', fontsize=14)
        
        # Draw sphere surface for reference
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')
        
        return fig, ax
    
    def __repr__(self):
        """String representation of the lattice."""
        total_states = len(self.points)
        return (f"PolarLattice(n_max={self.n_max}, ℓ_max={self.ℓ_max}, "
                f"total_states={total_states})")
