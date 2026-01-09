"""
Hamiltonian and operator construction for the polar lattice.

This module implements:
- Adjacency structure (angular and radial neighbors)
- Laplacian operators (angular and radial)
- Hamiltonian construction
- Eigenvalue/eigenvector computation
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional


class LatticeOperators:
    """
    Construct and analyze operators on the polar lattice.
    
    This class handles:
    - Graph connectivity (adjacency structure)
    - Discrete Laplacian operators
    - Hamiltonian construction and diagonalization
    - Eigenmode analysis
    """
    
    def __init__(self, lattice):
        """
        Initialize operators for a given lattice.
        
        Parameters
        ----------
        lattice : PolarLattice
            The polar lattice structure
        """
        self.lattice = lattice
        self.n_points = len(lattice.points)
        
        # Build index mappings
        self._build_index_maps()
        
        # Adjacency structures (built on demand)
        self._angular_adj = None
        self._radial_adj = None
        self._full_adj = None
        
        # Laplacian matrices (built on demand)
        self._laplacian_ang = None
        self._laplacian_rad = None
        self._laplacian_full = None
    
    def _build_index_maps(self):
        """Build mappings between (ℓ, j) and linear index."""
        self.index_map = {}  # (ℓ, j) -> linear index
        self.reverse_map = {}  # linear index -> (ℓ, j)
        
        idx = 0
        for p in self.lattice.points:
            ℓ, j = p['ℓ'], p['j']
            self.index_map[(ℓ, j)] = idx
            self.reverse_map[idx] = (ℓ, j)
            idx += 1
    
    def get_index(self, ℓ: int, j: int) -> int:
        """Get linear index for lattice site (ℓ, j)."""
        return self.index_map.get((ℓ, j), -1)
    
    def get_angular_neighbors(self, ℓ: int, j: int) -> List[Tuple[int, int]]:
        """
        Get angular neighbors on the same ring.
        
        Parameters
        ----------
        ℓ : int
            Ring number
        j : int
            Site index on ring
        
        Returns
        -------
        list of tuples
            List of (ℓ, j) pairs for angular neighbors
        """
        N_ℓ = 2 * (2 * ℓ + 1)  # Points on this ring
        
        # Periodic boundary conditions
        j_prev = (j - 1) % N_ℓ
        j_next = (j + 1) % N_ℓ
        
        return [(ℓ, j_prev), (ℓ, j_next)]
    
    def get_radial_neighbors(self, ℓ: int, j: int, 
                           method: str = 'angular_matching') -> List[Tuple[int, int]]:
        """
        Get radial neighbors on adjacent rings.
        
        Parameters
        ----------
        ℓ : int
            Ring number
        j : int
            Site index on ring
        method : str, default='angular_matching'
            Method for finding radial neighbors:
            - 'angular_matching': match by angular position
            - 'euclidean': closest by Euclidean distance
        
        Returns
        -------
        list of tuples
            List of (ℓ, j) pairs for radial neighbors
        """
        neighbors = []
        
        # Current point's angle
        N_ℓ = 2 * (2 * ℓ + 1)
        θ = 2 * np.pi * j / N_ℓ
        
        # Check inner ring (ℓ - 1)
        if ℓ > 0:
            neighbors.extend(self._find_neighbors_on_ring(ℓ - 1, θ, method))
        
        # Check outer ring (ℓ + 1)
        if ℓ < self.lattice.ℓ_max:
            neighbors.extend(self._find_neighbors_on_ring(ℓ + 1, θ, method))
        
        return neighbors
    
    def _find_neighbors_on_ring(self, ℓ_target: int, θ: float, 
                                method: str) -> List[Tuple[int, int]]:
        """Find nearest neighbors on a target ring."""
        if method == 'angular_matching':
            # Find the closest point(s) by angle
            N_target = 2 * (2 * ℓ_target + 1)
            
            # Map angle to continuous index
            j_float = θ / (2 * np.pi) * N_target
            
            # Get the two nearest integer indices
            j1 = int(np.floor(j_float)) % N_target
            j2 = int(np.ceil(j_float)) % N_target
            
            if j1 == j2:
                return [(ℓ_target, j1)]
            else:
                return [(ℓ_target, j1), (ℓ_target, j2)]
        
        elif method == 'euclidean':
            # Find closest by 2D Euclidean distance
            r_current = 1 + 2 * (ℓ_target if ℓ_target < self.lattice.ℓ_max else ℓ_target - 1)
            x_current = r_current * np.cos(θ)
            y_current = r_current * np.sin(θ)
            
            ring_points = self.lattice.get_ring(ℓ_target)
            distances = []
            
            for p in ring_points:
                dx = p['x_2d'] - x_current
                dy = p['y_2d'] - y_current
                dist = np.sqrt(dx**2 + dy**2)
                distances.append((dist, p['j']))
            
            # Return the closest 2 neighbors
            distances.sort()
            return [(ℓ_target, distances[0][1]), (ℓ_target, distances[1][1])]
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def build_angular_adjacency(self) -> sparse.csr_matrix:
        """
        Build angular adjacency matrix (same-ring connections).
        
        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse adjacency matrix for angular connections
        """
        if self._angular_adj is not None:
            return self._angular_adj
        
        row_indices = []
        col_indices = []
        
        for idx in range(self.n_points):
            ℓ, j = self.reverse_map[idx]
            neighbors = self.get_angular_neighbors(ℓ, j)
            
            for ℓ_n, j_n in neighbors:
                idx_n = self.get_index(ℓ_n, j_n)
                if idx_n >= 0:
                    row_indices.append(idx)
                    col_indices.append(idx_n)
        
        data = np.ones(len(row_indices))
        self._angular_adj = sparse.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(self.n_points, self.n_points)
        )
        
        return self._angular_adj
    
    def build_radial_adjacency(self, method: str = 'angular_matching') -> sparse.csr_matrix:
        """
        Build radial adjacency matrix (between-ring connections).
        
        Parameters
        ----------
        method : str
            Method for finding radial neighbors
        
        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse adjacency matrix for radial connections
        """
        row_indices = []
        col_indices = []
        
        for idx in range(self.n_points):
            ℓ, j = self.reverse_map[idx]
            neighbors = self.get_radial_neighbors(ℓ, j, method)
            
            for ℓ_n, j_n in neighbors:
                idx_n = self.get_index(ℓ_n, j_n)
                if idx_n >= 0:
                    row_indices.append(idx)
                    col_indices.append(idx_n)
        
        data = np.ones(len(row_indices))
        self._radial_adj = sparse.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(self.n_points, self.n_points)
        )
        
        return self._radial_adj
    
    def build_full_adjacency(self, radial_method: str = 'angular_matching') -> sparse.csr_matrix:
        """
        Build full adjacency matrix (angular + radial).
        
        Returns
        -------
        scipy.sparse.csr_matrix
            Full sparse adjacency matrix
        """
        A_ang = self.build_angular_adjacency()
        A_rad = self.build_radial_adjacency(radial_method)
        
        self._full_adj = A_ang + A_rad
        
        return self._full_adj
    
    def build_angular_laplacian(self) -> sparse.csr_matrix:
        """
        Build angular Laplacian operator (acts within each ring).
        
        The discrete Laplacian is: Δ ψ(j) = ψ(j+1) + ψ(j-1) - 2ψ(j)
        
        Returns
        -------
        scipy.sparse.csr_matrix
            Angular Laplacian matrix
        """
        if self._laplacian_ang is not None:
            return self._laplacian_ang
        
        A = self.build_angular_adjacency()
        
        # Degree matrix (diagonal with node degrees)
        degrees = np.array(A.sum(axis=1)).flatten()
        D = sparse.diags(degrees)
        
        # Laplacian: Δ = A - D
        # (Note: This is the unnormalized graph Laplacian)
        self._laplacian_ang = A - D
        
        return self._laplacian_ang
    
    def build_radial_laplacian(self, method: str = 'angular_matching',
                              weight: float = 1.0) -> sparse.csr_matrix:
        """
        Build radial Laplacian operator (acts between rings).
        
        Parameters
        ----------
        method : str
            Method for finding radial neighbors
        weight : float
            Relative weight for radial connections
        
        Returns
        -------
        scipy.sparse.csr_matrix
            Radial Laplacian matrix
        """
        A = self.build_radial_adjacency(method)
        
        # Degree matrix
        degrees = np.array(A.sum(axis=1)).flatten()
        D = sparse.diags(degrees)
        
        # Laplacian with weight
        self._laplacian_rad = weight * (A - D)
        
        return self._laplacian_rad
    
    def build_full_laplacian(self, angular_weight: float = 1.0,
                           radial_weight: float = 1.0,
                           radial_method: str = 'angular_matching') -> sparse.csr_matrix:
        """
        Build full Laplacian (angular + radial).
        
        Parameters
        ----------
        angular_weight : float
            Weight for angular part
        radial_weight : float
            Weight for radial part
        radial_method : str
            Method for radial connections
        
        Returns
        -------
        scipy.sparse.csr_matrix
            Full Laplacian matrix
        """
        Δ_ang = self.build_angular_laplacian()
        Δ_rad = self.build_radial_laplacian(radial_method, weight=1.0)
        
        self._laplacian_full = angular_weight * Δ_ang + radial_weight * Δ_rad
        
        return self._laplacian_full
    
    def get_ring_hamiltonian(self, ℓ: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get angular-only Hamiltonian for a single ring.
        
        H_ang(ℓ) = -Δ_ang acting only on ring ℓ
        
        Parameters
        ----------
        ℓ : int
            Ring number
        
        Returns
        -------
        H : np.ndarray
            Hamiltonian matrix (dense) for ring ℓ
        indices : np.ndarray
            Linear indices of points on this ring
        """
        # Get all points on this ring
        ring_points = self.lattice.get_ring(ℓ)
        N = len(ring_points)
        
        # Get their linear indices
        indices = np.array([self.get_index(ℓ, p['j']) for p in ring_points])
        
        # Extract submatrix of angular Laplacian
        Δ_ang = self.build_angular_laplacian()
        H_submatrix = Δ_ang[np.ix_(indices, indices)].toarray()
        
        # Hamiltonian: H = -Δ
        H = -H_submatrix
        
        return H, indices
    
    def solve_ring_hamiltonian(self, ℓ: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve eigenvalue problem for single ring.
        
        Parameters
        ----------
        ℓ : int
            Ring number
        
        Returns
        -------
        eigenvalues : np.ndarray
            Sorted eigenvalues
        eigenvectors : np.ndarray
            Corresponding eigenvectors (columns)
        """
        H, _ = self.get_ring_hamiltonian(ℓ)
        
        # Solve eigenvalue problem
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        
        return eigenvalues, eigenvectors
    
    def build_hamiltonian(self, potential: Optional[np.ndarray] = None,
                         kinetic_weight: float = -0.5,
                         angular_weight: float = 1.0,
                         radial_weight: float = 1.0) -> sparse.csr_matrix:
        """
        Build full Hamiltonian H = T + V.
        
        T = kinetic_weight * (angular_weight * Δ_ang + radial_weight * Δ_rad)
        V = potential (diagonal)
        
        Parameters
        ----------
        potential : np.ndarray, optional
            Potential energy at each site (length n_points)
        kinetic_weight : float
            Overall kinetic energy coefficient (typically -1/2)
        angular_weight : float
            Weight for angular kinetic term
        radial_weight : float
            Weight for radial kinetic term
        
        Returns
        -------
        scipy.sparse.csr_matrix
            Full Hamiltonian matrix
        """
        # Kinetic energy: -½ Δ
        Δ = self.build_full_laplacian(angular_weight, radial_weight)
        T = kinetic_weight * Δ
        
        # Add potential
        if potential is not None:
            V = sparse.diags(potential)
            H = T + V
        else:
            H = T
        
        return H
    
    def solve_hamiltonian(self, H: sparse.csr_matrix, 
                         n_eig: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for lowest eigenvalues and eigenvectors.
        
        Parameters
        ----------
        H : sparse matrix
            Hamiltonian matrix
        n_eig : int
            Number of eigenvalues to compute
        
        Returns
        -------
        eigenvalues : np.ndarray
            Sorted eigenvalues
        eigenvectors : np.ndarray
            Corresponding eigenvectors (columns)
        """
        n_eig = min(n_eig, self.n_points - 2)
        
        eigenvalues, eigenvectors = eigsh(H, k=n_eig, which='SA')
        
        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors
    
    def get_degree_distribution(self) -> Dict[str, np.ndarray]:
        """
        Compute degree distribution for angular and radial graphs.
        
        Returns
        -------
        dict
            Dictionary with 'angular', 'radial', and 'full' degree arrays
        """
        result = {}
        
        A_ang = self.build_angular_adjacency()
        result['angular'] = np.array(A_ang.sum(axis=1)).flatten()
        
        A_rad = self.build_radial_adjacency()
        result['radial'] = np.array(A_rad.sum(axis=1)).flatten()
        
        A_full = self.build_full_adjacency()
        result['full'] = np.array(A_full.sum(axis=1)).flatten()
        
        return result
    
    def __repr__(self):
        """String representation."""
        return f"LatticeOperators(n_points={self.n_points}, n_max={self.lattice.n_max})"
