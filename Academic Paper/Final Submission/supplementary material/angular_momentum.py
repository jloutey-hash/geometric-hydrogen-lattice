"""
Angular momentum operators for the polar lattice.

This module implements:
- L_z operator (diagonal with m_ℓ eigenvalues)
- L_± ladder operators (raise/lower m_ℓ)
- L_x, L_y operators (from L_±)
- L² total angular momentum squared
- Commutation relations testing
"""

import numpy as np
from scipy import sparse
from typing import Tuple, Dict
import matplotlib.pyplot as plt


class AngularMomentumOperators:
    """
    Construct and analyze angular momentum operators on the polar lattice.
    
    The lattice naturally encodes angular momentum quantum numbers:
    - Each point has (ℓ, m_ℓ, m_s)
    - L_z is diagonal with eigenvalues m_ℓ
    - L_± shifts m_ℓ by ±1 (within same ℓ shell)
    - L² is diagonal with eigenvalues ℓ(ℓ+1)
    """
    
    def __init__(self, lattice):
        """
        Initialize angular momentum operators for a lattice.
        
        Parameters
        ----------
        lattice : PolarLattice
            The polar lattice structure
        """
        self.lattice = lattice
        self.n_points = len(lattice.points)
        
        # Build index mappings
        self._build_index_maps()
        
        # Operators (built on demand)
        self._L_z = None
        self._L_plus = None
        self._L_minus = None
        self._L_x = None
        self._L_y = None
        self._L_squared = None
    
    def _build_index_maps(self):
        """Build mappings for quantum numbers."""
        self.index_map = {}  # (ℓ, m_ℓ, m_s) -> linear index
        self.reverse_map = {}  # linear index -> (ℓ, m_ℓ, m_s)
        
        for idx, p in enumerate(self.lattice.points):
            key = (p['ℓ'], p['m_ℓ'], p['m_s'])
            self.index_map[key] = idx
            self.reverse_map[idx] = key
    
    def get_index(self, ℓ: int, m_ℓ: float, m_s: float) -> int:
        """Get linear index for quantum numbers (ℓ, m_ℓ, m_s)."""
        return self.index_map.get((ℓ, m_ℓ, m_s), -1)
    
    def build_Lz(self) -> sparse.csr_matrix:
        """
        Build L_z operator (diagonal).
        
        L_z |ℓ, m_ℓ, m_s⟩ = m_ℓ |ℓ, m_ℓ, m_s⟩
        
        Returns
        -------
        scipy.sparse.csr_matrix
            L_z operator (diagonal matrix)
        """
        if self._L_z is not None:
            return self._L_z
        
        # Diagonal matrix with m_ℓ values
        diagonal = np.array([p['m_ℓ'] for p in self.lattice.points])
        self._L_z = sparse.diags(diagonal, format='csr')
        
        return self._L_z
    
    def build_Lplus(self) -> sparse.csr_matrix:
        """
        Build L_+ raising operator.
        
        L_+ |ℓ, m_ℓ, m_s⟩ = √[ℓ(ℓ+1) - m_ℓ(m_ℓ+1)] |ℓ, m_ℓ+1, m_s⟩
        
        Returns
        -------
        scipy.sparse.csr_matrix
            L_+ operator (sparse matrix)
        """
        if self._L_plus is not None:
            return self._L_plus
        
        row_indices = []
        col_indices = []
        data = []
        
        for idx in range(self.n_points):
            ℓ, m_ℓ, m_s = self.reverse_map[idx]
            
            # Can only raise if m_ℓ < ℓ
            if m_ℓ < ℓ:
                m_ℓ_new = m_ℓ + 1
                idx_new = self.get_index(ℓ, m_ℓ_new, m_s)
                
                if idx_new >= 0:
                    # Matrix element: ⟨ℓ, m_ℓ+1| L_+ |ℓ, m_ℓ⟩
                    matrix_element = np.sqrt(ℓ*(ℓ+1) - m_ℓ*(m_ℓ+1))
                    
                    row_indices.append(idx_new)
                    col_indices.append(idx)
                    data.append(matrix_element)
        
        self._L_plus = sparse.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(self.n_points, self.n_points)
        )
        
        return self._L_plus
    
    def build_Lminus(self) -> sparse.csr_matrix:
        """
        Build L_- lowering operator.
        
        L_- |ℓ, m_ℓ, m_s⟩ = √[ℓ(ℓ+1) - m_ℓ(m_ℓ-1)] |ℓ, m_ℓ-1, m_s⟩
        
        Returns
        -------
        scipy.sparse.csr_matrix
            L_- operator (sparse matrix)
        """
        if self._L_minus is not None:
            return self._L_minus
        
        row_indices = []
        col_indices = []
        data = []
        
        for idx in range(self.n_points):
            ℓ, m_ℓ, m_s = self.reverse_map[idx]
            
            # Can only lower if m_ℓ > -ℓ
            if m_ℓ > -ℓ:
                m_ℓ_new = m_ℓ - 1
                idx_new = self.get_index(ℓ, m_ℓ_new, m_s)
                
                if idx_new >= 0:
                    # Matrix element: ⟨ℓ, m_ℓ-1| L_- |ℓ, m_ℓ⟩
                    matrix_element = np.sqrt(ℓ*(ℓ+1) - m_ℓ*(m_ℓ-1))
                    
                    row_indices.append(idx_new)
                    col_indices.append(idx)
                    data.append(matrix_element)
        
        self._L_minus = sparse.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(self.n_points, self.n_points)
        )
        
        return self._L_minus
    
    def build_Lx(self) -> sparse.csr_matrix:
        """
        Build L_x operator from ladder operators.
        
        L_x = (L_+ + L_-) / 2
        
        Returns
        -------
        scipy.sparse.csr_matrix
            L_x operator
        """
        if self._L_x is not None:
            return self._L_x
        
        L_plus = self.build_Lplus()
        L_minus = self.build_Lminus()
        
        self._L_x = 0.5 * (L_plus + L_minus)
        
        return self._L_x
    
    def build_Ly(self) -> sparse.csr_matrix:
        """
        Build L_y operator from ladder operators.
        
        L_y = (L_+ - L_-) / (2i)
        
        Returns
        -------
        scipy.sparse.csr_matrix
            L_y operator
        """
        if self._L_y is not None:
            return self._L_y
        
        L_plus = self.build_Lplus()
        L_minus = self.build_Lminus()
        
        self._L_y = -0.5j * (L_plus - L_minus)
        
        return self._L_y
    
    def build_L_squared(self) -> sparse.csr_matrix:
        """
        Build L² = L_x² + L_y² + L_z² operator.
        
        For states |ℓ, m_ℓ⟩: L² |ℓ, m_ℓ⟩ = ℓ(ℓ+1) |ℓ, m_ℓ⟩
        
        Returns
        -------
        scipy.sparse.csr_matrix
            L² operator (should be diagonal with eigenvalues ℓ(ℓ+1))
        """
        if self._L_squared is not None:
            return self._L_squared
        
        L_x = self.build_Lx()
        L_y = self.build_Ly()
        L_z = self.build_Lz()
        
        # L² = L_x² + L_y² + L_z²
        self._L_squared = L_x @ L_x + L_y @ L_y + L_z @ L_z
        
        return self._L_squared
    
    def compute_commutator(self, A: sparse.csr_matrix, 
                          B: sparse.csr_matrix) -> sparse.csr_matrix:
        """
        Compute commutator [A, B] = AB - BA.
        
        Parameters
        ----------
        A, B : sparse matrices
            Operators to compute commutator of
        
        Returns
        -------
        scipy.sparse.csr_matrix
            Commutator [A, B]
        """
        return A @ B - B @ A
    
    def test_commutation_relations(self) -> Dict[str, float]:
        """
        Test angular momentum commutation relations.
        
        Should satisfy:
        - [L_x, L_y] = i L_z
        - [L_y, L_z] = i L_x
        - [L_z, L_x] = i L_y
        
        Returns
        -------
        dict
            Dictionary of commutator norm deviations
        """
        L_x = self.build_Lx()
        L_y = self.build_Ly()
        L_z = self.build_Lz()
        
        results = {}
        
        # [L_x, L_y] - i L_z
        comm_xy = self.compute_commutator(L_x, L_y)
        expected_xy = 1j * L_z
        deviation_xy = sparse.linalg.norm(comm_xy - expected_xy)
        results['[L_x, L_y] - i*L_z'] = deviation_xy
        
        # [L_y, L_z] - i L_x
        comm_yz = self.compute_commutator(L_y, L_z)
        expected_yz = 1j * L_x
        deviation_yz = sparse.linalg.norm(comm_yz - expected_yz)
        results['[L_y, L_z] - i*L_x'] = deviation_yz
        
        # [L_z, L_x] - i L_y
        comm_zx = self.compute_commutator(L_z, L_x)
        expected_zx = 1j * L_y
        deviation_zx = sparse.linalg.norm(comm_zx - expected_zx)
        results['[L_z, L_x] - i*L_y'] = deviation_zx
        
        return results
    
    def test_ladder_operators(self, ℓ: int = 1) -> Dict[str, bool]:
        """
        Test ladder operator properties on a specific ℓ shell.
        
        Parameters
        ----------
        ℓ : int
            Angular momentum quantum number to test
        
        Returns
        -------
        dict
            Dictionary of test results
        """
        L_plus = self.build_Lplus()
        L_minus = self.build_Lminus()
        L_z = self.build_Lz()
        
        results = {}
        
        # Test that L_± changes m_ℓ by ±1
        for m_ℓ in range(-ℓ, ℓ+1):
            for m_s in [0.5, -0.5]:
                idx = self.get_index(ℓ, m_ℓ, m_s)
                if idx < 0:
                    continue
                
                # Create state vector
                state = np.zeros(self.n_points)
                state[idx] = 1.0
                
                # Apply L_+
                if m_ℓ < ℓ:
                    new_state = L_plus @ state
                    idx_new = self.get_index(ℓ, m_ℓ + 1, m_s)
                    if idx_new >= 0:
                        results[f'L_+ raises m_ℓ={m_ℓ}'] = abs(new_state[idx_new]) > 1e-10
                
                # Apply L_-
                if m_ℓ > -ℓ:
                    new_state = L_minus @ state
                    idx_new = self.get_index(ℓ, m_ℓ - 1, m_s)
                    if idx_new >= 0:
                        results[f'L_- lowers m_ℓ={m_ℓ}'] = abs(new_state[idx_new]) > 1e-10
        
        return results
    
    def get_L_squared_eigenvalues(self) -> np.ndarray:
        """
        Get eigenvalues of L² (should be ℓ(ℓ+1) for each ℓ shell).
        
        Returns
        -------
        np.ndarray
            Diagonal of L² (eigenvalues)
        """
        L_sq = self.build_L_squared()
        
        # L² should be diagonal (or nearly diagonal)
        # Extract diagonal
        diagonal = L_sq.diagonal()
        
        return np.real(diagonal)
    
    def verify_L_squared(self, tolerance: float = 1e-10) -> Dict[str, any]:
        """
        Verify that L² has correct eigenvalues ℓ(ℓ+1).
        
        Parameters
        ----------
        tolerance : float
            Tolerance for eigenvalue comparison
        
        Returns
        -------
        dict
            Verification results
        """
        L_sq_diag = self.get_L_squared_eigenvalues()
        
        results = {
            'is_diagonal': False,
            'correct_eigenvalues': False,
            'deviations': []
        }
        
        # Check if L² is diagonal
        L_sq = self.build_L_squared()
        L_sq_dense = L_sq.toarray()
        off_diag_norm = np.linalg.norm(L_sq_dense - np.diag(np.diagonal(L_sq_dense)))
        results['is_diagonal'] = off_diag_norm < tolerance
        results['off_diagonal_norm'] = off_diag_norm
        
        # Check eigenvalues
        correct_count = 0
        for idx in range(self.n_points):
            ℓ, m_ℓ, m_s = self.reverse_map[idx]
            expected = ℓ * (ℓ + 1)
            actual = L_sq_diag[idx]
            deviation = abs(actual - expected)
            
            if deviation < tolerance:
                correct_count += 1
            else:
                results['deviations'].append({
                    'idx': idx,
                    'ℓ': ℓ,
                    'm_ℓ': m_ℓ,
                    'm_s': m_s,
                    'expected': expected,
                    'actual': actual,
                    'deviation': deviation
                })
        
        results['correct_eigenvalues'] = (correct_count == self.n_points)
        results['correct_fraction'] = correct_count / self.n_points
        
        return results
    
    def plot_operator_matrix(self, operator_name: str, 
                            figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Visualize operator as matrix.
        
        Parameters
        ----------
        operator_name : str
            Name of operator: 'Lz', 'Lx', 'Ly', 'L+', 'L-', 'L²'
        figsize : tuple
            Figure size
        
        Returns
        -------
        matplotlib.figure.Figure
            Figure object
        """
        # Get operator
        if operator_name == 'Lz':
            op = self.build_Lz()
        elif operator_name == 'Lx':
            op = self.build_Lx()
        elif operator_name == 'Ly':
            op = self.build_Ly()
        elif operator_name == 'L+':
            op = self.build_Lplus()
        elif operator_name == 'L-':
            op = self.build_Lminus()
        elif operator_name == 'L²':
            op = self.build_L_squared()
        else:
            raise ValueError(f"Unknown operator: {operator_name}")
        
        # Convert to dense for visualization
        op_dense = op.toarray()
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Real part
        im1 = ax1.imshow(np.real(op_dense), cmap='RdBu', aspect='auto',
                        vmin=-np.abs(op_dense).max(), vmax=np.abs(op_dense).max())
        ax1.set_title(f'{operator_name} (Real part)', fontsize=12)
        ax1.set_xlabel('Column index', fontsize=10)
        ax1.set_ylabel('Row index', fontsize=10)
        plt.colorbar(im1, ax=ax1)
        
        # Imaginary part
        im2 = ax2.imshow(np.imag(op_dense), cmap='RdBu', aspect='auto',
                        vmin=-np.abs(op_dense).max(), vmax=np.abs(op_dense).max())
        ax2.set_title(f'{operator_name} (Imaginary part)', fontsize=12)
        ax2.set_xlabel('Column index', fontsize=10)
        ax2.set_ylabel('Row index', fontsize=10)
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        
        return fig
    
    def __repr__(self):
        """String representation."""
        return f"AngularMomentumOperators(n_points={self.n_points}, n_max={self.lattice.n_max})"
