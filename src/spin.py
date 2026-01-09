"""
Spin Operators and Multi-Particle Physics Module

This module implements spin operators and multi-particle physics for the discrete lattice model:
1. Spin-1/2 operators: S_z, S_±, S_x, S_y
2. Pauli exclusion principle and shell filling
3. Spin-orbit coupling H_SO = λ L·S
4. Total angular momentum J = L + S

Author: Quantum Lattice Project
Date: January 2026
Phase: 5 - Multi-Particle and Spin
"""

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

try:
    from .lattice import PolarLattice
    from .operators import LatticeOperators
    from .angular_momentum import AngularMomentumOperators
except ImportError:
    from lattice import PolarLattice
    from operators import LatticeOperators
    from angular_momentum import AngularMomentumOperators


class SpinOperators:
    """
    Spin-1/2 operators for the discrete lattice model.
    
    Uses the hemisphere structure:
    - North hemisphere (z > 0): spin up (m_s = +1/2)
    - South hemisphere (z < 0): spin down (m_s = -1/2)
    
    Implements:
    - S_z: diagonal operator with eigenvalues ±1/2
    - S_±: ladder operators that flip spin (swap hemispheres)
    - S_x, S_y: Cartesian spin components
    - Spin algebra verification: [S_i, S_j] = iε_{ijk} S_k
    
    Attributes:
        lattice: PolarLattice object
        operators: LatticeOperators object
        angular_momentum: AngularMomentumOperators object
        N_sites: Total number of lattice sites
        site_to_quantum: Mapping from site index to (ℓ, m_ℓ, m_s)
        hemisphere_pairs: Mapping between spin-up and spin-down partners
    """
    
    def __init__(self, lattice: PolarLattice, operators: LatticeOperators,
                 angular_momentum: Optional[AngularMomentumOperators] = None):
        """
        Initialize spin operators.
        
        Parameters:
            lattice: PolarLattice with quantum number structure
            operators: LatticeOperators for Hamiltonian construction
            angular_momentum: AngularMomentumOperators for L operators (optional, will create if not provided)
        """
        self.lattice = lattice
        self.operators = operators
        
        # Create angular momentum operators if not provided
        if angular_momentum is None:
            self.angular_momentum = AngularMomentumOperators(lattice)
        else:
            self.angular_momentum = angular_momentum
            
        self.N_sites = len(lattice.points)
        
        # Build quantum number mapping
        self._build_site_mapping()
        
        # Find hemisphere pairs (same ℓ, m_ℓ, opposite m_s)
        self._build_hemisphere_pairs()
    
    def _build_site_mapping(self):
        """
        Build mapping from site indices to quantum numbers.
        """
        self.site_to_quantum = {}
        for idx, point in enumerate(self.lattice.points):
            ell = point['ℓ']
            m_ell = point['m_ℓ']
            m_s = point['m_s']
            self.site_to_quantum[idx] = (ell, m_ell, m_s)
    
    def _build_hemisphere_pairs(self):
        """
        Find pairs of sites with same (ℓ, m_ℓ) but opposite m_s.
        
        Creates mapping: spin_up_site -> spin_down_site and vice versa.
        """
        self.hemisphere_pairs = {}
        
        # Group sites by (ℓ, m_ℓ)
        orbital_sites = {}
        for idx in range(self.N_sites):
            ell, m_ell, m_s = self.site_to_quantum[idx]
            key = (ell, m_ell)
            if key not in orbital_sites:
                orbital_sites[key] = {}
            orbital_sites[key][m_s] = idx
        
        # Create pairs
        for key, spin_dict in orbital_sites.items():
            if 0.5 in spin_dict and -0.5 in spin_dict:
                up_idx = spin_dict[0.5]
                down_idx = spin_dict[-0.5]
                self.hemisphere_pairs[up_idx] = down_idx
                self.hemisphere_pairs[down_idx] = up_idx
    
    def build_Sz(self) -> sparse.csr_matrix:
        """
        Build S_z operator (diagonal).
        
        S_z |ℓ, m_ℓ, m_s⟩ = m_s |ℓ, m_ℓ, m_s⟩
        
        Returns:
            Sparse diagonal matrix with eigenvalues m_s = ±1/2
        """
        diagonal = np.zeros(self.N_sites)
        
        for idx in range(self.N_sites):
            _, _, m_s = self.site_to_quantum[idx]
            diagonal[idx] = m_s
        
        return sparse.diags(diagonal, format='csr')
    
    def build_Splus(self) -> sparse.csr_matrix:
        """
        Build S_+ raising operator.
        
        S_+ |ℓ, m_ℓ, -1/2⟩ = |ℓ, m_ℓ, +1/2⟩
        S_+ |ℓ, m_ℓ, +1/2⟩ = 0
        
        Flips spin from down to up (south to north hemisphere).
        
        Returns:
            Sparse matrix implementing spin raising
        """
        row_indices = []
        col_indices = []
        data = []
        
        for idx in range(self.N_sites):
            _, _, m_s = self.site_to_quantum[idx]
            
            # Only acts on spin-down states
            if m_s < 0:
                # Find spin-up partner
                partner_idx = self.hemisphere_pairs.get(idx)
                if partner_idx is not None:
                    row_indices.append(partner_idx)  # Final state (up)
                    col_indices.append(idx)          # Initial state (down)
                    data.append(1.0)
        
        return sparse.csr_matrix((data, (row_indices, col_indices)),
                                shape=(self.N_sites, self.N_sites))
    
    def build_Sminus(self) -> sparse.csr_matrix:
        """
        Build S_- lowering operator.
        
        S_- |ℓ, m_ℓ, +1/2⟩ = |ℓ, m_ℓ, -1/2⟩
        S_- |ℓ, m_ℓ, -1/2⟩ = 0
        
        Flips spin from up to down (north to south hemisphere).
        
        Returns:
            Sparse matrix implementing spin lowering
        """
        row_indices = []
        col_indices = []
        data = []
        
        for idx in range(self.N_sites):
            _, _, m_s = self.site_to_quantum[idx]
            
            # Only acts on spin-up states
            if m_s > 0:
                # Find spin-down partner
                partner_idx = self.hemisphere_pairs.get(idx)
                if partner_idx is not None:
                    row_indices.append(partner_idx)  # Final state (down)
                    col_indices.append(idx)          # Initial state (up)
                    data.append(1.0)
        
        return sparse.csr_matrix((data, (row_indices, col_indices)),
                                shape=(self.N_sites, self.N_sites))
    
    def build_Sx(self) -> sparse.csr_matrix:
        """
        Build S_x operator.
        
        S_x = (S_+ + S_-) / 2
        
        Returns:
            Sparse matrix for x-component of spin
        """
        S_plus = self.build_Splus()
        S_minus = self.build_Sminus()
        return (S_plus + S_minus) / 2.0
    
    def build_Sy(self) -> sparse.csr_matrix:
        """
        Build S_y operator.
        
        S_y = (S_+ - S_-) / (2i)
        
        Returns:
            Sparse matrix for y-component of spin
        """
        S_plus = self.build_Splus()
        S_minus = self.build_Sminus()
        return (S_plus - S_minus) / (2.0j)
    
    def build_S_squared(self) -> sparse.csr_matrix:
        """
        Build S² operator (total spin squared).
        
        S² = S_x² + S_y² + S_z²
        
        For spin-1/2: S² = s(s+1) = (1/2)(3/2) = 3/4
        
        Returns:
            Sparse matrix (should be diagonal with eigenvalue 3/4)
        """
        S_x = self.build_Sx()
        S_y = self.build_Sy()
        S_z = self.build_Sz()
        
        return S_x @ S_x + S_y @ S_y + S_z @ S_z
    
    def test_spin_algebra(self) -> Dict[str, float]:
        """
        Test spin-1/2 commutation relations.
        
        Verifies:
        - [S_x, S_y] = i S_z
        - [S_y, S_z] = i S_x
        - [S_z, S_x] = i S_y
        - S² = 3/4 I (diagonal)
        
        Returns:
            Dictionary with deviation norms for each relation
        """
        S_x = self.build_Sx()
        S_y = self.build_Sy()
        S_z = self.build_Sz()
        S_sq = self.build_S_squared()
        
        # Commutators
        comm_xy = S_x @ S_y - S_y @ S_x
        comm_yz = S_y @ S_z - S_z @ S_y
        comm_zx = S_z @ S_x - S_x @ S_z
        
        # Expected values
        expected_xy = 1.0j * S_z
        expected_yz = 1.0j * S_x
        expected_zx = 1.0j * S_y
        
        # Compute deviations
        dev_xy = np.linalg.norm((comm_xy - expected_xy).toarray())
        dev_yz = np.linalg.norm((comm_yz - expected_yz).toarray())
        dev_zx = np.linalg.norm((comm_zx - expected_zx).toarray())
        
        # Check S² is diagonal with eigenvalue 3/4
        S_sq_dense = S_sq.toarray()
        off_diag_norm = np.linalg.norm(S_sq_dense - np.diag(np.diag(S_sq_dense)))
        eigenvals_deviation = np.linalg.norm(np.diag(S_sq_dense) - 0.75)
        
        # Check Hermiticity
        S_x_herm = np.linalg.norm((S_x - S_x.conjugate().transpose()).toarray())
        S_y_herm = np.linalg.norm((S_y - S_y.conjugate().transpose()).toarray())
        S_z_herm = np.linalg.norm((S_z - S_z.conjugate().transpose()).toarray())
        
        return {
            'commutator_xy': dev_xy,
            'commutator_yz': dev_yz,
            'commutator_zx': dev_zx,
            'S_squared_diagonal': off_diag_norm,
            'S_squared_eigenvalue': eigenvals_deviation,
            'Sx_hermitian': S_x_herm,
            'Sy_hermitian': S_y_herm,
            'Sz_hermitian': S_z_herm
        }
    
    def build_spin_orbit_coupling(self, lambda_so: float = 0.01) -> sparse.csr_matrix:
        """
        Build spin-orbit coupling Hamiltonian.
        
        H_SO = λ L·S = λ (L_x S_x + L_y S_y + L_z S_z)
        
        This couples orbital and spin angular momentum, leading to
        fine structure splitting.
        
        Parameters:
            lambda_so: Spin-orbit coupling strength
        
        Returns:
            Sparse matrix for H_SO
        """
        # Get angular momentum operators
        L_x = self.angular_momentum.build_Lx()
        L_y = self.angular_momentum.build_Ly()
        L_z = self.angular_momentum.build_Lz()
        
        # Get spin operators
        S_x = self.build_Sx()
        S_y = self.build_Sy()
        S_z = self.build_Sz()
        
        # Compute L·S
        L_dot_S = L_x @ S_x + L_y @ S_y + L_z @ S_z
        
        return lambda_so * L_dot_S
    
    def build_total_angular_momentum(self) -> Tuple[sparse.csr_matrix, 
                                                     sparse.csr_matrix,
                                                     sparse.csr_matrix,
                                                     sparse.csr_matrix]:
        """
        Build total angular momentum operators J = L + S.
        
        Returns:
            Tuple of (J_x, J_y, J_z, J²) operators
        """
        # Orbital angular momentum
        L_x = self.angular_momentum.build_Lx()
        L_y = self.angular_momentum.build_Ly()
        L_z = self.angular_momentum.build_Lz()
        L_sq = self.angular_momentum.build_L_squared()
        
        # Spin angular momentum
        S_x = self.build_Sx()
        S_y = self.build_Sy()
        S_z = self.build_Sz()
        S_sq = self.build_S_squared()
        
        # Total angular momentum
        J_x = L_x + S_x
        J_y = L_y + S_y
        J_z = L_z + S_z
        
        # J² = L² + S² + 2L·S
        L_dot_S = L_x @ S_x + L_y @ S_y + L_z @ S_z
        J_sq = L_sq + S_sq + 2.0 * L_dot_S
        
        return J_x, J_y, J_z, J_sq


class ShellFilling:
    """
    Multi-particle physics with Pauli exclusion principle.
    
    Implements sequential filling of single-particle states to study
    shell structure and electron configurations.
    
    Attributes:
        lattice: PolarLattice object
        single_particle_energies: Eigenvalues of single-particle Hamiltonian
        single_particle_states: Eigenvectors of single-particle Hamiltonian
        occupations: List of occupied state indices
    """
    
    def __init__(self, lattice: PolarLattice, 
                 energies: np.ndarray, 
                 states: np.ndarray):
        """
        Initialize shell filling calculator.
        
        Parameters:
            lattice: PolarLattice object
            energies: Single-particle eigenvalues (sorted)
            states: Single-particle eigenvectors (columns)
        """
        self.lattice = lattice
        self.single_particle_energies = energies
        self.single_particle_states = states
        self.N_states = len(energies)
        self.occupations = []
    
    def fill_electrons(self, N_electrons: int) -> Dict:
        """
        Fill lowest N_electrons states according to Pauli exclusion.
        
        Parameters:
            N_electrons: Number of electrons to add
        
        Returns:
            Dictionary with:
                - 'occupations': List of occupied state indices
                - 'total_energy': Sum of single-particle energies
                - 'configuration': Shell configuration description
                - 'shell_closures': List of shell closure points
        """
        if N_electrons > self.N_states:
            raise ValueError(f"Cannot fit {N_electrons} electrons in {self.N_states} states")
        
        # Fill lowest states
        self.occupations = list(range(N_electrons))
        total_energy = np.sum(self.single_particle_energies[:N_electrons])
        
        # Identify shell closures (magic numbers)
        # For hydrogen-like: 2, 8, 18, 32, 50, 72, ...
        magic_numbers = [2 * n**2 for n in range(1, 10)]
        shell_closures = [n for n in magic_numbers if n <= N_electrons]
        
        # Check if current configuration is closed shell
        is_closed = N_electrons in magic_numbers
        
        # Build configuration description
        config = self._describe_configuration(N_electrons)
        
        return {
            'occupations': self.occupations,
            'total_energy': total_energy,
            'N_electrons': N_electrons,
            'configuration': config,
            'shell_closures': shell_closures,
            'is_closed_shell': is_closed,
            'highest_occupied_energy': self.single_particle_energies[N_electrons-1],
            'lowest_unoccupied_energy': self.single_particle_energies[N_electrons] if N_electrons < self.N_states else np.inf
        }
    
    def _describe_configuration(self, N_electrons: int) -> str:
        """
        Generate electron configuration description.
        
        Parameters:
            N_electrons: Number of electrons
        
        Returns:
            String like "1s² 2s² 2p⁶ 3s²" (approximate, based on n values)
        """
        # Simple approximation: assign states to shells by energy
        # Real configuration would need quantum number analysis
        
        shells = []
        n = 1
        remaining = N_electrons
        
        while remaining > 0 and n < 10:
            shell_capacity = 2 * n**2
            electrons_in_shell = min(remaining, shell_capacity)
            shells.append(f"n={n}: {electrons_in_shell}/{shell_capacity}")
            remaining -= electrons_in_shell
            n += 1
        
        return ", ".join(shells)
    
    def compute_ionization_energies(self, max_electrons: int = None) -> np.ndarray:
        """
        Compute ionization energy for removing electrons sequentially.
        
        I_k = E(N-1) - E(N) for removing k-th electron
        
        Parameters:
            max_electrons: Maximum number of electrons to consider
        
        Returns:
            Array of ionization energies
        """
        if max_electrons is None:
            max_electrons = min(50, self.N_states)
        else:
            max_electrons = min(max_electrons, self.N_states)
        
        ionization_energies = []
        
        for N in range(1, max_electrons + 1):
            if N > self.N_states:
                break
            E_N = np.sum(self.single_particle_energies[:N])
            E_N_minus_1 = np.sum(self.single_particle_energies[:N-1]) if N > 1 else 0.0
            
            # Ionization energy = energy to remove one electron
            I = E_N_minus_1 - E_N
            ionization_energies.append(-I)  # Make positive (binding energy)
        
        return np.array(ionization_energies)
    
    def identify_shell_gaps(self, max_electrons: int = None) -> List[int]:
        """
        Identify shell closures by finding large gaps in energy spectrum.
        
        Parameters:
            max_electrons: Maximum number to check
        
        Returns:
            List of electron numbers where shells close
        """
        if max_electrons is None:
            max_electrons = min(50, self.N_states - 1)
        
        # Compute energy gaps between consecutive levels
        gaps = np.diff(self.single_particle_energies[:max_electrons])
        
        # Find large gaps (threshold = mean + 2*std)
        threshold = np.mean(gaps) + 2 * np.std(gaps)
        
        shell_closures = []
        for i, gap in enumerate(gaps):
            if gap > threshold:
                shell_closures.append(i + 1)  # Number of electrons at closure
        
        return shell_closures


def visualize_spin_operators(spin_ops: SpinOperators, 
                             figsize=(16, 5),
                             save_path: Optional[str] = None):
    """
    Visualize spin operator matrices.
    
    Parameters:
        spin_ops: SpinOperators object
        figsize: Figure size
        save_path: If provided, save figure to this path
    """
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    operators = [
        ('S_z', spin_ops.build_Sz()),
        ('S_+', spin_ops.build_Splus()),
        ('S_x', spin_ops.build_Sx()),
        ('S_y', spin_ops.build_Sy())
    ]
    
    for ax, (name, op) in zip(axes, operators):
        # Plot magnitude of matrix elements
        op_dense = op.toarray()
        im = ax.imshow(np.abs(op_dense), cmap='Blues', aspect='auto')
        ax.set_title(f'{name} Operator\n({op.nnz} non-zero)', fontsize=12)
        ax.set_xlabel('State Index', fontsize=10)
        ax.set_ylabel('State Index', fontsize=10)
        plt.colorbar(im, ax=ax, label='|Matrix Element|')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved spin operators visualization to {save_path}")
    
    return fig, axes


def visualize_shell_filling(shell_filling: ShellFilling,
                           max_electrons: int = 50,
                           figsize=(14, 5),
                           save_path: Optional[str] = None):
    """
    Visualize shell filling and energy gaps.
    
    Parameters:
        shell_filling: ShellFilling object
        max_electrons: Maximum number of electrons to plot
        figsize: Figure size
        save_path: If provided, save figure to this path
    """
    max_electrons = min(max_electrons, shell_filling.N_states)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Panel 1: Single-particle energy levels
    N_plot = min(max_electrons + 10, shell_filling.N_states)
    energies = shell_filling.single_particle_energies[:N_plot]
    
    for i, E in enumerate(energies):
        color = 'blue' if i < max_electrons else 'gray'
        alpha = 0.8 if i < max_electrons else 0.3
        ax1.hlines(E, i-0.4, i+0.4, colors=color, alpha=alpha, linewidth=2)
    
    # Mark shell closures
    magic_numbers = [2 * n**2 for n in range(1, 10) if 2 * n**2 <= N_plot]
    for n in magic_numbers:
        if n <= N_plot:
            ax1.axvline(n - 0.5, color='red', linestyle='--', alpha=0.5)
            ax1.text(n - 0.5, ax1.get_ylim()[1], f'n={int(np.sqrt(n/2))}', 
                    ha='center', va='bottom', fontsize=9, color='red')
    
    ax1.set_xlabel('State Index', fontsize=12)
    ax1.set_ylabel('Energy (a.u.)', fontsize=12)
    ax1.set_title(f'Single-Particle Energy Levels\n(First {max_electrons} occupied)', fontsize=13)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Panel 2: Ionization energies
    ionization = shell_filling.compute_ionization_energies(max_electrons)
    
    ax2.plot(range(1, len(ionization) + 1), ionization, 'o-', markersize=4, linewidth=1)
    
    # Highlight shell closures
    for n in magic_numbers:
        if n <= len(ionization):
            ax2.axvline(n, color='red', linestyle='--', alpha=0.5)
            ax2.plot(n, ionization[n-1], 'ro', markersize=8)
    
    ax2.set_xlabel('Number of Electrons', fontsize=12)
    ax2.set_ylabel('Ionization Energy (a.u.)', fontsize=12)
    ax2.set_title('Ionization Energies\n(Peaks at shell closures)', fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved shell filling visualization to {save_path}")
    
    return fig, (ax1, ax2)
