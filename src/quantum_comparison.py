"""
Quantum Comparison Module

This module compares the discrete lattice model with continuous quantum mechanics,
specifically focusing on:
1. Spherical harmonics Y_ℓ^m evaluation and overlap with lattice eigenmodes
2. Energy eigenvalue comparison with hydrogen atom
3. Dipole selection rules and transition matrix elements

Author: Quantum Lattice Project
Date: January 2026
Phase: 4 - Comparison with Quantum Mechanics
"""

import numpy as np
from scipy.special import sph_harm
from scipy.linalg import eigh
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

try:
    from .lattice import PolarLattice
    from .operators import LatticeOperators
except ImportError:
    from lattice import PolarLattice
    from operators import LatticeOperators


class QuantumComparison:
    """
    Compare discrete lattice eigenmodes with continuous quantum mechanical functions.
    
    This class provides tools to:
    - Sample spherical harmonics Y_ℓ^m at lattice points
    - Compute overlap integrals between discrete modes and continuous functions
    - Compare energy eigenvalues to hydrogen atom predictions
    - Test dipole selection rules for transitions
    
    Attributes:
        lattice: PolarLattice object defining the geometric structure
        operators: LatticeOperators object for Hamiltonians
        spherical_coords: Dictionary mapping site indices to (theta, phi) coordinates
        ylm_samples: Cache of evaluated spherical harmonics
    """
    
    def __init__(self, lattice: PolarLattice, operators: LatticeOperators):
        """
        Initialize quantum comparison tools.
        
        Parameters:
            lattice: PolarLattice with spherical coordinates defined
            operators: LatticeOperators for Hamiltonian and eigenmode access
        """
        self.lattice = lattice
        self.operators = operators
        self.spherical_coords = {}
        self.ylm_samples = {}
        
        # Build convenient data structures
        self._build_site_mapping()
        
        # Extract spherical coordinates from lattice
        self._compute_spherical_coordinates()
    
    def _build_site_mapping(self):
        """
        Build mapping from site indices to quantum numbers.
        """
        self.site_to_quantum = {}
        idx = 0
        for ℓ in range(self.lattice.ℓ_max + 1):
            N_ℓ = 2 * (2 * ℓ + 1)
            for j in range(N_ℓ):
                ℓ_val, m_ℓ, m_s = self.lattice.get_quantum_numbers(ℓ, j)
                self.site_to_quantum[idx] = (ℓ_val, m_ℓ, m_s)
                idx += 1
        
        self.N_total = idx
    
    def _compute_spherical_coordinates(self):
        """
        Extract (theta, phi) spherical coordinates for each lattice site.
        
        Uses the spherical coordinates already computed in lattice.points.
        Converts from Cartesian (x, y, z) to spherical angles (theta, phi).
        
        Stores results in self.spherical_coords as:
            site_index -> (theta, phi)
        """
        for idx, point in enumerate(self.lattice.points):
            x = point['x_3d']
            y = point['y_3d']
            z = point['z_3d']
            
            # Convert to spherical angles
            # theta: colatitude (0 at north pole, pi at south pole)
            # phi: azimuth (0 to 2pi)
            r = np.sqrt(x**2 + y**2 + z**2)
            
            if r < 1e-10:
                # Origin point (shouldn't happen on sphere, but handle gracefully)
                theta, phi = 0.0, 0.0
            else:
                theta = np.arccos(np.clip(z / r, -1.0, 1.0))
                phi = np.arctan2(y, x)
                if phi < 0:
                    phi += 2 * np.pi
            
            self.spherical_coords[idx] = (theta, phi)
    
    def sample_spherical_harmonic(self, ell: int, m: int, 
                                  ring_filter: Optional[int] = None) -> np.ndarray:
        """
        Evaluate spherical harmonic Y_ℓ^m at all lattice points.
        
        Computes Y_ℓ^m(theta, phi) using scipy.special.sph_harm and returns
        values as a vector in the lattice basis.
        
        Parameters:
            ell: Angular momentum quantum number (ℓ ≥ 0)
            m: Magnetic quantum number (-ℓ ≤ m ≤ ℓ)
            ring_filter: If provided, only evaluate on specified ring ℓ
        
        Returns:
            ylm_vector: Complex vector of length N_sites with Y_ℓ^m values
        
        Notes:
            - scipy.sph_harm uses convention: sph_harm(m, ell, phi, theta)
            - Result is normalized: ∫ |Y_ℓ^m|² dΩ = 1 on continuous sphere
            - Lattice discretization means ∑ |Y_ℓ^m(i)|² ≠ 1 in general
        """
        if abs(m) > ell:
            raise ValueError(f"Invalid m={m} for ell={ell}. Need |m| <= ell.")
        
        # Check cache
        cache_key = (ell, m, ring_filter)
        if cache_key in self.ylm_samples:
            return self.ylm_samples[cache_key]
        
        # Build vector
        N_sites = self.N_total
        ylm_vector = np.zeros(N_sites, dtype=complex)
        
        for site_idx in range(N_sites):
            # Check ring filter
            if ring_filter is not None:
                site_ell = self.site_to_quantum[site_idx][0]
                if site_ell != ring_filter:
                    continue
            
            theta, phi = self.spherical_coords[site_idx]
            
            # scipy.sph_harm(m, l, phi, theta)
            # Note: order is (m, l, phi, theta)!
            ylm_vector[site_idx] = sph_harm(m, ell, phi, theta)
        
        # Cache result
        self.ylm_samples[cache_key] = ylm_vector
        
        return ylm_vector
    
    def compute_overlap_matrix(self, eigenmodes: np.ndarray, 
                               ell_max: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Compute overlap integrals between eigenmodes and spherical harmonics.
        
        For each discrete eigenmode ψ_i and each Y_ℓ^m, compute:
            O_{i, (ℓ,m)} = |⟨ψ_i | Y_ℓ^m⟩|²
        
        where the inner product is:
            ⟨ψ | Y⟩ = ∑_j ψ_j^* Y(j)
        
        Parameters:
            eigenmodes: Array of shape (N_sites, N_modes) with eigenvectors as columns
            ell_max: Maximum ℓ to include (default: lattice n_max - 1)
        
        Returns:
            Dictionary with:
                'overlap_matrix': Shape (N_modes, N_ylm) with overlap magnitudes
                'ylm_labels': List of (ℓ, m) tuples for each column
                'mode_quantum': List of quantum numbers for each mode (if available)
        """
        N_sites, N_modes = eigenmodes.shape
        
        if ell_max is None:
            ell_max = self.lattice.n_max - 1
        
        # Build list of (ℓ, m) pairs
        ylm_labels = []
        for ell in range(ell_max + 1):
            for m in range(-ell, ell + 1):
                ylm_labels.append((ell, m))
        
        N_ylm = len(ylm_labels)
        overlap_matrix = np.zeros((N_modes, N_ylm))
        
        # Compute overlaps
        for j, (ell, m) in enumerate(ylm_labels):
            ylm_vec = self.sample_spherical_harmonic(ell, m)
            
            for i in range(N_modes):
                mode = eigenmodes[:, i]
                # Inner product: ⟨mode | ylm⟩ = ∑_k mode[k]^* ylm[k]
                overlap = np.vdot(mode, ylm_vec)
                overlap_matrix[i, j] = np.abs(overlap)**2
        
        return {
            'overlap_matrix': overlap_matrix,
            'ylm_labels': ylm_labels,
            'N_modes': N_modes,
            'N_ylm': N_ylm
        }
    
    def identify_quantum_numbers(self, eigenmodes: np.ndarray, 
                                 eigenvalues: np.ndarray,
                                 ell_max: Optional[int] = None) -> List[Dict]:
        """
        Identify (n, ℓ, m) quantum numbers for each eigenmode by maximum overlap.
        
        For each eigenmode, find the spherical harmonic Y_ℓ^m with highest overlap.
        This assigns provisional (ℓ, m) labels to each mode.
        
        Parameters:
            eigenmodes: Array of shape (N_sites, N_modes) with eigenvectors
            eigenvalues: Array of energies corresponding to eigenmodes
            ell_max: Maximum ℓ to search (default: n_max - 1)
        
        Returns:
            List of dictionaries with keys:
                'mode_index': Index in eigenmode array
                'energy': Eigenvalue
                'ell_best': Best-matching ℓ
                'm_best': Best-matching m
                'overlap_max': Maximum overlap value
                'purity': Ratio of max overlap to sum of overlaps (measure of mixing)
        """
        overlap_data = self.compute_overlap_matrix(eigenmodes, ell_max)
        overlap_matrix = overlap_data['overlap_matrix']
        ylm_labels = overlap_data['ylm_labels']
        
        quantum_ids = []
        
        for i in range(overlap_data['N_modes']):
            overlaps = overlap_matrix[i, :]
            j_max = np.argmax(overlaps)
            overlap_max = overlaps[j_max]
            overlap_sum = np.sum(overlaps)
            
            ell_best, m_best = ylm_labels[j_max]
            purity = overlap_max / overlap_sum if overlap_sum > 0 else 0.0
            
            quantum_ids.append({
                'mode_index': i,
                'energy': eigenvalues[i],
                'ell_best': ell_best,
                'm_best': m_best,
                'overlap_max': overlap_max,
                'purity': purity
            })
        
        return quantum_ids
    
    def compare_to_hydrogen(self, eigenvalues: np.ndarray, 
                           quantum_ids: List[Dict],
                           Z: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Compare lattice eigenvalues to hydrogen atom energy levels.
        
        Hydrogen energy levels: E_n = -Z² × 13.6 eV / n²
        Or in atomic units: E_n = -Z² / (2n²)
        
        Parameters:
            eigenvalues: Lattice eigenvalues (assumed in atomic units)
            quantum_ids: Quantum number identification from identify_quantum_numbers()
            Z: Nuclear charge (Z=1 for hydrogen)
        
        Returns:
            Dictionary with:
                'n_assignments': Principal quantum number n for each mode
                'hydrogen_energies': E_n = -Z²/(2n²) for each assigned n
                'lattice_energies': Original eigenvalues
                'energy_errors': |E_lattice - E_hydrogen|
                'relative_errors': Errors relative to |E_hydrogen|
        """
        N_modes = len(eigenvalues)
        
        # Try to assign n based on energy
        # For hydrogen: E_n = -Z²/(2n²)
        # Solve for n: n = sqrt(-Z²/(2E))
        
        n_assignments = np.zeros(N_modes, dtype=int)
        hydrogen_energies = np.zeros(N_modes)
        energy_errors = np.zeros(N_modes)
        relative_errors = np.zeros(N_modes)
        
        for i in range(N_modes):
            E_lattice = eigenvalues[i]
            ell_best = quantum_ids[i]['ell_best']
            
            # Estimate n from energy
            if E_lattice < 0:
                n_est = np.sqrt(-Z**2 / (2 * E_lattice))
                n_assigned = max(int(np.round(n_est)), ell_best + 1)
            else:
                # Positive energy: continuum state, assign based on ℓ
                n_assigned = ell_best + 1
            
            n_assignments[i] = n_assigned
            E_hydrogen = -Z**2 / (2 * n_assigned**2)
            hydrogen_energies[i] = E_hydrogen
            energy_errors[i] = abs(E_lattice - E_hydrogen)
            
            if abs(E_hydrogen) > 1e-10:
                relative_errors[i] = energy_errors[i] / abs(E_hydrogen)
            else:
                relative_errors[i] = energy_errors[i]
        
        return {
            'n_assignments': n_assignments,
            'hydrogen_energies': hydrogen_energies,
            'lattice_energies': eigenvalues,
            'energy_errors': energy_errors,
            'relative_errors': relative_errors,
            'quantum_ids': quantum_ids
        }
    
    def compute_dipole_matrix_elements(self, eigenmodes: np.ndarray,
                                      quantum_ids: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Compute dipole transition matrix elements ⟨f|r|i⟩.
        
        The position operator r has components (x, y, z) evaluated at lattice points.
        For selection rules, we compute:
            ⟨ψ_f | x | ψ_i⟩, ⟨ψ_f | y | ψ_i⟩, ⟨ψ_f | z | ψ_i⟩
        
        Selection rules predict:
            - Δℓ = ±1
            - Δm = 0 (for z), Δm = ±1 (for x, y)
        
        Parameters:
            eigenmodes: Array of shape (N_sites, N_modes) with eigenvectors
            quantum_ids: Quantum number assignments for each mode
        
        Returns:
            Dictionary with:
                'dipole_x': Matrix elements ⟨f|x|i⟩ (N_modes × N_modes)
                'dipole_y': Matrix elements ⟨f|y|i⟩
                'dipole_z': Matrix elements ⟨f|z|i⟩
                'transition_strength': |⟨f|r|i⟩|² = |⟨f|x|i⟩|² + |⟨f|y|i⟩|² + |⟨f|z|i⟩|²
                'delta_ell': Δℓ for each transition
                'delta_m': Δm for each transition
                'selection_allowed': Boolean mask for Δℓ=±1, Δm=0,±1
        """
        N_sites, N_modes = eigenmodes.shape
        
        # Build position operators (diagonal in site basis)
        x_vec = np.zeros(N_sites)
        y_vec = np.zeros(N_sites)
        z_vec = np.zeros(N_sites)
        
        for site_idx in range(N_sites):
            if site_idx in self.spherical_coords:
                coords_3d = self.lattice.points[site_idx]
                x_vec[site_idx] = coords_3d['x_3d']
                y_vec[site_idx] = coords_3d['y_3d']
                z_vec[site_idx] = coords_3d['z_3d']
        
        # Compute matrix elements
        dipole_x = np.zeros((N_modes, N_modes), dtype=complex)
        dipole_y = np.zeros((N_modes, N_modes), dtype=complex)
        dipole_z = np.zeros((N_modes, N_modes), dtype=complex)
        
        for i in range(N_modes):
            for f in range(N_modes):
                # ⟨f|x|i⟩ = ∑_k ψ_f[k]^* x[k] ψ_i[k]
                dipole_x[f, i] = np.vdot(eigenmodes[:, f], x_vec * eigenmodes[:, i])
                dipole_y[f, i] = np.vdot(eigenmodes[:, f], y_vec * eigenmodes[:, i])
                dipole_z[f, i] = np.vdot(eigenmodes[:, f], z_vec * eigenmodes[:, i])
        
        # Transition strength |⟨f|r|i⟩|²
        transition_strength = (np.abs(dipole_x)**2 + 
                              np.abs(dipole_y)**2 + 
                              np.abs(dipole_z)**2)
        
        # Compute Δℓ and Δm
        delta_ell = np.zeros((N_modes, N_modes), dtype=int)
        delta_m = np.zeros((N_modes, N_modes), dtype=int)
        
        for i in range(N_modes):
            for f in range(N_modes):
                delta_ell[f, i] = quantum_ids[f]['ell_best'] - quantum_ids[i]['ell_best']
                delta_m[f, i] = quantum_ids[f]['m_best'] - quantum_ids[i]['m_best']
        
        # Selection rules: Δℓ = ±1, Δm ∈ {-1, 0, 1}
        selection_allowed = (np.abs(delta_ell) == 1) & (np.abs(delta_m) <= 1)
        
        return {
            'dipole_x': dipole_x,
            'dipole_y': dipole_y,
            'dipole_z': dipole_z,
            'transition_strength': transition_strength,
            'delta_ell': delta_ell,
            'delta_m': delta_m,
            'selection_allowed': selection_allowed,
            'quantum_ids': quantum_ids
        }
    
    def test_selection_rules(self, dipole_data: Dict, 
                            threshold: float = 1e-3) -> Dict[str, float]:
        """
        Test adherence to dipole selection rules.
        
        Check what fraction of strong transitions obey Δℓ=±1, Δm=0,±1.
        
        Parameters:
            dipole_data: Output from compute_dipole_matrix_elements()
            threshold: Transitions with strength > threshold are considered "strong"
        
        Returns:
            Dictionary with:
                'fraction_obey_rules': Fraction of strong transitions obeying rules
                'fraction_violate_rules': Fraction of strong transitions violating rules
                'total_strong': Total number of strong transitions
                'max_violation_strength': Largest strength among violating transitions
        """
        transition_strength = dipole_data['transition_strength']
        selection_allowed = dipole_data['selection_allowed']
        
        # Find strong transitions
        strong_mask = transition_strength > threshold
        N_strong = np.sum(strong_mask)
        
        if N_strong == 0:
            return {
                'fraction_obey_rules': 0.0,
                'fraction_violate_rules': 0.0,
                'total_strong': 0,
                'max_violation_strength': 0.0
            }
        
        # Check which strong transitions obey rules
        obey_mask = strong_mask & selection_allowed
        violate_mask = strong_mask & ~selection_allowed
        
        N_obey = np.sum(obey_mask)
        N_violate = np.sum(violate_mask)
        
        fraction_obey = N_obey / N_strong
        fraction_violate = N_violate / N_strong
        
        # Find maximum violating strength
        if N_violate > 0:
            max_violation = np.max(transition_strength[violate_mask])
        else:
            max_violation = 0.0
        
        return {
            'fraction_obey_rules': fraction_obey,
            'fraction_violate_rules': fraction_violate,
            'total_strong': N_strong,
            'N_obey': N_obey,
            'N_violate': N_violate,
            'max_violation_strength': max_violation,
            'threshold': threshold
        }


def visualize_overlap_matrix(overlap_data: Dict, figsize=(12, 8), 
                            cmap='Blues', save_path: Optional[str] = None):
    """
    Visualize overlap matrix between eigenmodes and spherical harmonics.
    
    Parameters:
        overlap_data: Output from compute_overlap_matrix()
        figsize: Figure size
        cmap: Colormap for heatmap
        save_path: If provided, save figure to this path
    """
    overlap_matrix = overlap_data['overlap_matrix']
    ylm_labels = overlap_data['ylm_labels']
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(overlap_matrix, aspect='auto', cmap=cmap, 
                   interpolation='nearest', origin='lower')
    
    ax.set_xlabel('Spherical Harmonic Index', fontsize=12)
    ax.set_ylabel('Eigenmode Index', fontsize=12)
    ax.set_title('Overlap Matrix: |⟨ψ_i | Y_ℓ^m⟩|²', fontsize=14)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('|Overlap|²', fontsize=12)
    
    # Add tick labels for some Y_ℓ^m
    if len(ylm_labels) < 50:
        ylm_tick_labels = [f"Y_{ell}^{m}" for ell, m in ylm_labels]
        ax.set_xticks(range(len(ylm_labels)))
        ax.set_xticklabels(ylm_tick_labels, rotation=90, fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved overlap matrix to {save_path}")
    
    return fig, ax


def visualize_energy_comparison(hydrogen_data: Dict, figsize=(12, 6),
                                save_path: Optional[str] = None):
    """
    Visualize comparison between lattice eigenvalues and hydrogen energies.
    
    Parameters:
        hydrogen_data: Output from compare_to_hydrogen()
        figsize: Figure size
        save_path: If provided, save figure to this path
    """
    n_assignments = hydrogen_data['n_assignments']
    hydrogen_energies = hydrogen_data['hydrogen_energies']
    lattice_energies = hydrogen_data['lattice_energies']
    relative_errors = hydrogen_data['relative_errors']
    quantum_ids = hydrogen_data['quantum_ids']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Panel 1: Energy levels comparison
    mode_indices = np.arange(len(lattice_energies))
    
    ax1.scatter(mode_indices, lattice_energies, label='Lattice', alpha=0.7, s=40)
    ax1.scatter(mode_indices, hydrogen_energies, label='Hydrogen', 
               alpha=0.7, s=40, marker='x')
    
    # Draw lines connecting same mode
    for i in mode_indices:
        ax1.plot([i, i], [lattice_energies[i], hydrogen_energies[i]], 
                'k-', alpha=0.2, linewidth=0.5)
    
    ax1.set_xlabel('Eigenmode Index', fontsize=12)
    ax1.set_ylabel('Energy (a.u.)', fontsize=12)
    ax1.set_title('Energy Level Comparison', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Relative errors by n
    unique_n = np.unique(n_assignments)
    for n in unique_n:
        mask = n_assignments == n
        ax2.scatter(mode_indices[mask], relative_errors[mask], 
                   label=f'n={n}', alpha=0.7, s=40)
    
    ax2.set_xlabel('Eigenmode Index', fontsize=12)
    ax2.set_ylabel('Relative Error |ΔE/E|', fontsize=12)
    ax2.set_title('Relative Energy Errors', fontsize=14)
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved energy comparison to {save_path}")
    
    return fig, (ax1, ax2)


def visualize_selection_rules(dipole_data: Dict, figsize=(14, 5),
                              strength_threshold: float = 1e-3,
                              save_path: Optional[str] = None):
    """
    Visualize dipole selection rules.
    
    Parameters:
        dipole_data: Output from compute_dipole_matrix_elements()
        figsize: Figure size
        strength_threshold: Threshold for "strong" transitions
        save_path: If provided, save figure to this path
    """
    transition_strength = dipole_data['transition_strength']
    delta_ell = dipole_data['delta_ell']
    delta_m = dipole_data['delta_m']
    selection_allowed = dipole_data['selection_allowed']
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Panel 1: Full transition matrix
    im1 = ax1.imshow(np.log10(transition_strength + 1e-10), 
                     aspect='auto', cmap='viridis', origin='lower')
    ax1.set_xlabel('Initial State i', fontsize=11)
    ax1.set_ylabel('Final State f', fontsize=11)
    ax1.set_title('Transition Strength\nlog₁₀|⟨f|r|i⟩|²', fontsize=12)
    plt.colorbar(im1, ax=ax1, label='log₁₀(strength)')
    
    # Panel 2: Δℓ vs Δm scatter (only strong transitions)
    strong_mask = transition_strength > strength_threshold
    allowed_and_strong = strong_mask & selection_allowed
    forbidden_and_strong = strong_mask & ~selection_allowed
    
    # Plot allowed transitions
    ax2.scatter(delta_m[allowed_and_strong], delta_ell[allowed_and_strong],
               c=np.log10(transition_strength[allowed_and_strong]), 
               cmap='Greens', s=50, alpha=0.7, label='Allowed')
    
    # Plot forbidden transitions
    if np.any(forbidden_and_strong):
        ax2.scatter(delta_m[forbidden_and_strong], delta_ell[forbidden_and_strong],
                   c=np.log10(transition_strength[forbidden_and_strong]), 
                   cmap='Reds', s=50, alpha=0.7, marker='x', label='Forbidden')
    
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=-1, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=-1, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Δm', fontsize=11)
    ax2.set_ylabel('Δℓ', fontsize=11)
    ax2.set_title(f'Selection Rules\n(strength > {strength_threshold:.0e})', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Histogram of transition strengths by rule
    allowed_strengths = transition_strength[selection_allowed & (transition_strength > 0)]
    forbidden_strengths = transition_strength[~selection_allowed & (transition_strength > 0)]
    
    bins = np.logspace(-10, 0, 50)
    ax3.hist(allowed_strengths, bins=bins, alpha=0.6, label='Allowed', color='green')
    ax3.hist(forbidden_strengths, bins=bins, alpha=0.6, label='Forbidden', color='red')
    
    ax3.set_xlabel('Transition Strength |⟨f|r|i⟩|²', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('Strength Distribution', fontsize=12)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved selection rules visualization to {save_path}")
    
    return fig, (ax1, ax2, ax3)
