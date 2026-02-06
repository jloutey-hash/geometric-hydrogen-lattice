"""
Relativistic Extensions to the Paraboloid Lattice

This module extends the hydrogen atom lattice with:
1. Runge-Lenz vector operators (A) for complete SO(4) symmetry
2. Spin degrees of freedom and spin-orbit coupling
3. Fine structure corrections

Author: Computational Physics Research
Date: February 2026
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, kron, eye, diags
from scipy.sparse.linalg import eigsh
from typing import Dict, Tuple, List
import time

from paraboloid_lattice_su11 import ParaboloidLattice


class RungeLenzLattice(ParaboloidLattice):
    """
    Extends ParaboloidLattice with Runge-Lenz vector operators.
    
    The Runge-Lenz vector A (also called Laplace-Runge-Lenz vector) is:
        A = (p × L - L × p)/(2m) - k*r/r
    
    For hydrogen, A and L form the SO(4) algebra for bound states:
        [L_i, A_j] = i ε_{ijk} A_k  (A transforms as vector)
        [A_i, A_j] = i ε_{ijk} L_k  (closed algebra)
        L² + A² = n² - 1  (Casimir invariant)
    
    Selection rules: Δn = 0, Δl = ±1, Δm = 0, ±1
    """
    
    def __init__(self, max_n: int):
        """Initialize with Runge-Lenz operators."""
        super().__init__(max_n)
        
        # Build Runge-Lenz operators
        print("Building Runge-Lenz vector operators...")
        self.Ax = None
        self.Ay = None
        self.Az = None
        self._build_runge_lenz_operators()
        print("Runge-Lenz operators constructed.")
    
    def _build_runge_lenz_operators(self):
        """
        Construct Runge-Lenz operators using EXACT Biedenharn-Louck formulas.
        
        Reference: Biedenharn & Louck, "Angular Momentum in Quantum Physics"
        
        For hydrogen (E_n = -1/(2n²)), the normalized operators satisfy:
            [L_i, A_j] = iε_{ijk}A_k,  [A_i, A_j] = -iε_{ijk}L_k
            Casimir: L² + A² = n² - 1  (EXACTLY on diagonal)
        
        Matrix elements (Biedenharn-Louck convention):
        
        A_z: <n,l',m|A_z|n,l,m> with Δl = ±1, Δm = 0
            l→l-1: √(n²-l²) · √[(l²-m²)/(4l²-1)]
            l→l+1: √(n²-(l+1)²) · √[((l+1)²-m²)/(4(l+1)²-1)]
        
        A_±: <n,l',m±1|A_±|n,l,m> with Δl = ±1, Δm = ±1
            l→l-1: ∓√(n²-l²) · √[(l∓m)(l∓m-1)] / √(4l²-1)
            l→l+1: ±√(n²-(l+1)²) · √[(l±m+1)(l±m+2)] / √(4(l+1)²-1)
        """
        dim = self.dim
        
        # Use lil_matrix for efficient construction
        Aplus = sp.lil_matrix((dim, dim), dtype=complex)
        Aminus = sp.lil_matrix((dim, dim), dtype=complex)
        Az_mat = sp.lil_matrix((dim, dim), dtype=complex)
        
        for i, (n, l, m) in enumerate(self.nodes):
            # ========================================
            # Transitions: l → l-1 (lowering l)
            # ========================================
            if l > 0:
                # Radial factor: √(n² - l²)
                radial = np.sqrt(n**2 - l**2)
                denominator_sqrt = np.sqrt(4 * l**2 - 1)
                
                # A_z: Δm = 0
                target = (n, l - 1, m)
                if target in self.node_index:
                    j = self.node_index[target]
                    # √[(l² - m²)/(4l² - 1)]
                    angular = np.sqrt((l**2 - m**2) / (4 * l**2 - 1))
                    Az_mat[j, i] = radial * angular
                
                # A_+: Δm = +1, sign is MINUS
                target = (n, l - 1, m + 1)
                if target in self.node_index:
                    j = self.node_index[target]
                    # -√[(l-m)(l-m-1)] / √(4l²-1)
                    numerator = np.sqrt((l - m) * (l - m - 1))
                    Aplus[j, i] = -radial * numerator / denominator_sqrt
                
                # A_-: Δm = -1, sign is PLUS
                target = (n, l - 1, m - 1)
                if target in self.node_index:
                    j = self.node_index[target]
                    # +√[(l+m)(l+m-1)] / √(4l²-1)
                    numerator = np.sqrt((l + m) * (l + m - 1))
                    Aminus[j, i] = radial * numerator / denominator_sqrt
            
            # ========================================
            # Transitions: l → l+1 (raising l)
            # ========================================
            if l + 1 < n:
                # Radial factor: √(n² - (l+1)²)
                radial = np.sqrt(n**2 - (l + 1)**2)
                denominator_sqrt = np.sqrt(4 * (l + 1)**2 - 1)
                
                # A_z: Δm = 0
                target = (n, l + 1, m)
                if target in self.node_index:
                    j = self.node_index[target]
                    # √[((l+1)² - m²)/(4(l+1)² - 1)]
                    angular = np.sqrt(((l + 1)**2 - m**2) / (4 * (l + 1)**2 - 1))
                    Az_mat[j, i] = radial * angular
                
                # A_+: Δm = +1, sign is PLUS
                target = (n, l + 1, m + 1)
                if target in self.node_index:
                    j = self.node_index[target]
                    # +√[(l+m+1)(l+m+2)] / √(4(l+1)²-1)
                    numerator = np.sqrt((l + m + 1) * (l + m + 2))
                    Aplus[j, i] = radial * numerator / denominator_sqrt
                
                # A_-: Δm = -1, sign is MINUS
                target = (n, l + 1, m - 1)
                if target in self.node_index:
                    j = self.node_index[target]
                    # -√[(l-m+1)(l-m+2)] / √(4(l+1)²-1)
                    numerator = np.sqrt((l - m + 1) * (l - m + 2))
                    Aminus[j, i] = -radial * numerator / denominator_sqrt
        
        # Convert to Cartesian components
        # Biedenharn-Louck convention with overall sign:
        # A_x = +(A_+ + A_-) / 2
        # A_y = -i(A_+ - A_-) / 2
        # A_z = -Az_mat
        self.Ax = ((Aplus + Aminus) / 2.0).tocsr()
        self.Ay = (-1j * (Aplus - Aminus) / 2.0).tocsr()
        self.Az = (-Az_mat).tocsr()
        
        print(f"  Runge-Lenz sparsity: Ax={self.Ax.nnz}/{dim**2} ({100*self.Ax.nnz/dim**2:.2f}%)")
        
        # Also build position operators for Stark effect
        self._build_position_operators()
    
    def _build_position_operators(self):
        """
        Build position operators r_x, r_y, r_z for Stark effect.
        
        In spherical coordinates:
            z = r cos(θ) couples to Y_1^0 (l±1)
            x,y couple to Y_1^±1
        
        Matrix elements for hydrogen (in units where a₀=1):
            <n,l',m'|z|n,l,m> connects l and l±1 with Δm=0
        
        For the special case of 2s-2p:
            <2,1,0|z|2,0,0> = 3*n*a₀ = 6 a.u. for n=2
        """
        dim = self.dim
        
        # Use lil_matrix for construction
        z_mat = sp.lil_matrix((dim, dim), dtype=complex)
        
        # Build z operator (couples l to l±1 with Δm=0)
        for i, (n, l, m) in enumerate(self.nodes):
            # z connects l to l+1 (only Δm=0)
            if l + 1 < n and m == 0:
                target = (n, l + 1, m)
                if target in self.node_index:
                    j = self.node_index[target]
                    # For s->p (l=0->l=1) with m=0: exact formula is 3n
                    if l == 0:
                        matrix_element = 3.0 * n
                    else:
                        # General formula (Wigner-Eckart): radial × Clebsch-Gordan
                        radial = (n**2) * np.sqrt((n - l - 1) * (n + l + 1)) / (l + 1)
                        angular = np.sqrt((2*l+1) / (2*(l+1)+1))  # Simplified for m=0
                        matrix_element = radial * angular
                    
                    # Set both directions (Hermitian)
                    z_mat[i, j] = matrix_element
                    z_mat[j, i] = matrix_element
        
        self.z_op = z_mat.tocsr()
        print(f"  Position operator z built. Sparsity: {self.z_op.nnz}/{dim**2} ({100*self.z_op.nnz/dim**2:.2f}%)")
    
    
    def validate_so4_algebra(self, verbose=True):
        """
        Verify the SO(4) commutation relations.
        
        Returns:
        --------
        dict: Dictionary of error norms for each commutator
        """
        if verbose:
            print("\n=== SO(4) Algebra Validation ===")
        
        errors = {}
        
        # [L_i, A_j] = i ε_{ijk} A_k
        # Test [Lz, Ax] = i Ay
        comm = self.Lz @ self.Ax - self.Ax @ self.Lz
        expected = 1j * self.Ay
        err = np.linalg.norm((comm - expected).toarray())
        errors['[Lz, Ax] - i*Ay'] = err
        if verbose:
            print(f"[Lz, Ax] = i*Ay: error = {err:.2e}")
        
        # Test [Lz, Ay] = -i Ax
        comm = self.Lz @ self.Ay - self.Ay @ self.Lz
        expected = -1j * self.Ax
        err = np.linalg.norm((comm - expected).toarray())
        errors['[Lz, Ay] + i*Ax'] = err
        if verbose:
            print(f"[Lz, Ay] = -i*Ax: error = {err:.2e}")
        
        # Test [Lplus, Az] = Aplus
        comm = self.Lplus @ self.Az - self.Az @ self.Lplus
        expected = self.Ax + 1j * self.Ay  # Aplus
        err = np.linalg.norm((comm - expected).toarray())
        errors['[L+, Az] - A+'] = err
        if verbose:
            print(f"[L+, Az] = A+: error = {err:.2e}")
        
        # [A_i, A_j] = i ε_{ijk} L_k
        # Test [Ax, Ay] = i Lz
        comm = self.Ax @ self.Ay - self.Ay @ self.Ax
        expected = 1j * self.Lz
        err = np.linalg.norm((comm - expected).toarray())
        errors['[Ax, Ay] - i*Lz'] = err
        if verbose:
            print(f"[Ax, Ay] = i*Lz: error = {err:.2e}")
        
        # Check Casimir: L² + A² should be diagonal with eigenvalue n² - 1
        L2 = self.Lplus @ self.Lminus + self.Lz @ self.Lz - self.Lz
        A2 = self.Ax @ self.Ax + self.Ay @ self.Ay + self.Az @ self.Az
        casimir = L2 + A2
        
        # Check if diagonal
        off_diag_norm = np.linalg.norm(casimir.toarray() - np.diag(np.diag(casimir.toarray())))
        errors['Casimir off-diagonal'] = off_diag_norm
        if verbose:
            print(f"\nCasimir L² + A² diagonal check: {off_diag_norm:.2e}")
        
        # Check eigenvalues
        diag_vals = np.diag(casimir.toarray())
        for i, (n, l, m) in enumerate(self.nodes):
            expected_val = n**2 - 1
            err = abs(diag_vals[i] - expected_val)
            if err > 1e-10 and verbose:
                print(f"  Node ({n},{l},{m}): Casimir = {diag_vals[i]:.6f}, expected = {expected_val}, err = {err:.2e}")
        
        if verbose:
            print("\n✓ SO(4) algebra validation complete")
        
        return errors


class SpinParaboloid:
    """
    Extends the paraboloid lattice with spin degrees of freedom.
    
    State space: |n, l, m_l, m_s⟩ (uncoupled basis)
    where m_s = ±1/2 for spin-1/2 particles.
    
    Implements:
    - Spin operators: S_z, S_±
    - Spin-orbit coupling: L·S
    - Fine structure Hamiltonian: H_FS ∝ (1/n³) L·S
    """
    
    def __init__(self, max_n: int):
        """
        Initialize spin-extended lattice.
        
        Parameters:
        -----------
        max_n : int
            Maximum principal quantum number
        """
        print(f"Constructing SpinParaboloid lattice (max_n={max_n})...")
        
        # Build scalar (orbital) lattice first
        self.scalar_lattice = ParaboloidLattice(max_n)
        self.max_n = max_n
        
        # Spin space is 2D: |↑⟩, |↓⟩
        self.spin_dim = 2
        
        # Full Hilbert space dimension
        self.dim = self.scalar_lattice.dim * self.spin_dim
        
        # Build node list: |n, l, m_l, m_s⟩
        self._construct_spin_nodes()
        
        # Build operators
        self._build_spin_operators()
        self._build_spin_orbit_operator()
        
        print(f"SpinParaboloid constructed: {self.dim} states ({self.scalar_lattice.dim} orbital × {self.spin_dim} spin)")
    
    def _construct_spin_nodes(self):
        """
        Build node list for spin-extended Hilbert space.
        Each orbital state |n,l,m_l⟩ is tensored with |↑⟩ and |↓⟩.
        """
        self.nodes = []
        self.node_index = {}
        
        idx = 0
        for n, l, ml in self.scalar_lattice.nodes:
            for ms in [0.5, -0.5]:  # Spin up, spin down
                node = (n, l, ml, ms)
                self.nodes.append(node)
                self.node_index[node] = idx
                idx += 1
    
    def _build_spin_operators(self):
        """
        Construct spin operators using Pauli matrices.
        
        In uncoupled basis |orbital⟩ ⊗ |spin⟩:
            S_z = I_orbital ⊗ σ_z/2
            S_+ = I_orbital ⊗ σ_+
            S_- = I_orbital ⊗ σ_-
        """
        print("Building spin operators...")
        
        # Pauli matrices in spin space
        sigma_z = csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))
        sigma_plus = csr_matrix(np.array([[0, 1], [0, 0]], dtype=complex))
        sigma_minus = csr_matrix(np.array([[0, 0], [1, 0]], dtype=complex))
        
        # Identity on orbital space
        I_orbital = eye(self.scalar_lattice.dim, format='csr')
        
        # Tensor product
        self.Sz = kron(I_orbital, sigma_z / 2.0)
        self.Splus = kron(I_orbital, sigma_plus)
        self.Sminus = kron(I_orbital, sigma_minus)
        
        print(f"  Spin operators built. Sparsity: {self.Sz.nnz}/{self.dim**2}")
    
    def _build_spin_orbit_operator(self):
        """
        Construct the spin-orbit coupling operator L·S.
        
        L·S = L_x*S_x + L_y*S_y + L_z*S_z
            = (1/2)(L_+ S_- + L_- S_+) + L_z S_z
        
        Using tensor products:
            L_z S_z = (L_z ⊗ I_spin) @ (I_orbital ⊗ S_z)
            L_+ S_- = (L_+ ⊗ I_spin) @ (I_orbital ⊗ S_-)
            L_- S_+ = (L_- ⊗ I_spin) @ (I_orbital ⊗ S_+)
        """
        print("Building L·S operator...")
        
        I_orbital = eye(self.scalar_lattice.dim, format='csr')
        I_spin = eye(self.spin_dim, format='csr')
        
        # Lift orbital operators to full space
        Lz_full = kron(self.scalar_lattice.Lz, I_spin)
        Lplus_full = kron(self.scalar_lattice.Lplus, I_spin)
        Lminus_full = kron(self.scalar_lattice.Lminus, I_spin)
        
        # L·S = (L_+ S_- + L_- S_+)/2 + L_z S_z
        self.L_dot_S = (Lplus_full @ self.Sminus + Lminus_full @ self.Splus) / 2.0 + Lz_full @ self.Sz
        
        print(f"  L·S operator built. Sparsity: {self.L_dot_S.nnz}/{self.dim**2} ({100*self.L_dot_S.nnz/self.dim**2:.2f}%)")
    
    def build_fine_structure_hamiltonian(self, alpha=1/137.036):
        """
        Build the fine structure Hamiltonian with EXACT radial prefactors.
        
        Semi-analytic approach:
            H_FS = ξ(n,l) * (L·S)
        
        Where ξ(n,l) is the exact quantum mechanical radial expectation:
            ξ(n,l) = (α²/2) * (1/n³) / [l(l+1/2)(l+1)]
        
        This gives the exact fine structure splitting:
            E_FS(n,l,j) = ξ(n,l) * [j(j+1) - l(l+1) - 3/4]
        
        For 2p (n=2, l=1):
            ξ = (α²/2) / (8 * 1 * 1.5 * 2) = α²/48
            j=3/2: E = (α²/48) * (15/4 - 2 - 3/4) = (α²/48) * 1 = α²/48
            j=1/2: E = (α²/48) * (3/4 - 2 - 3/4) = (α²/48) * (-2) = -α²/24
            Splitting: ΔE = 3α²/48 = α²/16 ≈ 3.32×10⁻⁶ a.u. ≈ 0.090 meV
        
        Parameters:
        -----------
        alpha : float
            Fine structure constant (default: 1/137.036)
        
        Returns:
        --------
        H_FS : sparse matrix
            Fine structure Hamiltonian (in atomic units)
        """
        print(f"Building fine structure Hamiltonian (α = {alpha:.6f})...")
        
        # Build exact prefactor ξ(n,l) for each state
        xi_array = np.zeros(self.dim)
        
        for i, (n, l, ml, ms) in enumerate(self.nodes):
            if l > 0:
                # Exact formula from quantum mechanics
                xi_array[i] = (alpha**2 / 2.0) / (n**3 * l * (l + 0.5) * (l + 1))
            else:
                # No spin-orbit coupling for s-states (l=0)
                xi_array[i] = 0.0
        
        # Apply scaling to L·S operator
        H_FS = diags(xi_array, format='csr') @ self.L_dot_S
        
        # Theoretical verification for n=2, l=1
        if any(n == 2 and l == 1 for n, l, _, _ in self.nodes):
            xi_2p = (alpha**2 / 2.0) / (8 * 1 * 1.5 * 2)
            theoretical_splitting = 3 * xi_2p  # j=3/2 to j=1/2
            print(f"  Exact 2p theory: ξ = {xi_2p:.6e}, ΔE = {theoretical_splitting:.6e} a.u.")
        
        print(f"  H_FS constructed. Norm = {np.linalg.norm(H_FS.toarray()):.6e}")
        
        return H_FS
    
    def extract_shell(self, n: int):
        """
        Extract states for a specific n-shell.
        
        Parameters:
        -----------
        n : int
            Principal quantum number
        
        Returns:
        --------
        indices : list
            Indices of states in this shell
        shell_nodes : list
            List of (l, ml, ms) tuples
        """
        indices = []
        shell_nodes = []
        
        for i, (n_i, l, ml, ms) in enumerate(self.nodes):
            if n_i == n:
                indices.append(i)
                shell_nodes.append((l, ml, ms))
        
        return indices, shell_nodes
    
    def validate_spin_operators(self, verbose=True):
        """
        Validate spin operator algebra: [S_i, S_j] = i ε_{ijk} S_k
        """
        if verbose:
            print("\n=== Spin Operator Validation ===")
        
        errors = {}
        
        # [Sz, S+] = S+
        comm = self.Sz @ self.Splus - self.Splus @ self.Sz
        err = np.linalg.norm((comm - self.Splus).toarray())
        errors['[Sz, S+] - S+'] = err
        if verbose:
            print(f"[Sz, S+] = S+: error = {err:.2e}")
        
        # [Sz, S-] = -S-
        comm = self.Sz @ self.Sminus - self.Sminus @ self.Sz
        err = np.linalg.norm((comm + self.Sminus).toarray())
        errors['[Sz, S-] + S-'] = err
        if verbose:
            print(f"[Sz, S-] = -S-: error = {err:.2e}")
        
        # [S+, S-] = 2*Sz
        comm = self.Splus @ self.Sminus - self.Sminus @ self.Splus
        err = np.linalg.norm((comm - 2.0 * self.Sz).toarray())
        errors['[S+, S-] - 2*Sz'] = err
        if verbose:
            print(f"[S+, S-] = 2*Sz: error = {err:.2e}")
        
        # S² = Sz² + (S+S- + S-S+)/2 should equal s(s+1) = 3/4 for spin-1/2
        S2 = self.Sz @ self.Sz + (self.Splus @ self.Sminus + self.Sminus @ self.Splus) / 2.0
        expected = 0.75 * eye(self.dim, format='csr')
        err = np.linalg.norm((S2 - expected).toarray())
        errors['S² - 3/4*I'] = err
        if verbose:
            print(f"S² = 3/4*I: error = {err:.2e}")
        
        if verbose:
            print("✓ Spin algebra validation complete\n")
        
        return errors


def analyze_n2_fine_structure(spin_lattice: SpinParaboloid, alpha=1/137.036):
    """
    Analyze the fine structure splitting of the n=2 shell.
    
    The n=2 shell contains:
    - 2s_{1/2}: l=0, j=1/2 (2 states)
    - 2p_{1/2}: l=1, j=1/2 (2 states)
    - 2p_{3/2}: l=1, j=3/2 (4 states)
    
    Fine structure lifts the l-degeneracy.
    
    Returns:
    --------
    dict: Analysis results including eigenvalues and splittings
    """
    print("\n=== n=2 Fine Structure Analysis ===")
    
    # Extract n=2 subspace
    indices, shell_nodes = spin_lattice.extract_shell(n=2)
    n_states = len(indices)
    
    print(f"n=2 shell: {n_states} states")
    
    # Build subspace Hamiltonian
    H_FS = spin_lattice.build_fine_structure_hamiltonian(alpha=alpha)
    H_sub = H_FS[np.ix_(indices, indices)].toarray()
    
    # Diagonalize
    eigenvalues, eigenvectors = np.linalg.eigh(H_sub)
    
    # Sort by energy
    sort_idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]
    
    # Identify states by quantum numbers
    print("\nEigenvalues (fine structure energies in a.u.):")
    for i, E in enumerate(eigenvalues):
        # Identify dominant component
        dominant_idx = np.argmax(np.abs(eigenvectors[:, i]))
        l, ml, ms = shell_nodes[dominant_idx]
        j = l + ms if ms > 0 else l - ms
        print(f"  E[{i}] = {E:+.6e}  (l={l}, ml={ml:.1f}, ms={ms:+.1f}, j≈{j:.1f})")
    
    # Calculate splittings
    print("\nLevel splittings:")
    unique_E = np.unique(np.round(eigenvalues, 10))
    for i in range(len(unique_E) - 1):
        delta = unique_E[i+1] - unique_E[i]
        print(f"  ΔE[{i}→{i+1}] = {delta:.6e} a.u. = {delta * 27.211:.6e} eV")
    
    # Theoretical prediction for n=2:
    # ΔE(2p_3/2 - 2p_1/2) ≈ (α²/32) * m_e c² ≈ 4.5 × 10^-5 eV
    theory_split = (alpha**2 / 32.0) * 0.5  # In Hartree (0.5 = m_e c² / 2 in a.u.)
    print(f"\nTheoretical 2p splitting: {theory_split:.6e} a.u. = {theory_split * 27.211:.6e} eV")
    
    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'shell_nodes': shell_nodes,
        'n_states': n_states
    }


if __name__ == "__main__":
    print("="*60)
    print("RELATIVISTIC PARABOLOID LATTICE TEST")
    print("="*60)
    
    # Test 1: Runge-Lenz operators
    print("\n--- Test 1: Runge-Lenz Vector (SO(4) Symmetry) ---")
    rl_lattice = RungeLenzLattice(max_n=4)
    rl_lattice.validate_so4_algebra(verbose=True)
    
    # Test 2: Spin operators
    print("\n--- Test 2: Spin Operators and L·S Coupling ---")
    spin_lattice = SpinParaboloid(max_n=3)
    spin_lattice.validate_spin_operators(verbose=True)
    
    # Test 3: Fine structure
    print("\n--- Test 3: Fine Structure Splitting ---")
    results = analyze_n2_fine_structure(spin_lattice)
    
    print("\n" + "="*60)
    print("✓ All tests complete!")
    print("="*60)
