"""
PHYSICS LAPLACIAN FIX: Correct Graph Laplacian Construction
============================================================
Critical Fix: Use proper Graph Laplacian T = D - A, not just adjacency A.

Previous Error: Kinetic energy had zero diagonal → wrong energy scale
Correction: Graph Laplacian includes degree matrix D (on-site inertia)

Proper Hamiltonian:
  H = beta * (D - A) + V
  where:
    D = degree matrix (D_ii = sum_j A_ij)
    A = adjacency matrix (transition weights)
    V = -1/n^2 (Coulomb potential)

Author: Spectral Graph Theory Team
Date: February 2026
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh
from paraboloid_lattice_su11 import ParaboloidLattice
import time
from typing import Tuple, Dict


class LaplacianHamiltonian:
    """
    Correct construction of the Paraboloid Lattice Hamiltonian
    using the Graph Laplacian for kinetic energy.
    """
    
    def __init__(self, max_n: int = 10):
        """
        Initialize Laplacian Hamiltonian.
        
        Parameters:
        -----------
        max_n : int
            Maximum principal quantum number
        """
        self.max_n = max_n
        self.lattice = None
        self.A = None  # Adjacency matrix
        self.D = None  # Degree matrix
        self.L = None  # Laplacian L = D - A
        self.V = None  # Potential
        self.H = None  # Full Hamiltonian
        self.beta = 1.0  # Scaling factor
        
        # Known hydrogen energies (Hartree units)
        self.E_hydrogen = {
            1: -0.5000,
            2: -0.1250,
            3: -0.0556
        }
        
    def build_lattice(self):
        """Build the paraboloid lattice."""
        print(f"Building lattice with n <= {self.max_n}...")
        self.lattice = ParaboloidLattice(max_n=self.max_n)
        print(f"  Lattice size: {len(self.lattice.nodes)} nodes")
        
    def build_adjacency_matrix(self):
        """
        Build adjacency matrix from transition operators.
        
        The adjacency matrix represents all possible transitions:
        A_ij = weight of transition from state i to state j
        
        We sum contributions from all ladder operators.
        """
        print("\nBuilding adjacency matrix...")
        
        n_nodes = len(self.lattice.nodes)
        A = lil_matrix((n_nodes, n_nodes), dtype=float)
        
        # Get all transition operators
        operators = [
            self.lattice.Tplus,
            self.lattice.Tminus,
            self.lattice.Lplus,
            self.lattice.Lminus
        ]
        
        for op in operators:
            if op is not None:
                # Add absolute value of operator elements
                # This gives undirected edge weights
                A += np.abs(op)
        
        # Make symmetric (undirected graph)
        A = (A + A.T) / 2.0
        
        self.A = A.tocsr()
        
        # Compute statistics
        nnz = self.A.nnz
        density = nnz / (n_nodes * n_nodes) * 100
        
        print(f"  Adjacency matrix: {self.A.shape}")
        print(f"  Non-zero elements: {nnz}")
        print(f"  Density: {density:.2f}%")
        print(f"  Edge weight range: [{self.A.data.min():.3f}, {self.A.data.max():.3f}]")
        
    def build_degree_matrix(self):
        """
        Build degree matrix D.
        
        D is diagonal with D_ii = sum of row i of A
        This represents the "connectivity" or "coordination number" of each node.
        """
        print("\nBuilding degree matrix...")
        
        # Sum each row of A
        degrees = np.array(self.A.sum(axis=1)).flatten()
        
        n_nodes = len(self.lattice.nodes)
        D = lil_matrix((n_nodes, n_nodes), dtype=float)
        
        for i in range(n_nodes):
            D[i, i] = degrees[i]
        
        self.D = D.tocsr()
        
        print(f"  Degree matrix: {self.D.shape}")
        print(f"  Degree range: [{degrees.min():.3f}, {degrees.max():.3f}]")
        print(f"  Mean degree: {degrees.mean():.3f}")
        
    def build_laplacian(self):
        """
        Build graph Laplacian L = D - A.
        
        This is the discrete analog of -nabla^2 (kinetic energy).
        Properties:
        - L is symmetric and positive semi-definite
        - L has zero row/column sums
        - Eigenvalues are >= 0
        """
        print("\nBuilding graph Laplacian...")
        
        self.L = self.D - self.A
        
        # Convert to dense for eigenvalue check (small matrix)
        if self.L.shape[0] < 500:
            L_dense = self.L.toarray()
            eigvals = np.linalg.eigvalsh(L_dense)
            min_eig = eigvals[0]
            max_eig = eigvals[-1]
            print(f"  Laplacian: {self.L.shape}")
            print(f"  Smallest eigenvalue: {min_eig:.6f} (should be ~0)")
            print(f"  Largest eigenvalue: {max_eig:.3f}")
            print(f"  Diagonal range: [{self.L.diagonal().min():.3f}, {self.L.diagonal().max():.3f}]")
        else:
            print(f"  Laplacian: {self.L.shape}")
            print(f"  Diagonal range: [{self.L.diagonal().min():.3f}, {self.L.diagonal().max():.3f}]")
        
    def build_potential(self):
        """
        Build potential energy operator.
        
        V_ii = -1/n_i^2 (Coulomb potential in atomic units)
        """
        print("\nBuilding potential energy...")
        
        n_nodes = len(self.lattice.nodes)
        V = lil_matrix((n_nodes, n_nodes), dtype=float)
        
        for i, (n, l, m) in enumerate(self.lattice.nodes):
            V[i, i] = -1.0 / (n * n)
        
        self.V = V.tocsr()
        
        v_diag = self.V.diagonal()
        print(f"  Potential: {self.V.shape}")
        print(f"  Potential range: [{v_diag.min():.6f}, {v_diag.max():.6f}]")
        
    def calibrate_hamiltonian(self, target_E1: float = -0.5):
        """
        Calibrate the Hamiltonian scaling factor beta.
        
        We want H = beta * L + V such that the ground state energy
        matches the hydrogen ground state E_1 = -0.5 Hartree.
        
        Strategy: Try different beta values and find the one that gives
        the best match to E_1.
        """
        print("\n" + "="*80)
        print("HAMILTONIAN CALIBRATION")
        print("="*80)
        print(f"\nTarget: E_1 = {target_E1:.6f} Hartree")
        
        # Try a range of beta values
        beta_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
        
        results = []
        
        print("\nScanning beta values...")
        print("-"*80)
        print(f"{'Beta':>10} {'E_1':>12} {'E_2':>12} {'E_3':>12} {'Error_1':>12}")
        print("-"*80)
        
        for beta in beta_values:
            # Build Hamiltonian
            H = beta * self.L + self.V
            
            # Diagonalize
            try:
                n_eigs = min(20, H.shape[0] - 2)
                evals, evecs = eigsh(H, k=n_eigs, which='SA', sigma=-1.0)
                evals = np.sort(evals)
                
                E_1 = evals[0]
                E_2 = evals[1] if len(evals) > 1 else 0.0
                E_3 = evals[2] if len(evals) > 2 else 0.0
                
                error_1 = abs(E_1 - target_E1)
                
                results.append({
                    'beta': beta,
                    'E_1': E_1,
                    'E_2': E_2,
                    'E_3': E_3,
                    'error': error_1,
                    'evals': evals,
                    'evecs': evecs
                })
                
                print(f"{beta:>10.4f} {E_1:>12.6f} {E_2:>12.6f} {E_3:>12.6f} {error_1:>12.6f}")
                
            except Exception as e:
                print(f"{beta:>10.4f} [Failed: {e}]")
        
        print("-"*80)
        
        # Find best beta
        if results:
            best = min(results, key=lambda x: x['error'])
            self.beta = best['beta']
            
            print(f"\nOptimal beta: {self.beta:.6f}")
            print(f"Achieved E_1: {best['E_1']:.6f} (target: {target_E1:.6f})")
            print(f"Error: {best['error']:.6f}")
            
            # Build final Hamiltonian
            self.H = self.beta * self.L + self.V
            
            return best
        else:
            print("\nERROR: All calibration attempts failed!")
            return None
            
    def test_splitting(self, calibration_result):
        """
        Test 2s/2p splitting with calibrated Hamiltonian.
        """
        print("\n" + "="*80)
        print("DEGENERACY BREAKING TEST")
        print("="*80)
        
        evals = calibration_result['evals']
        evecs = calibration_result['evecs']
        
        # Find 2s and 2p states
        try:
            idx_2s = self.lattice.nodes.index((2, 0, 0))
            idx_2p = self.lattice.nodes.index((2, 1, 0))
        except ValueError as e:
            print(f"ERROR: Required states not found: {e}")
            return None
        
        print(f"\nState (2,0,0) [2s] is node #{idx_2s}")
        print(f"State (2,1,0) [2p] is node #{idx_2p}")
        
        # Find eigenvectors with maximum overlap
        overlaps_2s = np.abs(evecs[idx_2s, :])**2
        overlaps_2p = np.abs(evecs[idx_2p, :])**2
        
        best_2s_idx = np.argmax(overlaps_2s)
        best_2p_idx = np.argmax(overlaps_2p)
        
        E_2s = evals[best_2s_idx]
        E_2p = evals[best_2p_idx]
        
        overlap_2s = overlaps_2s[best_2s_idx]
        overlap_2p = overlaps_2p[best_2p_idx]
        
        print(f"\n2s state:")
        print(f"  Eigenvalue: {E_2s:.6f}")
        print(f"  Eigenvector: #{best_2s_idx}")
        print(f"  Overlap: {overlap_2s:.4f}")
        
        print(f"\n2p state:")
        print(f"  Eigenvalue: {E_2p:.6f}")
        print(f"  Eigenvector: #{best_2p_idx}")
        print(f"  Overlap: {overlap_2p:.4f}")
        
        # Compute splitting
        delta_E = E_2p - E_2s
        
        # Compare to hydrogen n=2 level
        E_hydrogen_n2 = self.E_hydrogen[2]
        avg_E = (E_2s + E_2p) / 2
        
        print(f"\n" + "-"*80)
        print("RESULTS:")
        print("-"*80)
        print(f"  E(2s) = {E_2s:.6f}")
        print(f"  E(2p) = {E_2p:.6f}")
        print(f"  Delta E = {delta_E:.6f}")
        print(f"  Average E = {avg_E:.6f}")
        print(f"  Hydrogen E(n=2) = {E_hydrogen_n2:.6f}")
        print(f"  |Delta E / E_avg| = {abs(delta_E/avg_E)*100:.2f}%")
        
        # Verdict
        print(f"\n" + "-"*80)
        if delta_E > 1e-6:
            print("VERDICT: DEGENERACY BROKEN")
            print("  E(2p) > E(2s) - Centrifugal barrier effect confirmed")
            if delta_E > 0.01:
                print("  Splitting is SIGNIFICANT")
            else:
                print("  Splitting is small but measurable")
        elif delta_E < -1e-6:
            print("VERDICT: INVERTED ORDER")
            print("  E(2s) > E(2p) - Unexpected! Physics is backwards")
        else:
            print("VERDICT: DEGENERATE")
            print("  No measurable splitting - Geometric effect not present")
        
        return {
            'E_2s': E_2s,
            'E_2p': E_2p,
            'delta_E': delta_E,
            'overlap_2s': overlap_2s,
            'overlap_2p': overlap_2p,
            'avg_E': avg_E
        }
        
    def generate_report(self, calibration, splitting, output_file="laplacian_report.txt"):
        """Generate comprehensive report."""
        
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("LAPLACIAN HAMILTONIAN FIX: COMPREHENSIVE REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("OBJECTIVE:\n")
            f.write("Fix the Hamiltonian construction using proper Graph Laplacian.\n")
            f.write("Previous error: Used adjacency matrix A instead of L = D - A.\n\n")
            
            f.write("="*80 + "\n")
            f.write("LATTICE STRUCTURE\n")
            f.write("="*80 + "\n\n")
            f.write(f"Maximum n:           {self.max_n}\n")
            f.write(f"Total nodes:         {len(self.lattice.nodes)}\n")
            f.write(f"Hamiltonian size:    {self.H.shape[0]} x {self.H.shape[1]}\n\n")
            
            f.write("="*80 + "\n")
            f.write("CALIBRATION RESULTS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Optimal scaling factor (beta): {self.beta:.6f}\n\n")
            
            f.write("Energy Levels (Calibrated vs Hydrogen):\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Level':<10} {'Hydrogen':<15} {'Calculated':<15} {'Error':<15}\n")
            f.write("-"*80 + "\n")
            
            E_1 = calibration['E_1']
            E_2 = calibration['E_2']
            E_3 = calibration['E_3']
            
            f.write(f"n=1{'':<7} {self.E_hydrogen[1]:<15.6f} {E_1:<15.6f} {abs(E_1-self.E_hydrogen[1]):<15.6f}\n")
            f.write(f"n=2{'':<7} {self.E_hydrogen[2]:<15.6f} {E_2:<15.6f} {abs(E_2-self.E_hydrogen[2]):<15.6f}\n")
            f.write(f"n=3{'':<7} {self.E_hydrogen[3]:<15.6f} {E_3:<15.6f} {abs(E_3-self.E_hydrogen[3]):<15.6f}\n")
            f.write("\n")
            
            if splitting:
                f.write("="*80 + "\n")
                f.write("DEGENERACY BREAKING ANALYSIS\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"E(2s) = {splitting['E_2s']:.6f}\n")
                f.write(f"E(2p) = {splitting['E_2p']:.6f}\n")
                f.write(f"Delta E = {splitting['delta_E']:.6f}\n")
                f.write(f"Relative splitting: {abs(splitting['delta_E']/splitting['avg_E'])*100:.2f}%\n\n")
                
                if splitting['delta_E'] > 1e-6:
                    f.write("CONCLUSION: Geometric degeneracy breaking CONFIRMED.\n")
                    f.write("The Graph Laplacian naturally creates a centrifugal barrier.\n")
                else:
                    f.write("CONCLUSION: No significant splitting observed.\n")
            
        print(f"\n[OK] Report saved to {output_file}")


def main():
    """Main routine."""
    
    print("="*80)
    print("LAPLACIAN HAMILTONIAN FIX")
    print("="*80)
    print("\nCorrecting the Hamiltonian using Graph Laplacian T = D - A\n")
    
    # Initialize
    ham = LaplacianHamiltonian(max_n=10)
    
    # Step 1: Build lattice
    print("Step 1: Building lattice...")
    ham.build_lattice()
    
    # Step 2: Build adjacency matrix
    print("\nStep 2: Building adjacency matrix...")
    ham.build_adjacency_matrix()
    
    # Step 3: Build degree matrix
    print("\nStep 3: Computing degree matrix...")
    ham.build_degree_matrix()
    
    # Step 4: Build Laplacian
    print("\nStep 4: Computing graph Laplacian...")
    ham.build_laplacian()
    
    # Step 5: Build potential
    print("\nStep 5: Building potential energy...")
    ham.build_potential()
    
    # Step 6: Calibrate
    print("\nStep 6: Calibrating Hamiltonian...")
    calibration = ham.calibrate_hamiltonian(target_E1=-0.5)
    
    if calibration is None:
        print("\nERROR: Calibration failed!")
        return
    
    # Step 7: Test splitting
    print("\nStep 7: Testing 2s/2p splitting...")
    splitting = ham.test_splitting(calibration)
    
    # Step 8: Generate report
    print("\nStep 8: Generating report...")
    ham.generate_report(calibration, splitting)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    if splitting:
        print(f"\nFinal Result:")
        print(f"  Beta (scaling): {ham.beta:.6f}")
        print(f"  E(2s): {splitting['E_2s']:.6f}")
        print(f"  E(2p): {splitting['E_2p']:.6f}")
        print(f"  Splitting: {splitting['delta_E']:.6f}")
        
        if splitting['delta_E'] > 1e-6:
            print("\n  >>> GEOMETRIC BARRIER CONFIRMED <<<")
        else:
            print("\n  >>> No geometric splitting <<<")
    
    print(f"\nResults saved to: laplacian_report.txt\n")


if __name__ == "__main__":
    main()
