"""
PHYSICS KILL SWITCH: Rigorous Eigenvalue Test
==============================================
Critical Test: Does the Hamiltonian ACTUALLY break degeneracy?

Skeptical Review: "Your 'Lamb Shift' is not real. You measured diagonal
elements (connectivity) instead of true eigenvalues. When you diagonalize
properly, the degeneracy will return."

This script performs the definitive test:
1. Construct full Hamiltonian H = T + V
2. Diagonalize exactly (sparse eigenvalue solver)
3. Identify eigenvectors for |2s> and |2p>
4. Compare TRUE eigenvalues

If Delta lambda < 1e-10: Theory is wrong (connectivity artifact)
If Delta lambda > 1e-4:  Theory survives (genuine splitting)

Author: Skeptical Computational Physics
Date: February 2026
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh
from paraboloid_lattice_su11 import ParaboloidLattice
import time


class KillSwitch:
    """
    Rigorous test of degeneracy breaking via exact diagonalization.
    """
    
    def __init__(self, max_n: int = 30):
        """
        Initialize kill switch test.
        
        Parameters:
        -----------
        max_n : int
            Maximum principal quantum number
        """
        self.max_n = max_n
        self.lattice = None
        self.H = None
        self.eigenvalues = None
        self.eigenvectors = None
        
    def construct_hamiltonian(self) -> csr_matrix:
        """
        Construct the full Hamiltonian H = T + V.
        
        T: Kinetic energy (Graph Laplacian)
        V: Potential energy (Coulomb -1/n^2)
        
        Returns:
        --------
        H : csr_matrix
            Full Hamiltonian matrix
        """
        print("\n" + "="*80)
        print("CONSTRUCTING HAMILTONIAN")
        print("="*80)
        print(f"\nBuilding lattice with n <= {self.max_n}...")
        
        # Build lattice
        self.lattice = ParaboloidLattice(max_n=self.max_n)
        N = len(self.lattice.nodes)
        
        print(f"Lattice size: {N} nodes")
        
        # Kinetic energy: Graph Laplacian
        print("\nBuilding kinetic energy operator...")
        A = (self.lattice.Lplus + self.lattice.Lminus + 
             self.lattice.Tplus + self.lattice.Tminus)
        
        # Symmetrize
        A = (A + A.conj().T) / 2.0
        
        # Degree matrix
        degrees = np.array(np.abs(A).sum(axis=1)).flatten()
        D = sp.diags(degrees, 0, format='csr', dtype=np.complex128)
        
        # Laplacian
        T = D - A
        
        print(f"  Kinetic energy constructed: {T.shape}")
        print(f"  Degree range: [{degrees.min():.3f}, {degrees.max():.3f}]")
        
        # Potential energy: Coulomb
        print("\nBuilding potential energy operator...")
        V_diag = []
        for n, l, m in self.lattice.nodes:
            V_ii = -1.0 / (n**2)
            V_diag.append(V_ii)
        
        V = sp.diags(V_diag, 0, format='csr', dtype=np.complex128)
        
        print(f"  Potential energy constructed: {V.shape}")
        print(f"  Potential range: [{min(V_diag):.6f}, {max(V_diag):.6f}]")
        
        # Total Hamiltonian
        H = T + V
        
        print(f"\nTotal Hamiltonian constructed: {H.shape}")
        print(f"  Matrix is Hermitian: {np.allclose(H.toarray(), H.conj().T.toarray())}")
        print(f"  Sparsity: {H.nnz / (H.shape[0]**2) * 100:.2f}%")
        
        self.H = H
        return H
    
    def diagonalize(self, k: int = 20) -> tuple:
        """
        Diagonalize Hamiltonian to find lowest eigenvalues.
        
        Parameters:
        -----------
        k : int
            Number of eigenvalues to compute
        
        Returns:
        --------
        eigenvalues : np.ndarray
            Sorted eigenvalues (ascending)
        eigenvectors : np.ndarray
            Corresponding eigenvectors (columns)
        """
        print("\n" + "="*80)
        print("EXACT DIAGONALIZATION")
        print("="*80)
        print(f"\nComputing lowest {k} eigenvalues...")
        print("This may take a few minutes...\n")
        
        if self.H is None:
            raise ValueError("Hamiltonian not constructed. Call construct_hamiltonian() first.")
        
        # Find lowest eigenvalues
        # Using 'SA' for smallest algebraic (most negative)
        start_time = time.time()
        
        try:
            eigenvalues, eigenvectors = eigsh(
                self.H, 
                k=k, 
                which='SA',  # Smallest algebraic
                return_eigenvectors=True
            )
        except Exception as e:
            print(f"ERROR: Diagonalization failed: {e}")
            print("Trying with smaller k...")
            k = min(10, k)
            eigenvalues, eigenvectors = eigsh(
                self.H, 
                k=k, 
                which='SA',
                return_eigenvectors=True
            )
        
        elapsed = time.time() - start_time
        
        print(f"[OK] Diagonalization complete in {elapsed:.2f} seconds")
        print(f"\nLowest {len(eigenvalues)} eigenvalues:")
        print("-"*80)
        
        for i, E in enumerate(eigenvalues):
            print(f"  E[{i:2d}] = {E:16.10f}")
        
        print("-"*80)
        
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        
        return eigenvalues, eigenvectors
    
    def identify_state(self, target_state: tuple) -> dict:
        """
        Identify which eigenvector corresponds to a specific quantum state.
        
        Method: Find eigenvector with maximum overlap with target state.
        
        Parameters:
        -----------
        target_state : tuple
            Quantum numbers (n, l, m)
        
        Returns:
        --------
        result : dict
            Eigenvalue, eigenvector index, overlap
        """
        if target_state not in self.lattice.node_index:
            return {
                'found': False,
                'state': target_state,
                'eigenvalue': np.nan,
                'index': -1,
                'overlap': 0.0
            }
        
        # Get index of target state in lattice
        state_idx = self.lattice.node_index[target_state]
        
        # Find eigenvector with largest component at this index
        overlaps = np.abs(self.eigenvectors[state_idx, :])
        best_eigenvector_idx = np.argmax(overlaps)
        max_overlap = overlaps[best_eigenvector_idx]
        
        eigenvalue = self.eigenvalues[best_eigenvector_idx]
        
        return {
            'found': True,
            'state': target_state,
            'eigenvalue': eigenvalue,
            'index': best_eigenvector_idx,
            'overlap': max_overlap,
            'eigenvector': self.eigenvectors[:, best_eigenvector_idx]
        }
    
    def compare_states(self, state1: tuple, state2: tuple) -> dict:
        """
        Compare eigenvalues of two quantum states.
        
        Parameters:
        -----------
        state1, state2 : tuple
            Quantum numbers (n, l, m)
        
        Returns:
        --------
        comparison : dict
            Results of comparison
        """
        print("\n" + "="*80)
        print("STATE COMPARISON: EIGENVALUE TEST")
        print("="*80)
        
        # Identify states
        print(f"\nIdentifying state {state1}...")
        result1 = self.identify_state(state1)
        
        print(f"Identifying state {state2}...")
        result2 = self.identify_state(state2)
        
        if not result1['found'] or not result2['found']:
            print("\nERROR: One or both states not found in lattice")
            return {}
        
        # Display results
        print("\n" + "-"*80)
        print("RESULTS:")
        print("-"*80)
        print(f"\nState 1: {result1['state']}")
        print(f"  Eigenvalue:      {result1['eigenvalue']:.15f}")
        print(f"  Eigenvector:     #{result1['index']}")
        print(f"  Overlap:         {result1['overlap']:.6f}")
        
        print(f"\nState 2: {result2['state']}")
        print(f"  Eigenvalue:      {result2['eigenvalue']:.15f}")
        print(f"  Eigenvector:     #{result2['index']}")
        print(f"  Overlap:         {result2['overlap']:.6f}")
        
        # Compute difference
        delta_E = result2['eigenvalue'] - result1['eigenvalue']
        
        print(f"\n" + "="*80)
        print("SPLITTING:")
        print("="*80)
        print(f"\nDelta E = E({result2['state']}) - E({result1['state']})")
        print(f"        = {result2['eigenvalue']:.15f} - {result1['eigenvalue']:.15f}")
        print(f"        = {delta_E:.15f}")
        print(f"\n|Delta E| = {abs(delta_E):.15e}")
        
        # Interpretation
        print("\n" + "="*80)
        print("INTERPRETATION:")
        print("="*80)
        
        if abs(delta_E) < 1e-10:
            print("\n** DEGENERACY PRESERVED **")
            print("   |Delta E| < 1e-10 (machine precision)")
            print("   The states are degenerate.")
            print("   The 'Lamb Shift' was a calculation artifact.")
            print("\n   VERDICT: Theory needs major revision.")
        elif abs(delta_E) < 1e-6:
            print("\n** WEAK SPLITTING **")
            print("   1e-10 < |Delta E| < 1e-6")
            print("   Splitting exists but is very small.")
            print("   May be numerical error or genuine effect.")
            print("\n   VERDICT: Inconclusive - needs higher precision.")
        elif abs(delta_E) < 0.1:
            print("\n** MODERATE SPLITTING **")
            print("   1e-6 < |Delta E| < 0.1")
            print("   Splitting is significant.")
            print("   This is a genuine eigenvalue difference.")
            print("\n   VERDICT: Effect is real but smaller than expected.")
        else:
            print("\n** STRONG SPLITTING **")
            print("   |Delta E| > 0.1")
            print("   Splitting is large and unambiguous.")
            print("   Degeneracy is definitively broken.")
            print("\n   VERDICT: Geometric Lamb Shift confirmed!")
        
        # Additional checks
        print("\n" + "="*80)
        print("VALIDATION CHECKS:")
        print("="*80)
        
        # Check if states are in same eigenvector (would indicate degeneracy)
        if result1['index'] == result2['index']:
            print("\n  WARNING: Both states have maximum overlap with SAME eigenvector!")
            print("           This suggests they are degenerate combinations.")
        else:
            print("\n  PASS: States correspond to different eigenvectors")
        
        # Check overlap quality
        if result1['overlap'] < 0.5:
            print(f"  WARNING: State 1 overlap is low ({result1['overlap']:.3f})")
            print("           State may not be well-localized")
        
        if result2['overlap'] < 0.5:
            print(f"  WARNING: State 2 overlap is low ({result2['overlap']:.3f})")
            print("           State may not be well-localized")
        
        comparison = {
            'state1': result1,
            'state2': result2,
            'delta_E': delta_E,
            'abs_delta_E': abs(delta_E),
            'degenerate': abs(delta_E) < 1e-10
        }
        
        return comparison
    
    def generate_report(self, comparison: dict, filename: str = "kill_switch_report.txt"):
        """
        Generate kill switch report.
        """
        print(f"\n\nGenerating report: {filename}")
        
        with open(filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PHYSICS KILL SWITCH: RIGOROUS EIGENVALUE TEST\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("OBJECTIVE:\n")
            f.write("Test if the 'Geometric Lamb Shift' is real by exact diagonalization.\n\n")
            
            f.write("METHOD:\n")
            f.write("1. Construct Hamiltonian H = T + V\n")
            f.write("2. Diagonalize using sparse eigenvalue solver\n")
            f.write("3. Identify eigenvectors for |2s> and |2p>\n")
            f.write("4. Compare TRUE eigenvalues\n\n")
            
            f.write("="*80 + "\n")
            f.write("LATTICE INFORMATION\n")
            f.write("="*80 + "\n\n")
            f.write(f"Maximum n:           {self.max_n}\n")
            f.write(f"Total nodes:         {len(self.lattice.nodes)}\n")
            f.write(f"Hamiltonian size:    {self.H.shape[0]} x {self.H.shape[1]}\n\n")
            
            f.write("="*80 + "\n")
            f.write("EIGENVALUE RESULTS\n")
            f.write("="*80 + "\n\n")
            
            if comparison:
                state1 = comparison['state1']
                state2 = comparison['state2']
                
                f.write(f"State 1: {state1['state']}\n")
                f.write(f"  Eigenvalue:      {state1['eigenvalue']:.15f}\n")
                f.write(f"  Eigenvector #:   {state1['index']}\n")
                f.write(f"  Overlap:         {state1['overlap']:.6f}\n\n")
                
                f.write(f"State 2: {state2['state']}\n")
                f.write(f"  Eigenvalue:      {state2['eigenvalue']:.15f}\n")
                f.write(f"  Eigenvector #:   {state2['index']}\n")
                f.write(f"  Overlap:         {state2['overlap']:.6f}\n\n")
                
                f.write("ENERGY SPLITTING:\n")
                f.write("-"*80 + "\n")
                f.write(f"Delta E = E({state2['state']}) - E({state1['state']})\n")
                f.write(f"        = {comparison['delta_E']:.15f}\n\n")
                f.write(f"|Delta E| = {comparison['abs_delta_E']:.15e}\n\n")
                
                f.write("="*80 + "\n")
                f.write("VERDICT\n")
                f.write("="*80 + "\n\n")
                
                if comparison['degenerate']:
                    f.write("** DEGENERACY PRESERVED **\n\n")
                    f.write("The eigenvalues are identical within machine precision.\n")
                    f.write("The 'Lamb Shift' observed in previous calculations was an artifact\n")
                    f.write("of measuring diagonal elements instead of true eigenvalues.\n\n")
                    f.write("CONCLUSION: The Geometric Lamb Shift hypothesis is REJECTED.\n")
                elif comparison['abs_delta_E'] > 0.1:
                    f.write("** DEGENERACY BROKEN **\n\n")
                    f.write("The eigenvalues are significantly different.\n")
                    f.write("This is a genuine splitting of the quantum states.\n\n")
                    f.write("CONCLUSION: The Geometric Lamb Shift hypothesis is CONFIRMED.\n")
                else:
                    f.write("** WEAK SPLITTING DETECTED **\n\n")
                    f.write("The eigenvalues differ but the splitting is small.\n")
                    f.write("Further investigation required.\n\n")
                    f.write("CONCLUSION: INCONCLUSIVE - needs higher precision analysis.\n")
        
        print(f"[OK] Report saved to {filename}")


def main():
    """
    Main execution: The Kill Switch Test.
    """
    print("\n" + "="*80)
    print("PHYSICS KILL SWITCH")
    print("Rigorous Test of Degeneracy Breaking")
    print("="*80 + "\n")
    
    print("This is the definitive test.")
    print("If the degeneracy is real, the theory survives.")
    print("If not, we abandon the 'Geometric Lamb Shift' claim.\n")
    
    # Initialize (reduced size for faster computation)
    test = KillSwitch(max_n=15)
    
    # Construct Hamiltonian
    print("\nStep 1: Constructing Hamiltonian...")
    H = test.construct_hamiltonian()
    
    # Diagonalize
    print("\nStep 2: Exact diagonalization...")
    eigenvalues, eigenvectors = test.diagonalize(k=20)
    
    # Compare states
    print("\nStep 3: Comparing quantum states...")
    comparison = test.compare_states(
        state1=(2, 0, 0),  # 2s
        state2=(2, 1, 0)   # 2p
    )
    
    # Generate report
    test.generate_report(comparison, "kill_switch_report.txt")
    
    # Final summary
    print("\n" + "="*80)
    print("KILL SWITCH TEST COMPLETE")
    print("="*80)
    
    if comparison:
        if comparison['degenerate']:
            print("\n** RESULT: DEGENERACY PRESERVED **")
            print("   The 'Geometric Lamb Shift' was an artifact.")
            print("   Theory requires major revision.")
        elif comparison['abs_delta_E'] > 0.1:
            print("\n** RESULT: DEGENERACY BROKEN **")
            print(f"   Splitting: {comparison['abs_delta_E']:.6f}")
            print("   The 'Geometric Lamb Shift' is REAL!")
        else:
            print("\n** RESULT: INCONCLUSIVE **")
            print(f"   Weak splitting: {comparison['abs_delta_E']:.6e}")
    
    print("\nResults saved to: kill_switch_report.txt")
    print("\n")


if __name__ == "__main__":
    main()
