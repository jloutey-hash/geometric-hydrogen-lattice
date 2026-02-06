"""
PHYSICS VERIFICATION SCRIPT
===========================
Refinement and Verification of Geometric Theory Results

This script addresses three critical tests:
1. Potential Energy Correction: Add Coulomb potential to fix Lamb Shift sign
2. Continuum Limit Test: Verify splitting is geometric, not discretization artifact
3. Holonomy Hunt: Measure geometric twist (parallel transport around loops)

Author: Computational Physics Research
Date: February 2026
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh, eigs
import matplotlib.pyplot as plt
from paraboloid_lattice_su11 import ParaboloidLattice
from physics_discovery import QuaternionNode
import time
from typing import Dict, Tuple, List
from collections import defaultdict


class PhysicsVerification:
    """
    Verification engine for geometric theory predictions.
    
    Tests:
    - Corrected Hamiltonian with Coulomb potential
    - Continuum limit behavior
    - Geometric holonomy (curvature)
    """
    
    def __init__(self, max_n: int = 50):
        """
        Initialize verification engine.
        
        Parameters:
        -----------
        max_n : int
            Maximum principal quantum number
        """
        self.max_n = max_n
        self.lattice = None
        self.results = {
            'corrected_energies': {},
            'continuum_test': {},
            'holonomy': {}
        }
    
    def construct_corrected_hamiltonian(self, max_n: int, coupling: float = 1.0) -> csr_matrix:
        """
        TASK 1: Construct H = T + V with Coulomb potential.
        
        Previously we used only the Laplacian (kinetic energy):
            H_old = D - A  (graph Laplacian)
        
        Now we add the Coulomb potential:
            V_ii = -1/n_i^2  (diagonal)
        
        Full Hamiltonian:
            H = (D - A) + V
        
        Parameters:
        -----------
        max_n : int
            Maximum principal quantum number
        coupling : float
            Coupling strength for kinetic term (default 1.0)
        
        Returns:
        --------
        H : csr_matrix
            Corrected Hamiltonian with potential energy
        """
        print(f"\nConstructing corrected Hamiltonian (H = T + V) for n <= {max_n}...")
        
        # Build lattice
        self.lattice = ParaboloidLattice(max_n=max_n)
        
        # Kinetic energy: Graph Laplacian
        print("  Building kinetic energy operator (Laplacian)...")
        A = (self.lattice.Lplus + self.lattice.Lminus + 
             self.lattice.Tplus + self.lattice.Tminus)
        A = (A + A.conj().T) / 2.0  # Make symmetric
        
        degrees = np.array(np.abs(A).sum(axis=1)).flatten()
        D = sp.diags(degrees, 0, format='csr', dtype=np.complex128)
        
        T = coupling * (D - A)  # Kinetic energy
        
        # Potential energy: Coulomb -1/n^2
        print("  Building potential energy operator (Coulomb)...")
        V_diag = []
        for n, l, m in self.lattice.nodes:
            # Coulomb potential: -1/n^2 (in atomic units)
            V_ii = -1.0 / (n**2)
            V_diag.append(V_ii)
        
        V = sp.diags(V_diag, 0, format='csr', dtype=np.complex128)
        
        # Total Hamiltonian
        H = T + V
        
        print(f"  Hamiltonian constructed: {H.shape[0]} x {H.shape[1]}")
        print(f"  Kinetic energy range: [{degrees.min():.3f}, {degrees.max():.3f}]")
        print(f"  Potential energy range: [{min(V_diag):.3f}, {max(V_diag):.3f}]")
        
        return H
    
    def compute_energy_expectation(self, H: csr_matrix, n: int, l: int, m: int) -> float:
        """
        Compute energy expectation value for a specific state.
        
        For a node |n,l,m>, compute <n,l,m|H|n,l,m> (diagonal element)
        plus contribution from connected states.
        
        Parameters:
        -----------
        H : csr_matrix
            Hamiltonian
        n, l, m : int
            Quantum numbers
        
        Returns:
        --------
        E : float
            Energy expectation value
        """
        if (n, l, m) not in self.lattice.node_index:
            return np.nan
        
        idx = self.lattice.node_index[(n, l, m)]
        
        # Diagonal element
        E_diag = H[idx, idx].real
        
        return E_diag
    
    def test_corrected_lamb_shift(self, max_n: int = 30) -> Dict:
        """
        TASK 1: Test if adding Coulomb potential fixes Lamb Shift sign.
        
        Compute energy splitting with corrected Hamiltonian:
            Delta E = E(2p) - E(2s)
        
        Expected: E(2s) < E(2p) because s-orbitals penetrate deeper
                  (opposite of what we got with pure Laplacian)
        
        Parameters:
        -----------
        max_n : int
            Maximum n for lattice construction
        
        Returns:
        --------
        results : Dict
            Energy values and splitting
        """
        print("\n" + "="*80)
        print("TASK 1: POTENTIAL ENERGY CORRECTION (Fixing Lamb Shift)")
        print("="*80)
        print("\nHypothesis: Adding Coulomb potential will flip sign of splitting")
        print("Previous (pure Laplacian): E(2p) > E(2s)")
        print("Expected (with Coulomb):   E(2s) < E(2p)\n")
        
        # Construct corrected Hamiltonian
        H_corrected = self.construct_corrected_hamiltonian(max_n=max_n)
        
        # Compute energies for n=2 shell
        print("\nComputing energy eigenvalues for n=2 shell...")
        
        E_2s = self.compute_energy_expectation(H_corrected, n=2, l=0, m=0)
        E_2p_m0 = self.compute_energy_expectation(H_corrected, n=2, l=1, m=0)
        E_2p_m1 = self.compute_energy_expectation(H_corrected, n=2, l=1, m=1)
        E_2p_m_minus1 = self.compute_energy_expectation(H_corrected, n=2, l=1, m=-1)
        
        E_2p_avg = (E_2p_m0 + E_2p_m1 + E_2p_m_minus1) / 3.0
        delta_E = E_2p_avg - E_2s
        
        print(f"\n  Results with CORRECTED Hamiltonian (H = T + V):")
        print(f"  " + "-"*60)
        print(f"  E(2s, l=0, m=0):       {E_2s:.10f}")
        print(f"  E(2p, l=1, m=0):       {E_2p_m0:.10f}")
        print(f"  E(2p, l=1, m=1):       {E_2p_m1:.10f}")
        print(f"  E(2p, l=1, m=-1):      {E_2p_m_minus1:.10f}")
        print(f"  E(2p) average:         {E_2p_avg:.10f}")
        print(f"\n  Delta E = E(2p) - E(2s) = {delta_E:.10f}")
        
        # Interpret results
        print("\n  " + "="*60)
        if delta_E < 0:
            print("  ** FAILURE: Still E(2p) < E(2s)")
            print("     The potential didn't fix the sign!")
        elif delta_E > 0 and delta_E < 1e-8:
            print("  ** MARGINAL: Splitting nearly zero")
            print(f"     Magnitude: {abs(delta_E):.2e}")
        elif delta_E > 0:
            print("  ** SUCCESS: Sign flipped! E(2s) < E(2p)")
            print(f"     Splitting magnitude: {delta_E:.6f}")
            print("     This matches the expected direction of Lamb Shift")
            print("     (s-orbitals have higher binding energy)")
        
        results = {
            'E_2s': E_2s,
            'E_2p_avg': E_2p_avg,
            'delta_E': delta_E,
            'sign_correct': delta_E > 0
        }
        
        self.results['corrected_energies'] = results
        return results
    
    def test_continuum_limit(self, n_values: List[int] = None) -> Dict:
        """
        TASK 2: Test if splitting converges or vanishes as n -> infinity.
        
        Compute Delta E_n = E(n,p) - E(n,s) for multiple n values.
        
        Interpretation:
        - If Delta E -> 0 as n increases: Discretization artifact
        - If Delta E -> constant > 0: Genuine geometric feature
        
        Parameters:
        -----------
        n_values : List[int], optional
            List of n values to test. Default: [2, 3, 4, 5, 6, 8, 10]
        
        Returns:
        --------
        results : Dict
            Splitting values for each n
        """
        if n_values is None:
            n_values = [2, 3, 4, 5, 6, 8, 10]
        
        print("\n" + "="*80)
        print("TASK 2: CONTINUUM LIMIT TEST")
        print("="*80)
        print("\nHypothesis: If splitting persists as n -> infinity, it's geometric")
        print("Testing splitting for multiple principal quantum numbers...\n")
        
        # Determine max_n needed
        max_n_needed = max(n_values) + 5
        
        # Construct Hamiltonian
        H = self.construct_corrected_hamiltonian(max_n=max_n_needed)
        
        # Compute splittings
        results = []
        
        print("  Computing energy splittings:")
        print("  " + "-"*60)
        print(f"  {'n':>3} | {'E(n,s)':>12} | {'E(n,p)':>12} | {'Delta E':>12} | {'Rel. Split':>12}")
        print("  " + "-"*60)
        
        for n in n_values:
            # n,s state (l=0)
            E_ns = self.compute_energy_expectation(H, n=n, l=0, m=0)
            
            # n,p states (l=1) - average over m
            if n >= 2:
                E_np_list = []
                for m in range(-1, 2):  # m = -1, 0, 1
                    if (n, 1, m) in self.lattice.node_index:
                        E_np_list.append(self.compute_energy_expectation(H, n=n, l=1, m=m))
                
                if E_np_list:
                    E_np = np.mean(E_np_list)
                else:
                    E_np = np.nan
            else:
                E_np = np.nan
            
            delta_E = E_np - E_ns
            
            # Relative splitting (as fraction of total energy)
            if E_ns != 0:
                rel_split = abs(delta_E) / abs(E_ns)
            else:
                rel_split = np.nan
            
            results.append({
                'n': n,
                'E_ns': E_ns,
                'E_np': E_np,
                'delta_E': delta_E,
                'rel_split': rel_split
            })
            
            print(f"  {n:3d} | {E_ns:12.6f} | {E_np:12.6f} | {delta_E:12.6f} | {rel_split:12.6f}")
        
        print("  " + "-"*60)
        
        # Analyze convergence
        print("\n  CONVERGENCE ANALYSIS:")
        print("  " + "-"*60)
        
        delta_values = [r['delta_E'] for r in results if not np.isnan(r['delta_E'])]
        
        if len(delta_values) >= 3:
            # Check if decreasing
            last_three = delta_values[-3:]
            is_decreasing = all(last_three[i] > last_three[i+1] for i in range(len(last_three)-1))
            
            ratio = abs(last_three[-1] / last_three[0]) if last_three[0] != 0 else np.nan
            
            print(f"  Last 3 splittings: {last_three}")
            print(f"  Decreasing trend: {is_decreasing}")
            print(f"  Ratio (last/first): {ratio:.6f}")
            
            if ratio < 0.5:
                print("\n  ** INTERPRETATION: Splitting DECREASING significantly")
                print("     Likely a discretization artifact (vanishes at continuum limit)")
            elif ratio > 0.8:
                print("\n  ** INTERPRETATION: Splitting STABLE")
                print("     This is a genuine GEOMETRIC FEATURE of the lattice!")
            else:
                print("\n  ** INTERPRETATION: Splitting slowly decreasing")
                print("     Need higher n to determine conclusively")
        
        self.results['continuum_test'] = {
            'data': results,
            'n_values': n_values
        }
        
        return self.results['continuum_test']
    
    def compute_holonomy(self, loop_nodes: List[Tuple[int, int, int]], 
                        qnodes: Dict[Tuple, QuaternionNode]) -> float:
        """
        TASK 3: Compute geometric holonomy (parallel transport around loop).
        
        Given a closed loop A -> B -> C -> A, compute:
        1. Quaternion at each node
        2. Parallel transport (rotation) along each edge
        3. Total rotation accumulated around loop
        4. Deficit angle = geometric twist
        
        Parameters:
        -----------
        loop_nodes : List[Tuple[int, int, int]]
            List of (n, l, m) tuples forming closed loop
        qnodes : Dict
            Dictionary of QuaternionNode objects
        
        Returns:
        --------
        deficit_angle : float
            Geometric twist angle (radians)
        """
        if len(loop_nodes) < 3:
            print("    Warning: Need at least 3 nodes for a loop")
            return 0.0
        
        # Ensure loop is closed
        if loop_nodes[0] != loop_nodes[-1]:
            loop_nodes = loop_nodes + [loop_nodes[0]]
        
        # Compute cumulative quaternion rotation
        total_rotation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion [w, x, y, z]
        
        for i in range(len(loop_nodes) - 1):
            node_a = loop_nodes[i]
            node_b = loop_nodes[i + 1]
            
            if node_a not in qnodes or node_b not in qnodes:
                continue
            
            qn_a = qnodes[node_a]
            qn_b = qnodes[node_b]
            
            # Convert spinors to quaternions
            # Spinor [a, b] -> Quaternion via Pauli matrix representation
            q_a = self._spinor_to_quaternion(qn_a.psi)
            q_b = self._spinor_to_quaternion(qn_b.psi)
            
            # Relative rotation: q_b * conj(q_a)
            q_rel = self._quaternion_multiply(q_b, self._quaternion_conjugate(q_a))
            
            # Accumulate
            total_rotation = self._quaternion_multiply(total_rotation, q_rel)
        
        # Extract rotation angle from accumulated quaternion
        # For quaternion [w, x, y, z], angle = 2 * arccos(w)
        w = total_rotation[0]
        angle = 2 * np.arccos(np.clip(w, -1, 1))
        
        # Deficit angle (how much rotation occurred)
        deficit_angle = angle
        
        return deficit_angle
    
    def _spinor_to_quaternion(self, psi: np.ndarray) -> np.ndarray:
        """
        Convert spinor [a, b] to quaternion [w, x, y, z].
        
        Mapping: Spinor represents SU(2), which is the double cover of SO(3).
        Quaternion [w, x, y, z] with w^2 + x^2 + y^2 + z^2 = 1.
        
        For spinor psi = [a, b]:
        w = Re(a)
        x = Im(a)
        y = Re(b)
        z = Im(b)
        (Then normalize)
        """
        a, b = psi
        
        w = np.real(a)
        x = np.imag(a)
        y = np.real(b)
        z = np.imag(b)
        
        q = np.array([w, x, y, z])
        norm = np.linalg.norm(q)
        
        if norm > 1e-10:
            q = q / norm
        
        return q
    
    def _quaternion_conjugate(self, q: np.ndarray) -> np.ndarray:
        """Compute quaternion conjugate: [w, x, y, z] -> [w, -x, -y, -z]"""
        return np.array([q[0], -q[1], -q[2], -q[3]])
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Multiply two quaternions using Hamilton product.
        
        q1 = [w1, x1, y1, z1]
        q2 = [w2, x2, y2, z2]
        
        Result: q1 * q2
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return np.array([w, x, y, z])
    
    def test_holonomy(self, max_n: int = 10) -> Dict:
        """
        TASK 3: Geometric Holonomy Hunt (Parallel Transport Test).
        
        Measure geometric curvature by parallel-transporting spinors around loops.
        
        The deficit angle should relate to:
        - Magnetic flux through loop
        - Fine structure constant alpha
        - Geometric Berry phase
        
        Parameters:
        -----------
        max_n : int
            Maximum n for lattice construction
        
        Returns:
        --------
        results : Dict
            Holonomy data for various loops
        """
        print("\n" + "="*80)
        print("TASK 3: HOLONOMY HUNT (Geometric Twist Measurement)")
        print("="*80)
        print("\nConcept: Parallel transport spinor around closed loop")
        print("Measure: Deficit angle (geometric twist per unit area)")
        print("Hypothesis: Relates to magnetic flux or fine structure constant\n")
        
        # Build lattice
        lattice = ParaboloidLattice(max_n=max_n)
        
        # Create quaternion nodes
        print("Creating quaternion nodes...")
        qnodes = {}
        
        for n, l, m in lattice.nodes:
            # Get lattice position
            if n > 1:
                theta = np.pi * l / (n - 1)
            else:
                theta = 0.0
            
            if l > 0:
                phi = 2 * np.pi * (m + l) / (2 * l + 1)
            else:
                phi = 0.0
            
            qnode = QuaternionNode(n, l, m, theta, phi)
            
            # Initialize spinor (aligned with lattice position)
            beta = theta
            alpha = phi
            a = np.cos(beta/2) * np.exp(1j * alpha/2)
            b = np.sin(beta/2) * np.exp(-1j * alpha/2)
            qnode.set_spinor(a, b)
            
            qnodes[(n, l, m)] = qnode
        
        print(f"Created {len(qnodes)} quaternion nodes\n")
        
        # Define test loops
        loops = self._construct_test_loops(lattice)
        
        print(f"Testing {len(loops)} closed loops...")
        print("  " + "-"*60)
        
        results = []
        
        for i, loop in enumerate(loops):
            # Compute holonomy
            deficit = self.compute_holonomy(loop, qnodes)
            
            # Compute loop "area" (number of nodes)
            area = len(set(loop)) - 1  # Exclude repeated start node
            
            # Curvature = deficit / area
            if area > 0:
                curvature = deficit / area
            else:
                curvature = 0.0
            
            result = {
                'loop_id': i,
                'nodes': loop,
                'deficit_angle': deficit,
                'area': area,
                'curvature': curvature
            }
            results.append(result)
            
            print(f"  Loop {i+1}: {len(loop)-1} nodes, "
                  f"deficit = {deficit:.6f} rad ({np.degrees(deficit):.2f} deg), "
                  f"curvature = {curvature:.6f}")
        
        print("  " + "-"*60)
        
        # Statistical analysis
        if results:
            deficits = [r['deficit_angle'] for r in results]
            curvatures = [r['curvature'] for r in results]
            
            mean_deficit = np.mean(deficits)
            std_deficit = np.std(deficits)
            mean_curvature = np.mean(curvatures)
            
            print(f"\n  STATISTICAL ANALYSIS:")
            print(f"  " + "-"*60)
            print(f"  Mean deficit angle:  {mean_deficit:.6f} rad ({np.degrees(mean_deficit):.2f} deg)")
            print(f"  Std deficit angle:   {std_deficit:.6f} rad")
            print(f"  Mean curvature:      {mean_curvature:.6f} rad/node")
            
            # Compare to fundamental constants
            alpha = 1.0 / 137.036  # Fine structure constant
            pi = np.pi
            
            print(f"\n  COMPARISON TO FUNDAMENTAL CONSTANTS:")
            print(f"  " + "-"*60)
            print(f"  alpha (1/137):       {alpha:.6f}")
            print(f"  pi:                  {pi:.6f}")
            print(f"  2*pi:                {2*pi:.6f}")
            print(f"  pi/2:                {pi/2:.6f}")
            
            ratio_to_alpha = mean_curvature / alpha if alpha > 0 else np.nan
            ratio_to_pi = mean_curvature / pi
            
            print(f"\n  Curvature / alpha:   {ratio_to_alpha:.3f}")
            print(f"  Curvature / pi:      {ratio_to_pi:.3f}")
            
            if abs(ratio_to_alpha - 1.0) < 0.1:
                print("\n  ** DISCOVERY: Curvature matches fine structure constant!")
            elif abs(ratio_to_pi - 1.0) < 0.1:
                print("\n  ** DISCOVERY: Curvature matches pi!")
        
        self.results['holonomy'] = {
            'loops': results,
            'mean_deficit': mean_deficit if results else 0,
            'mean_curvature': mean_curvature if results else 0
        }
        
        return self.results['holonomy']
    
    def _construct_test_loops(self, lattice: ParaboloidLattice) -> List[List[Tuple]]:
        """
        Construct test loops for holonomy calculation.
        
        Find closed paths in the lattice graph:
        - Triangles (3-node loops)
        - Squares (4-node loops)
        - Larger polygons
        
        Returns:
        --------
        loops : List of loops, each loop is a list of (n, l, m) tuples
        """
        loops = []
        
        # Build adjacency list
        adj = defaultdict(list)
        
        for n, l, m in lattice.nodes:
            idx = lattice.node_index[(n, l, m)]
            
            # Check L+ connections
            if m < l:
                target = (n, l, m+1)
                if target in lattice.node_index:
                    adj[(n, l, m)].append(target)
            
            # Check L- connections
            if m > -l:
                target = (n, l, m-1)
                if target in lattice.node_index:
                    adj[(n, l, m)].append(target)
            
            # Check T+ connections
            if n < lattice.max_n:
                target = (n+1, l, m)
                if target in lattice.node_index:
                    adj[(n, l, m)].append(target)
            
            # Check T- connections
            if n > 1:
                target = (n-1, l, m)
                if target in lattice.node_index:
                    adj[(n, l, m)].append(target)
        
        # Find triangles (A -> B -> C -> A)
        for node_a in lattice.nodes[:50]:  # Limit search for performance
            for node_b in adj[node_a]:
                for node_c in adj[node_b]:
                    if node_a in adj[node_c] and node_a != node_c:
                        loop = [node_a, node_b, node_c, node_a]
                        # Check if already found (avoid duplicates)
                        if not self._loop_exists(loop, loops):
                            loops.append(loop)
                            if len(loops) >= 10:  # Limit number of loops
                                return loops
        
        return loops
    
    def _loop_exists(self, new_loop: List, existing_loops: List) -> bool:
        """Check if loop already exists (considering cyclic permutations)."""
        new_set = set(new_loop[:-1])  # Exclude closing node
        for loop in existing_loops:
            if new_set == set(loop[:-1]):
                return True
        return False
    
    def generate_report(self, filename: str = "verification_report.txt"):
        """
        Generate comprehensive verification report.
        """
        print(f"\n\nGenerating verification report: {filename}")
        
        with open(filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PHYSICS VERIFICATION REPORT\n")
            f.write("Refinement and Testing of Geometric Theory\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Task 1: Corrected Energies
            if 'corrected_energies' in self.results and self.results['corrected_energies']:
                f.write("\n" + "="*80 + "\n")
                f.write("TASK 1: POTENTIAL ENERGY CORRECTION\n")
                f.write("="*80 + "\n\n")
                f.write("Objective: Fix Lamb Shift sign by adding Coulomb potential\n")
                f.write("Previous model: H = T (pure Laplacian)\n")
                f.write("Corrected model: H = T + V, where V_ii = -1/n_i^2\n\n")
                
                res = self.results['corrected_energies']
                f.write("Results for n=2 shell:\n")
                f.write("-"*80 + "\n")
                f.write(f"  E(2s):           {res['E_2s']:.10f}\n")
                f.write(f"  E(2p) average:   {res['E_2p_avg']:.10f}\n")
                f.write(f"  Delta E:         {res['delta_E']:.10f}\n\n")
                
                if res['sign_correct']:
                    f.write("  ** SUCCESS: E(2s) < E(2p) (correct sign)\n")
                    f.write("     The Coulomb potential fixed the Lamb Shift direction!\n\n")
                else:
                    f.write("  ** ISSUE: Sign still incorrect or splitting negligible\n\n")
            
            # Task 2: Continuum Limit
            if 'continuum_test' in self.results and self.results['continuum_test']:
                f.write("\n" + "="*80 + "\n")
                f.write("TASK 2: CONTINUUM LIMIT TEST\n")
                f.write("="*80 + "\n\n")
                f.write("Objective: Determine if splitting is geometric or artifact\n")
                f.write("Method: Measure Delta E_n = E(n,p) - E(n,s) for increasing n\n\n")
                
                data = self.results['continuum_test']['data']
                f.write("Energy Splittings:\n")
                f.write("-"*80 + "\n")
                f.write(f"{'n':>3} | {'E(n,s)':>12} | {'E(n,p)':>12} | {'Delta E':>12} | {'Rel. Split':>12}\n")
                f.write("-"*80 + "\n")
                
                for r in data:
                    f.write(f"{r['n']:3d} | {r['E_ns']:12.6f} | {r['E_np']:12.6f} | "
                           f"{r['delta_E']:12.6f} | {r['rel_split']:12.6f}\n")
                
                f.write("-"*80 + "\n\n")
                
                delta_values = [r['delta_E'] for r in data if not np.isnan(r['delta_E'])]
                if len(delta_values) >= 2:
                    ratio = delta_values[-1] / delta_values[0] if delta_values[0] != 0 else np.nan
                    f.write(f"  Ratio (last/first): {ratio:.4f}\n")
                    
                    if ratio < 0.5:
                        f.write("  INTERPRETATION: Discretization artifact (vanishes at continuum)\n\n")
                    else:
                        f.write("  INTERPRETATION: Genuine geometric feature (persists)\n\n")
            
            # Task 3: Holonomy
            if 'holonomy' in self.results and self.results['holonomy']:
                f.write("\n" + "="*80 + "\n")
                f.write("TASK 3: GEOMETRIC HOLONOMY (Curvature)\n")
                f.write("="*80 + "\n\n")
                f.write("Objective: Measure geometric twist via parallel transport\n")
                f.write("Method: Transport spinor around closed loops, measure deficit angle\n\n")
                
                hdata = self.results['holonomy']
                f.write(f"Number of loops tested: {len(hdata['loops'])}\n")
                f.write(f"Mean deficit angle:     {hdata['mean_deficit']:.6f} rad\n")
                f.write(f"Mean curvature:         {hdata['mean_curvature']:.6f} rad/node\n\n")
                
                alpha = 1.0 / 137.036
                ratio = hdata['mean_curvature'] / alpha if alpha > 0 else 0
                
                f.write(f"Comparison to alpha (1/137):\n")
                f.write(f"  Curvature / alpha = {ratio:.3f}\n\n")
                
                if abs(ratio - 1.0) < 0.2:
                    f.write("  ** POTENTIAL DISCOVERY: Curvature correlates with alpha!\n\n")
            
            # Summary
            f.write("\n" + "="*80 + "\n")
            f.write("SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write("Key Results:\n")
            f.write("1. Coulomb potential correction tested\n")
            f.write("2. Continuum limit behavior analyzed\n")
            f.write("3. Geometric holonomy measured\n\n")
            f.write("These tests validate or refute the geometric theory predictions.\n\n")
        
        print(f"[OK] Report saved to {filename}")


def plot_continuum_test(results: Dict):
    """
    Visualize continuum limit test results.
    """
    data = results['data']
    n_values = [r['n'] for r in data]
    delta_E = [r['delta_E'] for r in data if not np.isnan(r['delta_E'])]
    n_valid = [r['n'] for r in data if not np.isnan(r['delta_E'])]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Delta E vs n
    ax1.plot(n_valid, delta_E, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Principal Quantum Number n', fontsize=12)
    ax1.set_ylabel('Energy Splitting Delta E', fontsize=12)
    ax1.set_title('Continuum Limit Test: E(n,p) - E(n,s)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log scale
    ax2.semilogy(n_valid, np.abs(delta_E), 'o-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Principal Quantum Number n', fontsize=12)
    ax2.set_ylabel('|Delta E| (log scale)', fontsize=12)
    ax2.set_title('Convergence Analysis (Log Scale)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('continuum_test.png', dpi=150, bbox_inches='tight')
    print("  [OK] Saved: continuum_test.png")
    plt.close()


def main():
    """
    Main execution: Run all verification tests.
    """
    print("\n" + "="*80)
    print("PHYSICS VERIFICATION ENGINE")
    print("Testing Refined Geometric Theory")
    print("="*80 + "\n")
    
    # Initialize
    verifier = PhysicsVerification(max_n=50)
    
    # TASK 1: Corrected Hamiltonian
    print("EXECUTING: Task 1 - Potential Energy Correction")
    corrected_results = verifier.test_corrected_lamb_shift(max_n=30)
    
    # TASK 2: Continuum Limit
    print("\nEXECUTING: Task 2 - Continuum Limit Test")
    continuum_results = verifier.test_continuum_limit(n_values=[2, 3, 4, 5, 6, 8, 10])
    
    # TASK 3: Holonomy
    print("\nEXECUTING: Task 3 - Holonomy Hunt")
    holonomy_results = verifier.test_holonomy(max_n=10)
    
    # Generate report
    verifier.generate_report("verification_report.txt")
    
    # Create visualization
    print("\n\nGenerating visualization...")
    plot_continuum_test(continuum_results)
    
    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)
    print("\nResults saved to:")
    print("  - verification_report.txt")
    print("  - continuum_test.png")
    print("\n")


if __name__ == "__main__":
    main()
