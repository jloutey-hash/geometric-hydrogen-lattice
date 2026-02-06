"""
GEOMETRIC HOLONOMY HUNTER
=========================
Corrected Berry Phase Measurement on Paraboloid Lattice

Key Insight: The lattice has "Manhattan" connectivity (Radial + Angular).
Fundamental loops are SQUARE PLAQUETTES, not triangles.

Algorithm:
1. Construct square loops: n,l,m -> T+ -> L+ -> T- -> L- -> back
2. Parallel transport spinor around loop
3. Measure geometric twist (Berry phase)
4. Test if twist density relates to fine structure constant alpha

Author: Geometric Physics Research
Date: February 2026
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from paraboloid_lattice_su11 import ParaboloidLattice
from physics_discovery import QuaternionNode
import time
from typing import List, Tuple, Dict, Optional


class PlaquetteHolonomy:
    """
    Geometric curvature measurement via square plaquettes.
    
    The lattice has Manhattan connectivity:
    - Radial: T+, T- (change n)
    - Angular: L+, L- (change m)
    
    Elementary loops are 4-node squares:
    |n,l,m> --T+--> |n+1,l,m> --L+--> |n+1,l,m+1> --T---> |n,l,m+1> --L---> |n,l,m>
    """
    
    def __init__(self, max_n: int = 20):
        """
        Initialize holonomy calculator.
        
        Parameters:
        -----------
        max_n : int
            Maximum principal quantum number
        """
        self.max_n = max_n
        self.lattice = None
        self.qnodes = {}
        self.plaquettes = []
        self.results = {
            'plaquettes': [],
            'statistics': {},
            'alpha_comparison': {}
        }
    
    def construct_plaquettes(self) -> List[Dict]:
        """
        TASK 1: Construct all valid square plaquettes.
        
        A plaquette is valid if all 4 nodes exist and form a closed loop:
        
        Start: |n, l, m>
        Step 1 (Up):    T+ -> |n+1, l, m>
        Step 2 (Right): L+ -> |n+1, l, m+1>
        Step 3 (Down):  T- -> |n, l, m+1>
        Step 4 (Left):  L- -> |n, l, m>  (back to start)
        
        Returns:
        --------
        plaquettes : List[Dict]
            List of valid plaquette structures
        """
        print("\n" + "="*80)
        print("TASK 1: PLAQUETTE SCANNER")
        print("="*80)
        print("\nSearching for square plaquettes in lattice...")
        print("Pattern: n,l,m -> T+ -> L+ -> T- -> L- -> back\n")
        
        # Build lattice
        self.lattice = ParaboloidLattice(max_n=self.max_n)
        print(f"Lattice constructed: {len(self.lattice.nodes)} nodes for n <= {self.max_n}")
        
        plaquettes = []
        
        # Scan all possible starting nodes
        for n, l, m in self.lattice.nodes:
            # Check if we can build a plaquette starting here
            
            # Node A: Start |n, l, m>
            node_a = (n, l, m)
            if node_a not in self.lattice.node_index:
                continue
            
            # Node B: Apply T+ -> |n+1, l, m>
            node_b = (n+1, l, m)
            if node_b not in self.lattice.node_index:
                continue
            
            # Node C: Apply L+ -> |n+1, l, m+1>
            # Check angular momentum constraint: m+1 <= l
            if m+1 > l:
                continue
            node_c = (n+1, l, m+1)
            if node_c not in self.lattice.node_index:
                continue
            
            # Node D: Apply T- -> |n, l, m+1>
            node_d = (n, l, m+1)
            if node_d not in self.lattice.node_index:
                continue
            
            # Verify closure: L- from D should return to A
            # L- changes m -> m-1, so D(m+1) -> A(m) ✓
            
            # Get connection weights from operators
            idx_a = self.lattice.node_index[node_a]
            idx_b = self.lattice.node_index[node_b]
            idx_c = self.lattice.node_index[node_c]
            idx_d = self.lattice.node_index[node_d]
            
            # Extract operator matrix elements
            weight_AB = self._get_operator_weight(self.lattice.Tplus, idx_a, idx_b)
            weight_BC = self._get_operator_weight(self.lattice.Lplus, idx_b, idx_c)
            weight_CD = self._get_operator_weight(self.lattice.Tminus, idx_c, idx_d)
            weight_DA = self._get_operator_weight(self.lattice.Lminus, idx_d, idx_a)
            
            # Only include if all connections exist
            if all([abs(w) > 1e-10 for w in [weight_AB, weight_BC, weight_CD, weight_DA]]):
                plaquette = {
                    'nodes': [node_a, node_b, node_c, node_d],
                    'path': 'T+ -> L+ -> T- -> L-',
                    'weights': [weight_AB, weight_BC, weight_CD, weight_DA],
                    'center_n': n + 0.5,
                    'center_l': l,
                    'center_m': m + 0.5
                }
                plaquettes.append(plaquette)
        
        self.plaquettes = plaquettes
        
        print(f"\n[OK] Found {len(plaquettes)} valid square plaquettes")
        
        if len(plaquettes) > 0:
            print(f"\nExample plaquette:")
            p = plaquettes[0]
            print(f"  Nodes: {p['nodes']}")
            print(f"  Weights: {[abs(w) for w in p['weights']]}")
        
        return plaquettes
    
    def _get_operator_weight(self, operator: csr_matrix, idx_from: int, idx_to: int) -> complex:
        """
        Extract matrix element <idx_to|Op|idx_from>.
        """
        return operator[idx_to, idx_from]
    
    def initialize_spinors(self):
        """
        Initialize quaternion nodes with spinors.
        
        Default: Spinors aligned with local coordinate axes.
        """
        print("\nInitializing spinor field on lattice...")
        
        for n, l, m in self.lattice.nodes:
            # Compute spherical angles
            if n > 1:
                theta = np.pi * l / (n - 1)
            else:
                theta = 0.0
            
            if l > 0:
                phi = 2 * np.pi * (m + l) / (2 * l + 1)
            else:
                phi = 0.0
            
            # Create quaternion node
            qnode = QuaternionNode(n, l, m, theta, phi)
            
            # Initialize spinor: aligned with position
            # Spinor in |theta, phi> direction
            beta = theta
            alpha = phi
            a = np.cos(beta/2) * np.exp(1j * alpha/2)
            b = np.sin(beta/2) * np.exp(-1j * alpha/2)
            qnode.set_spinor(a, b)
            
            self.qnodes[(n, l, m)] = qnode
        
        print(f"[OK] Initialized {len(self.qnodes)} spinor nodes")
    
    def parallel_transport_spinor(self, psi_start: np.ndarray, 
                                  node_from: Tuple, node_to: Tuple,
                                  weight: complex) -> np.ndarray:
        """
        TASK 2: Parallel transport spinor along connection.
        
        IMPROVED CONNECTION: Use geometric structure of paraboloid lattice.
        
        The connection 1-form should encode the curvature of the paraboloid.
        We compute the angle change in the coordinate system when moving
        between nodes.
        
        Parameters:
        -----------
        psi_start : np.ndarray
            Initial spinor [a, b]
        node_from : Tuple
            Starting node (n, l, m)
        node_to : Tuple
            Target node (n, l, m)
        weight : complex
            Connection weight from operator
        
        Returns:
        --------
        psi_transported : np.ndarray
            Transported spinor
        """
        # Get coordinate differences
        n1, l1, m1 = node_from
        n2, l2, m2 = node_to
        
        dn = n2 - n1
        dl = l2 - l1
        dm = m2 - m1
        
        # Get quaternion nodes for geometric information
        qnode_from = self.qnodes[node_from]
        qnode_to = self.qnodes[node_to]
        
        # Compute coordinate angles at each node
        theta1 = qnode_from.theta_lattice
        phi1 = qnode_from.phi_lattice
        theta2 = qnode_to.theta_lattice
        phi2 = qnode_to.phi_lattice
        
        # Compute connection based on move type
        if dm != 0:  # Angular step (L+/L-)
            # Moving in azimuthal direction (phi changes)
            # Connection = Berry connection for rotation about z-axis
            
            # Geometric phase from azimuthal transport at angle theta
            # A_phi = (1/2)(1 - cos(theta)) for monopole connection
            # Phase accumulated = A_phi * dphi
            
            dphi = phi2 - phi1
            if abs(dphi) < 1e-10:  # Avoid numerical issues
                dphi = 2 * np.pi * dm / (2 * l1 + 1) if l1 > 0 else 0
            
            # Berry connection for azimuthal transport
            A_phi = 0.5 * (1 - np.cos(theta1))
            geometric_phase = A_phi * dphi
            
            # Apply SU(2) rotation
            # For spinor, phase -> phase/2 (spin-1/2)
            psi_transported = psi_start * np.exp(1j * geometric_phase / 2)
        
        elif dn != 0:  # Radial step (T+/T-)
            # Moving in radial direction (n changes, theta may change)
            # Connection depends on how polar angle changes
            
            dtheta = theta2 - theta1
            
            # For radial transport, we rotate the spinor by half the polar angle change
            # This is the geometric connection for moving along meridian
            
            # Rotation about tangent axis
            # Use weight magnitude as coupling
            weight_mag = abs(weight)
            
            # Geometric phase accumulation
            if abs(dtheta) > 1e-10:
                # Rotation matrix for spinor in polar direction
                # This is a Wigner rotation
                alpha = dtheta / 2
                
                # Apply rotation: mix up and down components
                cos_a = np.cos(alpha)
                sin_a = np.sin(alpha)
                
                a_new = cos_a * psi_start[0] - sin_a * psi_start[1]
                b_new = sin_a * psi_start[0] + cos_a * psi_start[1]
                
                psi_transported = np.array([a_new, b_new])
            else:
                # Theta constant: pure radial motion
                # Add phase from radial connection
                # A_r = (n-dependent phase)
                radial_phase = np.pi * (n2 - n1) / (n1 + n2)
                psi_transported = psi_start * np.exp(1j * radial_phase / 2)
        
        else:
            # No transport
            psi_transported = psi_start.copy()
        
        # Normalize
        norm = np.linalg.norm(psi_transported)
        if norm > 1e-10:
            psi_transported = psi_transported / norm
        
        return psi_transported
    
    def compute_plaquette_holonomy(self, plaquette: Dict) -> Dict:
        """
        TASK 2: Compute geometric twist for a plaquette.
        
        Algorithm:
        1. Start with spinor at node A
        2. Transport around loop: A -> B -> C -> D -> A
        3. Compare final spinor to initial spinor
        4. Deficit angle = phase difference
        
        Parameters:
        -----------
        plaquette : Dict
            Plaquette structure with nodes and weights
        
        Returns:
        --------
        result : Dict
            Holonomy data (deficit angle, area, etc.)
        """
        nodes = plaquette['nodes']
        weights = plaquette['weights']
        
        # Get initial spinor at node A
        node_a = nodes[0]
        psi_initial = self.qnodes[node_a].psi.copy()
        
        # Parallel transport around loop
        psi_current = psi_initial.copy()
        
        for i in range(4):
            node_from = nodes[i]
            node_to = nodes[(i+1) % 4]
            weight = weights[i]
            
            # Transport
            psi_current = self.parallel_transport_spinor(
                psi_current, node_from, node_to, weight
            )
        
        # Final spinor after loop
        psi_final = psi_current
        
        # Compute deficit angle (phase difference)
        # For spinors: phase = arg(psi_final^dagger * psi_initial)
        overlap = np.vdot(psi_final, psi_initial)
        deficit_angle = np.angle(overlap)
        
        # Berry phase = accumulated geometric phase
        berry_phase = deficit_angle
        
        # Compute plaquette "area" (product of link amplitudes)
        area_product = np.prod([abs(w) for w in weights])
        area_sum = np.sum([abs(w) for w in weights])
        
        # Average area measure
        area = area_product ** 0.25  # Geometric mean
        
        result = {
            'nodes': nodes,
            'deficit_angle': deficit_angle,
            'berry_phase': berry_phase,
            'area_product': area_product,
            'area_sum': area_sum,
            'area': area,
            'curvature': berry_phase / area if area > 0 else 0
        }
        
        return result
    
    def measure_all_holonomies(self) -> List[Dict]:
        """
        Compute holonomy for all plaquettes.
        
        Returns:
        --------
        results : List[Dict]
            Holonomy data for each plaquette
        """
        print("\n" + "="*80)
        print("TASK 2: GEOMETRIC TWIST MEASUREMENT")
        print("="*80)
        print("\nComputing Berry phase for all plaquettes...\n")
        
        results = []
        
        for i, plaquette in enumerate(self.plaquettes):
            result = self.compute_plaquette_holonomy(plaquette)
            results.append(result)
            
            if i < 5 or i % 100 == 0:  # Show first few and periodic updates
                print(f"  Plaquette {i+1:4d}: "
                      f"Berry phase = {result['berry_phase']:8.5f} rad, "
                      f"Area = {result['area']:8.5f}, "
                      f"Curvature = {result['curvature']:8.5f}")
        
        print(f"\n[OK] Computed holonomy for {len(results)} plaquettes")
        
        self.results['plaquettes'] = results
        return results
    
    def analyze_alpha_connection(self) -> Dict:
        """
        TASK 3: Search for Fine Structure Constant.
        
        Hypothesis: Twist density relates to alpha ~ 1/137.
        
        Test ratios:
        - Mean curvature / alpha
        - Mean Berry phase / alpha
        - Scaled geometric invariants
        
        Returns:
        --------
        analysis : Dict
            Statistical analysis and alpha comparison
        """
        print("\n" + "="*80)
        print("TASK 3: THE SEARCH FOR ALPHA")
        print("="*80)
        print("\nHypothesis: Geometric twist density ~ Fine Structure Constant\n")
        
        results = self.results['plaquettes']
        
        if not results:
            print("ERROR: No plaquettes to analyze")
            return {}
        
        # Extract data
        berry_phases = [r['berry_phase'] for r in results]
        curvatures = [r['curvature'] for r in results]
        areas = [r['area'] for r in results]
        
        # Statistics
        mean_berry = np.mean(berry_phases)
        std_berry = np.std(berry_phases)
        mean_curv = np.mean(curvatures)
        std_curv = np.std(curvatures)
        mean_area = np.mean(areas)
        
        print("STATISTICAL SUMMARY:")
        print("-"*80)
        print(f"  Number of plaquettes:     {len(results)}")
        print(f"  Mean Berry phase:         {mean_berry:.8f} rad")
        print(f"  Std Berry phase:          {std_berry:.8f} rad")
        print(f"  Mean curvature:           {mean_curv:.8f}")
        print(f"  Std curvature:            {std_curv:.8f}")
        print(f"  Mean area:                {mean_area:.8f}")
        
        # Fundamental constants
        alpha = 1.0 / 137.035999084  # Fine structure constant
        pi = np.pi
        
        print(f"\n\nFUNDAMENTAL CONSTANTS:")
        print("-"*80)
        print(f"  alpha (1/137):            {alpha:.10f}")
        print(f"  pi:                       {pi:.10f}")
        print(f"  2*pi:                     {2*pi:.10f}")
        print(f"  pi/2:                     {pi/2:.10f}")
        print(f"  sqrt(alpha):              {np.sqrt(alpha):.10f}")
        
        # Compute ratios
        ratio_berry_alpha = mean_berry / alpha
        ratio_curv_alpha = mean_curv / alpha
        ratio_berry_pi = mean_berry / pi
        ratio_curv_pi = mean_curv / pi
        
        print(f"\n\nRATIO ANALYSIS:")
        print("-"*80)
        print(f"  Mean Berry / alpha:       {ratio_berry_alpha:.6f}")
        print(f"  Mean Curvature / alpha:   {ratio_curv_alpha:.6f}")
        print(f"  Mean Berry / pi:          {ratio_berry_pi:.6f}")
        print(f"  Mean Curvature / pi:      {ratio_curv_pi:.6f}")
        
        # Test if any ratio is close to 1, 2, or simple fraction
        print(f"\n\nCOINCIDENCE TESTS:")
        print("-"*80)
        
        tests = [
            ('Berry / alpha', ratio_berry_alpha, 1.0),
            ('Curvature / alpha', ratio_curv_alpha, 1.0),
            ('Berry / (2*alpha)', mean_berry / (2*alpha), 1.0),
            ('Curvature * 137', mean_curv * 137, 1.0),
            ('Berry / pi', ratio_berry_pi, 1.0),
            ('Curvature / (pi/2)', mean_curv / (pi/2), 1.0),
        ]
        
        for name, ratio, target in tests:
            deviation = abs(ratio - target) / target * 100
            status = "***MATCH***" if deviation < 10 else ""
            print(f"  {name:25s} = {ratio:10.6f}  (dev: {deviation:6.2f}%)  {status}")
        
        # Advanced tests
        print(f"\n\nADVANCED CORRELATIONS:")
        print("-"*80)
        
        # Test: Curvature * Area ~ constant
        curv_area_products = [r['curvature'] * r['area'] for r in results]
        mean_curv_area = np.mean(curv_area_products)
        print(f"  Mean(Curvature * Area):   {mean_curv_area:.8f}")
        print(f"  Ratio to alpha:           {mean_curv_area/alpha:.6f}")
        
        # Test: Berry phase independence
        correlation = np.corrcoef(berry_phases, areas)[0, 1]
        print(f"  Correlation(Berry, Area): {correlation:.6f}")
        
        analysis = {
            'mean_berry_phase': mean_berry,
            'mean_curvature': mean_curv,
            'ratio_to_alpha': ratio_curv_alpha,
            'alpha': alpha,
            'tests': tests,
            'correlation': correlation
        }
        
        self.results['alpha_comparison'] = analysis
        
        return analysis
    
    def generate_report(self, filename: str = "holonomy_report.txt"):
        """
        Generate comprehensive holonomy report.
        """
        print(f"\n\nGenerating holonomy report: {filename}")
        
        with open(filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GEOMETRIC HOLONOMY REPORT\n")
            f.write("Berry Phase Measurement on Paraboloid Lattice\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Plaquette statistics
            f.write("\n" + "="*80 + "\n")
            f.write("TASK 1: PLAQUETTE CONSTRUCTION\n")
            f.write("="*80 + "\n\n")
            f.write(f"Lattice size:            n <= {self.max_n}\n")
            f.write(f"Total nodes:             {len(self.lattice.nodes)}\n")
            f.write(f"Square plaquettes found: {len(self.plaquettes)}\n\n")
            
            if self.plaquettes:
                f.write("Example plaquette:\n")
                p = self.plaquettes[0]
                f.write(f"  Path: {p['nodes'][0]} -> {p['nodes'][1]} -> {p['nodes'][2]} -> {p['nodes'][3]} -> {p['nodes'][0]}\n")
                f.write(f"  Operators: {p['path']}\n\n")
            
            # Berry phase results
            if 'plaquettes' in self.results and self.results['plaquettes']:
                f.write("\n" + "="*80 + "\n")
                f.write("TASK 2: BERRY PHASE MEASUREMENTS\n")
                f.write("="*80 + "\n\n")
                
                results = self.results['plaquettes']
                berry_phases = [r['berry_phase'] for r in results]
                curvatures = [r['curvature'] for r in results]
                
                f.write(f"Number of measurements:  {len(results)}\n\n")
                f.write(f"Berry Phase Statistics:\n")
                f.write(f"  Mean:                  {np.mean(berry_phases):.10f} rad\n")
                f.write(f"  Std:                   {np.std(berry_phases):.10f} rad\n")
                f.write(f"  Min:                   {np.min(berry_phases):.10f} rad\n")
                f.write(f"  Max:                   {np.max(berry_phases):.10f} rad\n\n")
                
                f.write(f"Curvature Statistics:\n")
                f.write(f"  Mean:                  {np.mean(curvatures):.10f}\n")
                f.write(f"  Std:                   {np.std(curvatures):.10f}\n")
                f.write(f"  Min:                   {np.min(curvatures):.10f}\n")
                f.write(f"  Max:                   {np.max(curvatures):.10f}\n\n")
            
            # Alpha connection
            if 'alpha_comparison' in self.results and self.results['alpha_comparison']:
                f.write("\n" + "="*80 + "\n")
                f.write("TASK 3: FINE STRUCTURE CONSTANT CONNECTION\n")
                f.write("="*80 + "\n\n")
                
                ana = self.results['alpha_comparison']
                f.write(f"Fine structure constant: alpha = {ana['alpha']:.10f}\n\n")
                f.write(f"Geometric measurements:\n")
                f.write(f"  Mean Berry phase:      {ana['mean_berry_phase']:.10f} rad\n")
                f.write(f"  Mean curvature:        {ana['mean_curvature']:.10f}\n\n")
                f.write(f"Key ratios:\n")
                f.write(f"  Curvature / alpha:     {ana['ratio_to_alpha']:.6f}\n\n")
                
                f.write("Coincidence tests:\n")
                f.write("-"*80 + "\n")
                for name, ratio, target in ana['tests']:
                    deviation = abs(ratio - target) / target * 100
                    f.write(f"  {name:25s} = {ratio:10.6f}  (deviation: {deviation:6.2f}%)\n")
                f.write("\n")
            
            # Summary
            f.write("\n" + "="*80 + "\n")
            f.write("SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write("Key findings:\n")
            f.write("1. Square plaquettes successfully constructed from quantum numbers\n")
            f.write("2. Berry phase measured via parallel transport\n")
            f.write("3. Geometric curvature computed and compared to fundamental constants\n\n")
            f.write("This analysis reveals the intrinsic geometric structure of the lattice.\n\n")
        
        print(f"[OK] Report saved to {filename}")
    
    def plot_results(self):
        """
        Visualize holonomy results.
        """
        if not self.results['plaquettes']:
            print("No data to plot")
            return
        
        results = self.results['plaquettes']
        
        berry_phases = [r['berry_phase'] for r in results]
        curvatures = [r['curvature'] for r in results]
        areas = [r['area'] for r in results]
        
        # Extract plaquette centers for spatial plot
        centers_n = [r['nodes'][0][0] + 0.5 for r in results]
        centers_m = [r['nodes'][0][2] + 0.5 for r in results]
        
        fig = plt.figure(figsize=(15, 10))
        
        # Plot 1: Berry phase histogram
        ax1 = plt.subplot(2, 3, 1)
        ax1.hist(berry_phases, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(np.mean(berry_phases), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(berry_phases):.4f}')
        ax1.set_xlabel('Berry Phase (rad)', fontsize=11)
        ax1.set_ylabel('Count', fontsize=11)
        ax1.set_title('Berry Phase Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Curvature histogram
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(curvatures, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(np.mean(curvatures), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(curvatures):.4f}')
        ax2.set_xlabel('Curvature', fontsize=11)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_title('Curvature Distribution', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Berry vs Area
        ax3 = plt.subplot(2, 3, 3)
        ax3.scatter(areas, berry_phases, alpha=0.5, s=20, color='purple')
        ax3.set_xlabel('Plaquette Area', fontsize=11)
        ax3.set_ylabel('Berry Phase (rad)', fontsize=11)
        ax3.set_title('Berry Phase vs Area', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Spatial distribution of Berry phase
        ax4 = plt.subplot(2, 3, 4)
        scatter = ax4.scatter(centers_m, centers_n, c=berry_phases, s=50, cmap='RdYlBu', alpha=0.7)
        plt.colorbar(scatter, ax=ax4, label='Berry Phase (rad)')
        ax4.set_xlabel('m (angular)', fontsize=11)
        ax4.set_ylabel('n (radial)', fontsize=11)
        ax4.set_title('Spatial Berry Phase Map', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Spatial distribution of curvature
        ax5 = plt.subplot(2, 3, 5)
        scatter2 = ax5.scatter(centers_m, centers_n, c=curvatures, s=50, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter2, ax=ax5, label='Curvature')
        ax5.set_xlabel('m (angular)', fontsize=11)
        ax5.set_ylabel('n (radial)', fontsize=11)
        ax5.set_title('Spatial Curvature Map', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Comparison to alpha
        ax6 = plt.subplot(2, 3, 6)
        alpha = 1.0 / 137.035999084
        mean_curv = np.mean(curvatures)
        
        x_labels = ['alpha', 'Mean\nCurvature', 'Mean Berry\n/ 137']
        values = [alpha, mean_curv, np.mean(berry_phases)/137]
        colors = ['red', 'green', 'blue']
        
        bars = ax6.bar(x_labels, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax6.set_ylabel('Value', fontsize=11)
        ax6.set_title('Comparison to Fine Structure Constant', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.6f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('holonomy_analysis.png', dpi=150, bbox_inches='tight')
        print("  [OK] Saved: holonomy_analysis.png")
        plt.close()


def main():
    """
    Main execution: Complete holonomy analysis.
    """
    print("\n" + "="*80)
    print("GEOMETRIC HOLONOMY HUNTER")
    print("Berry Phase Measurement on Paraboloid Lattice")
    print("="*80 + "\n")
    
    # Initialize
    calc = PlaquetteHolonomy(max_n=20)
    
    # TASK 1: Find plaquettes
    print("EXECUTING: Task 1 - Plaquette Scanner")
    plaquettes = calc.construct_plaquettes()
    
    if len(plaquettes) == 0:
        print("\nERROR: No plaquettes found. Cannot proceed.")
        return
    
    # Initialize spinors
    calc.initialize_spinors()
    
    # TASK 2: Measure holonomies
    print("\nEXECUTING: Task 2 - Geometric Twist Measurement")
    holonomy_results = calc.measure_all_holonomies()
    
    # TASK 3: Search for alpha
    print("\nEXECUTING: Task 3 - Search for Fine Structure Constant")
    alpha_analysis = calc.analyze_alpha_connection()
    
    # Generate outputs
    calc.generate_report("holonomy_report.txt")
    calc.plot_results()
    
    print("\n" + "="*80)
    print("HOLONOMY ANALYSIS COMPLETE")
    print("="*80)
    print("\nResults saved to:")
    print("  - holonomy_report.txt")
    print("  - holonomy_analysis.png")
    print("\n")


if __name__ == "__main__":
    main()
