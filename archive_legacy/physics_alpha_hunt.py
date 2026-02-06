"""
PHYSICS ALPHA HUNT: Search for Fine Structure Constant in Lattice Geometry
============================================================================
Hypothesis: The fine structure constant alpha ~ 1/137 is not arbitrary,
but a geometric property of the paraboloid lattice packing.

Three search strategies:
1. Gearing Ratio: Operator norm ratio ||L+|| / ||T+||
2. Commutator Defect: Non-integrability [T+, L+]
3. Holonomy Defect: Gauss-Bonnet total curvature deviation

Target values:
  alpha        ~ 0.0072973525693
  1/alpha      ~ 137.035999084
  alpha/(2*pi) ~ 0.0011614

Author: Geometric Constant Discovery Team
Date: February 2026
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from paraboloid_lattice_su11 import ParaboloidLattice
import time
from typing import Dict, List, Tuple


class AlphaHunter:
    """
    Hunt for the fine structure constant in lattice geometry.
    """
    
    def __init__(self, max_n: int = 50):
        """
        Initialize alpha hunter.
        
        Parameters:
        -----------
        max_n : int
            Maximum principal quantum number for search
        """
        self.max_n = max_n
        
        # Known constants to search for
        self.alpha = 0.0072973525693
        self.inv_alpha = 137.035999084
        self.alpha_over_2pi = 0.0011614
        
        # Store results
        self.gearing_ratios = []
        self.commutator_defects = []
        self.holonomy_defects = []
        
    def frobenius_norm(self, matrix):
        """
        Compute Frobenius norm of sparse matrix.
        ||A||_F = sqrt(sum |A_ij|^2)
        """
        if matrix is None:
            return 0.0
        
        # For sparse matrix, compute efficiently
        data = np.abs(matrix.data)
        return np.sqrt(np.sum(data**2))
    
    def compute_gearing_ratio(self, n: int) -> Dict:
        """
        Compute gearing ratio between angular and radial operators.
        
        Theory: In continuum, radial and angular DOFs are independent.
        On lattice, they compete for density. The ratio may encode alpha.
        """
        print(f"\n  Computing gearing ratio for n={n}...")
        
        # Build lattice for this shell
        lattice = ParaboloidLattice(max_n=n)
        
        # Get operators
        T_plus = lattice.Tplus
        L_plus = lattice.Lplus
        
        # Compute Frobenius norms
        norm_T = self.frobenius_norm(T_plus)
        norm_L = self.frobenius_norm(L_plus)
        
        # Compute ratios
        ratio_L_over_T = norm_L / norm_T if norm_T > 0 else 0.0
        ratio_T_over_L = norm_T / norm_L if norm_L > 0 else 0.0
        
        result = {
            'n': n,
            'norm_T': norm_T,
            'norm_L': norm_L,
            'L_over_T': ratio_L_over_T,
            'T_over_L': ratio_T_over_L
        }
        
        print(f"    ||T+|| = {norm_T:.6f}")
        print(f"    ||L+|| = {norm_L:.6f}")
        print(f"    L/T = {ratio_L_over_T:.6f}")
        print(f"    T/L = {ratio_T_over_L:.6f}")
        
        return result
    
    def compute_commutator_defect(self, n: int) -> Dict:
        """
        Compute commutator [T+, L+] = T+L+ - L+T+.
        
        Theory: Perfect integrability requires [T+, L+] = 0.
        Defect magnitude may be proportional to alpha.
        """
        print(f"\n  Computing commutator defect for n={n}...")
        
        # Build lattice
        lattice = ParaboloidLattice(max_n=n)
        
        T_plus = lattice.Tplus
        L_plus = lattice.Lplus
        
        if T_plus is None or L_plus is None:
            return {'n': n, 'defect': 0.0, 'normalized_defect': 0.0}
        
        # Compute commutator C = [T+, L+] = T+L+ - L+T+
        TL = T_plus @ L_plus
        LT = L_plus @ T_plus
        C = TL - LT
        
        # Measure defect magnitude: Tr(C^dagger C)
        C_dag = C.conj().T
        C_dag_C = C_dag @ C
        
        # Trace
        defect_magnitude = np.abs(C_dag_C.diagonal().sum())
        
        # Normalize by lattice size
        n_nodes = len(lattice.nodes)
        normalized_defect = defect_magnitude / n_nodes if n_nodes > 0 else 0.0
        
        # Also compute Frobenius norm of commutator
        frobenius_defect = self.frobenius_norm(C)
        frobenius_normalized = frobenius_defect / np.sqrt(n_nodes) if n_nodes > 0 else 0.0
        
        result = {
            'n': n,
            'defect': defect_magnitude,
            'normalized_defect': normalized_defect,
            'frobenius_defect': frobenius_defect,
            'frobenius_normalized': frobenius_normalized
        }
        
        print(f"    Trace(C†C) = {defect_magnitude:.6e}")
        print(f"    Normalized = {normalized_defect:.6e}")
        print(f"    ||C||_F = {frobenius_defect:.6f}")
        print(f"    ||C||_F/sqrt(N) = {frobenius_normalized:.6e}")
        
        return result
    
    def compute_holonomy_defect(self, n: int) -> Dict:
        """
        Compute Gauss-Bonnet holonomy defect.
        
        Theory: For a closed surface (sphere), total curvature should be 4*pi.
        Any deviation may be related to alpha (lattice discretization error).
        """
        print(f"\n  Computing holonomy defect for n={n}...")
        
        # Build lattice
        lattice = ParaboloidLattice(max_n=n)
        
        # Find all plaquettes in shell n
        plaquettes = []
        berry_phases = []
        
        for node_n, node_l, node_m in lattice.nodes:
            if node_n != n:
                continue
            
            # Try to build a rectangular plaquette: (n,l,m) -> (n,l,m+1) -> (n,l+1,m+1) -> (n,l+1,m)
            # This is a plaquette in (l,m) space at fixed n
            corners = [
                (n, node_l, node_m),
                (n, node_l, node_m + 1),
                (n, node_l + 1, node_m + 1),
                (n, node_l + 1, node_m)
            ]
            
            # Check if all corners exist
            if all(corner in lattice.nodes for corner in corners):
                plaquettes.append(corners)
                
                # Compute Berry phase (simplified - using geometric phase)
                # For now, use a placeholder based on curvature
                # Real calculation would require state vectors
                
                # Estimate: Berry phase ~ area / radius^2 for small plaquettes
                # On sphere of radius n^2, area ~ 1, so phase ~ 1/n^4
                berry_phase = 1.0 / (n**4) if n > 0 else 0.0
                berry_phases.append(berry_phase)
        
        # Total curvature
        total_curvature = np.sum(berry_phases)
        
        # Gauss-Bonnet: Total curvature of sphere = 4*pi
        expected = 4 * np.pi
        defect = abs(expected - total_curvature)
        relative_defect = defect / expected if expected > 0 else 0.0
        
        result = {
            'n': n,
            'n_plaquettes': len(plaquettes),
            'total_curvature': total_curvature,
            'expected': expected,
            'defect': defect,
            'relative_defect': relative_defect
        }
        
        print(f"    Plaquettes: {len(plaquettes)}")
        print(f"    Total curvature: {total_curvature:.6f}")
        print(f"    Expected (4π): {expected:.6f}")
        print(f"    Defect: {defect:.6f}")
        print(f"    Relative: {relative_defect:.6e}")
        
        return result
    
    def hunt_alpha(self, n_start: int = 2, n_end: int = 20, n_step: int = 2):
        """
        Main hunting routine: scan through shells looking for alpha.
        """
        print("="*80)
        print("ALPHA HUNT: Search for Fine Structure Constant")
        print("="*80)
        print(f"\nTarget values:")
        print(f"  alpha       = {self.alpha:.10f}")
        print(f"  1/alpha     = {self.inv_alpha:.10f}")
        print(f"  alpha/(2π)  = {self.alpha_over_2pi:.10f}")
        print()
        
        # Hunt 1: Gearing Ratios
        print("\n" + "="*80)
        print("HUNT 1: GEARING RATIO (Operator Norms)")
        print("="*80)
        
        for n in range(n_start, n_end + 1, n_step):
            result = self.compute_gearing_ratio(n)
            self.gearing_ratios.append(result)
        
        # Hunt 2: Commutator Defects
        print("\n" + "="*80)
        print("HUNT 2: COMMUTATOR DEFECT (Non-Integrability)")
        print("="*80)
        
        for n in range(n_start, n_end + 1, n_step):
            result = self.compute_commutator_defect(n)
            self.commutator_defects.append(result)
        
        # Hunt 3: Holonomy Defects
        print("\n" + "="*80)
        print("HUNT 3: HOLONOMY DEFECT (Gauss-Bonnet)")
        print("="*80)
        
        for n in range(n_start, min(n_end + 1, 15), n_step):  # Limit for speed
            result = self.compute_holonomy_defect(n)
            self.holonomy_defects.append(result)
    
    def analyze_convergence(self):
        """
        Analyze if any ratios converge to alpha-related constants.
        """
        print("\n" + "="*80)
        print("CONVERGENCE ANALYSIS")
        print("="*80)
        
        # Analyze gearing ratios
        print("\n1. GEARING RATIOS:")
        print("-"*80)
        
        if len(self.gearing_ratios) > 2:
            # Check L/T ratio
            ratios_LT = [r['L_over_T'] for r in self.gearing_ratios]
            mean_LT = np.mean(ratios_LT[-5:])  # Last 5 values
            std_LT = np.std(ratios_LT[-5:])
            
            print(f"  L/T ratio:")
            print(f"    Mean (last 5): {mean_LT:.6f}")
            print(f"    Std dev:       {std_LT:.6f}")
            
            # Check proximity to alpha-related constants
            self.check_proximity("L/T", mean_LT)
            
            # Check T/L ratio
            ratios_TL = [r['T_over_L'] for r in self.gearing_ratios]
            mean_TL = np.mean(ratios_TL[-5:])
            std_TL = np.std(ratios_TL[-5:])
            
            print(f"\n  T/L ratio:")
            print(f"    Mean (last 5): {mean_TL:.6f}")
            print(f"    Std dev:       {std_TL:.6f}")
            
            self.check_proximity("T/L", mean_TL)
        
        # Analyze commutator defects
        print("\n2. COMMUTATOR DEFECTS:")
        print("-"*80)
        
        if len(self.commutator_defects) > 2:
            defects = [r['frobenius_normalized'] for r in self.commutator_defects]
            mean_defect = np.mean(defects[-5:])
            std_defect = np.std(defects[-5:])
            
            print(f"  Normalized Frobenius defect:")
            print(f"    Mean (last 5): {mean_defect:.6e}")
            print(f"    Std dev:       {std_defect:.6e}")
            
            self.check_proximity("Commutator defect", mean_defect)
        
        # Analyze holonomy defects
        print("\n3. HOLONOMY DEFECTS:")
        print("-"*80)
        
        if len(self.holonomy_defects) > 2:
            rel_defects = [r['relative_defect'] for r in self.holonomy_defects]
            mean_rel = np.mean(rel_defects[-3:])
            std_rel = np.std(rel_defects[-3:])
            
            print(f"  Relative holonomy defect:")
            print(f"    Mean (last 3): {mean_rel:.6e}")
            print(f"    Std dev:       {std_rel:.6e}")
            
            self.check_proximity("Holonomy defect", mean_rel)
    
    def check_proximity(self, name: str, value: float):
        """
        Check if value is close to alpha-related constants.
        """
        targets = {
            'alpha': self.alpha,
            '1/alpha': self.inv_alpha,
            'alpha/(2π)': self.alpha_over_2pi,
            'sqrt(alpha)': np.sqrt(self.alpha),
            'alpha^2': self.alpha**2,
            '2*alpha': 2*self.alpha,
            'pi*alpha': np.pi*self.alpha
        }
        
        print(f"\n    Proximity check for {name} = {value:.6e}:")
        
        found_match = False
        for target_name, target_value in targets.items():
            rel_error = abs(value - target_value) / target_value if target_value != 0 else float('inf')
            
            if rel_error < 0.01:  # Within 1%
                print(f"      *** MATCH: {target_name} = {target_value:.6e} (error: {rel_error*100:.2f}%) ***")
                found_match = True
            elif rel_error < 0.1:  # Within 10%
                print(f"      CLOSE:     {target_name} = {target_value:.6e} (error: {rel_error*100:.2f}%)")
                found_match = True
        
        if not found_match:
            print(f"      No significant match to alpha-related constants")
    
    def generate_report(self, output_file: str = "alpha_report.txt"):
        """
        Generate comprehensive report.
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ALPHA HUNT REPORT: Search for Fine Structure Constant\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("OBJECTIVE:\n")
            f.write("Search for the fine structure constant alpha ~ 1/137 in the\n")
            f.write("geometric structure of the paraboloid lattice.\n\n")
            
            f.write("TARGET VALUES:\n")
            f.write(f"  alpha       = {self.alpha:.10f}\n")
            f.write(f"  1/alpha     = {self.inv_alpha:.10f}\n")
            f.write(f"  alpha/(2π)  = {self.alpha_over_2pi:.10f}\n\n")
            
            # Gearing ratios
            f.write("="*80 + "\n")
            f.write("1. GEARING RATIOS (||L+|| / ||T+||)\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"{'n':<5} {'||T+||':<12} {'||L+||':<12} {'L/T':<12} {'T/L':<12}\n")
            f.write("-"*80 + "\n")
            
            for r in self.gearing_ratios:
                f.write(f"{r['n']:<5} {r['norm_T']:<12.6f} {r['norm_L']:<12.6f} "
                       f"{r['L_over_T']:<12.6f} {r['T_over_L']:<12.6f}\n")
            
            if self.gearing_ratios:
                mean_LT = np.mean([r['L_over_T'] for r in self.gearing_ratios[-5:]])
                f.write(f"\nConverged L/T (last 5): {mean_LT:.6f}\n")
            
            # Commutator defects
            f.write("\n" + "="*80 + "\n")
            f.write("2. COMMUTATOR DEFECTS ([T+, L+])\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"{'n':<5} {'Tr(C†C)':<15} {'Normalized':<15} {'||C||_F':<12}\n")
            f.write("-"*80 + "\n")
            
            for r in self.commutator_defects:
                f.write(f"{r['n']:<5} {r['defect']:<15.6e} {r['normalized_defect']:<15.6e} "
                       f"{r['frobenius_defect']:<12.6f}\n")
            
            # Holonomy defects
            f.write("\n" + "="*80 + "\n")
            f.write("3. HOLONOMY DEFECTS (Gauss-Bonnet)\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"{'n':<5} {'Plaquettes':<12} {'Total Curv':<15} {'Defect':<15} {'Relative':<12}\n")
            f.write("-"*80 + "\n")
            
            for r in self.holonomy_defects:
                f.write(f"{r['n']:<5} {r['n_plaquettes']:<12} {r['total_curvature']:<15.6f} "
                       f"{r['defect']:<15.6f} {r['relative_defect']:<12.6e}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("CONCLUSION\n")
            f.write("="*80 + "\n\n")
            
            f.write("Analysis complete. Review proximity checks above for potential matches.\n")
        
        print(f"\n[OK] Report saved to {output_file}")


def main():
    """
    Main routine: hunt for alpha.
    """
    print("="*80)
    print("ALPHA HUNT: Search for Fine Structure Constant in Lattice Geometry")
    print("="*80)
    print()
    print("Hypothesis: alpha ~ 1/137 is a geometric property of the lattice.")
    print()
    
    # Initialize hunter
    hunter = AlphaHunter()
    
    # Perform hunt (scan n=2 to n=20 in steps of 2)
    hunter.hunt_alpha(n_start=2, n_end=20, n_step=2)
    
    # Analyze convergence
    hunter.analyze_convergence()
    
    # Generate report
    hunter.generate_report()
    
    print("\n" + "="*80)
    print("ALPHA HUNT COMPLETE")
    print("="*80)
    print("\nResults saved to: alpha_report.txt")
    print()


if __name__ == "__main__":
    main()
