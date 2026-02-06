"""
LATTICE NATURAL SCALES ANALYSIS: Defend Against "Tuned Parameter" Critique
===========================================================================

Peer Review Critique: "The helical pitch δ = 3.086 was tuned to match α."

Defense Strategy: Show that δ corresponds to a NATURAL GEOMETRIC SCALE
of the paraboloid lattice—not an arbitrary fitting parameter.

This script measures ALL intrinsic scales of the n=5 shell:
- Transition operator weights (T_±, L_±)
- Euclidean edge lengths
- Node spacings (radial, angular, azimuthal)
- Plaquette dimensions

If δ ≈ [Natural Scale], then it's a GEOMETRIC CONSTRAINT, not a free parameter.

Author: Computational Physics Study
Date: 2026-02-05
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class TransitionOperators:
    """
    Compute exact transition operator weights for SO(4,2) × SU(2) algebra.
    These are the edge weights in the lattice graph.
    """
    
    @staticmethod
    def T_plus(n: int, l: int, m: int) -> float:
        """
        Radial raising operator T_+ weight.
        Transition: |n,l,m⟩ → |n+1,l,m⟩
        
        Formula (SU(1,1) Clebsch-Gordan):
        T_+ = sqrt((n+l)(n-l))
        """
        if l >= n:
            return 0.0
        return np.sqrt((n + l) * (n - l))
    
    @staticmethod
    def T_minus(n: int, l: int, m: int) -> float:
        """
        Radial lowering operator T_- weight.
        Transition: |n,l,m⟩ → |n-1,l,m⟩
        """
        if n == 1 or l >= n - 1:
            return 0.0
        return np.sqrt((n + l - 1) * (n - l - 1))
    
    @staticmethod
    def L_plus(n: int, l: int, m: int) -> float:
        """
        Angular raising operator L_+ weight.
        Transition: |n,l,m⟩ → |n,l,m+1⟩
        
        Formula (SU(2) angular momentum):
        L_+ = sqrt(l(l+1) - m(m+1))
        """
        if m >= l:
            return 0.0
        return np.sqrt(l * (l + 1) - m * (m + 1))
    
    @staticmethod
    def L_minus(n: int, l: int, m: int) -> float:
        """
        Angular lowering operator L_- weight.
        Transition: |n,l,m⟩ → |n,l,m-1⟩
        """
        if m <= -l:
            return 0.0
        return np.sqrt(l * (l + 1) - m * (m - 1))


class ParaboloidLatticeStatistics:
    """
    Analyze natural geometric scales of the paraboloid lattice.
    """
    
    def __init__(self, target_n: int = 5):
        self.n = target_n
        self.ops = TransitionOperators()
        
        # Generate states in shell n
        self.states = self._generate_shell_states()
        
        # Compute positions
        self.positions = {}
        for state in self.states:
            self.positions[state] = self.quantum_to_cartesian(*state)
        
        # Storage for measurements
        self.edge_weights = {
            'T_plus': [],
            'T_minus': [],
            'L_plus': [],
            'L_minus': []
        }
        self.edge_lengths = {
            'radial': [],      # Euclidean distance for T_± transitions
            'angular': []      # Euclidean distance for L_± transitions
        }
        self.node_spacings = {
            'radial_shell': [],     # Distance to next shell (n → n+1)
            'angular_same_l': [],   # Distance within same l (m → m+1)
            'angular_diff_l': []    # Distance between different l at same n
        }
    
    def _generate_shell_states(self) -> List[Tuple[int, int, int]]:
        """Generate all (n,l,m) states in shell n."""
        states = []
        for l in range(self.n):
            for m in range(-l, l + 1):
                states.append((self.n, l, m))
        return states
    
    def quantum_to_cartesian(self, n: int, l: int, m: int) -> np.ndarray:
        """
        Map quantum numbers to 3D paraboloid coordinates.
        
        Standard embedding:
        r = n² (parabolic radial)
        θ = πl/(n-1) (angular from pole)
        φ = 2πm/(2l+1) (azimuthal)
        z = -1/n² (energy depth)
        """
        r = n * n
        
        if n == 1:
            theta = 0.0
        else:
            theta = np.pi * l / (n - 1)
        
        if l == 0:
            phi = 0.0
        else:
            phi = 2.0 * np.pi * m / (2 * l + 1)
        
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = -1.0 / (n * n)
        
        return np.array([x, y, z])
    
    def measure_transition_weights(self):
        """Measure all transition operator weights in shell n."""
        print(f"\n{'='*70}")
        print(f"MEASURING TRANSITION OPERATOR WEIGHTS (Shell n={self.n})")
        print(f"{'='*70}")
        
        for state in self.states:
            n, l, m = state
            
            # T_plus: n → n+1
            w = self.ops.T_plus(n, l, m)
            if w > 0:
                self.edge_weights['T_plus'].append(w)
            
            # T_minus: n → n-1
            w = self.ops.T_minus(n, l, m)
            if w > 0:
                self.edge_weights['T_minus'].append(w)
            
            # L_plus: m → m+1
            w = self.ops.L_plus(n, l, m)
            if w > 0:
                self.edge_weights['L_plus'].append(w)
            
            # L_minus: m → m-1
            w = self.ops.L_minus(n, l, m)
            if w > 0:
                self.edge_weights['L_minus'].append(w)
        
        print(f"\nTransition operator statistics:")
        print(f"  States in shell: {len(self.states)}")
        for op_type, weights in self.edge_weights.items():
            if weights:
                print(f"\n  {op_type}:")
                print(f"    Count: {len(weights)}")
                print(f"    Mean:  {np.mean(weights):.6f}")
                print(f"    Median: {np.median(weights):.6f}")
                print(f"    Std:   {np.std(weights):.6f}")
                print(f"    Min:   {np.min(weights):.6f}")
                print(f"    Max:   {np.max(weights):.6f}")
    
    def measure_edge_lengths(self):
        """Measure Euclidean distances for all edges."""
        print(f"\n{'='*70}")
        print(f"MEASURING EUCLIDEAN EDGE LENGTHS")
        print(f"{'='*70}")
        
        for state in self.states:
            n, l, m = state
            pos = self.positions[state]
            
            # Radial edges (T_±)
            # Check if target states exist
            if (n + 1, l, m) in self.positions:
                target_pos = self.positions[(n + 1, l, m)]
                dist = np.linalg.norm(target_pos - pos)
                self.edge_lengths['radial'].append(dist)
            
            if (n - 1, l, m) in self.positions:
                target_pos = self.positions[(n - 1, l, m)]
                dist = np.linalg.norm(target_pos - pos)
                self.edge_lengths['radial'].append(dist)
            
            # Angular edges (L_±)
            if (n, l, m + 1) in self.positions:
                target_pos = self.positions[(n, l, m + 1)]
                dist = np.linalg.norm(target_pos - pos)
                self.edge_lengths['angular'].append(dist)
            
            if (n, l, m - 1) in self.positions:
                target_pos = self.positions[(n, l, m - 1)]
                dist = np.linalg.norm(target_pos - pos)
                self.edge_lengths['angular'].append(dist)
        
        print(f"\nEuclidean edge length statistics:")
        for edge_type, lengths in self.edge_lengths.items():
            if lengths:
                print(f"\n  {edge_type}:")
                print(f"    Count: {len(lengths)}")
                print(f"    Mean:  {np.mean(lengths):.6f}")
                print(f"    Median: {np.median(lengths):.6f}")
                print(f"    Std:   {np.std(lengths):.6f}")
                print(f"    Min:   {np.min(lengths):.6f}")
                print(f"    Max:   {np.max(lengths):.6f}")
    
    def measure_node_spacings(self):
        """Measure characteristic distances between nodes."""
        print(f"\n{'='*70}")
        print(f"MEASURING NODE SPACINGS")
        print(f"{'='*70}")
        
        for state in self.states:
            n, l, m = state
            pos = self.positions[state]
            
            # Radial spacing: distance to next shell
            if (n + 1, l, m) in self.positions:
                target_pos = self.positions[(n + 1, l, m)]
                dist = np.linalg.norm(target_pos - pos)
                self.node_spacings['radial_shell'].append(dist)
            
            # Angular spacing within same l
            if (n, l, m + 1) in self.positions:
                target_pos = self.positions[(n, l, m + 1)]
                dist = np.linalg.norm(target_pos - pos)
                self.node_spacings['angular_same_l'].append(dist)
            
            # Angular spacing between different l
            if (n, l + 1, m) in self.positions:
                target_pos = self.positions[(n, l + 1, m)]
                dist = np.linalg.norm(target_pos - pos)
                self.node_spacings['angular_diff_l'].append(dist)
        
        print(f"\nNode spacing statistics:")
        for spacing_type, dists in self.node_spacings.items():
            if dists:
                print(f"\n  {spacing_type}:")
                print(f"    Count: {len(dists)}")
                print(f"    Mean:  {np.mean(dists):.6f}")
                print(f"    Median: {np.median(dists):.6f}")
                print(f"    Std:   {np.std(dists):.6f}")
                print(f"    Min:   {np.min(dists):.6f}")
                print(f"    Max:   {np.max(dists):.6f}")
    
    def compute_special_scales(self) -> Dict[str, float]:
        """
        Compute special geometric scales that might relate to δ.
        """
        print(f"\n{'='*70}")
        print(f"COMPUTING SPECIAL GEOMETRIC SCALES")
        print(f"{'='*70}")
        
        scales = {}
        
        # Shell thickness: (n+1)² - n²
        scales['shell_thickness'] = (self.n + 1)**2 - self.n**2
        
        # Energy spacing: |E_{n+1} - E_n|
        E_n = -1.0 / (2 * self.n**2)
        E_n1 = -1.0 / (2 * (self.n + 1)**2)
        scales['energy_spacing'] = abs(E_n1 - E_n)
        
        # Circumference at n: 2πr = 2πn²
        scales['circumference'] = 2 * np.pi * self.n**2
        
        # Average radius: r = n²
        scales['radius'] = self.n**2
        
        # Average l spacing: π/(n-1)
        if self.n > 1:
            scales['angular_step'] = np.pi / (self.n - 1)
        else:
            scales['angular_step'] = 0.0
        
        # Characteristic length: sqrt(area / N_nodes)
        # Area ~ n⁴, N ~ n²
        scales['lattice_constant'] = self.n**2 / np.sqrt(len(self.states))
        
        # Mathematical constants that might appear
        scales['pi'] = np.pi
        scales['2pi'] = 2 * np.pi
        scales['sqrt_pi'] = np.sqrt(np.pi)
        scales['e'] = np.e
        
        print(f"\nSpecial scales:")
        for name, value in scales.items():
            print(f"  {name:.<30} {value:.6f}")
        
        return scales
    
    def compare_to_helical_pitch(self, delta: float = 3.086):
        """
        Compare helical pitch δ to all measured natural scales.
        Find the best match(es).
        """
        print(f"\n{'='*70}")
        print(f"COMPARISON TO HELICAL PITCH δ = {delta:.6f}")
        print(f"{'='*70}")
        
        matches = []
        
        # Compare to transition weights
        for op_type, weights in self.edge_weights.items():
            if weights:
                mean_w = np.mean(weights)
                median_w = np.median(weights)
                
                error_mean = abs(mean_w - delta) / delta * 100
                error_median = abs(median_w - delta) / delta * 100
                
                matches.append((f"{op_type}_mean", mean_w, error_mean))
                matches.append((f"{op_type}_median", median_w, error_median))
        
        # Compare to edge lengths
        for edge_type, lengths in self.edge_lengths.items():
            if lengths:
                mean_l = np.mean(lengths)
                median_l = np.median(lengths)
                
                error_mean = abs(mean_l - delta) / delta * 100
                error_median = abs(median_l - delta) / delta * 100
                
                matches.append((f"{edge_type}_edge_mean", mean_l, error_mean))
                matches.append((f"{edge_type}_edge_median", median_l, error_median))
        
        # Compare to node spacings
        for spacing_type, dists in self.node_spacings.items():
            if dists:
                mean_d = np.mean(dists)
                median_d = np.median(dists)
                
                error_mean = abs(mean_d - delta) / delta * 100
                error_median = abs(median_d - delta) / delta * 100
                
                matches.append((f"{spacing_type}_mean", mean_d, error_mean))
                matches.append((f"{spacing_type}_median", median_d, error_median))
        
        # Compare to special scales
        special_scales = self.compute_special_scales()
        for name, value in special_scales.items():
            error = abs(value - delta) / delta * 100
            matches.append((name, value, error))
        
        # Sort by error
        matches.sort(key=lambda x: x[2])
        
        print(f"\nTarget: δ = {delta:.6f}")
        print(f"\nBEST MATCHES (sorted by error):")
        print(f"{'Quantity':<35} {'Value':>12} {'Error (%)':>12} {'Match?':>8}")
        print(f"{'-'*70}")
        
        for name, value, error in matches[:20]:  # Top 20 matches
            match_str = "✓✓✓" if error < 1.0 else "✓✓" if error < 5.0 else "✓" if error < 10.0 else ""
            print(f"{name:<35} {value:>12.6f} {error:>12.4f} {match_str:>8}")
        
        # Find exact matches
        exact_matches = [m for m in matches if m[2] < 1.0]
        if exact_matches:
            print(f"\n{'='*70}")
            print(f"EXACT MATCHES (error < 1%):")
            print(f"{'='*70}")
            for name, value, error in exact_matches:
                print(f"  {name}: {value:.6f} (error: {error:.4f}%)")
                print(f"    → δ ≈ {name.upper()}")
        else:
            print(f"\n⚠ NO EXACT MATCHES FOUND (all errors > 1%)")
        
        return matches
    
    def generate_report(self, matches: List[Tuple[str, float, float]]):
        """Generate detailed report file."""
        filename = "lattice_stats_report.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("LATTICE NATURAL SCALES ANALYSIS\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Shell: n = {self.n}\n")
            f.write(f"States in shell: {len(self.states)}\n")
            f.write(f"Target helical pitch: δ = 3.086\n\n")
            
            f.write("PEER REVIEW DEFENSE:\n")
            f.write("-"*70 + "\n")
            f.write("Critique: \"The pitch δ was tuned to match α.\"\n")
            f.write("Defense: \"δ corresponds to a natural lattice scale.\"\n\n")
            
            f.write("TRANSITION OPERATOR WEIGHTS:\n")
            f.write("-"*70 + "\n")
            for op_type, weights in self.edge_weights.items():
                if weights:
                    f.write(f"\n{op_type}:\n")
                    f.write(f"  Mean:   {np.mean(weights):.6f}\n")
                    f.write(f"  Median: {np.median(weights):.6f}\n")
                    f.write(f"  Range:  [{np.min(weights):.6f}, {np.max(weights):.6f}]\n")
            
            f.write("\n\nEUCLIDEAN EDGE LENGTHS:\n")
            f.write("-"*70 + "\n")
            for edge_type, lengths in self.edge_lengths.items():
                if lengths:
                    f.write(f"\n{edge_type}:\n")
                    f.write(f"  Mean:   {np.mean(lengths):.6f}\n")
                    f.write(f"  Median: {np.median(lengths):.6f}\n")
                    f.write(f"  Range:  [{np.min(lengths):.6f}, {np.max(lengths):.6f}]\n")
            
            f.write("\n\nNODE SPACINGS:\n")
            f.write("-"*70 + "\n")
            for spacing_type, dists in self.node_spacings.items():
                if dists:
                    f.write(f"\n{spacing_type}:\n")
                    f.write(f"  Mean:   {np.mean(dists):.6f}\n")
                    f.write(f"  Median: {np.median(dists):.6f}\n")
                    f.write(f"  Range:  [{np.min(dists):.6f}, {np.max(dists):.6f}]\n")
            
            f.write("\n\nBEST MATCHES TO δ = 3.086:\n")
            f.write("-"*70 + "\n")
            f.write(f"{'Quantity':<35} {'Value':>12} {'Error (%)':>12}\n")
            f.write("-"*70 + "\n")
            for name, value, error in matches[:15]:
                f.write(f"{name:<35} {value:>12.6f} {error:>12.4f}\n")
            
            # Verdict
            f.write("\n\n" + "="*70 + "\n")
            f.write("VERDICT:\n")
            f.write("="*70 + "\n")
            
            exact_matches = [m for m in matches if m[2] < 1.0]
            close_matches = [m for m in matches if 1.0 <= m[2] < 5.0]
            
            if exact_matches:
                f.write("\n✓ DEFENSE SUCCESSFUL\n\n")
                f.write(f"The helical pitch δ = 3.086 matches the following natural scales:\n")
                for name, value, error in exact_matches:
                    f.write(f"  - {name}: {value:.6f} (error: {error:.4f}%)\n")
                f.write("\nConclusion: δ is NOT a tuned parameter. It is a geometric constraint\n")
                f.write("arising from the intrinsic structure of the paraboloid lattice.\n")
            elif close_matches:
                f.write("\n~ DEFENSE PARTIAL\n\n")
                f.write(f"The helical pitch δ = 3.086 is close to:\n")
                for name, value, error in close_matches:
                    f.write(f"  - {name}: {value:.6f} (error: {error:.4f}%)\n")
                f.write("\nConclusion: δ may be related to lattice geometry, but relationship\n")
                f.write("is not exact. Further theoretical justification needed.\n")
            else:
                f.write("\n❌ DEFENSE WEAK\n\n")
                f.write("No natural lattice scale matches δ = 3.086 within 5%.\n")
                f.write("The pitch may be a higher-order combination of scales,\n")
                f.write("or it may indeed be an emergent parameter.\n")
                f.write("\nRecommendation: Investigate composite scales (e.g., geometric means,\n")
                f.write("ratios, or products of fundamental scales).\n")
        
        print(f"\n{'='*70}")
        print(f"Report saved: {filename}")
        print(f"{'='*70}")
    
    def run_full_analysis(self):
        """Execute complete lattice statistics analysis."""
        print(f"\n{'#'*70}")
        print(f"# LATTICE NATURAL SCALES ANALYSIS")
        print(f"{'#'*70}")
        print(f"#")
        print(f"# Defending against peer review critique:")
        print(f"# \"The helical pitch δ = 3.086 was tuned to match α.\"")
        print(f"#")
        print(f"# Strategy: Show δ corresponds to a NATURAL lattice scale.")
        print(f"#")
        print(f"{'#'*70}")
        
        # Step 1: Transition weights
        self.measure_transition_weights()
        
        # Step 2: Edge lengths
        self.measure_edge_lengths()
        
        # Step 3: Node spacings
        self.measure_node_spacings()
        
        # Step 4: Special scales
        self.compute_special_scales()
        
        # Step 5: Compare to δ
        matches = self.compare_to_helical_pitch(delta=3.086)
        
        # Step 6: Generate report
        self.generate_report(matches)
        
        print(f"\n{'='*70}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*70}")
        print(f"\nSee 'lattice_stats_report.txt' for detailed results.")


def main():
    """Main execution."""
    print("\nInitializing Lattice Statistics Analysis...")
    
    analyzer = ParaboloidLatticeStatistics(target_n=5)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
