"""
ALPHA DEEP SEARCH: Extended Hunt for Fine Structure Constant
==============================================================
The initial hunt revealed:
  - Gearing ratio L/T converges to ~1.77 (not 137)
  - Commutator [T+, L+] = 0 (perfect integrability - the algebra is closed!)
  - Holonomy defect ~100% (lattice curvature << expected sphere curvature)

New strategies:
1. DIMENSIONAL ANALYSIS: Ratios of lattice dimensions vs quantum numbers
2. CONNECTIVITY RATIOS: Degree patterns across shells
3. SPECTRAL GAPS: Eigenvalue spacing ratios
4. CROSS-OPERATOR PRODUCTS: ||T+||·||L+|| / N combinations
5. HIGHER-ORDER COMMUTATORS: [[T+, L+], ...] brackets

Author: Geometric Constant Discovery Team
Date: February 2026
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from paraboloid_lattice_su11 import ParaboloidLattice
from typing import Dict, List
import time


class AlphaDeepSearch:
    """
    Enhanced search for fine structure constant in deeper geometric patterns.
    """
    
    def __init__(self):
        self.alpha = 0.0072973525693
        self.inv_alpha = 137.035999084
        self.alpha_over_2pi = 0.0011614
        
        self.results = {}
    
    def compute_connectivity_ratio(self, n: int) -> Dict:
        """
        Strategy: Alpha may appear in connectivity density ratios.
        
        Check: (Total edges) / (N nodes)^2
               (Degree variance) / (Mean degree)
        """
        print(f"\n[n={n}] Computing connectivity metrics...")
        
        lattice = ParaboloidLattice(max_n=n)
        n_nodes = len(lattice.nodes)
        
        # Build adjacency
        A = abs(lattice.Tplus) + abs(lattice.Tminus) + abs(lattice.Lplus) + abs(lattice.Lminus)
        
        # Count edges (undirected, so divide by 2)
        n_edges = A.nnz / 2
        
        # Compute degrees
        degrees = np.array(A.sum(axis=1)).flatten()
        mean_degree = np.mean(degrees)
        var_degree = np.var(degrees)
        std_degree = np.std(degrees)
        
        # Connectivity ratios
        edge_density = n_edges / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0
        degree_ratio = var_degree / mean_degree if mean_degree > 0 else 0
        cv_degree = std_degree / mean_degree if mean_degree > 0 else 0  # Coefficient of variation
        
        result = {
            'n': n,
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'edge_density': edge_density,
            'mean_degree': mean_degree,
            'var_degree': var_degree,
            'std_degree': std_degree,
            'degree_ratio': degree_ratio,
            'cv_degree': cv_degree
        }
        
        print(f"  Nodes: {n_nodes}, Edges: {n_edges}")
        print(f"  Edge density: {edge_density:.6e}")
        print(f"  Degree CV: {cv_degree:.6f}")
        print(f"  Var/Mean: {degree_ratio:.6f}")
        
        return result
    
    def compute_dimensional_ratios(self, n: int) -> Dict:
        """
        Strategy: Alpha may relate quantum numbers to lattice structure.
        
        Check: n^2 / N_nodes (states vs lattice size)
               sqrt(N_nodes) / n (dimensionality scaling)
        """
        print(f"\n[n={n}] Computing dimensional ratios...")
        
        lattice = ParaboloidLattice(max_n=n)
        n_nodes = len(lattice.nodes)
        
        # Theoretical: N(n) = n(n+1)(2n+1)/6 for sum of squares
        # Actual: Count states with principal quantum number <= n
        
        ratio_1 = n**2 / n_nodes if n_nodes > 0 else 0
        ratio_2 = np.sqrt(n_nodes) / n if n > 0 else 0
        ratio_3 = n_nodes / n**3 if n > 0 else 0
        ratio_4 = n / np.sqrt(n_nodes) if n_nodes > 0 else 0
        
        result = {
            'n': n,
            'n_nodes': n_nodes,
            'n2_over_N': ratio_1,
            'sqrtN_over_n': ratio_2,
            'N_over_n3': ratio_3,
            'n_over_sqrtN': ratio_4
        }
        
        print(f"  n²/N = {ratio_1:.6f}")
        print(f"  sqrt(N)/n = {ratio_2:.6f}")
        print(f"  N/n³ = {ratio_3:.6f}")
        print(f"  n/sqrt(N) = {ratio_4:.6f}")
        
        return result
    
    def compute_operator_products(self, n: int) -> Dict:
        """
        Strategy: Alpha may appear in cross products of operator norms.
        
        Check: ||T+|| · ||L+|| / N
               ||T+||/||L+|| · sqrt(N)
               (||T+|| + ||L+||) / (||T+|| · ||L+||)
        """
        print(f"\n[n={n}] Computing operator products...")
        
        lattice = ParaboloidLattice(max_n=n)
        n_nodes = len(lattice.nodes)
        
        # Frobenius norms
        norm_T = np.sqrt(np.sum(np.abs(lattice.Tplus.data)**2))
        norm_L = np.sqrt(np.sum(np.abs(lattice.Lplus.data)**2))
        
        # Products and ratios
        product = norm_T * norm_L
        product_over_N = product / n_nodes if n_nodes > 0 else 0
        product_over_N2 = product / (n_nodes**2) if n_nodes > 0 else 0
        
        ratio_times_sqrtN = (norm_T / norm_L) * np.sqrt(n_nodes) if norm_L > 0 else 0
        
        harmonic_mean = 2 * norm_T * norm_L / (norm_T + norm_L) if (norm_T + norm_L) > 0 else 0
        
        result = {
            'n': n,
            'norm_T': norm_T,
            'norm_L': norm_L,
            'product': product,
            'product_over_N': product_over_N,
            'product_over_N2': product_over_N2,
            'ratio_times_sqrtN': ratio_times_sqrtN,
            'harmonic_mean': harmonic_mean
        }
        
        print(f"  ||T||·||L|| = {product:.6f}")
        print(f"  (||T||·||L||)/N = {product_over_N:.6f}")
        print(f"  (T/L)·sqrt(N) = {ratio_times_sqrtN:.6f}")
        
        return result
    
    def compute_spectral_gaps(self, n: int) -> Dict:
        """
        Strategy: Alpha may appear in eigenvalue spacing patterns.
        
        Compute first few eigenvalues of adjacency matrix and measure gaps.
        """
        if n > 12:  # Too expensive for large n
            return {'n': n, 'skipped': True}
        
        print(f"\n[n={n}] Computing spectral gaps...")
        
        lattice = ParaboloidLattice(max_n=n)
        
        # Adjacency matrix
        A = abs(lattice.Tplus) + abs(lattice.Tminus) + abs(lattice.Lplus) + abs(lattice.Lminus)
        
        # Compute top eigenvalues
        try:
            eigenvalues = eigsh(A, k=min(10, len(lattice.nodes)-2), which='LA', return_eigenvectors=False)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
            
            # Compute gaps
            gaps = np.diff(eigenvalues)
            
            # Gap ratios (Wigner surmise for chaotic systems gives specific distribution)
            gap_ratios = []
            for i in range(len(gaps)-1):
                r = min(gaps[i], gaps[i+1]) / max(gaps[i], gaps[i+1]) if max(gaps[i], gaps[i+1]) > 0 else 0
                gap_ratios.append(r)
            
            mean_gap = np.mean(gaps) if len(gaps) > 0 else 0
            mean_ratio = np.mean(gap_ratios) if len(gap_ratios) > 0 else 0
            
            result = {
                'n': n,
                'eigenvalues': eigenvalues.tolist(),
                'gaps': gaps.tolist(),
                'gap_ratios': gap_ratios,
                'mean_gap': mean_gap,
                'mean_ratio': mean_ratio
            }
            
            print(f"  Top eigenvalue: {eigenvalues[0]:.6f}")
            print(f"  Mean gap: {mean_gap:.6f}")
            print(f"  Mean gap ratio: {mean_ratio:.6f}")
            
            return result
            
        except Exception as e:
            print(f"  [FAILED] {e}")
            return {'n': n, 'error': str(e)}
    
    def compute_curvature_ratios(self, n: int) -> Dict:
        """
        Strategy: Alpha may appear in curvature vs dimension ratios.
        
        The Berry phase scaling is ~n^(-2.11). Check if 2.11/alpha or similar.
        """
        print(f"\n[n={n}] Computing curvature metrics...")
        
        # Berry phase estimate (from previous results: theta ~ n^(-2.11))
        k_berry = 2.113
        A_berry = 2.323
        
        theta_estimate = A_berry * (n ** (-k_berry))
        
        # Ratios
        k_over_alpha = k_berry / self.alpha
        k_times_alpha = k_berry * self.alpha
        
        # Check if k relates to alpha
        result = {
            'n': n,
            'k_berry': k_berry,
            'theta_estimate': theta_estimate,
            'k_over_alpha': k_over_alpha,
            'k_times_alpha': k_times_alpha,
            'k_minus_2': k_berry - 2.0
        }
        
        print(f"  k_Berry = {k_berry:.6f}")
        print(f"  k/alpha = {k_over_alpha:.6f}")
        print(f"  k·alpha = {k_times_alpha:.6f}")
        print(f"  k-2 = {k_berry - 2.0:.6f}")
        
        return result
    
    def check_alpha_proximity(self, name: str, value: float) -> bool:
        """
        Check if value is within 10% of any alpha-related constant.
        """
        targets = {
            'alpha': self.alpha,
            '1/alpha': self.inv_alpha,
            'alpha/(2π)': self.alpha_over_2pi,
            'sqrt(alpha)': np.sqrt(self.alpha),
            'alpha^2': self.alpha**2,
            '2*alpha': 2*self.alpha,
            'pi*alpha': np.pi*self.alpha,
            '4*pi*alpha': 4*np.pi*self.alpha,
            'alpha/pi': self.alpha/np.pi
        }
        
        found = False
        for target_name, target_value in targets.items():
            rel_error = abs(value - target_value) / target_value if target_value != 0 else float('inf')
            
            if rel_error < 0.05:  # Within 5%
                print(f"    *** MATCH: {name} ≈ {target_name} (error: {rel_error*100:.2f}%) ***")
                found = True
            elif rel_error < 0.15:  # Within 15%
                print(f"    CLOSE: {name} ≈ {target_name} (error: {rel_error*100:.2f}%)")
                found = True
        
        return found
    
    def run_deep_search(self):
        """
        Execute comprehensive search.
        """
        print("="*80)
        print("ALPHA DEEP SEARCH")
        print("="*80)
        print(f"\nSearching for alpha = {self.alpha:.10f}")
        print(f"                1/alpha = {self.inv_alpha:.10f}")
        print()
        
        # Store all results
        connectivity_results = []
        dimensional_results = []
        operator_product_results = []
        spectral_results = []
        
        # Scan shells
        for n in range(2, 21, 2):
            print(f"\n{'='*80}")
            print(f"SHELL n = {n}")
            print(f"{'='*80}")
            
            # Run all analyses
            conn = self.compute_connectivity_ratio(n)
            connectivity_results.append(conn)
            
            dim = self.compute_dimensional_ratios(n)
            dimensional_results.append(dim)
            
            op = self.compute_operator_products(n)
            operator_product_results.append(op)
            
            if n <= 12:
                spec = self.compute_spectral_gaps(n)
                spectral_results.append(spec)
        
        # Curvature analysis (constant for all n)
        curv = self.compute_curvature_ratios(10)
        
        # Final convergence analysis
        print("\n" + "="*80)
        print("CONVERGENCE ANALYSIS - SEARCHING FOR ALPHA")
        print("="*80)
        
        # Check connectivity patterns
        print("\n1. CONNECTIVITY RATIOS:")
        if connectivity_results:
            cv_values = [r['cv_degree'] for r in connectivity_results[-5:]]
            mean_cv = np.mean(cv_values)
            print(f"  Mean CV (last 5): {mean_cv:.6f}")
            self.check_alpha_proximity("CV", mean_cv)
        
        # Check dimensional ratios
        print("\n2. DIMENSIONAL RATIOS:")
        if dimensional_results:
            n2_N = [r['n2_over_N'] for r in dimensional_results[-5:]]
            mean_n2_N = np.mean(n2_N)
            print(f"  Mean n²/N (last 5): {mean_n2_N:.6f}")
            self.check_alpha_proximity("n²/N", mean_n2_N)
            
            n_sqrtN = [r['n_over_sqrtN'] for r in dimensional_results[-5:]]
            mean_n_sqrtN = np.mean(n_sqrtN)
            print(f"  Mean n/sqrt(N) (last 5): {mean_n_sqrtN:.6f}")
            self.check_alpha_proximity("n/sqrt(N)", mean_n_sqrtN)
        
        # Check operator products
        print("\n3. OPERATOR PRODUCTS:")
        if operator_product_results:
            prod_N = [r['product_over_N'] for r in operator_product_results[-5:]]
            mean_prod_N = np.mean(prod_N)
            print(f"  Mean (||T||·||L||)/N (last 5): {mean_prod_N:.6f}")
            self.check_alpha_proximity("(||T||·||L||)/N", mean_prod_N)
        
        # Check spectral gaps
        print("\n4. SPECTRAL STATISTICS:")
        if spectral_results:
            valid_ratios = [r['mean_ratio'] for r in spectral_results if 'mean_ratio' in r]
            if valid_ratios:
                mean_gap_ratio = np.mean(valid_ratios)
                print(f"  Mean gap ratio: {mean_gap_ratio:.6f}")
                self.check_alpha_proximity("Gap ratio", mean_gap_ratio)
        
        # Check Berry phase exponent
        print("\n5. BERRY PHASE EXPONENT:")
        print(f"  k = {curv['k_berry']:.6f}")
        print(f"  k - 2 = {curv['k_minus_2']:.6f}")
        self.check_alpha_proximity("k-2", curv['k_minus_2'])
        self.check_alpha_proximity("k", curv['k_berry'])
        
        print("\n" + "="*80)
        print("DEEP SEARCH COMPLETE")
        print("="*80)


def main():
    searcher = AlphaDeepSearch()
    searcher.run_deep_search()


if __name__ == "__main__":
    main()
