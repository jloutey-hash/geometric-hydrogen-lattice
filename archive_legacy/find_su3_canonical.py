"""
SU(3) Canonical Representation Finder

Identify canonical SU(3) representations that play a role analogous to hydrogen n=5
in the U(1) electromagnetic calculation.

Searches for representations with extremal properties:
- Z_per_state minima/maxima
- Packing efficiency extrema
- Resonance detection
- Mixing index extrema
- C2-normalized impedance extrema

Author: Unified Geometry Framework
Date: February 5, 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
from scipy.signal import find_peaks, argrelextrema
from scipy.interpolate import griddata


class CanonicalRepFinder:
    """
    Find canonical SU(3) representations with extremal geometric properties.
    """
    
    def __init__(self, csv_file: str = 'su3_impedance_packing_scan_extended.csv'):
        """
        Initialize finder with extended dataset.
        
        Parameters
        ----------
        csv_file : str
            Path to CSV with impedance + packing data
        """
        self.csv_file = csv_file
        self.df = None
        self.candidates = []
        
    def load_and_compute_derived(self):
        """
        Load data and compute all derived quantities.
        """
        print("\n" + "="*80)
        print("CANONICAL SU(3) REPRESENTATION FINDER")
        print("="*80)
        print("\nLoading data from:", self.csv_file)
        
        self.df = pd.read_csv(self.csv_file)
        
        # Filter out non-finite values
        initial_count = len(self.df)
        self.df = self.df[np.isfinite(self.df['Z'])]
        filtered_count = initial_count - len(self.df)
        
        if filtered_count > 0:
            print(f"  Filtered {filtered_count} reps with non-finite Z")
        
        print(f"  Total representations: {len(self.df)}")
        print(f"  (p,q) range: p+q <= {self.df['p'].max() + self.df['q'].max()}")
        
        # Compute derived quantities
        print("\nComputing derived quantities...")
        
        # Basic normalized quantities
        self.df['Z_eff'] = self.df['Z']
        self.df['C_per_state'] = self.df['C_matter'] / self.df['dim']
        self.df['Z_per_state'] = self.df['Z_eff'] / self.df['dim']
        self.df['S_per_state'] = self.df['S_holonomy'] / self.df['dim']
        
        # Casimir-normalized
        self.df['Z_per_C2'] = self.df['Z_eff'] / self.df['C2']
        
        # Mixing indices
        self.df['min_pq'] = self.df[['p', 'q']].min(axis=1)
        self.df['max_pq'] = self.df[['p', 'q']].max(axis=1)
        self.df['mixing_index'] = self.df['p'] * self.df['q']
        self.df['symmetry_index'] = np.abs(self.df['p'] - self.df['q'])
        
        # Classification
        self.df['rep_type'] = self.df.apply(
            lambda row: 'pure' if row['p'] == 0 or row['q'] == 0 else 'mixed',
            axis=1
        )
        
        # Summary statistics
        print(f"  Derived quantities computed: {len(self.df.columns)} columns")
        print(f"  Pure reps: {(self.df['rep_type'] == 'pure').sum()}")
        print(f"  Mixed reps: {(self.df['rep_type'] == 'mixed').sum()}")
        
        # Print ranges
        print("\nKey quantity ranges:")
        for col in ['Z_eff', 'Z_per_state', 'Z_per_C2', 'packing_efficiency_mean', 'mixing_index']:
            if col in self.df.columns:
                print(f"  {col:30s}: [{self.df[col].min():.6f}, {self.df[col].max():.6f}]")
        
        return self.df
    
    def find_extrema(self, column: str, n_top: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Find local extrema (minima and maxima) in a column.
        
        Parameters
        ----------
        column : str
            Column name to analyze
        n_top : int
            Number of top extrema to return
        
        Returns
        -------
        minima, maxima : DataFrame
            Top n minima and maxima
        """
        # Sort by dimension to find local extrema
        df_sorted = self.df.sort_values('dim').reset_index(drop=True)
        
        # Find local minima and maxima
        values = df_sorted[column].values
        
        # Use scipy to find local extrema
        min_indices = argrelextrema(values, np.less, order=3)[0]
        max_indices = argrelextrema(values, np.greater, order=3)[0]
        
        # Also include global extrema
        global_min_idx = values.argmin()
        global_max_idx = values.argmax()
        
        min_indices = np.unique(np.append(min_indices, global_min_idx))
        max_indices = np.unique(np.append(max_indices, global_max_idx))
        
        # Get DataFrames
        df_minima = df_sorted.iloc[min_indices].copy()
        df_maxima = df_sorted.iloc[max_indices].copy()
        
        # Sort by value and take top n
        df_minima = df_minima.nsmallest(n_top, column)
        df_maxima = df_maxima.nlargest(n_top, column)
        
        return df_minima, df_maxima
    
    def detect_resonances(self, threshold: float = 2.0) -> pd.DataFrame:
        """
        Detect resonances: reps where Z differs significantly from neighbors.
        
        A representation (p,q) is a resonance if:
        |Z(p,q) - mean(Z_neighbors)| / std(Z_neighbors) > threshold
        
        Parameters
        ----------
        threshold : float
            Number of standard deviations for resonance detection
        
        Returns
        -------
        resonances : DataFrame
            Representations identified as resonances
        """
        print(f"\nDetecting resonances (threshold = {threshold} σ)...")
        
        resonances = []
        
        for idx, row in self.df.iterrows():
            p, q = int(row['p']), int(row['q'])
            Z = row['Z_eff']
            
            # Find neighbors: (p±1,q), (p,q±1), (p±1,q±1)
            neighbor_positions = [
                (p-1, q), (p+1, q),
                (p, q-1), (p, q+1),
                (p-1, q-1), (p+1, q+1),
                (p-1, q+1), (p+1, q-1)
            ]
            
            # Get neighbor Z values
            neighbor_Z = []
            for np_val, nq_val in neighbor_positions:
                if np_val >= 0 and nq_val >= 0:  # Valid rep
                    neighbor = self.df[(self.df['p'] == np_val) & (self.df['q'] == nq_val)]
                    if len(neighbor) > 0:
                        neighbor_Z.append(neighbor['Z_eff'].iloc[0])
            
            if len(neighbor_Z) >= 2:
                mean_neighbor = np.mean(neighbor_Z)
                std_neighbor = np.std(neighbor_Z)
                
                if std_neighbor > 0:
                    resonance_score = abs(Z - mean_neighbor) / std_neighbor
                    
                    if resonance_score > threshold:
                        resonances.append({
                            'p': p,
                            'q': q,
                            'Z_eff': Z,
                            'mean_neighbor_Z': mean_neighbor,
                            'resonance_score': resonance_score,
                            'dim': row['dim'],
                            'C2': row['C2']
                        })
        
        df_resonances = pd.DataFrame(resonances)
        
        if len(df_resonances) > 0:
            df_resonances = df_resonances.sort_values('resonance_score', ascending=False)
            print(f"  Found {len(df_resonances)} resonances")
        else:
            print("  No resonances found")
        
        return df_resonances
    
    def rank_candidates(self, n_top: int = 10) -> pd.DataFrame:
        """
        Rank all representations by multiple criteria.
        
        Scoring based on:
        1. Z_per_state extremality (distance from median)
        2. Packing efficiency extremality
        3. Mixing index
        4. Resonance score
        5. C2-normalized impedance extremality
        
        Parameters
        ----------
        n_top : int
            Number of top candidates to return
        
        Returns
        -------
        ranked : DataFrame
            Top candidates with scores
        """
        print("\nRanking candidates by multiple criteria...")
        
        df_work = self.df.copy()
        
        # Compute scores for each criterion (0-1 normalized)
        
        # 1. Z_per_state extremality (distance from median)
        median_zps = df_work['Z_per_state'].median()
        df_work['score_zps'] = np.abs(df_work['Z_per_state'] - median_zps) / df_work['Z_per_state'].std()
        
        # 2. Packing efficiency extremality
        median_pack = df_work['packing_efficiency_mean'].median()
        df_work['score_packing'] = np.abs(df_work['packing_efficiency_mean'] - median_pack) / df_work['packing_efficiency_mean'].std()
        
        # 3. Mixing index (prefer moderate mixing)
        max_mixing = df_work['mixing_index'].max()
        df_work['score_mixing'] = df_work['mixing_index'] / max_mixing if max_mixing > 0 else 0
        
        # 4. C2-normalized impedance extremality
        median_zc2 = df_work['Z_per_C2'].median()
        df_work['score_zc2'] = np.abs(df_work['Z_per_C2'] - median_zc2) / df_work['Z_per_C2'].std()
        
        # 5. Low dimension preference (easier to interpret)
        df_work['score_lowdim'] = 1.0 / (1.0 + df_work['dim'] / 10.0)
        
        # Composite score (weighted sum)
        weights = {
            'score_zps': 3.0,       # Most important
            'score_packing': 2.0,   
            'score_mixing': 1.5,
            'score_zc2': 2.0,
            'score_lowdim': 1.0
        }
        
        df_work['composite_score'] = sum(
            df_work[col] * weight for col, weight in weights.items()
        )
        
        # Normalize composite score
        df_work['composite_score'] = df_work['composite_score'] / sum(weights.values())
        
        # Sort by composite score
        ranked = df_work.sort_values('composite_score', ascending=False).head(n_top)
        
        print(f"  Top {n_top} candidates identified")
        
        return ranked
    
    def identify_canonical_candidates(self) -> Dict[str, pd.DataFrame]:
        """
        Main analysis: identify canonical representation candidates.
        
        Returns
        -------
        results : dict
            Dictionary with:
            - 'ranked': Top candidates by composite score
            - 'z_per_state_min': Minimum Z_per_state reps
            - 'z_per_state_max': Maximum Z_per_state reps
            - 'packing_max': Maximum packing efficiency
            - 'resonances': Detected resonances
        """
        print("\n" + "="*80)
        print("IDENTIFYING CANONICAL CANDIDATES")
        print("="*80)
        
        # Find extrema in different quantities
        print("\n1. Z_per_state extrema...")
        zps_min, zps_max = self.find_extrema('Z_per_state', n_top=5)
        
        print("\n2. Packing efficiency extrema...")
        pack_min, pack_max = self.find_extrema('packing_efficiency_mean', n_top=5)
        
        print("\n3. Z_per_C2 extrema...")
        zc2_min, zc2_max = self.find_extrema('Z_per_C2', n_top=5)
        
        print("\n4. Detecting resonances...")
        resonances = self.detect_resonances(threshold=1.5)
        
        print("\n5. Computing composite ranking...")
        ranked = self.rank_candidates(n_top=10)
        
        results = {
            'ranked': ranked,
            'z_per_state_min': zps_min,
            'z_per_state_max': zps_max,
            'packing_min': pack_min,
            'packing_max': pack_max,
            'z_per_c2_min': zc2_min,
            'z_per_c2_max': zc2_max,
            'resonances': resonances
        }
        
        return results
    
    def print_candidate_summary(self, results: Dict[str, pd.DataFrame]):
        """
        Print summary of canonical candidates.
        """
        print("\n" + "="*80)
        print("CANONICAL CANDIDATE SUMMARY")
        print("="*80)
        
        print("\n" + "-"*80)
        print("TOP 5 COMPOSITE SCORE CANDIDATES")
        print("-"*80)
        cols = ['p', 'q', 'dim', 'C2', 'Z_eff', 'Z_per_state', 'packing_efficiency_mean', 
                'mixing_index', 'composite_score']
        print(results['ranked'][cols].head(5).to_string(index=False))
        
        print("\n" + "-"*80)
        print("MINIMUM Z_PER_STATE (Most Efficient)")
        print("-"*80)
        cols = ['p', 'q', 'dim', 'C2', 'Z_eff', 'Z_per_state', 'rep_type']
        print(results['z_per_state_min'][cols].head(5).to_string(index=False))
        
        print("\n" + "-"*80)
        print("MAXIMUM Z_PER_STATE (Most Resistant)")
        print("-"*80)
        print(results['z_per_state_max'][cols].head(5).to_string(index=False))
        
        print("\n" + "-"*80)
        print("MAXIMUM PACKING EFFICIENCY")
        print("-"*80)
        cols = ['p', 'q', 'dim', 'packing_efficiency_mean', 'Z_per_state', 'rep_type']
        print(results['packing_max'][cols].head(5).to_string(index=False))
        
        if len(results['resonances']) > 0:
            print("\n" + "-"*80)
            print("RESONANCES (Anomalous Z values)")
            print("-"*80)
            print(results['resonances'].head(5).to_string(index=False))
    
    def plot_analysis(self, output_prefix: str = 'su3_canonical'):
        """
        Generate comprehensive visualization suite.
        """
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Separate pure and mixed reps
        df_pure = self.df[self.df['rep_type'] == 'pure']
        df_mixed = self.df[self.df['rep_type'] == 'mixed']
        
        # 1. Z vs dim (log-log)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(df_pure['dim'], df_pure['Z_eff'], s=50, alpha=0.6, 
                   label='Pure', marker='o', color='blue')
        ax1.scatter(df_mixed['dim'], df_mixed['Z_eff'], s=80, alpha=0.6,
                   label='Mixed', marker='s', color='green')
        ax1.set_xlabel('Dimension', fontsize=10)
        ax1.set_ylabel('Z_eff', fontsize=10)
        ax1.set_title('Z vs Dimension', fontsize=11, fontweight='bold')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Z_per_state vs dim
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(df_pure['dim'], df_pure['Z_per_state'], s=50, alpha=0.6,
                   label='Pure', marker='o', color='blue')
        ax2.scatter(df_mixed['dim'], df_mixed['Z_per_state'], s=80, alpha=0.6,
                   label='Mixed', marker='s', color='green')
        ax2.set_xlabel('Dimension', fontsize=10)
        ax2.set_ylabel('Z per state', fontsize=10)
        ax2.set_title('Normalized Impedance', fontsize=11, fontweight='bold')
        ax2.set_yscale('log')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. Z vs mixing index
        ax3 = fig.add_subplot(gs[0, 2])
        sc = ax3.scatter(self.df['mixing_index'], self.df['Z_eff'], 
                        c=self.df['dim'], s=60, alpha=0.6, cmap='viridis')
        ax3.set_xlabel('Mixing Index (p×q)', fontsize=10)
        ax3.set_ylabel('Z_eff', fontsize=10)
        ax3.set_title('Z vs Mixing', fontsize=11, fontweight='bold')
        ax3.set_yscale('log')
        plt.colorbar(sc, ax=ax3, label='dim')
        ax3.grid(True, alpha=0.3)
        
        # 4. Z vs packing efficiency
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.scatter(df_pure['packing_efficiency_mean'], df_pure['Z_eff'], 
                   s=50, alpha=0.6, label='Pure', marker='o', color='blue')
        ax4.scatter(df_mixed['packing_efficiency_mean'], df_mixed['Z_eff'],
                   s=80, alpha=0.6, label='Mixed', marker='s', color='green')
        ax4.set_xlabel('Packing Efficiency', fontsize=10)
        ax4.set_ylabel('Z_eff', fontsize=10)
        ax4.set_title('Z vs Packing', fontsize=11, fontweight='bold')
        ax4.set_yscale('log')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. Heatmap: Z over (p,q) grid
        ax5 = fig.add_subplot(gs[1, 1])
        
        # Create grid for heatmap
        p_max = int(self.df['p'].max())
        q_max = int(self.df['q'].max())
        
        Z_grid = np.full((q_max+1, p_max+1), np.nan)
        
        for _, row in self.df.iterrows():
            p_idx = int(row['p'])
            q_idx = int(row['q'])
            Z_grid[q_idx, p_idx] = row['Z_eff']
        
        # Use log scale for better visualization
        Z_grid_log = np.log10(Z_grid + 1e-10)
        
        im = ax5.imshow(Z_grid_log, origin='lower', aspect='auto', cmap='RdYlBu_r',
                       interpolation='nearest')
        ax5.set_xlabel('p', fontsize=10)
        ax5.set_ylabel('q', fontsize=10)
        ax5.set_title('log10(Z) Heatmap', fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax5, label='log10(Z)')
        
        # 6. Z_per_C2 vs C2
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.scatter(df_pure['C2'], df_pure['Z_per_C2'], s=50, alpha=0.6,
                   label='Pure', marker='o', color='blue')
        ax6.scatter(df_mixed['C2'], df_mixed['Z_per_C2'], s=80, alpha=0.6,
                   label='Mixed', marker='s', color='green')
        ax6.set_xlabel('Casimir C2', fontsize=10)
        ax6.set_ylabel('Z / C2', fontsize=10)
        ax6.set_title('C2-Normalized Impedance', fontsize=11, fontweight='bold')
        ax6.set_yscale('log')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
        
        # 7. Dimension histogram
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.hist(df_pure['dim'], bins=20, alpha=0.6, label='Pure', color='blue')
        ax7.hist(df_mixed['dim'], bins=20, alpha=0.6, label='Mixed', color='green')
        ax7.set_xlabel('Dimension', fontsize=10)
        ax7.set_ylabel('Count', fontsize=10)
        ax7.set_title('Dimension Distribution', fontsize=11, fontweight='bold')
        ax7.legend(fontsize=8)
        ax7.grid(True, alpha=0.3, axis='y')
        
        # 8. Z_per_state distribution
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.hist(np.log10(df_pure['Z_per_state']), bins=20, alpha=0.6, 
                label='Pure', color='blue')
        ax8.hist(np.log10(df_mixed['Z_per_state']), bins=20, alpha=0.6,
                label='Mixed', color='green')
        ax8.set_xlabel('log10(Z per state)', fontsize=10)
        ax8.set_ylabel('Count', fontsize=10)
        ax8.set_title('Z per State Distribution', fontsize=11, fontweight='bold')
        ax8.legend(fontsize=8)
        ax8.grid(True, alpha=0.3, axis='y')
        
        # 9. Symmetry index vs Z
        ax9 = fig.add_subplot(gs[2, 2])
        sc = ax9.scatter(self.df['symmetry_index'], self.df['Z_eff'],
                        c=self.df['mixing_index'], s=60, alpha=0.6, cmap='plasma')
        ax9.set_xlabel('Symmetry Index |p-q|', fontsize=10)
        ax9.set_ylabel('Z_eff', fontsize=10)
        ax9.set_title('Z vs Symmetry', fontsize=11, fontweight='bold')
        ax9.set_yscale('log')
        plt.colorbar(sc, ax=ax9, label='Mixing (p×q)')
        ax9.grid(True, alpha=0.3)
        
        plt.suptitle('SU(3) Canonical Representation Analysis', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        output_file = f'{output_prefix}_analysis.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        
        plt.close()


def run_canonical_finder():
    """
    Main execution: find canonical SU(3) representations.
    """
    # Initialize finder
    finder = CanonicalRepFinder('su3_impedance_packing_scan_extended.csv')
    
    # Load and compute derived quantities
    df = finder.load_and_compute_derived()
    
    # Identify candidates
    results = finder.identify_canonical_candidates()
    
    # Print summary
    finder.print_candidate_summary(results)
    
    # Generate visualizations
    finder.plot_analysis('su3_canonical')
    
    # Save detailed results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save full derived dataset
    output_csv = 'su3_canonical_derived.csv'
    df.to_csv(output_csv, index=False)
    print(f"  Full dataset: {output_csv}")
    
    # Save top candidates
    candidates_csv = 'su3_canonical_candidates.csv'
    results['ranked'].to_csv(candidates_csv, index=False)
    print(f"  Top candidates: {candidates_csv}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return finder, results


if __name__ == "__main__":
    finder, results = run_canonical_finder()
