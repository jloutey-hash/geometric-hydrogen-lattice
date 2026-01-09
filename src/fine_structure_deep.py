"""
Fine structure constant: Deep investigation with combined factors.

This module explores combinations of geometric quantities since
individual ratios showed >400% error. We test:
- Products of geometric factors
- Ratios between different quantities  
- Power series combinations
- Convergence with larger systems
"""

import numpy as np
from scipy import sparse
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from src.fine_structure import ALPHA_FINE


class DeepFineStructureExplorer:
    """
    Deep exploration of fine structure constant from combined geometric factors.
    """
    
    def __init__(self, lattice, operators=None, angular_momentum=None, spin_ops=None):
        """Initialize deep explorer."""
        self.lattice = lattice
        self.operators = operators
        self.angular_momentum = angular_momentum
        self.spin_ops = spin_ops
        self.results = {}
        
        # Pre-compute key geometric quantities
        self._compute_base_quantities()
    
    def _compute_base_quantities(self):
        """Pre-compute fundamental geometric quantities."""
        self.base_quantities = {}
        
        # Magic numbers
        self.base_quantities['magic_numbers'] = [2 * n**2 for n in range(1, self.lattice.n_max + 1)]
        
        # Ring parameters
        self.base_quantities['ring_radii'] = [1 + 2*ℓ for ℓ in range(self.lattice.ℓ_max + 1)]
        self.base_quantities['ring_points'] = [2*(2*ℓ+1) for ℓ in range(self.lattice.ℓ_max + 1)]
        
        # Overlap efficiency (from Phase 4)
        self.base_quantities['eta_overlap'] = 0.82
        
        # Selection rule compliance (from Phase 4)
        self.base_quantities['selection_compliance'] = 0.31
        
        # Convergence rate (from Phase 6)
        self.base_quantities['alpha_convergence'] = 0.19
        
        # Spin-orbit geometric coupling
        self.base_quantities['lambda_geom'] = []
        for ℓ in range(1, min(6, self.lattice.ℓ_max + 1)):
            theta_north = np.pi * (ℓ + 0.5) / (self.lattice.ℓ_max + 1)
            z_north = np.cos(theta_north)
            z_south = -z_north
            hemisphere_sep = abs(z_north - z_south)
            ring_spacing = 2
            self.base_quantities['lambda_geom'].append(hemisphere_sep / ring_spacing)
        
        # Angular momentum sums
        self.base_quantities['L_sum'] = []
        for n in range(1, min(6, self.lattice.n_max + 1)):
            sum_sqrt_l = sum(np.sqrt(ℓ*(ℓ+1)) for ℓ in range(n))
            self.base_quantities['L_sum'].append(sum_sqrt_l / n**2)
    
    def explore_products(self, verbose=False):
        """
        Explore products of geometric quantities.
        """
        if verbose:
            print("\n" + "="*70)
            print("INVESTIGATING PRODUCT COMBINATIONS")
            print("="*70)
        
        results = {
            'name': 'Product Combinations',
            'candidates': []
        }
        
        eta = self.base_quantities['eta_overlap']
        sel = self.base_quantities['selection_compliance']
        alpha_conv = self.base_quantities['alpha_convergence']
        
        # Two-factor products
        two_factor = [
            ('eta * selection', eta * sel),
            ('eta * alpha_conv', eta * alpha_conv),
            ('selection * alpha_conv', sel * alpha_conv),
            ('(1-eta) * selection', (1-eta) * sel),
            ('(1-eta) * alpha_conv', (1-eta) * alpha_conv),
            ('eta² * selection', eta**2 * sel),
            ('eta * selection²', eta * sel**2),
            ('√eta * √selection', np.sqrt(eta) * np.sqrt(sel)),
            ('√(eta * selection)', np.sqrt(eta * sel)),
        ]
        
        for desc, value in two_factor:
            results['candidates'].append({
                'description': f'Product: {desc}',
                'value': value,
                'deviation_from_alpha': abs(value - ALPHA_FINE),
                'relative_error': abs(value - ALPHA_FINE) / ALPHA_FINE
            })
        
        # Three-factor products
        three_factor = [
            ('eta * selection * alpha_conv', eta * sel * alpha_conv),
            ('(1-eta) * selection * alpha_conv', (1-eta) * sel * alpha_conv),
            ('√eta * √selection * √alpha_conv', np.sqrt(eta) * np.sqrt(sel) * np.sqrt(alpha_conv)),
            ('eta² * selection * alpha_conv', eta**2 * sel * alpha_conv),
            ('eta * selection² * alpha_conv', eta * sel**2 * alpha_conv),
        ]
        
        for desc, value in three_factor:
            results['candidates'].append({
                'description': f'Product: {desc}',
                'value': value,
                'deviation_from_alpha': abs(value - ALPHA_FINE),
                'relative_error': abs(value - ALPHA_FINE) / ALPHA_FINE
            })
        
        # With lambda_geom
        if self.base_quantities['lambda_geom']:
            for i, lam in enumerate(self.base_quantities['lambda_geom'][:3]):
                products = [
                    (f'lambda[{i+1}]² * eta', lam**2 * eta),
                    (f'lambda[{i+1}]² * selection', lam**2 * sel),
                    (f'lambda[{i+1}]² * alpha_conv', lam**2 * alpha_conv),
                    (f'lambda[{i+1}] * eta * selection', lam * eta * sel),
                ]
                
                for desc, value in products:
                    results['candidates'].append({
                        'description': f'Product: {desc}',
                        'value': value,
                        'deviation_from_alpha': abs(value - ALPHA_FINE),
                        'relative_error': abs(value - ALPHA_FINE) / ALPHA_FINE
                    })
        
        if verbose:
            best = min(results['candidates'], key=lambda x: x['relative_error'])
            print(f"\nBest product: {best['description']}")
            print(f"  Value: {best['value']:.8f}")
            print(f"  Error: {best['relative_error']*100:.2f}%")
        
        return results
    
    def explore_ratios(self, verbose=False):
        """
        Explore ratios between different geometric quantities.
        """
        if verbose:
            print("\n" + "="*70)
            print("INVESTIGATING RATIO COMBINATIONS")
            print("="*70)
        
        results = {
            'name': 'Ratio Combinations',
            'candidates': []
        }
        
        eta = self.base_quantities['eta_overlap']
        sel = self.base_quantities['selection_compliance']
        alpha_conv = self.base_quantities['alpha_convergence']
        
        # Complex ratios
        ratio_combos = [
            ('selection / eta', sel / eta),
            ('alpha_conv / eta', alpha_conv / eta),
            ('selection / (1-eta)', sel / (1-eta)),
            ('alpha_conv / (1-eta)', alpha_conv / (1-eta)),
            ('(1-eta) / (1+eta)', (1-eta) / (1+eta)),
            ('selection / (selection + eta)', sel / (sel + eta)),
            ('alpha_conv / (alpha_conv + eta)', alpha_conv / (alpha_conv + eta)),
            ('eta / (eta + 1/eta)', eta / (eta + 1/eta)),
            ('√(selection/eta)', np.sqrt(sel/eta)),
            ('√(alpha_conv/eta)', np.sqrt(alpha_conv/eta)),
        ]
        
        for desc, value in ratio_combos:
            results['candidates'].append({
                'description': f'Ratio: {desc}',
                'value': value,
                'deviation_from_alpha': abs(value - ALPHA_FINE),
                'relative_error': abs(value - ALPHA_FINE) / ALPHA_FINE
            })
        
        # Magic number ratios with other quantities
        magic = self.base_quantities['magic_numbers']
        if len(magic) >= 3:
            combos = [
                ('magic[0]/magic[1] * eta', magic[0]/magic[1] * eta),
                ('magic[0]/magic[1] * selection', magic[0]/magic[1] * sel),
                ('magic[0]/magic[2] * eta', magic[0]/magic[2] * eta),
                ('(magic[1]-magic[0])/magic[1] * alpha_conv', (magic[1]-magic[0])/magic[1] * alpha_conv),
            ]
            
            for desc, value in combos:
                results['candidates'].append({
                    'description': f'Ratio: {desc}',
                    'value': value,
                    'deviation_from_alpha': abs(value - ALPHA_FINE),
                    'relative_error': abs(value - ALPHA_FINE) / ALPHA_FINE
                })
        
        if verbose:
            best = min(results['candidates'], key=lambda x: x['relative_error'])
            print(f"\nBest ratio: {best['description']}")
            print(f"  Value: {best['value']:.8f}")
            print(f"  Error: {best['relative_error']*100:.2f}%")
        
        return results
    
    def explore_power_series(self, verbose=False):
        """
        Explore power series expansions.
        """
        if verbose:
            print("\n" + "="*70)
            print("INVESTIGATING POWER SERIES")
            print("="*70)
        
        results = {
            'name': 'Power Series',
            'candidates': []
        }
        
        # Series over shells
        for n_max_test in range(2, min(8, self.lattice.n_max + 1)):
            # Sum 1/n²
            series_1 = sum(1/n**2 for n in range(1, n_max_test + 1))
            results['candidates'].append({
                'description': f'Σ(1/n²) for n=1..{n_max_test}',
                'value': series_1,
                'deviation_from_alpha': abs(series_1 - ALPHA_FINE),
                'relative_error': abs(series_1 - ALPHA_FINE) / ALPHA_FINE
            })
            
            # Sum 1/n³
            series_2 = sum(1/n**3 for n in range(1, n_max_test + 1))
            results['candidates'].append({
                'description': f'Σ(1/n³) for n=1..{n_max_test}',
                'value': series_2,
                'deviation_from_alpha': abs(series_2 - ALPHA_FINE),
                'relative_error': abs(series_2 - ALPHA_FINE) / ALPHA_FINE
            })
            
            # Sum 1/(2n)²
            series_3 = sum(1/(2*n)**2 for n in range(1, n_max_test + 1))
            results['candidates'].append({
                'description': f'Σ(1/(2n)²) for n=1..{n_max_test}',
                'value': series_3,
                'deviation_from_alpha': abs(series_3 - ALPHA_FINE),
                'relative_error': abs(series_3 - ALPHA_FINE) / ALPHA_FINE
            })
        
        # Series over ℓ
        for ℓ_max_test in range(3, min(10, self.lattice.ℓ_max + 1)):
            # Sum 1/(2ℓ+1)²
            series_4 = sum(1/(2*ℓ+1)**2 for ℓ in range(ℓ_max_test + 1))
            results['candidates'].append({
                'description': f'Σ(1/(2ℓ+1)²) for ℓ=0..{ℓ_max_test}',
                'value': series_4,
                'deviation_from_alpha': abs(series_4 - ALPHA_FINE),
                'relative_error': abs(series_4 - ALPHA_FINE) / ALPHA_FINE
            })
            
            # Sum 1/(ℓ²+1)
            series_5 = sum(1/(ℓ**2+1) for ℓ in range(1, ℓ_max_test + 1))
            results['candidates'].append({
                'description': f'Σ(1/(ℓ²+1)) for ℓ=1..{ℓ_max_test}',
                'value': series_5,
                'deviation_from_alpha': abs(series_5 - ALPHA_FINE),
                'relative_error': abs(series_5 - ALPHA_FINE) / ALPHA_FINE
            })
        
        if verbose:
            best = min(results['candidates'], key=lambda x: x['relative_error'])
            print(f"\nBest series: {best['description']}")
            print(f"  Value: {best['value']:.8f}")
            print(f"  Error: {best['relative_error']*100:.2f}%")
        
        return results
    
    def explore_weighted_combinations(self, verbose=False):
        """
        Explore weighted linear combinations with optimization.
        """
        if verbose:
            print("\n" + "="*70)
            print("INVESTIGATING WEIGHTED COMBINATIONS")
            print("="*70)
        
        results = {
            'name': 'Weighted Combinations',
            'candidates': []
        }
        
        eta = self.base_quantities['eta_overlap']
        sel = self.base_quantities['selection_compliance']
        alpha_conv = self.base_quantities['alpha_convergence']
        
        # Grid search for a*x + b*y = α
        quantities = {
            'eta': eta,
            'selection': sel,
            'alpha_conv': alpha_conv,
            '1-eta': 1-eta,
            '√eta': np.sqrt(eta),
            '√selection': np.sqrt(sel),
        }
        
        # Two-term linear combinations
        keys = list(quantities.keys())
        for i, key1 in enumerate(keys):
            for key2 in keys[i+1:]:
                x, y = quantities[key1], quantities[key2]
                
                # Try various rational weights
                for a in [1, 2, 3, 0.5, 0.25, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8]:
                    for b in [1, 2, 3, 0.5, 0.25, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8]:
                        value = a*x + b*y
                        if 0 < value < 0.1:  # Only reasonable values
                            results['candidates'].append({
                                'description': f'{a:.4f}*{key1} + {b:.4f}*{key2}',
                                'value': value,
                                'deviation_from_alpha': abs(value - ALPHA_FINE),
                                'relative_error': abs(value - ALPHA_FINE) / ALPHA_FINE
                            })
        
        # Product and sum: a*x*y + b*z
        if len(quantities) >= 3:
            for key1 in keys[:3]:
                for key2 in keys[:3]:
                    for key3 in keys[:3]:
                        if key1 != key2:
                            x, y, z = quantities[key1], quantities[key2], quantities[key3]
                            for a in [1, 0.5, 0.1, 0.05]:
                                for b in [1, 0.5, 0.1, 0.05]:
                                    value = a*x*y + b*z
                                    if 0 < value < 0.05:
                                        results['candidates'].append({
                                            'description': f'{a:.3f}*{key1}*{key2} + {b:.3f}*{key3}',
                                            'value': value,
                                            'deviation_from_alpha': abs(value - ALPHA_FINE),
                                            'relative_error': abs(value - ALPHA_FINE) / ALPHA_FINE
                                        })
        
        if verbose:
            if results['candidates']:
                best = min(results['candidates'], key=lambda x: x['relative_error'])
                print(f"\nBest combination: {best['description']}")
                print(f"  Value: {best['value']:.8f}")
                print(f"  Error: {best['relative_error']*100:.2f}%")
        
        return results
    
    def explore_transcendental(self, verbose=False):
        """
        Explore transcendental functions of geometric quantities.
        """
        if verbose:
            print("\n" + "="*70)
            print("INVESTIGATING TRANSCENDENTAL FUNCTIONS")
            print("="*70)
        
        results = {
            'name': 'Transcendental Functions',
            'candidates': []
        }
        
        eta = self.base_quantities['eta_overlap']
        sel = self.base_quantities['selection_compliance']
        alpha_conv = self.base_quantities['alpha_convergence']
        
        # Exponential combinations
        transcendental = [
            ('exp(-1/eta)', np.exp(-1/eta)),
            ('exp(-1/sel)', np.exp(-1/sel)),
            ('exp(-1/(1-eta))', np.exp(-1/(1-eta))),
            ('exp(-eta)', np.exp(-eta)),
            ('ln(1/eta)', np.log(1/eta)),
            ('ln(1/sel)', np.log(1/sel)),
            ('ln(1/alpha_conv)', np.log(1/alpha_conv)),
            ('1/ln(1/eta)', 1/np.log(1/eta)),
            ('1/ln(1/sel)', 1/np.log(1/sel)),
            ('sin(eta)', np.sin(eta)),
            ('sin(π*sel)', np.sin(np.pi*sel)),
            ('tan(eta/4)', np.tan(eta/4)),
            ('arcsin(sel)', np.arcsin(sel)),
            ('arctan(alpha_conv)', np.arctan(alpha_conv)),
        ]
        
        for desc, value in transcendental:
            if np.isfinite(value) and value > 0:
                results['candidates'].append({
                    'description': f'Transcendental: {desc}',
                    'value': value,
                    'deviation_from_alpha': abs(value - ALPHA_FINE),
                    'relative_error': abs(value - ALPHA_FINE) / ALPHA_FINE
                })
        
        # Pi-based combinations
        pi_combos = [
            ('π * sel / 10', np.pi * sel / 10),
            ('π * alpha_conv / 10', np.pi * alpha_conv / 10),
            ('1/π * eta / 10', eta / (np.pi * 10)),
            ('sel² * π', sel**2 * np.pi),
            ('alpha_conv² * π / 4', alpha_conv**2 * np.pi / 4),
        ]
        
        for desc, value in pi_combos:
            results['candidates'].append({
                'description': f'Pi-based: {desc}',
                'value': value,
                'deviation_from_alpha': abs(value - ALPHA_FINE),
                'relative_error': abs(value - ALPHA_FINE) / ALPHA_FINE
            })
        
        if verbose:
            if results['candidates']:
                best = min(results['candidates'], key=lambda x: x['relative_error'])
                print(f"\nBest transcendental: {best['description']}")
                print(f"  Value: {best['value']:.8f}")
                print(f"  Error: {best['relative_error']*100:.2f}%")
        
        return results
    
    def explore_all_deep(self, verbose=True):
        """
        Run all deep explorations.
        """
        if verbose:
            print("="*70)
            print("DEEP FINE STRUCTURE EXPLORATION")
            print("Testing combinations of geometric factors")
            print("="*70)
            print(f"\nTarget: α = {ALPHA_FINE:.10f} (1/137.036)")
            print(f"Lattice: n_max={self.lattice.n_max}, N={len(self.lattice.points)} points")
        
        explorations = [
            ('Products', self.explore_products),
            ('Ratios', self.explore_ratios),
            ('Power Series', self.explore_power_series),
            ('Weighted Combinations', self.explore_weighted_combinations),
            ('Transcendental', self.explore_transcendental),
        ]
        
        all_results = []
        
        for name, func in explorations:
            result = func(verbose=verbose)
            self.results[name] = result
            if 'candidates' in result:
                all_results.extend(result['candidates'])
        
        # Synthesize
        synthesis = self._synthesize_deep(all_results)
        self.results['Synthesis'] = synthesis
        
        if verbose:
            self._print_deep_synthesis(synthesis)
        
        return self.results
    
    def _synthesize_deep(self, all_candidates):
        """Synthesize deep exploration results."""
        sorted_candidates = sorted(all_candidates, key=lambda x: x['relative_error'])
        
        errors = [c['relative_error'] for c in all_candidates if np.isfinite(c['relative_error'])]
        
        statistics = {
            'total_candidates': len(all_candidates),
            'mean_relative_error': np.mean(errors) if errors else np.inf,
            'median_relative_error': np.median(errors) if errors else np.inf,
            'min_relative_error': np.min(errors) if errors else np.inf,
            'within_1_percent': sum(1 for e in errors if e < 0.01),
            'within_5_percent': sum(1 for e in errors if e < 0.05),
            'within_10_percent': sum(1 for e in errors if e < 0.10),
            'within_20_percent': sum(1 for e in errors if e < 0.20),
        }
        
        return {
            'best_candidates': sorted_candidates[:50],
            'statistics': statistics
        }
    
    def _print_deep_synthesis(self, synthesis):
        """Print deep synthesis results."""
        print("\n" + "="*70)
        print("DEEP SYNTHESIS")
        print("="*70)
        
        stats = synthesis['statistics']
        print(f"\nTotal candidates: {stats['total_candidates']}")
        print(f"Within 1%:  {stats['within_1_percent']}")
        print(f"Within 5%:  {stats['within_5_percent']}")
        print(f"Within 10%: {stats['within_10_percent']}")
        print(f"Within 20%: {stats['within_20_percent']}")
        
        print(f"\n{'='*70}")
        print("TOP 15 DEEP CANDIDATES")
        print('='*70)
        
        for i, candidate in enumerate(synthesis['best_candidates'][:15], 1):
            print(f"\n{i}. {candidate['description']}")
            print(f"   Value: {candidate['value']:.10f}")
            print(f"   alpha = {ALPHA_FINE:.10f}")
            print(f"   Error: {candidate['relative_error']*100:.4f}%")
    
    def plot_deep_results(self, save_path=None):
        """Visualize deep exploration results."""
        if 'Synthesis' not in self.results:
            print("Run explore_all_deep() first")
            return
        
        synthesis = self.results['Synthesis']
        candidates = synthesis['best_candidates'][:40]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Best candidates
        ax = axes[0, 0]
        values = [c['value'] for c in candidates]
        errors = [c['relative_error'] for c in candidates]
        colors = ['green' if e < 0.10 else 'orange' if e < 0.20 else 'red' for e in errors]
        
        ax.scatter(range(len(values)), values, c=colors, s=100, alpha=0.7, edgecolors='black')
        ax.axhline(ALPHA_FINE, color='blue', linestyle='--', linewidth=2, label=f'α = {ALPHA_FINE:.6f}')
        ax.set_xlabel('Candidate rank', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Top 40 Combined Factor Candidates', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Error distribution (log scale)
        ax = axes[0, 1]
        all_errors = [c['relative_error']*100 for c in synthesis['best_candidates']]
        ax.hist(all_errors, bins=50, edgecolor='black', alpha=0.7, log=True)
        ax.axvline(1, color='green', linestyle='--', linewidth=2, label='1% error')
        ax.axvline(5, color='orange', linestyle='--', linewidth=2, label='5% error')
        ax.axvline(10, color='red', linestyle='--', linewidth=2, label='10% error')
        ax.set_xlabel('Relative error (%)', fontsize=12)
        ax.set_ylabel('Count (log scale)', fontsize=12)
        ax.set_title('Error Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.set_xlim(0, 200)
        
        # Plot 3: Error vs value scatter
        ax = axes[1, 0]
        vals = [c['value'] for c in synthesis['best_candidates'][:100]]
        errs = [c['relative_error'] for c in synthesis['best_candidates'][:100]]
        ax.scatter(vals, errs, alpha=0.6, s=50)
        ax.axhline(0.01, color='green', linestyle='--', alpha=0.5)
        ax.axhline(0.10, color='orange', linestyle='--', alpha=0.5)
        ax.axvline(ALPHA_FINE, color='blue', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xlabel('Candidate value', fontsize=12)
        ax.set_ylabel('Relative error', fontsize=12)
        ax.set_title('Value vs Error', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.set_xlim(0, 0.05)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        stats = synthesis['statistics']
        best = candidates[0]
        
        summary_text = f"""
DEEP EXPLORATION RESULTS

Target: α = {ALPHA_FINE:.10f}
      = 1/137.036

Total candidates: {stats['total_candidates']}

Accuracy breakdown:
  Within 1%:  {stats['within_1_percent']}
  Within 5%:  {stats['within_5_percent']}
  Within 10%: {stats['within_10_percent']}
  Within 20%: {stats['within_20_percent']}

Best candidate:
{best['description'][:45]}
Value: {best['value']:.10f}
Error: {best['relative_error']*100:.4f}%

Lattice: n_max={self.lattice.n_max}
        """
        
        ax.text(0.05, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round',
                facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
        
        return fig
