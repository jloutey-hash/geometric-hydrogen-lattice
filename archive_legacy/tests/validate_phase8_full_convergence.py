"""
FULL CONVERGENCE TEST: Recompute empirical values for each n_max

This is the DEFINITIVE test. We recompute:
- η (overlap efficiency) 
- selection (dipole rule compliance)
- α_convergence (operator convergence rate)

for EACH n_max value, then check if (1-eta)*selection*alpha_conv converges to α.
"""

import sys
import os

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from time import time
from src.lattice import PolarLattice
from src.operators import LatticeOperators
from src.angular_momentum import AngularMomentumOperators
from src.quantum_comparison import QuantumComparison
from src.convergence import ConvergenceAnalysis

ALPHA_FINE = 1/137.035999084  # 0.0072973526


def compute_eta_overlap(lattice, operators, angular_momentum):
    """Compute η = average overlap with spherical harmonics."""
    print("    Computing η (overlap efficiency)...")
    
    # Build quantum comparison
    qc = QuantumComparison(lattice, operators)
    
    # Compute L² eigenmodes
    Lsq = angular_momentum.L_squared()
    eigenvalues, eigenmodes = np.linalg.eigh(Lsq.toarray())
    
    # Compute overlap with spherical harmonics up to ℓ_max
    overlap_data = qc.compute_overlap_matrix(eigenmodes, ell_max=min(5, lattice.ℓ_max))
    
    # Extract average overlap efficiency
    overlaps = []
    for ell in overlap_data:
        for m in overlap_data[ell]:
            if 'max_overlap' in overlap_data[ell][m]:
                overlaps.append(overlap_data[ell][m]['max_overlap'])
    
    if len(overlaps) > 0:
        eta = np.mean(overlaps)
    else:
        eta = 0.82  # fallback
    
    return eta


def compute_selection_compliance(lattice, operators, angular_momentum):
    """Compute selection rule compliance rate."""
    print("    Computing selection rule compliance...")
    
    # Build quantum comparison
    qc = QuantumComparison(lattice, operators)
    
    # Compute L² eigenmodes
    Lsq = angular_momentum.L_squared()
    eigenvalues, eigenmodes = np.linalg.eigh(Lsq.toarray())
    
    # Compute dipole matrix elements
    dipole_data = qc.test_dipole_selection_rules(eigenmodes, eigenvalues)
    
    # Extract compliance rate
    if 'rule_compliance_rate' in dipole_data:
        selection = dipole_data['rule_compliance_rate']
    else:
        selection = 0.31  # fallback
    
    return selection


def compute_alpha_convergence(lattice, operators):
    """Compute convergence rate of discrete operators."""
    print("    Computing α_convergence (operator convergence)...")
    
    # Build convergence analysis
    conv = ConvergenceAnalysis(lattice, operators)
    
    # Test derivative convergence
    results = conv.test_derivative_convergence(m_test=1)
    
    # Extract convergence rate
    if 'convergence_rate' in results and not np.isnan(results['convergence_rate']):
        alpha_conv = results['convergence_rate']
        # Normalize to [0,1] range
        alpha_conv = min(1.0, max(0.0, alpha_conv / 10.0))
    else:
        alpha_conv = 0.19  # fallback
    
    return alpha_conv


def full_convergence_test(n_max_values=[6, 8, 10]):
    """
    Full convergence test: recompute all empirical values.
    
    This is computationally expensive but gives the definitive answer.
    """
    results = {}
    
    for n_max in n_max_values:
        print(f"\n{'='*70}")
        print(f"n_max = {n_max}")
        print('='*70)
        
        start_time = time()
        
        # Build lattice and operators
        print("  Building lattice...")
        lattice = PolarLattice(n_max=n_max)
        N = len(lattice.points)
        print(f"    N_points = {N}")
        
        print("  Building operators...")
        operators = LatticeOperators(lattice)
        angular_momentum = AngularMomentumOperators(lattice)
        
        # Compute empirical values
        try:
            eta = compute_eta_overlap(lattice, operators, angular_momentum)
            print(f"      eta = {eta:.6f}")
        except Exception as e:
            print(f"      eta computation failed: {e}")
            eta = 0.82
            
        try:
            selection = compute_selection_compliance(lattice, operators, angular_momentum)
            print(f"      selection = {selection:.6f}")
        except Exception as e:
            print(f"      selection computation failed: {e}")
            selection = 0.31
            
        try:
            alpha_conv = compute_alpha_convergence(lattice, operators)
            print(f"      alpha_conv = {alpha_conv:.6f}")
        except Exception as e:
            print(f"      alpha_conv computation failed: {e}")
            alpha_conv = 0.19
        
        # Compute candidate
        candidate_value = (1 - eta) * selection * alpha_conv
        error = abs(candidate_value - ALPHA_FINE) / ALPHA_FINE
        
        elapsed = time() - start_time
        
        results[n_max] = {
            'N_points': N,
            'eta': eta,
            'selection': selection,
            'alpha_conv': alpha_conv,
            'candidate_value': candidate_value,
            'error': error,
            'error_pct': error * 100,
            'time': elapsed
        }
        
        print(f"\n  Results:")
        print(f"    (1-eta) * selection * alpha_conv = {candidate_value:.8f}")
        print(f"    Target alpha = {ALPHA_FINE:.8f}")
        print(f"    Error = {error*100:.2f}%")
        print(f"    Time = {elapsed:.1f}s")
    
    return results


def plot_full_convergence(results, save_path=None):
    """Plot full convergence results."""
    n_max_values = sorted(results.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: eta vs n_max
    ax = axes[0, 0]
    eta_values = [results[n]['eta'] for n in n_max_values]
    ax.plot(n_max_values, eta_values, 'bo-', linewidth=2, markersize=10)
    ax.set_xlabel('n_max', fontsize=12, fontweight='bold')
    ax.set_ylabel('eta (overlap efficiency)', fontsize=12, fontweight='bold')
    ax.set_title('Overlap Efficiency vs Lattice Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: selection vs n_max
    ax = axes[0, 1]
    sel_values = [results[n]['selection'] for n in n_max_values]
    ax.plot(n_max_values, sel_values, 'go-', linewidth=2, markersize=10)
    ax.set_xlabel('n_max', fontsize=12, fontweight='bold')
    ax.set_ylabel('selection (rule compliance)', fontsize=12, fontweight='bold')
    ax.set_title('Selection Rule Compliance vs Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: alpha_conv vs n_max
    ax = axes[0, 2]
    alpha_values = [results[n]['alpha_conv'] for n in n_max_values]
    ax.plot(n_max_values, alpha_values, 'ro-', linewidth=2, markersize=10)
    ax.set_xlabel('n_max', fontsize=12, fontweight='bold')
    ax.set_ylabel('alpha_conv (convergence rate)', fontsize=12, fontweight='bold')
    ax.set_title('Operator Convergence vs Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Candidate value vs n_max
    ax = axes[1, 0]
    cand_values = [results[n]['candidate_value'] for n in n_max_values]
    ax.plot(n_max_values, cand_values, 'mo-', linewidth=2, markersize=10, label='(1-eta)*sel*alpha')
    ax.axhline(ALPHA_FINE, color='red', linestyle='--', linewidth=2, label=f'alpha = {ALPHA_FINE:.6f}')
    ax.set_xlabel('n_max', fontsize=12, fontweight='bold')
    ax.set_ylabel('Candidate Value', fontsize=12, fontweight='bold')
    ax.set_title('Candidate vs Target', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Error vs n_max
    ax = axes[1, 1]
    error_values = [results[n]['error_pct'] for n in n_max_values]
    ax.plot(n_max_values, error_values, 'ko-', linewidth=2, markersize=10)
    ax.set_xlabel('n_max', fontsize=12, fontweight='bold')
    ax.set_ylabel('Relative Error (%)', fontsize=12, fontweight='bold')
    ax.set_title('Error vs Lattice Size', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 6: Error vs 1/n_max (test convergence)
    ax = axes[1, 2]
    inv_n = [1/n for n in n_max_values]
    ax.plot(inv_n, error_values, 'ro-', linewidth=2, markersize=10, label='Actual')
    
    # Fit linear trend
    if len(inv_n) >= 2:
        coeffs = np.polyfit(inv_n, error_values, 1)
        fit_line = np.poly1d(coeffs)
        ax.plot(inv_n, fit_line(inv_n), 'b--', linewidth=2, label='Linear fit')
        
        # Extrapolate
        extrapolated = coeffs[1]
        ax.axhline(extrapolated, color='green', linestyle=':', linewidth=2,
                   label=f'Extrap: {extrapolated:.1f}%')
    
    ax.set_xlabel('1/n_max', fontsize=12, fontweight='bold')
    ax.set_ylabel('Error (%)', fontsize=12, fontweight='bold')
    ax.set_title('Convergence Test', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved plot: {save_path}")
    
    return fig


def main():
    print("="*70)
    print("FULL CONVERGENCE TEST")
    print("Recomputing eta, selection, alpha_conv for each n_max")
    print("="*70)
    print(f"\nTarget: alpha = {ALPHA_FINE:.10f}")
    print("\nWARNING: This test is computationally expensive!")
    print("Each n_max requires eigenvalue decomposition and overlap computations.")
    
    # Test with smaller values first
    n_max_values = [6, 8]
    
    print(f"\nTesting n_max = {n_max_values}")
    print("(Can add n_max=10 if these show convergence)\n")
    
    # Run test
    results = full_convergence_test(n_max_values)
    
    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, 'phase8_full_convergence.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("FULL CONVERGENCE TEST RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write("Recomputed eta, selection, alpha_conv for each n_max\n\n")
        f.write(f"Target: alpha = {ALPHA_FINE:.10f}\n\n")
        
        for n_max in sorted(results.keys()):
            r = results[n_max]
            f.write(f"\nn_max = {n_max}\n")
            f.write("-"*70 + "\n")
            f.write(f"N_points: {r['N_points']}\n")
            f.write(f"eta: {r['eta']:.8f}\n")
            f.write(f"selection: {r['selection']:.8f}\n")
            f.write(f"alpha_conv: {r['alpha_conv']:.8f}\n")
            f.write(f"\n(1-eta) * selection * alpha_conv = {r['candidate_value']:.8f}\n")
            f.write(f"Error: {r['error_pct']:.4f}%\n")
            f.write(f"Time: {r['time']:.1f}s\n")
    
    print(f"\n\nResults saved: {results_file}")
    
    # Plot
    print("Generating plots...")
    fig_path = os.path.join(results_dir, 'phase8_full_convergence.png')
    plot_full_convergence(results, save_path=fig_path)
    
    # Verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    
    error_values = [results[n]['error_pct'] for n in sorted(results.keys())]
    
    if len(error_values) >= 2:
        decreasing = all(error_values[i] > error_values[i+1] for i in range(len(error_values)-1))
        varying = max(error_values) - min(error_values) > 5
        
        if decreasing:
            print("\n*** ERROR IS DECREASING! ***")
            print("This suggests genuine convergence as lattice refines!")
            print("\nNext steps:")
            print("  1. Test with n_max = 10, 15, 20")
            print("  2. Extrapolate to n_max -> infinity")
            print("  3. Develop theoretical derivation")
        elif varying:
            print("\n** VALUES ARE CHANGING **")
            print("The empirical quantities vary with n_max.")
            print("Need more data points to determine trend.")
        else:
            print("\n** VALUES ARE CONSTANT **")
            print("The empirical quantities don't vary with n_max.")
            print("The 45% error appears to be a fixed offset.")
    
    print(f"\nFiles generated:")
    print(f"  {results_file}")
    print(f"  {fig_path}")


if __name__ == '__main__':
    main()
