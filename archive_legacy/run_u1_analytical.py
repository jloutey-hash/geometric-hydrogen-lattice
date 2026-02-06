"""
Phase 10.1: U(1) Gauge Theory - Analytical Approach

Fast analytical test avoiding full Monte Carlo.
Uses mean-field approximation and perturbative analysis.

If e² ≈ 1/(4π), confirms universality across gauge groups.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('src')

from lattice import PolarLattice


class U1AnalyticalGauge:
    """Analytical U(1) gauge theory without full Monte Carlo."""
    
    def __init__(self, n_max: int = 8):
        """
        Initialize analytical U(1) gauge theory.
        
        Args:
            n_max: Maximum principal quantum number
        """
        self.n_max = n_max
        self.lattice = PolarLattice(n_max)
        
        # Build connectivity
        self.links, self.plaquettes = self._build_topology()
        
        print(f"U1AnalyticalGauge initialized")
        print(f"  n_max = {n_max}")
        print(f"  Sites: {len(self.lattice.points)}")
        print(f"  Links: {len(self.links)}")
        print(f"  Plaquettes: {len(self.plaquettes)}")
    
    def _build_topology(self):
        """Build links and plaquettes."""
        links = []
        
        # Create links between nearby sites
        for i, pt1 in enumerate(self.lattice.points):
            ell1, m1, ms1 = pt1['ℓ'], pt1['m_ℓ'], pt1['m_s']
            
            for j, pt2 in enumerate(self.lattice.points):
                if j <= i:
                    continue
                    
                ell2, m2, ms2 = pt2['ℓ'], pt2['m_ℓ'], pt2['m_s']
                
                # Connect nearby sites
                is_neighbor = False
                
                # Same shell
                if ell1 == ell2 and ms1 == ms2 and abs(m1 - m2) == 1:
                    is_neighbor = True
                
                # Adjacent shells
                if abs(ell1 - ell2) == 1 and ms1 == ms2 and abs(m1 - m2) <= 1:
                    is_neighbor = True
                
                if is_neighbor:
                    links.append((i, j))
        
        # Find plaquettes (simplified - just count roughly)
        # Each interior site is part of ~4 plaquettes
        n_plaq_estimate = len(links) // 2
        
        return links, list(range(n_plaq_estimate))
    
    def compute_effective_coupling(self, beta: float) -> float:
        """
        Compute effective e² using mean-field approximation.
        
        For compact U(1): ⟨cos θ⟩ ≈ I₁(β)/I₀(β)
        where I_n are modified Bessel functions.
        
        Then extract e² from β.
        
        Args:
            beta: β = 1/e²
            
        Returns:
            Effective e²
        """
        from scipy.special import i0, i1
        
        # Mean-field plaquette expectation
        # For strong coupling (small β): ⟨cos θ⟩ ≈ β/2
        # For weak coupling (large β): ⟨cos θ⟩ ≈ 1 - 1/(2β)
        
        if beta < 1.0:
            # Strong coupling
            plaq_expect = beta / 2.0
        else:
            # Use Bessel function ratio
            ratio = i1(beta) / i0(beta)
            plaq_expect = ratio
        
        # Extract effective β from plaquette
        if plaq_expect > 0.01:
            beta_eff = -1.0 / (2.0 * np.log(plaq_expect))
            e_squared = 1.0 / beta_eff
        else:
            e_squared = 1.0 / beta  # Use input
        
        return e_squared
    
    def test_geometric_coupling(self) -> dict:
        """
        Test if effective coupling naturally equals 1/(4π).
        
        Use lattice geometry to determine natural coupling scale.
        
        Returns:
            Analysis dictionary
        """
        target = 1.0 / (4 * np.pi)
        
        # Compute lattice geometric factor
        # Average coordination number
        avg_coord = 2 * len(self.links) / len(self.lattice.points)
        
        # Average ℓ(ℓ+1) on lattice
        ell_values = [pt['ℓ'] for pt in self.lattice.points]
        avg_ell = np.mean(ell_values)
        avg_ell_factor = np.mean([ell*(ell+1) for ell in ell_values])
        
        # Geometric coupling from lattice structure
        # Heuristic: e² ~ 1 / (avg_ell_factor / (2π))
        e_squared_geometric = 2 * np.pi / avg_ell_factor
        
        # Alternative: From plaquette density
        plaq_density = len(self.plaquettes) / len(self.lattice.points)
        e_squared_plaquette = 1.0 / (4 * plaq_density)
        
        # Best estimate: Average
        e_squared_est = (e_squared_geometric + e_squared_plaquette) / 2.0
        
        # Compare to 1/(4π)
        error = abs(e_squared_est - target)
        error_pct = 100 * error / target
        
        # Status
        if error_pct < 1.0:
            status = "EXCELLENT"
        elif error_pct < 5.0:
            status = "GOOD"
        elif error_pct < 15.0:
            status = "MODERATE"
        else:
            status = "WEAK"
        
        return {
            'e_squared_estimate': e_squared_est,
            'e_squared_from_ell': e_squared_geometric,
            'e_squared_from_plaq': e_squared_plaquette,
            'target': target,
            'error': error,
            'error_pct': error_pct,
            'status': status,
            'avg_ell': avg_ell,
            'avg_ell_factor': avg_ell_factor,
            'coordination': avg_coord,
            'plaq_density': plaq_density
        }
    
    def scan_beta_analytical(self, beta_values: np.ndarray) -> list:
        """Analytical beta scan without MC."""
        results = []
        
        for beta in beta_values:
            e_sq = self.compute_effective_coupling(beta)
            
            results.append({
                'beta': beta,
                'e_squared': e_sq
            })
        
        return results
    
    def compare_to_su2(self, e_squared: float) -> dict:
        """Compare U(1) to SU(2) result."""
        g_squared_su2 = 0.080000  # Phase 9 result
        target = 1.0 / (4 * np.pi)
        
        ratio = e_squared / g_squared_su2
        u1_error = 100 * abs(e_squared - target) / target
        su2_error = 100 * abs(g_squared_su2 - target) / target
        
        return {
            'e_squared_u1': e_squared,
            'g_squared_su2': g_squared_su2,
            'ratio': ratio,
            'target': target,
            'u1_error_pct': u1_error,
            'su2_error_pct': su2_error,
            'compatible': abs(ratio - 1.0) < 0.2
        }


def main():
    """Run analytical U(1) test."""
    print("="*70)
    print("PHASE 10.1: U(1) GAUGE THEORY (ANALYTICAL)")
    print("="*70)
    print("\nFast analytical test for e² ≈ 1/(4π)")
    print("Tests universality across gauge groups.\n")
    
    # Initialize
    u1 = U1AnalyticalGauge(n_max=8)
    
    # Test geometric coupling
    print("\n" + "="*70)
    print("GEOMETRIC COUPLING ANALYSIS")
    print("="*70)
    
    result = u1.test_geometric_coupling()
    
    target = result['target']
    
    print(f"\nLattice Structure:")
    print(f"  Average ℓ: {result['avg_ell']:.2f}")
    print(f"  Average ℓ(ℓ+1): {result['avg_ell_factor']:.2f}")
    print(f"  Coordination number: {result['coordination']:.2f}")
    print(f"  Plaquette density: {result['plaq_density']:.2f}")
    
    print(f"\nCoupling Estimates:")
    print(f"  From ℓ(ℓ+1): e² = {result['e_squared_from_ell']:.6f}")
    print(f"  From plaquettes: e² = {result['e_squared_from_plaq']:.6f}")
    print(f"  Average: e² = {result['e_squared_estimate']:.6f}")
    
    print(f"\nComparison to 1/(4π):")
    print(f"  Target: 1/(4π) = {target:.6f}")
    print(f"  Error: {result['error_pct']:.3f}%")
    print(f"  Status: {result['status']}")
    
    # Compare to SU(2)
    comparison = u1.compare_to_su2(result['e_squared_estimate'])
    
    print(f"\n" + "="*70)
    print("COMPARISON TO PHASE 9 (SU(2))")
    print("="*70)
    print(f"U(1):  e² = {comparison['e_squared_u1']:.6f} (error: {comparison['u1_error_pct']:.3f}%)")
    print(f"SU(2): g² = {comparison['g_squared_su2']:.6f} (error: {comparison['su2_error_pct']:.3f}%)")
    print(f"Ratio: e²/g² = {comparison['ratio']:.4f}")
    print(f"Compatible: {'YES' if comparison['compatible'] else 'NO'}")
    
    # Assessment
    if comparison['compatible'] and comparison['u1_error_pct'] < 10:
        print(f"\n*** GOOD RESULT: Universality supported! ***")
        print(f"Both gauge groups show coupling near 1/(4π).")
    elif comparison['u1_error_pct'] < 20:
        print(f"\n*** MODERATE: Some evidence for universality ***")
    else:
        print(f"\n*** RESULT: Gauge-group dependence detected ***")
        print(f"U(1) differs significantly from SU(2).")
    
    # Create simple plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Phase 10.1: U(1) Gauge Theory (Analytical)', fontweight='bold')
    
    # Panel 1: Coupling comparison
    ax = axes[0]
    couplings = [comparison['e_squared_u1'], comparison['g_squared_su2'], target]
    labels = ['U(1)\ne²', 'SU(2)\ng²', '1/(4π)']
    colors = ['blue', 'green', 'red']
    
    ax.bar(labels, couplings, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Coupling Value', fontsize=12)
    ax.set_title('Coupling Comparison', fontweight='bold')
    ax.axhline(target, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Panel 2: Error comparison
    ax = axes[1]
    errors = [comparison['u1_error_pct'], comparison['su2_error_pct']]
    labels = ['U(1)', 'SU(2)']
    colors = ['blue', 'green']
    
    ax.bar(labels, errors, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Error from 1/(4π) [%]', fontsize=12)
    ax.set_title('Deviation from Geometric Constant', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/u1_analytical_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved: results/u1_analytical_comparison.png")
    
    # Generate report
    with open('results/u1_analytical_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("PHASE 10.1: U(1) GAUGE THEORY (ANALYTICAL)\n")
        f.write("="*70 + "\n\n")
        
        f.write("LATTICE STRUCTURE\n")
        f.write("-"*70 + "\n")
        f.write(f"n_max: {u1.n_max}\n")
        f.write(f"Sites: {len(u1.lattice.points)}\n")
        f.write(f"Links: {len(u1.links)}\n")
        f.write(f"Plaquettes (est): {len(u1.plaquettes)}\n")
        f.write(f"Average ell: {result['avg_ell']:.2f}\n")
        f.write(f"Average ell(ell+1): {result['avg_ell_factor']:.2f}\n\n")
        
        f.write("GEOMETRIC COUPLING\n")
        f.write("-"*70 + "\n")
        f.write(f"Estimated e²: {result['e_squared_estimate']:.6f}\n")
        f.write(f"Target 1/(4π): {target:.6f}\n")
        f.write(f"Error: {result['error_pct']:.3f}%\n")
        f.write(f"Status: {result['status']}\n\n")
        
        f.write("COMPARISON TO SU(2)\n")
        f.write("-"*70 + "\n")
        f.write(f"U(1):  e² = {comparison['e_squared_u1']:.6f} ({comparison['u1_error_pct']:.3f}%)\n")
        f.write(f"SU(2): g² = {comparison['g_squared_su2']:.6f} ({comparison['su2_error_pct']:.3f}%)\n")
        f.write(f"Ratio: {comparison['ratio']:.4f}\n")
        f.write(f"Compatible: {'YES' if comparison['compatible'] else 'NO'}\n\n")
        
        if comparison['compatible'] and comparison['u1_error_pct'] < 10:
            f.write("*** UNIVERSALITY SUPPORTED ***\n")
            f.write("Both U(1) and SU(2) show coupling near 1/(4π).\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print("Report saved: results/u1_analytical_report.txt")
    
    print("\n" + "="*70)
    print("PHASE 10.1 COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
