"""
PHASE 27: YUKAWA COUPLINGS - SIMPLIFIED DEMONSTRATION
=======================================================

Fast demonstration of fermion mass generation from Higgs VEV.

This version skips Monte Carlo thermalization and uses pre-set Higgs VEV
to demonstrate the physics principle quickly.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def run_yukawa_demo():
    """Quick demonstration of Yukawa mass generation."""
    
    print("=" * 70)
    print("PHASE 27: YUKAWA COUPLINGS")
    print("FERMION MASS GENERATION FROM HIGGS VEV")
    print("=" * 70)
    
    # Create output directory
    output_path = Path("results/phase27")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Physics parameters
    higgs_vev = 0.12  # Measured from Phase 26 (typical value from 4⁴ lattice)
    yukawa_values = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 5.0])  # Range of couplings
    
    # Compute fermion masses: m = y × v
    fermion_masses = yukawa_values * higgs_vev
    
    print(f"\nHiggs VEV: <phi> = {higgs_vev:.4f}")
    print("\nYUKAWA COUPLING -> FERMION MASS:")
    print("-" * 70)
    for y, m in zip(yukawa_values, fermion_masses):
        print(f"  y = {y:.2f}  ->  m = y*v = {m:.4f}")
    
    # Standard Model comparison
    print("\n" + "=" * 70)
    print("STANDARD MODEL ANALOGY:")
    print("=" * 70)
    
    # Real SM values (approximate)
    sm_vev = 246  # GeV (Higgs VEV)
    
    particles = ['Electron', 'Muon', 'Bottom', 'Charm', 'Top']
    sm_masses = np.array([0.511, 105.7, 4180, 1270, 172760])  # MeV
    sm_yukawa = sm_masses / (sm_vev * 1000)  # Convert to dimensionless
    
    print("\nParticle        Mass (MeV)     Yukawa (y)")
    print("-" * 70)
    for particle, mass, y in zip(particles, sm_masses, sm_yukawa):
        print(f"{particle:12s}  {mass:10.1f}    {y:.2e}")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("=" * 70)
    print("ALL fermion masses come from the SAME Higgs VEV!")
    print("The mass hierarchy is determined by Yukawa couplings:")
    print("  • Electron is light (y ~ 10⁻⁶)")
    print("  • Top is heavy (y ~ 1)")
    print("  • All interact with SAME Higgs field!")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Phase 27: Yukawa Couplings - Fermion Mass Generation',
                 fontsize=16, fontweight='bold')
    
    # Panel 1: Linear relationship m = y·v
    ax1 = axes[0, 0]
    ax1.plot(yukawa_values, fermion_masses, 'o-', linewidth=2, markersize=10,
             color='blue', label=f'Lattice (v={higgs_vev:.3f})')
    ax1.plot([0, yukawa_values[-1]], [0, yukawa_values[-1] * higgs_vev], '--',
             color='gray', alpha=0.5, label='m = y·v (theory)')
    ax1.set_xlabel('Yukawa Coupling y', fontsize=12)
    ax1.set_ylabel('Fermion Mass m (lattice units)', fontsize=12)
    ax1.set_title('Mass Generation: m = y·v', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Panel 2: Log-log plot showing hierarchy
    ax2 = axes[0, 1]
    nonzero = yukawa_values > 0
    ax2.loglog(yukawa_values[nonzero], fermion_masses[nonzero], 'o-',
               linewidth=2, markersize=10, color='red')
    ax2.set_xlabel('Yukawa Coupling y', fontsize=12)
    ax2.set_ylabel('Fermion Mass m', fontsize=12)
    ax2.set_title('Mass Hierarchy (Log-Log)', fontweight='bold')
    ax2.grid(True, which='both', alpha=0.3)
    
    # Panel 3: Standard Model masses
    ax3 = axes[1, 0]
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    ax3.barh(particles, sm_masses, color=colors, alpha=0.7)
    ax3.set_xscale('log')
    ax3.set_xlabel('Mass (MeV)', fontsize=12)
    ax3.set_title('Standard Model Fermion Masses', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Panel 4: Standard Model Yukawa couplings
    ax4 = axes[1, 1]
    ax4.barh(particles, sm_yukawa, color=colors, alpha=0.7)
    ax4.set_xscale('log')
    ax4.set_xlabel('Yukawa Coupling y', fontsize=12)
    ax4.set_title('Standard Model Yukawa Couplings\n(6 orders of magnitude!)',
                 fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_path / "phase27_yukawa_summary.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {plot_file}")
    plt.close()
    
    # Save results
    results = {
        "higgs_vev": float(higgs_vev),
        "yukawa_values": yukawa_values.tolist(),
        "fermion_masses": fermion_masses.tolist(),
        "standard_model": {
            "vev_GeV": float(sm_vev),
            "particles": particles,
            "masses_MeV": sm_masses.tolist(),
            "yukawa_couplings": sm_yukawa.tolist()
        },
        "physics": {
            "mechanism": "Spontaneous symmetry breaking gives Higgs VEV",
            "mass_formula": "m_fermion = y_yukawa × v_higgs",
            "hierarchy": "Different y values explain mass hierarchy",
            "universality": "All fermions couple to SAME Higgs field"
        }
    }
    
    results_file = output_path / "phase27_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {results_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("PHASE 27 COMPLETE!")
    print("=" * 70)
    print("\nPHYSICS DEMONSTRATED:")
    print("1. Fermion mass generation: m = y·v")
    print("2. Mass hierarchy from Yukawa couplings")
    print("3. Universal Higgs interaction")
    print("\nNOBEL CONNECTION:")
    print("Part of 2013 Physics Nobel Prize (Higgs mechanism)")
    print("Explains origin of ALL fermion masses in Standard Model!")
    
    # Open visualization
    import os
    os.startfile(plot_file)
    
    return results


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("YUKAWA COUPLINGS: THE ORIGIN OF FERMION MASSES")
    print("=" * 70)
    print("\nThe Higgs doesn't just break symmetry...")
    print("It CREATES MASS for every quark and lepton!")
    print("\nLet's see how...\n")
    
    results = run_yukawa_demo()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAY:")
    print("=" * 70)
    print("The mass of a fermion (quark or lepton) is:")
    print("")
    print("    m = y × v")
    print("")
    print("where:")
    print("  * v = Higgs VEV (<phi> ~ 246 GeV in real world)")
    print("  * y = Yukawa coupling (determines which particle)")
    print("")
    print("This simple formula explains:")
    print("  * Why top quark is 340,000x heavier than electron")
    print("  * Why all particles get mass from Higgs")
    print("  * The origin of mass hierarchy in nature!")
    print("=" * 70)
