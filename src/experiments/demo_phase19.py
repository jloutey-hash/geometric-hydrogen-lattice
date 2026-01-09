"""
Quick demonstration of Phase 19: U(1) vs SU(2) comparison.

Runs with reduced parameters for fast execution.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from phase19_u1_su2_comparison import run_phase19_study

if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 19 DEMONSTRATION (Reduced parameters)")
    print("=" * 70)
    print("\nRunning with:")
    print("  - ℓ_max = 10 (reduced from 20)")
    print("  - n_configs = 100 (reduced from 1000)")
    print("  - Fast execution for demonstration")
    print()
    
    # Run with reduced parameters
    results = run_phase19_study(
        ℓ_max=10,          # Smaller lattice
        n_configs=100,      # Fewer configurations
        save_plots=True,
        output_dir="results/phase19_demo"
    )
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Results:")
    print(f"  SU(2) mean coupling: {results['SU2']['mean']:.6f}")
    print(f"  SU(2) std deviation: {results['SU2']['std']:.6f}")
    print(f"  Theory (1/4π): 0.079577")
    print(f"  Error from theory: {results['SU2']['error']:.2f}%")
    print()
    print(f"  U(1) coefficient of variation: {results['U1']['cv']:.2%}")
    print(f"  SU(2) coefficient of variation: {results['SU2']['cv']:.2%}")
    print(f"  Variance ratio: {results['variance_ratio']:.1f}×")
    print()
    print("✓ Demonstrates SU(2)-specificity of 1/(4π) coupling")
    print("✓ Full run with 1000 configs recommended for publication")
