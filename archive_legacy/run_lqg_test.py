"""
Test script for Phase 11.1: Loop Quantum Gravity Operators

LQG is fundamentally SU(2)-based, so this is where we expect strong matches!

Phase 9.1: SU(2) gauge g² = 0.080 (0.5% error from 1/(4π)) ✓
Phase 9.6: Spin networks matched 1/(4π) with 0.74% error ✓
Phase 10.1: U(1) did NOT match (Abelian ≠ non-Abelian) ✗
Phase 10.2: SU(3) did NOT match (different gauge group) ✗

Now testing full LQG: Does Immirzi parameter γ = 1/(4π)?
"""

import sys
sys.path.append('src')

from lqg_operators import main as run_lqg

def main():
    print("="*70)
    print("PHASE 11.1: LOOP QUANTUM GRAVITY TEST")
    print("="*70)
    print("\nLQG is SU(2)-based quantum gravity")
    print("This is where 1/(4π) should appear strongly!\n")
    
    print("Previous Results:")
    print("  ✓ Phase 8: Pure geometry → 1/(4π) (0.0015% error)")
    print("  ✓ Phase 9.1: SU(2) gauge → 1/(4π) (0.5% error)")
    print("  ✓ Phase 9.6: Spin networks → 1/(4π) (0.74% error)")
    print("  ✗ Phase 10.1: U(1) ≠ 1/(4π) (124% error)")
    print("  ✗ Phase 10.2: SU(3) ≠ 1/(4π) (889% error)")
    
    print("\nPattern: 1/(4π) appears in SU(2) & quantum geometry contexts")
    print("LQG combines both → expect strong match!\n")
    
    # Run analysis
    run_lqg()


if __name__ == '__main__':
    main()
