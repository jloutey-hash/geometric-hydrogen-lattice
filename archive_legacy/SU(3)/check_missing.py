"""
Check which irrep the "missing" states belong to.
"""

from lattice import SU3Lattice


def check_missing_states():
    """Check GT patterns that appear in transitions but aren't in (1,1)."""
    print("="*70)
    print("Checking missing states")
    print("="*70)
    
    lattice = SU3Lattice(max_p=2, max_q=2)
    
    missing_gts = [
        (2, 1, 0, 1, 1, 0),
        (2, 1, 0, 2, 1, 0),
    ]
    
    for gt in missing_gts:
        found = False
        for s in lattice.states:
            if s['gt'] == gt:
                found = True
                print(f"\nGT {gt}:")
                print(f"  Belongs to irrep ({s['p']},{s['q']})")
                print(f"  Index: {s['index']}")
                print(f"  Weight: I3={s['i3']:.3f}, Y={s['y']:.3f}")
        
        if not found:
            print(f"\nGT {gt}: NOT FOUND in any irrep!")


if __name__ == "__main__":
    check_missing_states()
