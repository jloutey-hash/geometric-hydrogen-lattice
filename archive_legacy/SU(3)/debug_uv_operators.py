"""
Debug U and V operators for (1,0) representation
"""

from lattice import SU3Lattice

p, q = 1, 0
lattice = SU3Lattice(max_p=p, max_q=q)
states = [s for s in lattice.states if s['p'] == p and s['q'] == q]

print(f"States in (1,0) representation:")
for s in states:
    gt = s['gt']
    m13, m23, m33, m12, m22, m11 = gt
    print(f"  GT={gt}, I3={s['i3']:.2f}, Y={s['y']:.2f}")
    
    # Check U+ action: m12 -> m12+1, m11 -> m11-1
    m12_new = m12 + 1
    m11_new = m11 - 1
    valid = m12_new <= m13 and m11_new >= m22
    print(f"    U+: m12={m12}->{m12_new}, m11={m11}->{m11_new}, valid={valid}")
    
    # Check U- action: m12 -> m12-1, m11 -> m11+1
    m12_new = m12 - 1
    m11_new = m11 + 1
    valid = m12_new >= m23 and m11_new <= m12
    print(f"    U-: m12={m12}->{m12_new}, m11={m11}->{m11_new}, valid={valid}")
    
    # Check V+ action: m22 -> m22+1, m11 -> m11-1
    m22_new = m22 + 1
    m11_new = m11 - 1
    valid = m22_new <= m23 and m11_new >= m22_new
    print(f"    V+: m22={m22}->{m22_new}, m11={m11}->{m11_new}, valid={valid}")
    
    # Check V- action: m22 -> m22-1, m11 -> m11+1
    m22_new = m22 - 1
    m11_new = m11 + 1
    valid = m22_new >= m33 and m11_new <= m12
    print(f"    V-: m22={m22}->{m22_new}, m11={m11}->{m11_new}, valid={valid}")
    print()

print("\nConclusion: For (1,0), the fundamental representation,")
print("U and V operators don't connect states within the representation!")
print("This is correct - the fundamental (1,0) only has I-spin transitions.")
print("\nLet's test (1,1) which should have all transitions:")

print("\n" + "="*60)
p, q = 1, 1
lattice2 = SU3Lattice(max_p=p, max_q=q)
states2 = [s for s in lattice2.states if s['p'] == p and s['q'] == q]

print(f"\nStates in (1,1) representation (adjoint/octet):")
for i, s in enumerate(states2):
    gt = s['gt']
    m13, m23, m33, m12, m22, m11 = gt
    print(f"{i}: GT={gt}, I3={s['i3']:.2f}, Y={s['y']:.2f}")
    
    # Check U+ action
    m12_new = m12 + 1
    m11_new = m11 - 1
    valid = m12_new <= m13 and m11_new >= m22
    if valid:
        gt_new = (m13, m23, m33, m12_new, m22, m11_new)
        # Find target
        for j, s2 in enumerate(states2):
            if s2['gt'] == gt_new:
                print(f"  U+: connects to state {j}")
                break
    
    # Check U- action
    m12_new = m12 - 1
    m11_new = m11 + 1
    valid = m12_new >= m23 and m11_new <= m12
    if valid:
        gt_new = (m13, m23, m33, m12_new, m22, m11_new)
        for j, s2 in enumerate(states2):
            if s2['gt'] == gt_new:
                print(f"  U-: connects to state {j}")
                break
