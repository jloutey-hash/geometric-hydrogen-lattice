"""
Test a single SU(3) representation to verify the algebra
"""

import numpy as np
from scipy.sparse import lil_matrix
from lattice import SU3Lattice
from operators_v2 import SU3Operators

# Test with fundamental representation (1, 0)
print("="*60)
print("Testing fundamental representation (1, 0)")
print("="*60)

p, q = 1, 0
lattice = SU3Lattice(max_p=p, max_q=q)
states = [s for s in lattice.states if s['p'] == p and s['q'] == q]

print(f"\nDimension: {len(states)} (theory: {(p+1)*(q+1)*(p+q+2)//2})")
print("\nStates:")
for s in states:
    gt = s['gt']
    print(f"  {s['index']}: GT={gt}, I3={s['i3']:.2f}, Y={s['y']:.2f}")

# Build operators for just this representation
ops = SU3Operators(lattice)

# Extract the sub-block for this representation
indices = [s['index'] for s in states]
dim = len(indices)

def extract_block(op, indices):
    """Extract the sub-block corresponding to these indices."""
    block = lil_matrix((dim, dim), dtype=complex)
    for i, idx_i in enumerate(indices):
        for j, idx_j in enumerate(indices):
            val = op[idx_i, idx_j]
            if abs(val) > 1e-14:
                block[i, j] = val
    return block.tocsr()

T3 = extract_block(ops.T3, indices)
T8 = extract_block(ops.T8, indices)
Ip = extract_block(ops.I_plus, indices)
Im = extract_block(ops.I_minus, indices)
Up = extract_block(ops.U_plus, indices)
Um = extract_block(ops.U_minus, indices)
Vp = extract_block(ops.V_plus, indices)
Vm = extract_block(ops.V_minus, indices)

print("\n" + "="*60)
print("Testing Commutators")
print("="*60)

# Test [I+, I-] = 2*T3
comm = (Ip @ Im - Im @ Ip).toarray()
expected = (2 * T3).toarray()
error = np.max(np.abs(comm - expected))
print(f"\n[I+, I-] = 2*T3: max error = {error:.2e}")
if error < 1e-13:
    print("  ✓ PASSED")
else:
    print("  ✗ FAILED")
    print("Commutator:")
    print(comm)
    print("Expected:")
    print(expected)

# Test [U+, U-] = -(3/2)*T3 + (sqrt(3)/2)*T8
comm = (Up @ Um - Um @ Up).toarray()
expected = (-1.5 * T3 + np.sqrt(3)/2 * T8).toarray()
error = np.max(np.abs(comm - expected))
print(f"\n[U+, U-] = -(3/2)*T3 + (sqrt(3)/2)*T8: max error = {error:.2e}")
if error < 1e-13:
    print("  ✓ PASSED")
else:
    print("  ✗ FAILED")
    print("Commutator:")
    print(comm)
    print("Expected:")
    print(expected)

# Test [V+, V-] = (3/2)*T3 + (sqrt(3)/2)*T8
comm = (Vp @ Vm - Vm @ Vp).toarray()
expected = (1.5 * T3 + np.sqrt(3)/2 * T8).toarray()
error = np.max(np.abs(comm - expected))
print(f"\n[V+, V-] = (3/2)*T3 + (sqrt(3)/2)*T8: max error = {error:.2e}")
if error < 1e-13:
    print("  ✓ PASSED")
else:
    print("  ✗ FAILED")
    print("Commutator:")
    print(comm)
    print("Expected:")
    print(expected)

# Test [I+, U-] = V-
comm = (Ip @ Um - Um @ Ip).toarray()
expected = Vm.toarray()
error = np.max(np.abs(comm - expected))
print(f"\n[I+, U-] = V-: max error = {error:.2e}")
if error < 1e-13:
    print("  ✓ PASSED")
else:
    print("  ✗ FAILED")
    
# Test Casimir
C2 = (T3 @ T3 + T8 @ T8 + 
      0.5 * (Ip @ Im + Im @ Ip) +
      0.5 * (Up @ Um + Um @ Up) +
      0.5 * (Vp @ Vm + Vm @ Vp))

C2_vals = np.diag(C2.toarray())
C2_theory = (p*p + q*q + 3*p + 3*q + p*q) / 3.0

print("\n" + "="*60)
print("Casimir Operator")
print("="*60)
print(f"\nTheory: C2 = {C2_theory:.6f}")
print(f"Computed: {C2_vals}")
print(f"Error: {np.max(np.abs(C2_vals - C2_theory)):.2e}")

# Now test (2, 1)
print("\n\n" + "="*60)
print("Testing representation (2, 1)")
print("="*60)

p, q = 2, 1
lattice2 = SU3Lattice(max_p=p, max_q=q)
states2 = [s for s in lattice2.states if s['p'] == p and s['q'] == q]

print(f"\nDimension: {len(states2)} (theory: {(p+1)*(q+1)*(p+q+2)//2})")

ops2 = SU3Operators(lattice2)
indices2 = [s['index'] for s in states2]

T3_2 = extract_block(ops2.T3, indices2)
T8_2 = extract_block(ops2.T8, indices2)
Ip_2 = extract_block(ops2.I_plus, indices2)
Im_2 = extract_block(ops2.I_minus, indices2)
Up_2 = extract_block(ops2.U_plus, indices2)
Um_2 = extract_block(ops2.U_minus, indices2)
Vp_2 = extract_block(ops2.V_plus, indices2)
Vm_2 = extract_block(ops2.V_minus, indices2)

# Test [I+, I-] = 2*T3
comm = (Ip_2 @ Im_2 - Im_2 @ Ip_2).toarray()
expected = (2 * T3_2).toarray()
error = np.max(np.abs(comm - expected))
print(f"\n[I+, I-] = 2*T3: max error = {error:.2e}")
if error < 1e-13:
    print("  ✓ PASSED")
else:
    print("  ✗ FAILED")

# Test [U+, U-]
comm = (Up_2 @ Um_2 - Um_2 @ Up_2).toarray()
expected = (-1.5 * T3_2 + np.sqrt(3)/2 * T8_2).toarray()
error = np.max(np.abs(comm - expected))
print(f"\n[U+, U-] = -(3/2)*T3 + (sqrt(3)/2)*T8: max error = {error:.2e}")
if error < 1e-13:
    print("  ✓ PASSED")
else:
    print("  ✗ FAILED")

# Test [V+, V-]
comm = (Vp_2 @ Vm_2 - Vm_2 @ Vp_2).toarray()
expected = (1.5 * T3_2 + np.sqrt(3)/2 * T8_2).toarray()
error = np.max(np.abs(comm - expected))
print(f"\n[V+, V-] = (3/2)*T3 + (sqrt(3)/2)*T8: max error = {error:.2e}")
if error < 1e-13:
    print("  ✓ PASSED")
else:
    print("  ✗ FAILED")

# Test Casimir
C2_2 = (T3_2 @ T3_2 + T8_2 @ T8_2 + 
        0.5 * (Ip_2 @ Im_2 + Im_2 @ Ip_2) +
        0.5 * (Up_2 @ Um_2 + Um_2 @ Up_2) +
        0.5 * (Vp_2 @ Vm_2 + Vm_2 @ Vp_2))

C2_vals_2 = np.diag(C2_2.toarray())
C2_theory_2 = (p*p + q*q + 3*p + 3*q + p*q) / 3.0

print("\n" + "="*60)
print("Casimir Operator")
print("="*60)
print(f"\nTheory: C2 = {C2_theory_2:.6f}")
print(f"Computed (unique values): {np.unique(np.round(C2_vals_2, 6))}")
print(f"Error: {np.max(np.abs(C2_vals_2 - C2_theory_2)):.2e}")
