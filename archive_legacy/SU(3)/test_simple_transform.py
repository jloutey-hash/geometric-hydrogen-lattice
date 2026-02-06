"""
Simple test of basis transformation for (1,0).
"""

import numpy as np
from weight_basis_gellmann import WeightBasisSU3
from lattice import SU3Lattice

print("="*80)
print("Testing (1,0) Basis Transformation")
print("="*80)

# Get weight basis operators
wb = WeightBasisSU3(1, 0)
print(f"\nWeight basis dim: {wb.dim}")

# Get quantum numbers from weight basis
I3_weight = np.diag(wb.T3).real
T8_weight = np.diag(wb.T8).real
Y_weight = T8_weight * 2 / np.sqrt(3)

print(f"\nWeight basis (I3, Y):")
for i in range(wb.dim):
    print(f"  State {i}: I3={I3_weight[i]:.4f}, Y={Y_weight[i]:.4f}")

# Get GT basis
lattice = SU3Lattice(max_p=1, max_q=0)
gt_states = [s for s in lattice.states if s['p'] == 1 and s['q'] == 0]

print(f"\nGT basis (I3, Y):")
for i, s in enumerate(gt_states):
    print(f"  State {i}: I3={s['i3']:.4f}, Y={s['y']:.4f}")

# Build permutation matrix
print(f"\nBuilding permutation matrix U...")
U = np.zeros((wb.dim, wb.dim))

for i_gt, gt_state in enumerate(gt_states):
    I3_gt = gt_state['i3']
    Y_gt = gt_state['y']
    
    # Find matching weight state
    for i_w in range(wb.dim):
        if abs(I3_gt - I3_weight[i_w]) < 1e-10 and abs(Y_gt - Y_weight[i_w]) < 1e-10:
            U[i_gt, i_w] = 1.0
            print(f"  GT state {i_gt} (I3={I3_gt:.2f}, Y={Y_gt:.2f}) ← Weight state {i_w}")
            break

print(f"\nPermutation matrix U:")
print(U)
print(f"\nU @ U.T:")
print(U @ U.T)

# Transform T3
T3_GT = U @ wb.T3 @ U.T
print(f"\nT3 in GT basis:")
print(T3_GT)
print(f"\nT3 diagonal: {np.diag(T3_GT)}")
print(f"Expected from GT: {[s['i3'] for s in gt_states]}")

# Transform T8
T8_GT = U @ wb.T8 @ U.T
print(f"\nT8 in GT basis:")
print(T8_GT)
print(f"\nT8 diagonal: {np.diag(T8_GT)}")
Y_GT_expected = [s['y'] for s in gt_states]
T8_expected = [y * np.sqrt(3) / 2 for y in Y_GT_expected]
print(f"Expected from GT: {T8_expected}")

# Transform E12
E12_GT = U @ wb.E12 @ U.T
print(f"\nE12 in GT basis:")
print(np.abs(E12_GT))

# Check if transformed operators preserve commutation
comm_T3T8 = T3_GT @ T8_GT - T8_GT @ T3_GT
print(f"\n[T3,T8] error: {np.max(np.abs(comm_T3T8))}")

# Check Casimir
C2_weight = wb.get_casimir()
print(f"\nCasimir in weight basis:")
print(f"Eigenvalues: {np.linalg.eigvalsh(C2_weight)}")

C2_GT = U @ C2_weight @ U.T
print(f"\nCasimir in GT basis:")
print(f"Eigenvalues: {np.linalg.eigvalsh(C2_GT)}")
