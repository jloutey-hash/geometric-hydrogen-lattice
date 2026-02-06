"""
Debug script to see why GT and weight basis states aren't matching.
"""

import numpy as np
from weight_basis_gellmann import WeightBasisSU3
from lattice import SU3Lattice

print("="*80)
print("Weight Basis States Analysis")
print("="*80)

# (1,0) representation
wb = WeightBasisSU3(1, 0)
print(f"\n(p,q) = (1,0), dim = {wb.dim}")
print(f"Weight (I3, Y) pairs: {wb.weights}")

# Get T3 and T8 quantum numbers from diagonal
T3_diag = np.diag(wb.T3)
T8_diag = np.diag(wb.T8)
Y_values = T8_diag * 2 / np.sqrt(3)

print(f"\nFrom diagonal operators:")
print(f"T3 eigenvalues: {T3_diag}")
print(f"T8 eigenvalues: {T8_diag}")
print(f"Y = T8 * 2/√3: {Y_values}")

print(f"\nDetailed (I3, Y) pairs:")
for i in range(wb.dim):
    print(f"  State {i}: I3={T3_diag[i]:.4f}, Y={Y_values[i]:.4f}")

print("\n" + "="*80)
print("GT Basis States Analysis")
print("="*80)

# Generate GT patterns using lattice
p, q = 1, 0
lattice = SU3Lattice(max_p=p, max_q=q)

# Filter states for this specific (p,q)
gt_states = [s for s in lattice.states if s['p'] == p and s['q'] == q]

print(f"\n(p,q) = ({p},{q}), dim = {len(gt_states)}")
print(f"GT patterns: {[s['gt'] for s in gt_states]}")

# Compute T3 and T8 from GT formulas
for i, state in enumerate(gt_states):
    gt = state['gt']
    m13, m23, m33, m12, m22, m11 = gt
    
    # T3 from GT formula
    T3_GT = m11 - (m12 + m22) / 2
    
    # Y from GT formula - from the lattice computation
    Y_GT = state['y']
    
    # T8 from Y
    T8_GT = Y_GT * np.sqrt(3) / 2
    
    print(f"  Pattern {i}: {gt}")
    print(f"    T3={T3_GT:.4f}, Y={Y_GT:.4f}, T8={T8_GT:.4f}")

print("\n" + "="*80)
print("(0,1) Representation")
print("="*80)

# (0,1) representation
wb = WeightBasisSU3(0, 1)
print(f"\n(p,q) = (0,1), dim = {wb.dim}")
print(f"Weight (I3, Y) pairs: {wb.weights}")

T3_diag = np.diag(wb.T3)
T8_diag = np.diag(wb.T8)
Y_values = T8_diag * 2 / np.sqrt(3)

print(f"\nFrom diagonal operators:")
print(f"T3 eigenvalues: {T3_diag}")
print(f"Y = T8 * 2/√3: {Y_values}")

print(f"\nDetailed (I3, Y) pairs:")
for i in range(wb.dim):
    print(f"  State {i}: I3={T3_diag[i]:.4f}, Y={Y_values[i]:.4f}")

# GT patterns
p, q = 0, 1
lattice = SU3Lattice(max_p=p, max_q=q)
gt_states = [s for s in lattice.states if s['p'] == p and s['q'] == q]

print(f"\nGT patterns: {[s['gt'] for s in gt_states]}")
for i, state in enumerate(gt_states):
    gt = state['gt']
    m13, m23, m33, m12, m22, m11 = gt
    T3_GT = m11 - (m12 + m22) / 2
    Y_GT = state['y']
    T8_GT = Y_GT * np.sqrt(3) / 2
    
    print(f"  Pattern {i}: {gt}")
    print(f"    T3={T3_GT:.4f}, Y={Y_GT:.4f}, T8={T8_GT:.4f}")
