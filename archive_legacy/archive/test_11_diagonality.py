"""
Debug (1,1) diagonality issue.

The (1,1) adjoint representation should have T3, T8 diagonal in the weight basis.
Let's check what's going wrong with the transformation.
"""
import sys
sys.path.insert(0, 'SU(3)')

import numpy as np
from general_rep_builder import GeneralRepBuilder
from su3_spherical_embedding import SU3SphericalEmbedding

# Get (1,1) operators
builder = GeneralRepBuilder()
operators_dict = builder.get_irrep_operators(1, 1)

T3_gt = operators_dict['T3']
T8_gt = operators_dict['T8']

print("=" * 80)
print("(1,1) Adjoint Representation - Diagonality Check")
print("=" * 80)

print("\n1. T3 in GT basis:")
print(f"   Diagonal: {np.diag(T3_gt)}")
print(f"   Max off-diagonal: {np.max(np.abs(T3_gt - np.diag(np.diag(T3_gt)))):.2e}")
print(f"   Matrix:\n{T3_gt}")

print("\n2. T8 in GT basis:")
print(f"   Diagonal: {np.diag(T8_gt)}")
print(f"   Max off-diagonal: {np.max(np.abs(T8_gt - np.diag(np.diag(T8_gt)))):.2e}")
print(f"   Matrix:\n{T8_gt}")

# Get GT patterns and quantum numbers
embedding = SU3SphericalEmbedding(1, 1)
gt_patterns = embedding.gt_patterns

print("\n3. GT Patterns and Quantum Numbers:")
for idx, gt in enumerate(gt_patterns):
    I3, Y, z = embedding._gt_to_quantum_numbers(gt)
    print(f"   State {idx}: GT = {gt}, I3 = {I3:+.3f}, Y = {Y:+.3f}, z = {z}")
    print(f"             T3 eigenvalue = {T3_gt[idx, idx]:+.3f}, T8 eigenvalue = {T8_gt[idx, idx]:+.3f}")

# Check if T3, T8 are already diagonal
print("\n4. Are T3, T8 already diagonal in GT basis?")
T3_is_diagonal = np.max(np.abs(T3_gt - np.diag(np.diag(T3_gt)))) < 1e-10
T8_is_diagonal = np.max(np.abs(T8_gt - np.diag(np.diag(T8_gt)))) < 1e-10
print(f"   T3 diagonal: {T3_is_diagonal}")
print(f"   T8 diagonal: {T8_is_diagonal}")

# If not diagonal, find eigenvectors
if not (T3_is_diagonal and T8_is_diagonal):
    print("\n5. T3 and T8 are NOT diagonal in GT basis.")
    print("   This means the GT basis != weight basis for (1,1).")
    
    # Find simultaneous eigenvectors of T3 and T8
    # They commute, so they can be simultaneously diagonalized
    print("\n6. Finding simultaneous eigenvectors...")
    
    # Diagonalize T3
    evals_t3, evecs_t3 = np.linalg.eigh(T3_gt)
    print(f"   T3 eigenvalues: {evals_t3}")
    
    # Check T8 in T3 eigenbasis
    T8_in_t3_basis = evecs_t3.T.conj() @ T8_gt @ evecs_t3
    print(f"   T8 in T3 eigenbasis (should be diagonal):")
    print(f"{T8_in_t3_basis}")
    print(f"   Max off-diagonal: {np.max(np.abs(T8_in_t3_basis - np.diag(np.diag(T8_in_t3_basis)))):.2e}")
    
    # This should be the correct transformation matrix
    U_correct = evecs_t3
    
    print("\n7. Correct transformation matrix:")
    print(f"   U shape: {U_correct.shape}")
    print(f"   U is unitary: {np.allclose(U_correct @ U_correct.T.conj(), np.eye(8))}")
    
    # Apply to T3 and T8
    T3_diag = U_correct.T.conj() @ T3_gt @ U_correct
    T8_diag = U_correct.T.conj() @ T8_gt @ U_correct
    
    print(f"\n8. After transformation:")
    print(f"   T3 max off-diagonal: {np.max(np.abs(T3_diag - np.diag(np.diag(T3_diag)))):.2e}")
    print(f"   T8 max off-diagonal: {np.max(np.abs(T8_diag - np.diag(np.diag(T8_diag)))):.2e}")
    print(f"   T3 eigenvalues: {np.diag(T3_diag)}")
    print(f"   T8 eigenvalues: {np.diag(T8_diag)}")
    
else:
    print("\n5. T3 and T8 ARE already diagonal - no transformation needed!")
