"""Simple hydrogen test focusing on geometric factor"""
import sys
sys.path.append('src')
from hydrogen_lattice import HydrogenLattice
import numpy as np

print('='*70)
print('PHASE 9.2: HYDROGEN ATOM - GEOMETRIC FACTOR TEST')
print('='*70)
print()

# Create solver with smaller lattice for speed
hydrogen = HydrogenLattice(ell_max=20, a_lattice=1.0)
print()

# Solve with hopping (better approximation)
print('Solving with radial hopping...')
E_hop, states = hydrogen.solve_with_hopping()
print(f'Ground state energy: {E_hop[0]:.6f} Ry')
print(f'Continuum (n=1):     {-0.5:.6f} Ry')
print(f'Error: {abs(E_hop[0] - (-0.5))/0.5 * 100:.1f}%')
print()

# Compare first few levels
print('Energy Level Comparison:')
print('-'*70)
print(f'{"n":>3} {"E_continuum":>12} {"E_lattice":>12} {"Error (%)":>12}')
print('-'*70)

for n in range(1, min(6, len(E_hop)+1)):
    E_cont = -0.5 / n**2
    E_lat = E_hop[n-1]
    error = abs(E_lat - E_cont) / abs(E_cont) * 100
    print(f'{n:3d} {E_cont:12.6f} {E_lat:12.6f} {error:12.3f}')
print()

# Geometric factor analysis
print('Searching for 1/(4π) in energy corrections...')
analysis = hydrogen.find_geometric_factor()
print()

print('Reference: 1/(4π) = {:.10f}'.format(analysis['one_over_4pi']))
print()

# Find best model
best_model_name = None
best_residual = float('inf')
for name, data in analysis['models'].items():
    if data['residual'] < best_residual:
        best_residual = data['residual']
        best_model_name = name

print(f'Best-fit model: {best_model_name}')
print(f'  Coefficient A = {analysis["models"][best_model_name]["A_diag"]:.6f}')
print(f'  A × 4π = {analysis["models"][best_model_name]["A_diag"] * 4 * np.pi:.6f}')
print(f'  Residual = {best_residual:.4f}')
print()

# Check all models
print('All models tested:')
for name, data in analysis['models'].items():
    A = data['A_diag']
    ratio_to_4pi = A * 4 * np.pi
    print(f'  {name:15s}: A×4π = {ratio_to_4pi:8.4f}, Residual = {data["residual"]:.4f}')
print()

print('='*70)
print('INTERPRETATION:')
print('='*70)
print('The energy corrections ΔE = E_lattice - E_continuum follow a scaling')
print('with n (principal quantum number). We test if ΔE ∝ 1/(4π) × f(n).')
print()
print('If A×4π ≈ integer, this suggests geometric factor involvement!')
print('='*70)
