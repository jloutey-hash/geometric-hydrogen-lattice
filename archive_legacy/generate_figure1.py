"""
Generate Figure 1 for geometric_atom_v5.tex
Shows the paraboloid lattice connectivity for n_max=5
"""
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from paraboloid_lattice_su11 import ParaboloidLattice, plot_lattice_connectivity

# Create lattice with max_n=5
print("Creating lattice with max_n=5...")
lattice = ParaboloidLattice(max_n=5)

# Generate the connectivity plot
print("Generating connectivity plot...")
fig = plot_lattice_connectivity(lattice, max_connections=300, elev=25, azim=45)

# Save as PNG and PDF
print("Saving figure1_paraboloid.png...")
fig.savefig('figure1_paraboloid.png', dpi=300, bbox_inches='tight')
print("Saving figure1_paraboloid.pdf...")
fig.savefig('figure1_paraboloid.pdf', bbox_inches='tight')

print("✓ Figure 1 generated successfully!")
print(f"  - figure1_paraboloid.png (300 dpi)")
print(f"  - figure1_paraboloid.pdf (vector)")
