"""
Generate Stark Map for Hydrogen (n=1,2,3) using Paraboloid Lattice
"""
import numpy as np
import matplotlib.pyplot as plt
from paraboloid_relativistic import RungeLenzLattice

# Physical constants
a0 = 0.529177210903e-10  # Bohr radius in meters
e = 1.602176634e-19      # electron charge in C
Eh = 27.211386245988     # Hartree energy in eV

# Build lattice for n=1,2,3
n_max = 3
lattice = RungeLenzLattice(n_max)

# Build z-dipole operator manually
# For hydrogen: <n,l,m|z|n,l',m> couples states with Δl=±1, Δm=0
# Use fact that z = r*cosθ, and Az provides the correct selection rules
# But we need actual dipole matrix elements, not Runge-Lenz matrix elements

# Build z operator from scratch with proper dipole coupling
from scipy.sparse import lil_matrix
z_operator = np.zeros((len(lattice.nodes), len(lattice.nodes)))

for i, (n1, l1, m1) in enumerate(lattice.nodes):
    for j, (n2, l2, m2) in enumerate(lattice.nodes):
        # z couples states within same n, with Δl=±1, Δm=0
        if n1 == n2 and m1 == m2 and abs(l1 - l2) == 1:
            # Approximate dipole matrix element in atomic units
            # <n,l,m|z|n,l±1,m> ~ n^2 * sqrt((l^2-m^2)/(4l^2-1)) for l→l-1
            if l2 == l1 - 1:  # j has lower l
                z_operator[i, j] = n1**2 * np.sqrt((l1**2 - m1**2) / max(4*l1**2 - 1, 1))
            elif l2 == l1 + 1:  # j has higher l  
                z_operator[i, j] = n1**2 * np.sqrt(((l1+1)**2 - m1**2) / max(4*(l1+1)**2 - 1, 1))

# Field strengths in kV/cm
field_strengths_kV = np.linspace(0, 200, 150)  # kV/cm

# Convert to atomic units
# 1 a.u. of E-field = e/(4πε₀a₀²) ≈ 5.142e11 V/m
F_au_to_SI = 5.142206707e11  # V/m per a.u.
field_strengths_V_m = field_strengths_kV * 1e3 * 100  # kV/cm -> V/m
field_au = field_strengths_V_m / F_au_to_SI

# Extract energy levels (just eigenvalues, let them evolve naturally)
energies_vs_field = []
for F_au in field_au:
    # Build Hamiltonian: H0 (diagonal) + F*z (off-diagonal)
    # H0 energies are -1/(2n^2) in atomic units
    H0 = np.zeros((len(lattice.nodes), len(lattice.nodes)))
    for i, (n, l, m) in enumerate(lattice.nodes):
        H0[i, i] = -1.0 / (2 * n**2)
    
    # Stark perturbation: H' = -e*F*z
    H_stark = -F_au * z_operator
    
    # Full Hamiltonian
    H_total = H0 + H_stark
    
    # Diagonalize and sort
    eigenvalues = np.sort(np.linalg.eigvalsh(H_total))
    energies_vs_field.append(eigenvalues)

energies_vs_field = np.array(energies_vs_field)

# Convert to eV for plotting
energies_eV = energies_vs_field * Eh

# Create plot
plt.figure(figsize=(10, 8))

# Color assignment based on initial (F=0) energies
E_initial = energies_eV[0]
colors_list = []
for E in E_initial:
    if abs(E - (-13.6)) < 0.1:
        colors_list.append('C0')  # n=1
    elif abs(E - (-13.6/4)) < 0.02:
        colors_list.append('C1')  # n=2
    elif abs(E - (-13.6/9)) < 0.01:
        colors_list.append('C2')  # n=3
    else:
        colors_list.append('gray')

# Plot each energy level
for i in range(len(lattice.nodes)):
    plt.plot(field_strengths_kV, energies_eV[:, i], 
             color=colors_list[i], linewidth=2, alpha=0.8)

# Add legend manually
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='C0', lw=2, label='n=1'),
                   Line2D([0], [0], color='C1', lw=2, label='n=2'),
                   Line2D([0], [0], color='C2', lw=2, label='n=3')]

plt.xlabel('Electric Field (kV/cm)', fontsize=14)
plt.ylabel('Energy (eV)', fontsize=14)
plt.title('Stark Map for Hydrogen (n=1,2,3) - Paraboloid Lattice', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(handles=legend_elements, fontsize=12, loc='lower left')

# Add annotations for key features
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=1)

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig('stark_map.png', dpi=300, bbox_inches='tight')
print("Stark map saved as stark_map.png")

# Print some statistics
print(f"\nNumber of states: {len(lattice.nodes)}")
print(f"\nEnergy spread at F=0 kV/cm:")
for n in [1, 2, 3]:
    E_n_initial = [E for E in E_initial if abs(E - (-13.6/n**2)) < 0.1/n**2]
    if len(E_n_initial) > 0:
        print(f"  n={n}: {len(E_n_initial)} states")

print(f"\nEnergy spread at F={field_strengths_kV[-1]:.0f} kV/cm:")
E_final = energies_eV[-1]
for n in [1, 2, 3]:
    # Find states that started in this shell
    E_n_final = [E_final[i] for i, E in enumerate(E_initial) if abs(E - (-13.6/n**2)) < 0.1/n**2]
    if len(E_n_final) > 0:
        print(f"  n={n}: {min(E_n_final):.6f} to {max(E_n_final):.6f} eV (spread: {max(E_n_final)-min(E_n_final):.6f} eV)")
