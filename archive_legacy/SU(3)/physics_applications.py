# SU(3) Physics Applications Framework
"""
Ready-to-use physics exploration tools for the SU(3) Ziggurat geometry.

The validation tests confirm the geometry is exact. Now we explore physics:
1. Color dynamics (quark interactions)
2. Confinement mechanisms
3. SU(3) Hamiltonians (chromodynamics)
4. Emergent gauge symmetry
5. Curvature via Wilson loops
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import expm
from weight_basis_gellmann import WeightBasisSU3
from gt_basis_transformed import GTBasisSU3
from adjoint_tensor_product import AdjointSU3, AdjointSU3_GT
from lattice import SU3Lattice


class SU3PhysicsLab:
    """
    Physics exploration laboratory for SU(3) Ziggurat geometry.
    """
    
    def __init__(self):
        """Initialize all representations and lattices."""
        # Representations
        self.fund = WeightBasisSU3(1, 0)  # Quarks (fundamental)
        self.anti = WeightBasisSU3(0, 1)  # Antiquarks
        self.adj = AdjointSU3()           # Gluons (adjoint)
        
        # GT basis for geometric visualization
        self.gt_fund = GTBasisSU3(1, 0)
        self.gt_anti = GTBasisSU3(0, 1)
        self.gt_adj = AdjointSU3_GT()
        
        # Lattices
        self.lattice_fund = SU3Lattice(1, 0)
        self.lattice_adj = SU3Lattice(1, 1)
        
        print("SU(3) Physics Lab initialized")
        print("  Fundamental (quarks): 3D")
        print("  Antifundamental (antiquarks): 3D")
        print("  Adjoint (gluons): 8D")
        print("  Ready for physics exploration!")
    
    # ========================================================================
    # 1. COLOR DYNAMICS
    # ========================================================================
    
    def color_charge_dynamics(self, initial_state='red', t_max=10, n_steps=100):
        """
        Simulate color charge evolution under SU(3) Hamiltonian.
        
        Models a quark in a color eigenstate evolving under:
        H = \u03c9\u2081 T\u2083 + \u03c9\u2082 T\u2088
        
        Args:
            initial_state: 'red', 'green', 'blue', or custom 3D vector
            t_max: Maximum time
            n_steps: Number of time steps
            
        Returns:
            times, states, color_charges
        """
        print("\n" + "="*70)
        print("COLOR DYNAMICS SIMULATION")
        print("="*70)
        
        # Initial state
        if initial_state == 'red':
            psi0 = np.array([1, 0, 0], dtype=complex)  # I3=+1/2, Y=+1/3
        elif initial_state == 'green':
            psi0 = np.array([0, 1, 0], dtype=complex)  # I3=-1/2, Y=+1/3
        elif initial_state == 'blue':
            psi0 = np.array([0, 0, 1], dtype=complex)  # I3=0, Y=-2/3
        else:
            psi0 = np.array(initial_state, dtype=complex)
            psi0 /= np.linalg.norm(psi0)
        
        # Hamiltonian (arbitrary frequencies for demonstration)
        omega1, omega2 = 1.0, 0.5
        H = omega1 * self.fund.T3 + omega2 * self.fund.T8
        
        # Time evolution
        times = np.linspace(0, t_max, n_steps)
        states = []
        I3_vals = []
        Y_vals = []
        C2_vals = []
        
        for t in times:
            # Evolve: |\u03c8(t)\u27e9 = exp(-iHt) |\u03c8(0)\u27e9
            U_t = expm(-1j * H * t)
            psi_t = U_t @ psi0
            states.append(psi_t)
            
            # Measure color charges
            I3 = np.real(psi_t.conj() @ self.fund.T3 @ psi_t)
            Y = 2/np.sqrt(3) * np.real(psi_t.conj() @ self.fund.T8 @ psi_t)
            I3_vals.append(I3)
            Y_vals.append(Y)
            
            # Casimir (should be conserved)
            ops = [self.fund.T3, self.fund.T8, self.fund.E12, self.fund.E21,
                   self.fund.E23, self.fund.E32, self.fund.E13, self.fund.E31]
            C2 = sum(T @ T for T in ops)
            C2_val = np.real(psi_t.conj() @ C2 @ psi_t)
            C2_vals.append(C2_val)
        
        print(f"\\nInitial state: {initial_state}")
        print(f"  I3(0) = {I3_vals[0]:.4f}")
        print(f"  Y(0) = {Y_vals[0]:.4f}")
        print(f"  C2(0) = {C2_vals[0]:.4f}")
        print(f"\\nFinal state:")
        print(f"  I3(T) = {I3_vals[-1]:.4f}")
        print(f"  Y(T) = {Y_vals[-1]:.4f}")
        print(f"  C2(T) = {C2_vals[-1]:.4f}")
        print(f"\\nCasimir conservation: \u0394C2 = {abs(C2_vals[-1] - C2_vals[0]):.2e}")
        
        return times, states, (I3_vals, Y_vals, C2_vals)
    
    def plot_color_dynamics(self, times, charges):
        """Plot color charge evolution."""
        I3_vals, Y_vals, C2_vals = charges
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # I3 vs time
        axes[0,0].plot(times, I3_vals, 'b-', lw=2)
        axes[0,0].set_xlabel('Time')
        axes[0,0].set_ylabel('I\u2083 (isospin)')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_title('Isospin Evolution')
        
        # Y vs time
        axes[0,1].plot(times, Y_vals, 'r-', lw=2)
        axes[0,1].set_xlabel('Time')
        axes[0,1].set_ylabel('Y (hypercharge)')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].set_title('Hypercharge Evolution')
        
        # Phase space trajectory
        axes[1,0].plot(I3_vals, Y_vals, 'g-', lw=2, alpha=0.7)
        axes[1,0].scatter([I3_vals[0]], [Y_vals[0]], c='green', s=100, marker='o', label='Start')
        axes[1,0].scatter([I3_vals[-1]], [Y_vals[-1]], c='red', s=100, marker='x', label='End')
        axes[1,0].set_xlabel('I\u2083')
        axes[1,0].set_ylabel('Y')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].legend()
        axes[1,0].set_title('Color Charge Trajectory')
        
        # Casimir conservation
        axes[1,1].plot(times, C2_vals, 'm-', lw=2)
        axes[1,1].set_xlabel('Time')
        axes[1,1].set_ylabel('C\u2082 (Casimir)')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].set_title('Casimir Operator (Conserved)')
        
        plt.tight_layout()
        return fig
    
    # ========================================================================
    # 2. CONFINEMENT MECHANISMS
    # ========================================================================
    
    def linear_potential_dynamics(self, separation_range=(0.1, 5.0), n_points=50):
        """
        Explore confinement via linear potential V(r) = \u03c3 r.
        
        Simulates the energy cost of separating color charges.
        
        Args:
            separation_range: (r_min, r_max) in lattice units
            n_points: Number of separation distances
            
        Returns:
            separations, energies, forces
        """
        print("\\n" + "="*70)
        print("CONFINEMENT: LINEAR POTENTIAL MODEL")
        print("="*70)
        
        # String tension (typical QCD value ~ 1 GeV/fm)
        sigma = 1.0  # In lattice units
        
        separations = np.linspace(*separation_range, n_points)
        energies = sigma * separations
        forces = np.full_like(separations, sigma)
        
        print(f"\\nString tension: \u03c3 = {sigma} (lattice units)")
        print(f"Separation range: {separation_range[0]:.2f} to {separation_range[1]:.2f}")
        print(f"\\nEnergy at r=1: E = {sigma:.3f}")
        print(f"Energy at r=5: E = {5*sigma:.3f}")
        print(f"Constant force: F = {sigma:.3f}")
        
        return separations, energies, forces
    
    def flux_tube_formation(self, quark_pos=(0,0,0), antiquark_pos=(5,0,0)):
        """
        Visualize chromoelectric flux tube between quark-antiquark pair.
        
        Models the gluon field configuration connecting color sources.
        """
        print("\\n" + "="*70)
        print("FLUX TUBE FORMATION")
        print("="*70)
        
        q1 = np.array(quark_pos)
        q2 = np.array(antiquark_pos)
        separation = np.linalg.norm(q2 - q1)
        
        print(f"\\nQuark position: {q1}")
        print(f"Antiquark position: {q2}")
        print(f"Separation: {separation:.3f}")
        
        # Generate flux tube path (simple linear interpolation)
        n_points = 50
        tube_path = np.array([q1 + t*(q2-q1) for t in np.linspace(0, 1, n_points)])
        
        # Field strength (constant in linear confinement)
        field_strength = 1.0
        
        return tube_path, field_strength
    
    # ========================================================================
    # 3. SU(3) HAMILTONIANS
    # ========================================================================
    
    def build_chromodynamic_hamiltonian(self, coupling_strength=1.0):
        """
        Construct SU(3) Hamiltonian for chromodynamics.
        
        H = g \u03a3_a T^a \u22c5 T^a = g C\u2082
        
        where C\u2082 is the quadratic Casimir operator.
        """
        print("\\n" + "="*70)
        print("CHROMODYNAMIC HAMILTONIAN")
        print("="*70)
        
        # Build Casimir for fundamental representation
        ops = [self.fund.T3, self.fund.T8, self.fund.E12, self.fund.E21,
               self.fund.E23, self.fund.E32, self.fund.E13, self.fund.E31]
        C2_fund = sum(T @ T for T in ops)
        H_fund = coupling_strength * C2_fund
        
        # Build Casimir for adjoint representation
        ops_adj = [self.adj.T3, self.adj.T8, self.adj.E12, self.adj.E21,
                   self.adj.E23, self.adj.E32, self.adj.E13, self.adj.E31]
        
        # Use Hermitian combinations
        lambda1_adj = self.adj.E12 + self.adj.E21
        lambda2_adj = -1j * (self.adj.E12 - self.adj.E21)
        lambda3_adj = 2 * self.adj.T3
        lambda4_adj = self.adj.E23 + self.adj.E32
        lambda5_adj = -1j * (self.adj.E23 - self.adj.E32)
        lambda6_adj = self.adj.E13 + self.adj.E31
        lambda7_adj = -1j * (self.adj.E13 - self.adj.E31)
        lambda8_adj = 2 * self.adj.T8
        
        C2_adj = (lambda1_adj @ lambda1_adj + lambda2_adj @ lambda2_adj +
                  lambda3_adj @ lambda3_adj + lambda4_adj @ lambda4_adj +
                  lambda5_adj @ lambda5_adj + lambda6_adj @ lambda6_adj +
                  lambda7_adj @ lambda7_adj + lambda8_adj @ lambda8_adj) / 4
        H_adj = coupling_strength * C2_adj
        
        # Eigenvalues
        E_fund = np.linalg.eigvalsh(H_fund)
        E_adj = np.linalg.eigvalsh(C2_adj)
        
        print(f"\\nCoupling strength: g = {coupling_strength}")
        print(f"\\nFundamental representation (quarks):")
        print(f"  Eigenvalues: {np.unique(np.round(E_fund, 6))}")
        print(f"  Energy splitting: \u0394E = {E_fund.max() - E_fund.min():.6f}")
        print(f"\\nAdjoint representation (gluons):")
        print(f"  Eigenvalues: {np.unique(np.round(E_adj, 6))}")
        print(f"  Energy splitting: \u0394E = {E_adj.max() - E_adj.min():.6f}")
        
        return H_fund, H_adj, (E_fund, E_adj)
    
    # ========================================================================
    # 4. EMERGENT GAUGE SYMMETRY
    # ========================================================================
    
    def gauge_transformation_invariance(self, n_random_transformations=10):
        """
        Test gauge invariance of observables under SU(3) transformations.
        
        Applies random SU(3) transformations and checks that physical
        observables (Casimir, norms, etc.) are invariant.
        """
        print("\\n" + "="*70)
        print("EMERGENT GAUGE SYMMETRY")
        print("="*70)
        
        # Initial state
        psi0 = np.array([1, 1, 1], dtype=complex)
        psi0 /= np.linalg.norm(psi0)
        
        # Measure initial observables
        C2 = sum(T @ T for T in [self.fund.T3, self.fund.T8, self.fund.E12,
                                  self.fund.E21, self.fund.E23, self.fund.E32,
                                  self.fund.E13, self.fund.E31])
        C2_initial = np.real(psi0.conj() @ C2 @ psi0)
        norm_initial = np.linalg.norm(psi0)
        
        print(f"\\nInitial state: {psi0}")
        print(f"  Norm: {norm_initial:.10f}")
        print(f"  Casimir: {C2_initial:.10f}")
        
        # Apply random gauge transformations
        max_norm_error = 0
        max_casimir_error = 0
        
        np.random.seed(42)
        for i in range(n_random_transformations):
            # Generate random SU(3) transformation (via exponential map)
            # g = exp(i \u03b8\u2090 T\u2090)
            thetas = 2*np.pi * (np.random.rand(8) - 0.5)
            generators = [self.fund.T3, self.fund.T8, self.fund.E12, self.fund.E21,
                         self.fund.E23, self.fund.E32, self.fund.E13, self.fund.E31]
            
            # Build transformation
            algebra_element = sum(theta * T for theta, T in zip(thetas, generators))
            g = expm(1j * algebra_element)
            
            # Transform state
            psi_transformed = g @ psi0
            
            # Measure transformed observables
            norm_transformed = np.linalg.norm(psi_transformed)
            C2_transformed = np.real(psi_transformed.conj() @ C2 @ psi_transformed)
            
            # Check invariance
            norm_error = abs(norm_transformed - norm_initial)
            casimir_error = abs(C2_transformed - C2_initial)
            
            max_norm_error = max(max_norm_error, norm_error)
            max_casimir_error = max(max_casimir_error, casimir_error)
        
        print(f"\nAfter {n_random_transformations} gauge transformations:")
        print(f"  Maximum norm deviation: {max_norm_error:.2e}")
        print(f"  Maximum Casimir deviation: {max_casimir_error:.2e}")
        
        if max_norm_error < 1e-10 and max_casimir_error < 1e-10:
            print("\n✓ Gauge invariance CONFIRMED at machine precision!")
        else:
            print("\n✗ Warning: Gauge invariance may be broken")
        
        return max_norm_error, max_casimir_error
    
    # ========================================================================
    # 5. WILSON LOOPS AND CURVATURE
    # ========================================================================
    
    def wilson_loop_curvature(self, loop_size=3, representation='adjoint'):
        """
        Compute Wilson loop to measure SU(3) field curvature.
        
        W(C) = Tr[P exp(ig \u222e_C A\u22c5dx)]
        
        For discrete lattice, this becomes a product of link operators.
        """
        print("\\n" + "="*70)
        print("WILSON LOOP CURVATURE MEASUREMENT")
        print("="*70)
        
        if representation == 'adjoint':
            ops = [self.adj.E12, self.adj.E23, self.adj.E31,  # Triangle path
                   self.adj.E21, self.adj.E32, self.adj.E13]  # Reverse
        else:
            ops = [self.fund.E12, self.fund.E23, self.fund.E31,
                   self.fund.E21, self.fund.E32, self.fund.E13]
        
        # Forward loop
        W_forward = ops[0] @ ops[1] @ ops[2]
        w_forward = np.trace(W_forward)
        
        # Reverse loop
        W_reverse = ops[3] @ ops[4] @ ops[5]
        w_reverse = np.trace(W_reverse)
        
        # Curvature indicator
        curvature = np.real(w_forward - w_reverse)
        
        print(f"\\nRepresentation: {representation}")
        print(f"Loop size: {loop_size} (triangular)")
        print(f"\\nWilson loop (forward): {w_forward:.6f}")
        print(f"Wilson loop (reverse): {w_reverse:.6f}")
        print(f"Curvature indicator: {curvature:.6f}")
        
        if abs(curvature) < 1e-10:
            print("\\nFlat connection (trivial curvature)")
        else:
            print(f"\\nNon-trivial curvature detected!")
        
        return w_forward, w_reverse, curvature


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    lab = SU3PhysicsLab()
    
    print("\\n" + "="*70)
    print("WELCOME TO THE SU(3) PHYSICS LABORATORY")
    print("="*70)
    print("\\nAvailable experiments:")
    print("  1. Color dynamics simulation")
    print("  2. Confinement mechanisms")
    print("  3. Chromodynamic Hamiltonians")
    print("  4. Gauge invariance tests")
    print("  5. Wilson loop curvature")
    print("\\nRun experiments individually or all at once.")
    print("="*70)
    
    # Run all experiments
    print("\\n\\nRUNNING ALL PHYSICS EXPERIMENTS...")
    
    # 1. Color dynamics
    times, states, charges = lab.color_charge_dynamics('red', t_max=10, n_steps=100)
    
    # 2. Confinement
    sep, energy, force = lab.linear_potential_dynamics()
    
    # 3. Hamiltonian
    H_f, H_a, E = lab.build_chromodynamic_hamiltonian(coupling_strength=0.5)
    
    # 4. Gauge symmetry
    err_norm, err_cas = lab.gauge_transformation_invariance(n_random_transformations=20)
    
    # 5. Wilson loops
    w_f, w_r, curv = lab.wilson_loop_curvature(representation='adjoint')
    
    print("\\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*70)
    print("\\nThe SU(3) Ziggurat geometry is ready for:")
    print("  \u2713 Quark dynamics simulations")
    print("  \u2713 Confinement studies")
    print("  \u2713 Gauge theory applications")
    print("  \u2713 Lattice QCD exploration")
    print("\\nYou've built the engine. Now drive it! \U0001f680")
