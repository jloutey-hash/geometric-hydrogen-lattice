"""
Wilson Gauge Fields on Discrete SU(2) Lattice

Implements SU(2) Yang-Mills gauge theory on the discrete angular momentum lattice.
This is the highest priority Phase 9 investigation.

Key test: Does the bare coupling constant g² involve the geometric factor 1/(4π)?

Physical setup:
- SU(2) link variables U_link ∈ SU(2) connecting lattice sites
- Wilson plaquette action: S = Σ[1 - (1/N_c)Re Tr(U_plaquette)]
- Monte Carlo sampling with Metropolis algorithm
- Measure effective coupling from ⟨U⟩

Phase 9.1 - Highest Priority Implementation
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# Import lattice structure
import sys
sys.path.append('src')
from lattice import PolarLattice


class SU2Element:
    """
    Element of SU(2) group represented as 2x2 complex unitary matrix.
    
    Parameterization: U = a₀·I + i·(a₁·σ₁ + a₂·σ₂ + a₃·σ₃)
    where a₀² + a₁² + a₂² + a₃² = 1
    """
    
    def __init__(self, params: Optional[np.ndarray] = None):
        """
        Initialize SU(2) element.
        
        Parameters:
        -----------
        params : np.ndarray, shape (4,)
            Pauli parameters (a₀, a₁, a₂, a₃) with norm = 1
            If None, initializes to identity
        """
        if params is None:
            self.params = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            # Normalize
            norm = np.linalg.norm(params)
            self.params = params / norm if norm > 0 else np.array([1.0, 0.0, 0.0, 0.0])
    
    @classmethod
    def identity(cls):
        """Create identity element."""
        return cls(np.array([1.0, 0.0, 0.0, 0.0]))
    
    @classmethod
    def random(cls, beta: float = 1.0):
        """
        Create random SU(2) element with given spread parameter.
        
        Parameters:
        -----------
        beta : float
            Inverse temperature parameter. β→∞ gives identity, β→0 gives uniform.
        """
        # Sample from heat kernel on SU(2)
        if beta > 10:
            # Small deviation from identity
            params = np.random.randn(4) / np.sqrt(beta)
            params[0] += 1.0
        else:
            # Uniform sampling
            params = np.random.randn(4)
        
        return cls(params)
    
    def to_matrix(self) -> np.ndarray:
        """
        Convert to 2x2 complex matrix.
        
        U = a₀·I + i·(a₁·σ_x + a₂·σ_y + a₃·σ_z)
        """
        a0, a1, a2, a3 = self.params
        return np.array([
            [a0 + 1j*a3, a2 + 1j*a1],
            [-a2 + 1j*a1, a0 - 1j*a3]
        ], dtype=complex)
    
    def conjugate(self):
        """Return adjoint (inverse for SU(2))."""
        a0, a1, a2, a3 = self.params
        return SU2Element(np.array([a0, -a1, -a2, -a3]))
    
    def __mul__(self, other):
        """Group multiplication."""
        U1 = self.to_matrix()
        U2 = other.to_matrix()
        U_prod = U1 @ U2
        
        # Extract parameters from product
        a0 = U_prod[0, 0].real
        a3 = U_prod[0, 0].imag
        a1 = U_prod[0, 1].imag
        a2 = U_prod[0, 1].real
        
        return SU2Element(np.array([a0, a1, a2, a3]))
    
    def trace(self) -> float:
        """Return trace of matrix representation."""
        return 2 * self.params[0]
    
    def __repr__(self):
        return f"SU2({self.params[0]:.3f}, {self.params[1]:.3f}, {self.params[2]:.3f}, {self.params[3]:.3f})"


class WilsonGaugeField:
    """
    SU(2) gauge field on discrete angular momentum lattice.
    
    Links connect nearby points on adjacent shells.
    Plaquettes are minimal closed loops.
    """
    
    def __init__(self, ell_max: int = 5, beta: float = 4.0):
        """
        Initialize gauge field.
        
        Parameters:
        -----------
        ell_max : int
            Maximum angular momentum
        beta : float
            Inverse coupling β = 2N_c/g² for SU(N_c)
            For SU(2): β = 4/g²
        """
        self.ell_max = ell_max
        self.beta = beta
        
        # Build underlying lattice
        self.lattice = PolarLattice(ell_max)
        
        # Store lattice structure
        self.ell_values = np.arange(0, ell_max + 1)
        self.r_ell = 1 + 2 * self.ell_values
        self.N_ell = 2 * (2 * self.ell_values + 1)
        
        print(f"Wilson Gauge Field initialized:")
        print(f"  ℓ_max = {ell_max}")
        print(f"  β = {beta:.3f}")
        print(f"  g² = {4/beta:.6f} (for SU(2))")
        print(f"  1/(4π) = {1/(4*np.pi):.6f}")
        print(f"  Ratio g²/(1/4π) = {(4/beta)/(1/(4*np.pi)):.6f}")
        
        # Initialize link variables (U_link = identity initially)
        self.links = self._initialize_links()
        
        # Build plaquette list
        self.plaquettes = self._build_plaquettes()
        
        print(f"  Number of links: {len(self.links)}")
        print(f"  Number of plaquettes: {len(self.plaquettes)}")
    
    def _initialize_links(self) -> Dict:
        """
        Initialize all link variables to identity.
        
        Returns:
        --------
        links : dict
            Dictionary mapping (ℓ₁, i₁, ℓ₂, i₂) → SU2Element
        """
        links = {}
        
        # Radial links: connect shell ℓ to ℓ+1
        for ell in range(self.ell_max):
            N1 = self.N_ell[ell]
            N2 = self.N_ell[ell + 1]
            
            # Connect nearest angular neighbors
            for i1 in range(N1):
                # Find closest point on next shell
                i2 = int(i1 * N2 / N1)  # Approximate angular matching
                
                # Create link
                link_key = (ell, i1, ell+1, i2)
                links[link_key] = SU2Element.identity()
        
        # Angular links: connect points within same shell
        for ell in range(self.ell_max + 1):
            N = self.N_ell[ell]
            
            for i in range(N):
                # Connect to next angular neighbor (periodic)
                j = (i + 1) % N
                link_key = (ell, i, ell, j)
                links[link_key] = SU2Element.identity()
        
        return links
    
    def _build_plaquettes(self) -> List[Tuple]:
        """
        Build list of all plaquettes (minimal closed loops).
        
        Each plaquette is a 4-tuple of link keys forming a loop.
        
        Returns:
        --------
        plaquettes : list of tuples
            Each element is 4 link keys forming oriented loop
        """
        plaquettes = []
        
        # Type 1: Radial-angular rectangles
        for ell in range(self.ell_max):
            N1 = self.N_ell[ell]
            N2 = self.N_ell[ell + 1]
            
            for i1 in range(N1):
                i2 = int(i1 * N2 / N1)
                j1 = (i1 + 1) % N1
                j2 = (i2 + 1) % N2
                
                # Create plaquette:
                # (ℓ,i) → (ℓ,j) → (ℓ+1,j') → (ℓ+1,i') → (ℓ,i)
                plaq = [
                    (ell, i1, ell, j1),      # Angular link on shell ℓ
                    (ell, j1, ell+1, j2),    # Radial link up
                    (ell+1, j2, ell+1, i2),  # Angular link on shell ℓ+1 (backward)
                    (ell+1, i2, ell, i1)     # Radial link down
                ]
                plaquettes.append(plaq)
        
        return plaquettes
    
    def get_link(self, key: Tuple) -> SU2Element:
        """Get link variable, computing reverse if needed."""
        if key in self.links:
            return self.links[key]
        else:
            # Try reverse link
            reverse_key = (key[2], key[3], key[0], key[1])
            if reverse_key in self.links:
                return self.links[reverse_key].conjugate()
            else:
                # Link doesn't exist, return identity
                return SU2Element.identity()
    
    def plaquette_product(self, plaq: List[Tuple]) -> SU2Element:
        """
        Compute ordered product around plaquette.
        
        Parameters:
        -----------
        plaq : list of tuples
            Link keys forming oriented loop
        
        Returns:
        --------
        U_plaq : SU2Element
            Product U₁ × U₂ × U₃ × U₄
        """
        U = SU2Element.identity()
        
        for link_key in plaq:
            U_link = self.get_link(link_key)
            U = U * U_link
        
        return U
    
    def wilson_action(self) -> float:
        """
        Compute Wilson plaquette action.
        
        S = β Σ_plaq [1 - (1/2)Re Tr(U_plaq)]
        
        For SU(2): N_c = 2, so (1/N_c) = 1/2
        
        Returns:
        --------
        S : float
            Total action
        """
        S = 0.0
        
        for plaq in self.plaquettes:
            U_plaq = self.plaquette_product(plaq)
            # For SU(2): Tr(U) = 2·a₀
            Re_Tr = U_plaq.trace()
            S += self.beta * (1 - 0.5 * Re_Tr)
        
        return S
    
    def average_plaquette(self) -> float:
        """
        Compute average plaquette: ⟨(1/2)Re Tr(U_plaq)⟩
        
        Returns:
        --------
        avg : float
            Average over all plaquettes
        """
        if len(self.plaquettes) == 0:
            return 1.0
        
        total = 0.0
        for plaq in self.plaquettes:
            U_plaq = self.plaquette_product(plaq)
            total += 0.5 * U_plaq.trace()
        
        return total / len(self.plaquettes)
    
    def effective_coupling(self) -> float:
        """
        Extract effective coupling from average plaquette.
        
        At weak coupling: ⟨U_plaq⟩ ≈ 1 - (g²_eff/4) + O(g⁴)
        
        Returns:
        --------
        g²_eff : float
            Effective coupling squared
        """
        avg_plaq = self.average_plaquette()
        
        # Weak coupling expansion
        g2_eff = 4 * (1 - avg_plaq)
        
        return g2_eff
    
    def metropolis_update(self, n_sweeps: int = 100, delta: float = 0.5):
        """
        Monte Carlo update using Metropolis algorithm.
        
        Parameters:
        -----------
        n_sweeps : int
            Number of full lattice sweeps
        delta : float
            Step size for random updates
        """
        n_accept = 0
        n_total = 0
        
        for sweep in range(n_sweeps):
            # Update each link
            for link_key in self.links:
                # Current action
                S_old = self.wilson_action()
                
                # Propose new link
                U_old = self.links[link_key]
                U_new = SU2Element.random(beta=1/delta) * U_old  # Small random change
                
                # Trial configuration
                self.links[link_key] = U_new
                S_new = self.wilson_action()
                
                # Metropolis acceptance
                dS = S_new - S_old
                accept = (dS < 0) or (np.random.rand() < np.exp(-dS))
                
                if accept:
                    n_accept += 1
                else:
                    # Reject: restore old link
                    self.links[link_key] = U_old
                
                n_total += 1
        
        acceptance_rate = n_accept / n_total if n_total > 0 else 0
        return acceptance_rate
    
    def thermalize(self, n_sweeps: int = 1000):
        """Run thermalization sweeps."""
        print(f"Thermalizing for {n_sweeps} sweeps...")
        
        for i in range(0, n_sweeps, 100):
            acc = self.metropolis_update(n_sweeps=100)
            if (i // 100) % 5 == 0:
                avg_plaq = self.average_plaquette()
                g2 = self.effective_coupling()
                print(f"  Sweep {i+100}/{n_sweeps}: ⟨U⟩ = {avg_plaq:.6f}, g² = {g2:.6f}, acc = {acc:.3f}")
        
        print("Thermalization complete!")
    
    def measure_observables(self, n_measurements: int = 100, n_sweeps_between: int = 10) -> Dict:
        """
        Measure observables after thermalization.
        
        Parameters:
        -----------
        n_measurements : int
            Number of independent measurements
        n_sweeps_between : int
            Sweeps between measurements (decorrelation)
        
        Returns:
        --------
        data : dict
            Measurement results
        """
        avg_plaq_list = []
        g2_list = []
        action_list = []
        
        print(f"Measuring observables ({n_measurements} samples)...")
        
        for i in range(n_measurements):
            # Evolve system
            self.metropolis_update(n_sweeps=n_sweeps_between)
            
            # Measure
            avg_plaq = self.average_plaquette()
            g2 = self.effective_coupling()
            S = self.wilson_action()
            
            avg_plaq_list.append(avg_plaq)
            g2_list.append(g2)
            action_list.append(S)
            
            if (i+1) % 20 == 0:
                print(f"  Sample {i+1}/{n_measurements}: ⟨U⟩ = {avg_plaq:.6f}, g² = {g2:.6f}")
        
        # Compute statistics
        data = {
            'avg_plaquette': np.array(avg_plaq_list),
            'g2_effective': np.array(g2_list),
            'action': np.array(action_list),
            'avg_plaq_mean': np.mean(avg_plaq_list),
            'avg_plaq_std': np.std(avg_plaq_list),
            'g2_mean': np.mean(g2_list),
            'g2_std': np.std(g2_list),
            'beta': self.beta,
            'g2_bare': 4 / self.beta,
            'one_over_4pi': 1 / (4 * np.pi)
        }
        
        return data


def analyze_coupling_vs_4pi(ell_max: int = 5, beta_values: List[float] = None):
    """
    Scan coupling constant and compare with 1/(4π).
    
    Parameters:
    -----------
    ell_max : int
        Lattice size
    beta_values : list of float
        Values of β = 4/g² to scan
    """
    if beta_values is None:
        # Scan around β = 4 / (1/(4π)) ≈ 50
        beta_values = [10, 20, 30, 40, 50, 60, 80, 100]
    
    results = []
    
    for beta in beta_values:
        print("\n" + "="*80)
        print(f"β = {beta:.2f}, g² = {4/beta:.6f}")
        print("="*80)
        
        # Create gauge field
        gauge = WilsonGaugeField(ell_max=ell_max, beta=beta)
        
        # Thermalize
        gauge.thermalize(n_sweeps=500)
        
        # Measure
        data = gauge.measure_observables(n_measurements=50)
        
        results.append(data)
        
        print(f"\nResults for β = {beta:.2f}:")
        print(f"  g²_bare = {data['g2_bare']:.6f}")
        print(f"  g²_eff  = {data['g2_mean']:.6f} ± {data['g2_std']:.6f}")
        print(f"  1/(4π)  = {data['one_over_4pi']:.6f}")
        print(f"  Ratio g²_eff / (1/4π) = {data['g2_mean']/data['one_over_4pi']:.6f}")
    
    return results


def main():
    """Main execution: Wilson gauge fields."""
    print("=" * 80)
    print("PHASE 9.1: WILSON GAUGE FIELDS ON DISCRETE LATTICE")
    print("=" * 80)
    print()
    
    # Test SU(2) operations
    print("Testing SU(2) group operations...")
    U1 = SU2Element.random()
    U2 = SU2Element.random()
    U3 = U1 * U2
    print(f"  U1 = {U1}")
    print(f"  U2 = {U2}")
    print(f"  U3 = U1*U2 = {U3}")
    print(f"  Tr(U1) = {U1.trace():.6f}")
    print()
    
    # Create small gauge field
    print("Creating Wilson gauge field...")
    gauge = WilsonGaugeField(ell_max=3, beta=50.0)
    print()
    
    # Initial state
    print("Initial state (cold start):")
    print(f"  ⟨U_plaq⟩ = {gauge.average_plaquette():.6f}")
    print(f"  g²_eff = {gauge.effective_coupling():.6f}")
    print(f"  S = {gauge.wilson_action():.2f}")
    print()
    
    # Thermalize
    gauge.thermalize(n_sweeps=500)
    print()
    
    # Measure
    data = gauge.measure_observables(n_measurements=100)
    print()
    
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Bare coupling:       g² = {data['g2_bare']:.6f}")
    print(f"Effective coupling:  g² = {data['g2_mean']:.6f} ± {data['g2_std']:.6f}")
    print(f"Geometric constant:  1/(4π) = {data['one_over_4pi']:.6f}")
    print(f"Ratio: g²_eff / (1/4π) = {data['g2_mean']/data['one_over_4pi']:.6f}")
    print("=" * 80)
    print()
    
    print("🔥 KEY QUESTION: Is g² proportional to 1/(4π)?")
    print()
    print("Phase 9.1 gauge field implementation complete!")
    print("Next: Scan β values to test hypothesis systematically")


if __name__ == '__main__':
    main()
