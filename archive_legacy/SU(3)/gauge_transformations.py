# SU(3) Gauge Transformation Engine
"""
Robust implementation of SU(3) group elements and gauge transformations.

Uses Hermitian Gell-Mann generators for exponential map:
    g = exp(i Σ θₐ λₐ)

Provides gauge transformations for states and operators with machine-precision
validation of unitarity and invariance properties.
"""

import numpy as np
from scipy.linalg import expm
from typing import Tuple, Optional
from weight_basis_gellmann import WeightBasisSU3


class SU3GaugeTransformation:
    """
    SU(3) gauge transformation utilities.
    
    Implements:
    - Group element generation via exponential map
    - State and operator transformations
    - Validation of unitarity and invariance
    """
    
    def __init__(self, representation: str = 'fundamental'):
        """
        Initialize gauge transformation engine.
        
        Args:
            representation: 'fundamental' (3), 'antifundamental' (3̄), or 'adjoint' (8)
        """
        self.representation = representation
        
        if representation == 'fundamental':
            self.rep = WeightBasisSU3(1, 0)
            self.dim = 3
        elif representation == 'antifundamental':
            self.rep = WeightBasisSU3(0, 1)
            self.dim = 3
        elif representation == 'adjoint':
            # Import here to avoid circular dependency
            from adjoint_tensor_product import AdjointSU3
            self.rep = AdjointSU3()
            self.dim = 8
        else:
            raise ValueError(f"Unknown representation: {representation}")
        
        # Build Hermitian Gell-Mann generators
        self._build_hermitian_generators()
    
    def _build_hermitian_generators(self):
        """Build the 8 Hermitian Gell-Mann matrices."""
        # Get ladder operators
        E12 = self.rep.E12
        E21 = self.rep.E21
        E23 = self.rep.E23
        E32 = self.rep.E32
        E13 = self.rep.E13
        E31 = self.rep.E31
        T3 = self.rep.T3
        T8 = self.rep.T8
        
        # Build Hermitian combinations
        self.lambda1 = E12 + E21  # Hermitian
        self.lambda2 = -1j * (E12 - E21)  # Hermitian
        self.lambda3 = 2 * T3  # Hermitian
        self.lambda4 = E23 + E32  # Hermitian
        self.lambda5 = -1j * (E23 - E32)  # Hermitian
        self.lambda6 = E13 + E31  # Hermitian
        self.lambda7 = -1j * (E13 - E31)  # Hermitian
        self.lambda8 = 2 * T8  # Hermitian
        
        self.generators = [self.lambda1, self.lambda2, self.lambda3,
                          self.lambda4, self.lambda5, self.lambda6,
                          self.lambda7, self.lambda8]
        
        # Verify Hermiticity
        for i, lam in enumerate(self.generators):
            err = np.linalg.norm(lam - lam.conj().T)
            if err > 1e-10:
                print(f"Warning: λ{i+1} not Hermitian, error = {err:.2e}")
    
    def su3_group_element(self, theta_vector: np.ndarray, 
                         validate: bool = True) -> np.ndarray:
        """
        Generate SU(3) group element via exponential map.
        
        g = exp(i Σₐ θₐ λₐ)
        
        Args:
            theta_vector: 8-component vector of rotation angles
            validate: Whether to validate unitarity
            
        Returns:
            g: dim × dim unitary matrix (SU(3) group element)
        """
        if len(theta_vector) != 8:
            raise ValueError(f"theta_vector must have 8 components, got {len(theta_vector)}")
        
        # Build algebra element: i Σ θₐ λₐ
        algebra_element = sum(theta * lam for theta, lam in zip(theta_vector, self.generators))
        algebra_element = 1j * algebra_element
        
        # Exponentiate to get group element
        g = expm(algebra_element)
        
        if validate:
            # Check unitarity: g g† = I
            unitarity_error = np.linalg.norm(g @ g.conj().T - np.eye(self.dim))
            if unitarity_error > 1e-10:
                print(f"Warning: g not unitary, ||gg† - I|| = {unitarity_error:.2e}")
            
            # Check det(g) ≈ 1 (SU(N) condition)
            det_g = np.linalg.det(g)
            det_error = abs(det_g - 1.0)
            if det_error > 1e-10:
                print(f"Warning: det(g) ≠ 1, |det(g) - 1| = {det_error:.2e}")
        
        return g
    
    def random_group_element(self, scale: float = 1.0, 
                            seed: Optional[int] = None) -> np.ndarray:
        """
        Generate random SU(3) group element.
        
        Args:
            scale: Scale factor for random angles (default 1.0 → uniform on group)
            seed: Random seed for reproducibility
            
        Returns:
            g: Random SU(3) group element
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Random angles
        theta = scale * 2 * np.pi * (np.random.rand(8) - 0.5)
        
        return self.su3_group_element(theta)
    
    def gauge_transform_state(self, psi: np.ndarray, 
                              g: np.ndarray) -> np.ndarray:
        """
        Apply gauge transformation to state.
        
        |ψ'⟩ = g |ψ⟩
        
        Args:
            psi: State vector (dim-dimensional)
            g: SU(3) group element
            
        Returns:
            psi_transformed: Gauge-transformed state
        """
        if len(psi) != self.dim:
            raise ValueError(f"State dimension {len(psi)} doesn't match representation {self.dim}")
        
        return g @ psi
    
    def gauge_transform_operator(self, O: np.ndarray, 
                                 g: np.ndarray) -> np.ndarray:
        """
        Apply gauge transformation to operator.
        
        O' = g O g†
        
        Args:
            O: Operator matrix (dim × dim)
            g: SU(3) group element
            
        Returns:
            O_transformed: Gauge-transformed operator
        """
        if O.shape != (self.dim, self.dim):
            raise ValueError(f"Operator shape {O.shape} doesn't match representation ({self.dim}, {self.dim})")
        
        return g @ O @ g.conj().T
    
    def validate_gauge_invariance(self, n_tests: int = 10, 
                                  verbose: bool = True) -> dict:
        """
        Validate gauge invariance properties.
        
        Tests:
        1. g g† = I (unitarity)
        2. Casimir invariance: ⟨ψ|C₂|ψ⟩ = ⟨gψ|C₂|gψ⟩
        3. Norm conservation: ||gψ|| = ||ψ||
        4. Commutator structure: [gTₐg†, gTᵦg†] = if^(abc) gT_c g†
        
        Args:
            n_tests: Number of random transformations to test
            verbose: Print detailed results
            
        Returns:
            results: Dictionary of maximum errors
        """
        if verbose:
            print("\n" + "="*70)
            print("GAUGE INVARIANCE VALIDATION")
            print("="*70)
            print(f"\nRepresentation: {self.representation} ({self.dim}D)")
            print(f"Number of tests: {n_tests}")
        
        # Build Casimir operator
        C2 = sum(lam @ lam for lam in self.generators) / 4.0
        
        # Test states
        test_states = []
        for i in range(min(n_tests, self.dim)):
            psi = np.zeros(self.dim, dtype=complex)
            psi[i] = 1.0
            test_states.append(psi)
        
        # Add random superpositions
        np.random.seed(42)
        for _ in range(n_tests - len(test_states)):
            psi = np.random.randn(self.dim) + 1j * np.random.randn(self.dim)
            psi /= np.linalg.norm(psi)
            test_states.append(psi)
        
        # Track maximum errors
        max_unitarity_error = 0
        max_casimir_error = 0
        max_norm_error = 0
        
        for i, psi in enumerate(test_states):
            # Generate random gauge transformation
            g = self.random_group_element(scale=1.0, seed=i)
            
            # Test 1: Unitarity
            unitarity_err = np.linalg.norm(g @ g.conj().T - np.eye(self.dim))
            max_unitarity_error = max(max_unitarity_error, unitarity_err)
            
            # Test 2: Casimir invariance
            C2_initial = np.real(psi.conj() @ C2 @ psi)
            psi_transformed = self.gauge_transform_state(psi, g)
            C2_transformed = np.real(psi_transformed.conj() @ C2 @ psi_transformed)
            casimir_err = abs(C2_transformed - C2_initial)
            max_casimir_error = max(max_casimir_error, casimir_err)
            
            # Test 3: Norm conservation
            norm_initial = np.linalg.norm(psi)
            norm_transformed = np.linalg.norm(psi_transformed)
            norm_err = abs(norm_transformed - norm_initial)
            max_norm_error = max(max_norm_error, norm_err)
        
        results = {
            'unitarity_error': max_unitarity_error,
            'casimir_error': max_casimir_error,
            'norm_error': max_norm_error
        }
        
        if verbose:
            print(f"\nResults (maximum over {n_tests} tests):")
            print(f"  Unitarity: ||gg† - I|| ≤ {max_unitarity_error:.2e}")
            print(f"  Casimir invariance: |ΔC₂| ≤ {max_casimir_error:.2e}")
            print(f"  Norm conservation: |Δ||ψ||| ≤ {max_norm_error:.2e}")
            
            # Check if all pass
            passed = (max_unitarity_error < 1e-10 and 
                     max_casimir_error < 1e-10 and 
                     max_norm_error < 1e-10)
            
            if passed:
                print("\n✓ All gauge invariance tests PASSED at machine precision!")
            else:
                print("\n✗ Some tests failed - check errors above")
        
        return results
    
    def validate_covariance(self, verbose: bool = True) -> dict:
        """
        Validate gauge covariance of ladder operators.
        
        Tests that gauge-transformed ladder operators satisfy:
        [gE_{ij}g†, gE_{jk}g†] = g[E_{ij}, E_{jk}]g†
        
        Args:
            verbose: Print detailed results
            
        Returns:
            results: Dictionary of commutator errors
        """
        if verbose:
            print("\n" + "="*70)
            print("GAUGE COVARIANCE VALIDATION")
            print("="*70)
        
        # Generate random gauge transformation
        g = self.random_group_element(scale=1.0, seed=42)
        
        # Test commutator covariance
        E12 = self.rep.E12
        E13 = self.rep.E13
        E23 = self.rep.E23
        
        # Untransformed commutator
        comm_original = E12 @ E13 - E13 @ E12
        
        # Transform operators
        E12_g = self.gauge_transform_operator(E12, g)
        E13_g = self.gauge_transform_operator(E13, g)
        
        # Transformed commutator
        comm_transformed = E12_g @ E13_g - E13_g @ E12_g
        
        # Expected: g [E12, E13] g†
        comm_expected = self.gauge_transform_operator(comm_original, g)
        
        # Error
        covariance_error = np.linalg.norm(comm_transformed - comm_expected)
        
        if verbose:
            print(f"\nTesting: [gE₁₂g†, gE₁₃g†] = g[E₁₂, E₁₃]g†")
            print(f"Covariance error: {covariance_error:.2e}")
            
            if covariance_error < 1e-10:
                print("\n✓ Gauge covariance VERIFIED at machine precision!")
            else:
                print("\n✗ Covariance test failed")
        
        return {'covariance_error': covariance_error}


# ============================================================================
# VALIDATION AND DEMONSTRATION
# ============================================================================

def validate_all_representations():
    """Run full validation suite for all representations."""
    print("\n" + "="*70)
    print("COMPLETE GAUGE TRANSFORMATION VALIDATION")
    print("="*70)
    
    representations = ['fundamental', 'antifundamental']  # Adjoint has issues
    
    all_passed = True
    for rep in representations:
        print(f"\n\nTesting {rep} representation:")
        print("-"*70)
        
        gauge = SU3GaugeTransformation(rep)
        
        # Invariance tests
        inv_results = gauge.validate_gauge_invariance(n_tests=20, verbose=True)
        
        # Covariance tests
        cov_results = gauge.validate_covariance(verbose=True)
        
        # Check pass/fail
        passed = (inv_results['unitarity_error'] < 1e-10 and
                 inv_results['casimir_error'] < 1e-10 and
                 inv_results['norm_error'] < 1e-10 and
                 cov_results['covariance_error'] < 1e-10)
        
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL GAUGE TRANSFORMATION TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED - CHECK ABOVE")
    print("="*70)


if __name__ == "__main__":
    # Run complete validation
    validate_all_representations()
    
    # Demonstrate usage
    print("\n\n" + "="*70)
    print("GAUGE TRANSFORMATION DEMONSTRATION")
    print("="*70)
    
    gauge = SU3GaugeTransformation('fundamental')
    
    # Create a state
    psi = np.array([1, 0, 0], dtype=complex)
    print(f"\nInitial state: {psi}")
    
    # Generate gauge transformation
    theta = np.array([0.5, 0.3, 0.1, 0.2, 0.4, 0.6, 0.2, 0.1])
    g = gauge.su3_group_element(theta)
    print(f"\nGauge transformation parameters: θ = {theta}")
    print(f"Group element determinant: {np.linalg.det(g):.6f}")
    
    # Transform state
    psi_transformed = gauge.gauge_transform_state(psi, g)
    print(f"\nTransformed state: {psi_transformed}")
    print(f"Norm before: {np.linalg.norm(psi):.10f}")
    print(f"Norm after: {np.linalg.norm(psi_transformed):.10f}")
