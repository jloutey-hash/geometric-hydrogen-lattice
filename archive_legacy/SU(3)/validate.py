"""
SU(3) Lattice Validation Script
Tests the commutation relations and Casimir eigenvalues.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import identity
from scipy.sparse.linalg import eigsh
from lattice import SU3Lattice
from operators_v2 import SU3Operators


class SU3Validator:
    """Validates the SU(3) lattice construction."""
    
    def __init__(self, max_p: int, max_q: int):
        """
        Initialize validator.
        
        Parameters:
        -----------
        max_p, max_q : int
            Maximum Dynkin labels for lattice construction
        """
        self.max_p = max_p
        self.max_q = max_q
        
        print(f"Initializing SU(3) lattice with max_p={max_p}, max_q={max_q}")
        self.lattice = SU3Lattice(max_p, max_q)
        self.lattice.print_summary()
        
        print("\nBuilding operators...")
        self.ops = SU3Operators(self.lattice)
        
        self.results = {
            'commutator_errors': {},
            'eigenvalue_errors': {},
            'max_commutator_error': 0.0,
            'max_eigenvalue_error': 0.0
        }
    
    def commutator(self, A, B):
        """Compute the commutator [A, B] = AB - BA."""
        return A @ B - B @ A
    
    def anticommutator(self, A, B):
        """Compute the anticommutator {A, B} = AB + BA."""
        return A @ B + B @ A
    
    def test_commutation_relations(self):
        """Test key SU(3) commutation relations."""
        print("\n" + "="*60)
        print("TESTING COMMUTATION RELATIONS")
        print("="*60)
        
        ops = self.ops.get_operators()
        T3 = ops['T3']
        T8 = ops['T8']
        I_plus = ops['I+']
        I_minus = ops['I-']
        U_plus = ops['U+']
        U_minus = ops['U-']
        V_plus = ops['V+']
        V_minus = ops['V-']
        
        errors = {}
        
        # Test 1: [I+, I-] = 2*T3
        print("\n1. Testing [I+, I-] = 2*T3")
        comm = self.commutator(I_plus, I_minus)
        expected = 2 * T3
        diff = comm - expected
        error = np.max(np.abs(diff.data)) if diff.nnz > 0 else 0.0
        errors['[I+, I-] = 2*T3'] = error
        print(f"   Max error: {error:.2e}")
        
        # Test 2: [T3, I+] = I+
        print("\n2. Testing [T3, I+] = I+")
        comm = self.commutator(T3, I_plus)
        expected = I_plus
        diff = comm - expected
        error = np.max(np.abs(diff.data)) if diff.nnz > 0 else 0.0
        errors['[T3, I+] = I+'] = error
        print(f"   Max error: {error:.2e}")
        
        # Test 3: [T3, I-] = -I-
        print("\n3. Testing [T3, I-] = -I-")
        comm = self.commutator(T3, I_minus)
        expected = -I_minus
        diff = comm - expected
        error = np.max(np.abs(diff.data)) if diff.nnz > 0 else 0.0
        errors['[T3, I-] = -I-'] = error
        print(f"   Max error: {error:.2e}")
        
        # Test 4: [U+, U-] should give combination of T3 and T8
        # [U+, U-] = -(3/2)*T3 + (sqrt(3)/2)*T8
        print("\n4. Testing [U+, U-] = -(3/2)*T3 + (sqrt(3)/2)*T8")
        comm = self.commutator(U_plus, U_minus)
        expected = -1.5 * T3 + (np.sqrt(3)/2.0) * T8
        diff = comm - expected
        error = np.max(np.abs(diff.data)) if diff.nnz > 0 else 0.0
        errors['[U+, U-] relation'] = error
        print(f"   Max error: {error:.2e}")
        
        # Test 5: [V+, V-] should give combination of T3 and T8
        # [V+, V-] = (3/2)*T3 + (sqrt(3)/2)*T8
        print("\n5. Testing [V+, V-] = (3/2)*T3 + (sqrt(3)/2)*T8")
        comm = self.commutator(V_plus, V_minus)
        expected = 1.5 * T3 + (np.sqrt(3)/2.0) * T8
        diff = comm - expected
        error = np.max(np.abs(diff.data)) if diff.nnz > 0 else 0.0
        errors['[V+, V-] relation'] = error
        print(f"   Max error: {error:.2e}")
        
        # Test 6: [T3, U+] = -0.5 * U+
        print("\n6. Testing [T3, U+] = -0.5 * U+")
        comm = self.commutator(T3, U_plus)
        expected = -0.5 * U_plus
        diff = comm - expected
        error = np.max(np.abs(diff.data)) if diff.nnz > 0 else 0.0
        errors['[T3, U+] = -0.5*U+'] = error
        print(f"   Max error: {error:.2e}")
        
        # Test 7: [T3, V+] = 0.5 * V+
        print("\n7. Testing [T3, V+] = 0.5 * V+")
        comm = self.commutator(T3, V_plus)
        expected = 0.5 * V_plus
        diff = comm - expected
        error = np.max(np.abs(diff.data)) if diff.nnz > 0 else 0.0
        errors['[T3, V+] = 0.5*V+'] = error
        print(f"   Max error: {error:.2e}")
        
        # Test 8: [I+, U-] = V-
        print("\n8. Testing [I+, U-] = V-")
        comm = self.commutator(I_plus, U_minus)
        expected = V_minus
        diff = comm - expected
        error = np.max(np.abs(diff.data)) if diff.nnz > 0 else 0.0
        errors['[I+, U-] = V-'] = error
        print(f"   Max error: {error:.2e}")
        
        # Test 9: [I+, V-] = -U-
        print("\n9. Testing [I+, V-] = -U-")
        comm = self.commutator(I_plus, V_minus)
        expected = -U_minus
        diff = comm - expected
        error = np.max(np.abs(diff.data)) if diff.nnz > 0 else 0.0
        errors['[I+, V-] = -U-'] = error
        print(f"   Max error: {error:.2e}")
        
        # Store results
        self.results['commutator_errors'] = errors
        self.results['max_commutator_error'] = max(errors.values())
        
        print("\n" + "-"*60)
        print(f"MAXIMUM COMMUTATOR ERROR: {self.results['max_commutator_error']:.2e}")
        print("-"*60)
        
        return errors
    
    def test_casimir_eigenvalues(self):
        """Test that C2 has correct eigenvalues."""
        print("\n" + "="*60)
        print("TESTING CASIMIR EIGENVALUES")
        print("="*60)
        
        C2 = self.ops.C2
        
        # Compute eigenvalues
        print("\nComputing eigenvalues of C2...")
        eigenvalues = np.linalg.eigvalsh(C2.toarray())
        
        # Group states by (p, q) and compute theoretical eigenvalue
        rep_eigenvalues = {}
        for state in self.lattice.states:
            p, q = state['p'], state['q']
            key = (p, q)
            if key not in rep_eigenvalues:
                # Theoretical Casimir eigenvalue
                C2_theory = (p*p + q*q + 3*p + 3*q + p*q) / 3.0
                rep_eigenvalues[key] = {
                    'theory': C2_theory,
                    'computed': [],
                    'count': 0
                }
            rep_eigenvalues[key]['count'] += 1
        
        # Match computed eigenvalues to representations
        print("\nEigenvalue comparison:")
        print(f"{'(p,q)':<10} {'Theory':<15} {'Computed (avg)':<15} {'Error':<15} {'Count':<10}")
        print("-"*70)
        
        errors = {}
        for (p, q), data in sorted(rep_eigenvalues.items()):
            theory = data['theory']
            count = data['count']
            
            # Find the eigenvalues closest to theory (should be degenerate)
            # Sort eigenvalues and take those close to theory
            close_eigs = [ev for ev in eigenvalues if abs(ev - theory) < 0.5]
            
            if close_eigs:
                computed_avg = np.mean(close_eigs[:count])
                error = abs(computed_avg - theory) / (abs(theory) + 1e-10)
            else:
                computed_avg = np.nan
                error = np.inf
            
            errors[(p, q)] = error
            
            print(f"({p},{q}){'':<6} {theory:<15.6f} {computed_avg:<15.6f} "
                  f"{error:<15.2e} {count:<10}")
        
        # Store results
        self.results['eigenvalue_errors'] = errors
        self.results['max_eigenvalue_error'] = max(errors.values())
        
        print("\n" + "-"*60)
        print(f"MAXIMUM EIGENVALUE RELATIVE ERROR: {self.results['max_eigenvalue_error']:.2e}")
        print("-"*60)
        
        return errors
    
    def visualize_lattice(self, save_path='lattice_visualization.png'):
        """Visualize the lattice points colored by representation."""
        print(f"\nGenerating lattice visualization...")
        
        coords = self.lattice.get_coordinates()
        
        # Create color map by (p, q)
        colors = []
        for state in self.lattice.states:
            p, q = state['p'], state['q']
            # Create a unique color for each representation
            color_val = p + q * 3
            colors.append(color_val)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=colors, 
                            cmap='viridis', s=50, alpha=0.7, edgecolors='black')
        plt.colorbar(scatter, label='p + 3q')
        plt.xlabel('I3 (Isospin)')
        plt.ylabel('Y (Hypercharge)')
        plt.title(f'SU(3) Weight Lattice (max_p={self.max_p}, max_q={self.max_q})')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Visualization saved to {save_path}")
        plt.close()
    
    def visualize_casimir_distribution(self, save_path='casimir_distribution.png'):
        """Visualize the lattice colored by C2 eigenvalue."""
        print(f"\nGenerating Casimir distribution visualization...")
        
        coords = self.lattice.get_coordinates()
        C2 = self.ops.C2
        
        # Get diagonal elements of C2 (should be close to eigenvalues for each state)
        c2_values = np.diag(C2.toarray())
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=c2_values,
                            cmap='plasma', s=50, alpha=0.7, edgecolors='black')
        plt.colorbar(scatter, label='C2 eigenvalue')
        plt.xlabel('I3 (Isospin)')
        plt.ylabel('Y (Hypercharge)')
        plt.title(f'SU(3) Lattice - Casimir Operator C2')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Casimir distribution saved to {save_path}")
        plt.close()
    
    def generate_report(self):
        """Generate a comprehensive validation report."""
        print("\n" + "="*60)
        print("VALIDATION REPORT SUMMARY")
        print("="*60)
        
        print(f"\nLattice Configuration:")
        print(f"  Max (p, q): ({self.max_p}, {self.max_q})")
        print(f"  Total states: {self.lattice.get_dimension()}")
        
        print(f"\nCommutation Relations:")
        print(f"  Max error: {self.results['max_commutator_error']:.2e}")
        if self.results['max_commutator_error'] < 1e-13:
            print(f"  Status: ✓ PASSED (< 10^-13)")
        else:
            print(f"  Status: ✗ FAILED (>= 10^-13)")
        
        print(f"\nCasimir Eigenvalues:")
        print(f"  Max relative error: {self.results['max_eigenvalue_error']:.2e}")
        if self.results['max_eigenvalue_error'] < 1e-12:
            print(f"  Status: ✓ PASSED (< 10^-12)")
        else:
            print(f"  Status: ✗ FAILED (>= 10^-12)")
        
        print("\n" + "="*60)
        if (self.results['max_commutator_error'] < 1e-13 and 
            self.results['max_eigenvalue_error'] < 1e-12):
            print("CONCLUSION: SU(3) LATTICE CONSTRUCTION SUCCESSFUL! ✓")
        else:
            print("CONCLUSION: SU(3) LATTICE NEEDS REFINEMENT ✗")
        print("="*60 + "\n")


def main():
    """Main validation routine."""
    print("="*60)
    print("SU(3) TRIANGULAR LATTICE VALIDATION")
    print("="*60)
    
    # Create validator with max_p=2, max_q=2
    validator = SU3Validator(max_p=2, max_q=2)
    
    # Test commutation relations
    validator.test_commutation_relations()
    
    # Test Casimir eigenvalues
    validator.test_casimir_eigenvalues()
    
    # Generate visualizations
    validator.visualize_lattice()
    validator.visualize_casimir_distribution()
    
    # Generate final report
    validator.generate_report()


if __name__ == "__main__":
    main()
