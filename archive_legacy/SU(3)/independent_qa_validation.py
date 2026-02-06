"""
Independent QA Validation of SU(3) Representation Framework
============================================================

This script acts as an independent QA engineer verifying the correctness
of the SU(3) implementation without assuming any component is correct.
All algebra relations, Casimir values, hermiticity, and transformations
are validated independently.
"""

import numpy as np
from typing import Dict, List, Tuple
import sys

# Import the modules to be tested
from weight_basis_gellmann import WeightBasisSU3
from gt_basis_transformed import GTBasisSU3
from adjoint_tensor_product import AdjointSU3
from lattice import SU3Lattice


class IndependentQAValidator:
    """Independent validation of SU(3) representations."""
    
    def __init__(self):
        """Initialize validator with results storage."""
        self.results = {}
        self.defects = []
        
    def validate_all(self):
        """Run full validation suite."""
        print("="*80)
        print("INDEPENDENT QA VALIDATION OF SU(3) FRAMEWORK")
        print("="*80)
        print()
        
        # Load and summarize representations
        self.load_and_summarize()
        
        # Validate each representation
        for rep_label in ["(1,0) fundamental", "(0,1) antifundamental", "(1,1) adjoint"]:
            print("\n" + "="*80)
            print(f"VALIDATING REPRESENTATION: {rep_label}")
            print("="*80)
            
            if rep_label == "(1,0) fundamental":
                p, q = 1, 0
            elif rep_label == "(0,1) antifundamental":
                p, q = 0, 1
            elif rep_label == "(1,1) adjoint":
                p, q = 1, 1
            
            # Validate in weight basis
            print(f"\n--- Weight Basis Validation ---")
            weight_results = self.validate_representation_weight(p, q)
            
            # Validate in GT basis
            print(f"\n--- GT Basis Validation ---")
            gt_results = self.validate_representation_gt(p, q)
            
            # Validate transformation
            print(f"\n--- Transformation Validation ---")
            transform_results = self.validate_transformation(p, q)
            
            # Store results
            self.results[rep_label] = {
                'weight': weight_results,
                'gt': gt_results,
                'transformation': transform_results
            }
        
        # Generate final report
        self.generate_final_report()
    
    def load_and_summarize(self):
        """Load files and print summary of representations detected."""
        print("Loading modules...")
        print(f"  - weight_basis_gellmann.py")
        print(f"  - gt_basis_transformed.py")
        print(f"  - adjoint_tensor_product.py")
        print(f"  - lattice.py")
        print()
        
        print("Representations detected:")
        
        # Check (1,0)
        try:
            ops = WeightBasisSU3(1, 0)
            print(f"  ✓ (1,0) fundamental - dimension {ops.dim}")
        except Exception as e:
            print(f"  ✗ (1,0) fundamental - ERROR: {e}")
            self.defects.append(f"(1,0) failed to load: {e}")
        
        # Check (0,1)
        try:
            ops = WeightBasisSU3(0, 1)
            print(f"  ✓ (0,1) antifundamental - dimension {ops.dim}")
        except Exception as e:
            print(f"  ✗ (0,1) antifundamental - ERROR: {e}")
            self.defects.append(f"(0,1) failed to load: {e}")
        
        # Check (1,1)
        try:
            ops = WeightBasisSU3(1, 1)
            print(f"  ✓ (1,1) adjoint - dimension {ops.dim}")
        except Exception as e:
            print(f"  ✗ (1,1) adjoint - ERROR: {e}")
            self.defects.append(f"(1,1) failed to load: {e}")
    
    def validate_representation_weight(self, p: int, q: int) -> Dict:
        """Validate a representation in weight basis."""
        results = {}
        
        try:
            ops = WeightBasisSU3(p, q)
            
            # A. Commutators
            results['commutators'] = self.check_commutators(ops)
            
            # B. Hermiticity
            results['hermiticity'] = self.check_hermiticity(ops)
            
            # C. Casimir
            results['casimir'] = self.check_casimir(ops, p, q)
            
            # D. Diagonality
            results['diagonality'] = self.check_diagonality(ops)
            
            # Print results
            self.print_test_results(results, "Weight Basis")
            
        except Exception as e:
            self.defects.append(f"Weight basis ({p},{q}) validation failed: {e}")
            print(f"  ✗ VALIDATION FAILED: {e}")
            results['error'] = str(e)
        
        return results
    
    def validate_representation_gt(self, p: int, q: int) -> Dict:
        """Validate a representation in GT basis."""
        results = {}
        
        try:
            ops = GTBasisSU3(p, q)
            
            # A. Commutators
            results['commutators'] = self.check_commutators(ops)
            
            # B. Hermiticity
            results['hermiticity'] = self.check_hermiticity(ops)
            
            # C. Casimir
            results['casimir'] = self.check_casimir(ops, p, q)
            
            # D. Diagonality
            results['diagonality'] = self.check_diagonality(ops)
            
            # Structural validation
            results['structure'] = self.check_gt_structure(ops, p, q)
            
            # Print results
            self.print_test_results(results, "GT Basis")
            
        except Exception as e:
            self.defects.append(f"GT basis ({p},{q}) validation failed: {e}")
            print(f"  ✗ VALIDATION FAILED: {e}")
            results['error'] = str(e)
        
        return results
    
    def validate_transformation(self, p: int, q: int) -> Dict:
        """Validate unitary transformation between bases."""
        results = {}
        
        try:
            gt_ops = GTBasisSU3(p, q)
            U = gt_ops.U
            
            # Check unitarity
            U_dag_U = U.conj().T @ U
            identity = np.eye(U.shape[0])
            unitarity_error = np.max(np.abs(U_dag_U - identity))
            results['unitarity'] = unitarity_error
            
            # Check transformation preserves operators
            weight_ops = gt_ops.weight_ops
            
            # Check O_GT = U† O_weight U for each operator
            transform_errors = {}
            
            for name in ['T3', 'T8', 'E12', 'E21', 'E23', 'E32', 'E13', 'E31']:
                O_weight = getattr(weight_ops, name)
                O_gt = getattr(gt_ops, name)
                O_transformed = U.conj().T @ O_weight @ U
                error = np.max(np.abs(O_gt - O_transformed))
                transform_errors[name] = error
            
            results['transformation_errors'] = transform_errors
            
            # Check condition number
            results['condition_number'] = np.linalg.cond(U)
            
            # Check norm preservation
            norm_errors = {}
            for name in ['T3', 'T8', 'E12', 'E21', 'E23', 'E32', 'E13', 'E31']:
                O_weight = getattr(weight_ops, name)
                O_gt = getattr(gt_ops, name)
                norm_weight = np.linalg.norm(O_weight, 'fro')
                norm_gt = np.linalg.norm(O_gt, 'fro')
                norm_errors[name] = abs(norm_weight - norm_gt)
            
            results['norm_preservation'] = norm_errors
            
            # Print results
            print(f"\nUnitarity (U†U - I): {unitarity_error:.2e}", 
                  "✓" if unitarity_error < 1e-10 else "✗")
            print(f"Condition number: {results['condition_number']:.2f}")
            
            print("\nTransformation accuracy (O_GT vs U†O_weight U):")
            for name, error in transform_errors.items():
                status = "✓" if error < 1e-10 else "✗"
                print(f"  {name}: {error:.2e} {status}")
            
            print("\nNorm preservation:")
            for name, error in norm_errors.items():
                status = "✓" if error < 1e-10 else "✗"
                print(f"  {name}: {error:.2e} {status}")
            
        except Exception as e:
            self.defects.append(f"Transformation ({p},{q}) validation failed: {e}")
            print(f"  ✗ VALIDATION FAILED: {e}")
            results['error'] = str(e)
        
        return results
    
    def check_commutators(self, ops) -> Dict:
        """Check all required commutation relations."""
        errors = {}
        
        # [T3, T8] = 0
        comm = ops.T3 @ ops.T8 - ops.T8 @ ops.T3
        errors['[T3,T8]'] = np.max(np.abs(comm))
        
        # [E12, E21] = 2*T3
        comm = ops.E12 @ ops.E21 - ops.E21 @ ops.E12
        expected = 2 * ops.T3
        errors['[E12,E21] - 2T3'] = np.max(np.abs(comm - expected))
        
        # [E23, E32] = T3 + sqrt(3)*T8
        comm = ops.E23 @ ops.E32 - ops.E32 @ ops.E23
        expected = ops.T3 + np.sqrt(3) * ops.T8
        errors['[E23,E32] - (T3+√3T8)'] = np.max(np.abs(comm - expected))
        
        # [E13, E31] = -T3 + sqrt(3)*T8
        comm = ops.E13 @ ops.E31 - ops.E31 @ ops.E13
        expected = -ops.T3 + np.sqrt(3) * ops.T8
        errors['[E13,E31] - (-T3+√3T8)'] = np.max(np.abs(comm - expected))
        
        return errors
    
    def check_hermiticity(self, ops) -> Dict:
        """Check hermiticity of all operators."""
        errors = {}
        
        # T3† = T3
        errors['T3 hermitian'] = np.max(np.abs(ops.T3 - ops.T3.conj().T))
        
        # T8† = T8
        errors['T8 hermitian'] = np.max(np.abs(ops.T8 - ops.T8.conj().T))
        
        # E21 = E12†
        errors['E21 - E12†'] = np.max(np.abs(ops.E21 - ops.E12.conj().T))
        
        # E32 = E23†
        errors['E32 - E23†'] = np.max(np.abs(ops.E32 - ops.E23.conj().T))
        
        # E31 = E13†
        errors['E31 - E13†'] = np.max(np.abs(ops.E31 - ops.E13.conj().T))
        
        return errors
    
    def check_casimir(self, ops, p: int, q: int) -> Dict:
        """Check Casimir operator."""
        results = {}
        
        # Compute C2 = sum of T_a @ T_a
        lambda1 = ops.E12 + ops.E21
        lambda2 = -1j * (ops.E12 - ops.E21)
        lambda3 = 2 * ops.T3
        lambda4 = ops.E23 + ops.E32
        lambda5 = -1j * (ops.E23 - ops.E32)
        lambda6 = ops.E13 + ops.E31
        lambda7 = -1j * (ops.E13 - ops.E31)
        lambda8 = 2 * ops.T8
        
        C2 = sum((lam/2) @ (lam/2) for lam in 
                 [lambda1, lambda2, lambda3, lambda4, lambda5, lambda6, lambda7, lambda8])
        
        # Get eigenvalues
        eigenvalues = np.linalg.eigvalsh(C2)
        
        # Check they're all identical
        results['eigenvalues'] = eigenvalues
        results['std_dev'] = np.std(eigenvalues)
        results['max_deviation'] = np.max(eigenvalues) - np.min(eigenvalues)
        
        # Compare to theoretical value
        theoretical = (p**2 + q**2 + p*q + 3*p + 3*q) / 3
        results['theoretical'] = theoretical
        results['mean_value'] = np.mean(eigenvalues)
        results['error_from_theory'] = abs(np.mean(eigenvalues) - theoretical)
        
        return results
    
    def check_diagonality(self, ops) -> Dict:
        """Check that T3 and T8 are diagonal."""
        errors = {}
        
        # Extract off-diagonal elements
        T3_diag = np.diag(np.diag(ops.T3))
        T3_off = ops.T3 - T3_diag
        errors['T3 off-diagonal'] = np.max(np.abs(T3_off))
        
        T8_diag = np.diag(np.diag(ops.T8))
        T8_off = ops.T8 - T8_diag
        errors['T8 off-diagonal'] = np.max(np.abs(T8_off))
        
        return errors
    
    def check_gt_structure(self, ops, p: int, q: int) -> Dict:
        """Check GT pattern structure."""
        results = {}
        
        # Check dimension matches theory
        dim_theory = (p + 1) * (q + 1) * (p + q + 2) // 2
        results['dimension'] = ops.dim
        results['dimension_theory'] = dim_theory
        results['dimension_match'] = (ops.dim == dim_theory)
        
        # Check GT pattern count
        results['gt_pattern_count'] = len(ops.gt_states)
        results['gt_patterns_match_dim'] = (len(ops.gt_states) == ops.dim)
        
        # Check quantum number correspondence
        I3_from_diag = np.diag(ops.T3).real
        T8_from_diag = np.diag(ops.T8).real
        
        I3_from_gt = ops.T3_GT_diag
        T8_from_gt = ops.T8_GT_diag
        
        results['I3_match'] = np.allclose(I3_from_diag, I3_from_gt, atol=1e-10)
        results['T8_match'] = np.allclose(T8_from_diag, T8_from_gt, atol=1e-10)
        
        return results
    
    def print_test_results(self, results: Dict, basis_name: str):
        """Print formatted test results."""
        
        if 'error' in results:
            return
        
        if 'commutators' in results:
            print("\nCommutator Tests:")
            for name, error in results['commutators'].items():
                status = "✓" if error < 1e-10 else "✗"
                print(f"  {name}: {error:.2e} {status}")
        
        if 'hermiticity' in results:
            print("\nHermiticity Tests:")
            for name, error in results['hermiticity'].items():
                status = "✓" if error < 1e-10 else "✗"
                print(f"  {name}: {error:.2e} {status}")
        
        if 'casimir' in results:
            print("\nCasimir Tests:")
            c = results['casimir']
            print(f"  Theoretical value: {c['theoretical']:.6f}")
            print(f"  Mean eigenvalue: {c['mean_value']:.6f}")
            print(f"  Error from theory: {c['error_from_theory']:.2e}", 
                  "✓" if c['error_from_theory'] < 1e-10 else "✗")
            print(f"  Std deviation: {c['std_dev']:.2e}",
                  "✓" if c['std_dev'] < 1e-10 else "✗")
            print(f"  Max deviation: {c['max_deviation']:.2e}",
                  "✓" if c['max_deviation'] < 1e-10 else "✗")
            print(f"  All eigenvalues: {c['eigenvalues']}")
        
        if 'diagonality' in results:
            print("\nDiagonality Tests:")
            for name, error in results['diagonality'].items():
                status = "✓" if error < 1e-10 else "✗"
                print(f"  {name}: {error:.2e} {status}")
        
        if 'structure' in results:
            print("\nStructural Tests:")
            s = results['structure']
            print(f"  Dimension: {s['dimension']} (theory: {s['dimension_theory']})",
                  "✓" if s['dimension_match'] else "✗")
            print(f"  GT pattern count: {s['gt_pattern_count']}",
                  "✓" if s['gt_patterns_match_dim'] else "✗")
            print(f"  I3 quantum numbers match:",
                  "✓" if s['I3_match'] else "✗")
            print(f"  T8 quantum numbers match:",
                  "✓" if s['T8_match'] else "✗")
    
    def generate_final_report(self):
        """Generate comprehensive final QA report."""
        print("\n" + "="*80)
        print("FINAL QA REPORT")
        print("="*80)
        
        # Summary table of all tests
        print("\n--- COMMUTATOR ERROR SUMMARY ---")
        print(f"{'Representation':<25} {'[T3,T8]':<12} {'[E12,E21]':<12} {'[E23,E32]':<12} {'[E13,E31]':<12}")
        print("-" * 80)
        
        for rep_label in ["(1,0) fundamental", "(0,1) antifundamental", "(1,1) adjoint"]:
            if rep_label not in self.results:
                continue
            
            row = f"{rep_label:<25}"
            
            for basis in ['weight', 'gt']:
                if basis not in self.results[rep_label]:
                    continue
                    
                comm = self.results[rep_label][basis].get('commutators', {})
                
                t3t8 = comm.get('[T3,T8]', float('nan'))
                e12e21 = comm.get('[E12,E21] - 2T3', float('nan'))
                e23e32 = comm.get('[E23,E32] - (T3+√3T8)', float('nan'))
                e13e31 = comm.get('[E13,E31] - (-T3+√3T8)', float('nan'))
                
                basis_name = "W" if basis == 'weight' else "GT"
                print(f"{rep_label:<25} ({basis_name:<2}) "
                      f"{t3t8:<12.2e} {e12e21:<12.2e} {e23e32:<12.2e} {e13e31:<12.2e}")
        
        print("\n--- CASIMIR EIGENVALUE SUMMARY ---")
        print(f"{'Representation':<25} {'Basis':<8} {'Theory':<12} {'Mean':<12} {'Error':<12} {'StdDev':<12}")
        print("-" * 95)
        
        for rep_label in ["(1,0) fundamental", "(0,1) antifundamental", "(1,1) adjoint"]:
            if rep_label not in self.results:
                continue
            
            for basis in ['weight', 'gt']:
                if basis not in self.results[rep_label]:
                    continue
                    
                cas = self.results[rep_label][basis].get('casimir', {})
                
                if 'theoretical' in cas:
                    theory = cas['theoretical']
                    mean = cas['mean_value']
                    error = cas['error_from_theory']
                    std = cas['std_dev']
                    
                    basis_name = "Weight" if basis == 'weight' else "GT"
                    print(f"{rep_label:<25} {basis_name:<8} "
                          f"{theory:<12.6f} {mean:<12.6f} {error:<12.2e} {std:<12.2e}")
        
        print("\n--- HERMITICITY ERROR SUMMARY ---")
        print(f"{'Representation':<25} {'Basis':<8} {'T3':<12} {'T8':<12} {'E21-E12†':<12} {'E32-E23†':<12} {'E31-E13†':<12}")
        print("-" * 105)
        
        for rep_label in ["(1,0) fundamental", "(0,1) antifundamental", "(1,1) adjoint"]:
            if rep_label not in self.results:
                continue
            
            for basis in ['weight', 'gt']:
                if basis not in self.results[rep_label]:
                    continue
                    
                herm = self.results[rep_label][basis].get('hermiticity', {})
                
                t3 = herm.get('T3 hermitian', float('nan'))
                t8 = herm.get('T8 hermitian', float('nan'))
                e12 = herm.get('E21 - E12†', float('nan'))
                e23 = herm.get('E32 - E23†', float('nan'))
                e13 = herm.get('E31 - E13†', float('nan'))
                
                basis_name = "Weight" if basis == 'weight' else "GT"
                print(f"{rep_label:<25} {basis_name:<8} "
                      f"{t3:<12.2e} {t8:<12.2e} {e12:<12.2e} {e23:<12.2e} {e13:<12.2e}")
        
        print("\n--- DIAGONALITY ERROR SUMMARY ---")
        print(f"{'Representation':<25} {'Basis':<8} {'T3 off-diag':<15} {'T8 off-diag':<15}")
        print("-" * 70)
        
        for rep_label in ["(1,0) fundamental", "(0,1) antifundamental", "(1,1) adjoint"]:
            if rep_label not in self.results:
                continue
            
            for basis in ['weight', 'gt']:
                if basis not in self.results[rep_label]:
                    continue
                    
                diag = self.results[rep_label][basis].get('diagonality', {})
                
                t3 = diag.get('T3 off-diagonal', float('nan'))
                t8 = diag.get('T8 off-diagonal', float('nan'))
                
                basis_name = "Weight" if basis == 'weight' else "GT"
                print(f"{rep_label:<25} {basis_name:<8} {t3:<15.2e} {t8:<15.2e}")
        
        print("\n--- UNITARY TRANSFORMATION VERIFICATION ---")
        print(f"{'Representation':<25} {'U†U - I':<15} {'Condition':<12}")
        print("-" * 60)
        
        for rep_label in ["(1,0) fundamental", "(0,1) antifundamental", "(1,1) adjoint"]:
            if rep_label not in self.results:
                continue
            
            trans = self.results[rep_label].get('transformation', {})
            
            if 'unitarity' in trans:
                unit = trans['unitarity']
                cond = trans['condition_number']
                print(f"{rep_label:<25} {unit:<15.2e} {cond:<12.2f}")
        
        # Pass/Fail Summary
        print("\n--- PASS/FAIL SUMMARY ---")
        
        all_passed = True
        threshold = 1e-10
        
        for rep_label in ["(1,0) fundamental", "(0,1) antifundamental", "(1,1) adjoint"]:
            if rep_label not in self.results:
                print(f"{rep_label}: FAIL (not tested)")
                all_passed = False
                continue
            
            rep_passed = True
            
            # Check weight basis
            if 'weight' in self.results[rep_label]:
                w = self.results[rep_label]['weight']
                
                # Check commutators
                if 'commutators' in w:
                    for error in w['commutators'].values():
                        if error > threshold:
                            rep_passed = False
                
                # Check hermiticity
                if 'hermiticity' in w:
                    for error in w['hermiticity'].values():
                        if error > threshold:
                            rep_passed = False
                
                # Check casimir
                if 'casimir' in w:
                    if w['casimir']['std_dev'] > threshold or w['casimir']['error_from_theory'] > threshold:
                        rep_passed = False
                
                # Check diagonality
                if 'diagonality' in w:
                    for error in w['diagonality'].values():
                        if error > threshold:
                            rep_passed = False
            
            # Check GT basis
            if 'gt' in self.results[rep_label]:
                g = self.results[rep_label]['gt']
                
                # Check commutators
                if 'commutators' in g:
                    for error in g['commutators'].values():
                        if error > threshold:
                            rep_passed = False
                
                # Check hermiticity
                if 'hermiticity' in g:
                    for error in g['hermiticity'].values():
                        if error > threshold:
                            rep_passed = False
                
                # Check casimir
                if 'casimir' in g:
                    if g['casimir']['std_dev'] > threshold or g['casimir']['error_from_theory'] > threshold:
                        rep_passed = False
                
                # Check diagonality
                if 'diagonality' in g:
                    for error in g['diagonality'].values():
                        if error > threshold:
                            rep_passed = False
            
            # Check transformation
            if 'transformation' in self.results[rep_label]:
                t = self.results[rep_label]['transformation']
                
                if 'unitarity' in t and t['unitarity'] > threshold:
                    rep_passed = False
                
                if 'transformation_errors' in t:
                    for error in t['transformation_errors'].values():
                        if error > threshold:
                            rep_passed = False
            
            status = "PASS ✓" if rep_passed else "FAIL ✗"
            print(f"{rep_label}: {status}")
            
            if not rep_passed:
                all_passed = False
        
        # Defects
        if self.defects:
            print("\n--- DEFECTS FOUND ---")
            for i, defect in enumerate(self.defects, 1):
                print(f"{i}. {defect}")
        
        # Final verdict
        print("\n" + "="*80)
        if all_passed and not self.defects:
            print("✓ FULL VERIFICATION COMPLETE - ALL TESTS PASSED")
            print("The SU(3) representation framework is mathematically correct.")
        else:
            print("✗ VERIFICATION FAILED - DEFECTS DETECTED")
            print("Please review the errors above and fix the defects.")
        print("="*80)


if __name__ == "__main__":
    validator = IndependentQAValidator()
    validator.validate_all()
