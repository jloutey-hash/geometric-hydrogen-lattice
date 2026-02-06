"""
Installation and Environment Verification Script

Run this script to verify that all dependencies and modules are properly installed
and ready for the geometric transformation research.

This checks:
1. Python version compatibility
2. Required packages (numpy, scipy, matplotlib)
3. Existing project modules (lattice, angular_momentum)
4. New geometric research module
5. File structure
6. Quick functionality test

Usage:
    python verify_geometric_research_setup.py

Author: Quantum Lattice Project
Date: January 2026
"""

import sys
import os
from pathlib import Path

print("=" * 70)
print("GEOMETRIC TRANSFORMATION RESEARCH - ENVIRONMENT VERIFICATION")
print("=" * 70)
print()

# Check 1: Python version
print("1. Checking Python version...")
version = sys.version_info
print(f"   Python {version.major}.{version.minor}.{version.micro}")
if version.major >= 3 and version.minor >= 7:
    print("   ✅ Compatible (Python 3.7+)")
else:
    print("   ❌ Incompatible (requires Python 3.7+)")
    print("   Please upgrade Python")
    sys.exit(1)
print()

# Check 2: Required packages
print("2. Checking required packages...")
required_packages = {
    'numpy': 'numpy',
    'scipy': 'scipy',
    'matplotlib': 'matplotlib'
}

all_packages_ok = True
for package_name, import_name in required_packages.items():
    try:
        __import__(import_name)
        print(f"   ✅ {package_name}")
    except ImportError:
        print(f"   ❌ {package_name} - NOT FOUND")
        all_packages_ok = False

if not all_packages_ok:
    print()
    print("   Missing packages detected. Install with:")
    print("   pip install numpy scipy matplotlib")
    print()
print()

# Check 3: Project structure
print("3. Checking project structure...")
required_dirs = ['src', 'results']
required_files = [
    'src/lattice.py',
    'src/angular_momentum.py',
    'src/geometric_transform_research.py',
    'run_phase1_geometric_diagnostic.py',
    'run_phase2_geometric_transform_test.py',
    'run_phase3_geometric_validation.py',
    'run_phase4_geometric_optimization.py',
    'run_geometric_research_complete.py',
    'GEOMETRIC_RESEARCH_README.md',
    'GEOMETRIC_RESEARCH_QUICKSTART.md'
]

structure_ok = True

# Check directories
for dir_name in required_dirs:
    if os.path.isdir(dir_name):
        print(f"   ✅ {dir_name}/")
    else:
        print(f"   ⚠️  {dir_name}/ - creating...")
        try:
            os.makedirs(dir_name, exist_ok=True)
            print(f"      Created {dir_name}/")
        except Exception as e:
            print(f"      ❌ Failed to create: {e}")
            structure_ok = False

# Check files
for file_path in required_files:
    if os.path.isfile(file_path):
        print(f"   ✅ {file_path}")
    else:
        print(f"   ❌ {file_path} - MISSING")
        structure_ok = False

if not structure_ok:
    print()
    print("   ⚠️  Some files are missing. Please ensure all files are present.")
    print()
print()

# Check 4: Module imports
print("4. Checking module imports...")
sys.path.insert(0, 'src')

module_imports_ok = True
modules_to_test = [
    ('lattice', 'PolarLattice'),
    ('angular_momentum', 'AngularMomentumOperators'),
    ('geometric_transform_research', 'GeometricTransformResearch'),
]

for module_name, class_name in modules_to_test:
    try:
        module = __import__(module_name)
        if hasattr(module, class_name):
            print(f"   ✅ {module_name}.{class_name}")
        else:
            print(f"   ⚠️  {module_name}.{class_name} - class not found")
            module_imports_ok = False
    except ImportError as e:
        print(f"   ❌ {module_name} - IMPORT FAILED: {e}")
        module_imports_ok = False
    except Exception as e:
        print(f"   ❌ {module_name} - ERROR: {e}")
        module_imports_ok = False

if not module_imports_ok:
    print()
    print("   ⚠️  Some module imports failed. Check error messages above.")
    print()
print()

# Check 5: Quick functionality test
print("5. Running quick functionality test...")
try:
    from lattice import PolarLattice
    from angular_momentum import AngularMomentumOperators
    from geometric_transform_research import GeometricTransformResearch
    from scipy import sparse
    import numpy as np
    
    # Create small lattice
    print("   Creating lattice (n_max=3)...")
    lattice = PolarLattice(n_max=3)
    print(f"   ✅ Lattice created: {len(lattice.points)} points")
    
    # Create operators
    print("   Building angular momentum operators...")
    angular_ops = AngularMomentumOperators(lattice)
    L_squared = angular_ops.build_L_squared()
    print(f"   ✅ L² operator: {L_squared.shape} sparse matrix")
    
    # Create research object
    print("   Initializing research module...")
    research = GeometricTransformResearch(lattice, angular_ops)
    print(f"   ✅ Research module initialized")
    
    # Compute a test eigenvector
    print("   Computing test eigenvector (ℓ=1)...")
    eigenvalues, eigenvectors = sparse.linalg.eigsh(L_squared, k=5, which='SM')
    idx = np.argmin(np.abs(eigenvalues - 2.0))  # ℓ=1: L²=2
    psi = eigenvectors[:, idx]
    print(f"   ✅ Eigenvector computed")
    
    # Test transformation
    print("   Testing stereographic transformation...")
    J = research.compute_jacobian('stereographic')
    psi_corrected = research.apply_geometric_correction(psi, 'stereographic', 'forward')
    print(f"   ✅ Transformation applied")
    
    # Test overlap computation
    print("   Computing overlap with Y_1^0...")
    overlap = research.compute_overlap_with_Ylm(psi, 1, 0)
    print(f"   ✅ Overlap computed: {overlap:.4%}")
    
    # Test eigenvalue preservation
    print("   Verifying eigenvalue preservation...")
    error, preserved = research.verify_eigenvalue_preservation(psi_corrected, 1)
    print(f"   ✅ Eigenvalue error: {error:.2e} (preserved: {preserved})")
    
    print()
    print("   🎉 All functionality tests PASSED!")
    print()

except Exception as e:
    print()
    print(f"   ❌ Functionality test FAILED:")
    print(f"      {type(e).__name__}: {e}")
    print()
    import traceback
    print("   Full traceback:")
    print("   " + "\n   ".join(traceback.format_exc().split('\n')))
    print()
    sys.exit(1)

# Final summary
print("=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)
print()

if all_packages_ok and structure_ok and module_imports_ok:
    print("✅ ENVIRONMENT READY!")
    print()
    print("Your environment is properly configured for geometric transformation research.")
    print()
    print("Next steps:")
    print("  1. Read GEOMETRIC_RESEARCH_QUICKSTART.md for quick start")
    print("  2. Run: python run_geometric_research_complete.py")
    print("  3. Review results in results/ directory")
    print()
    print("Or run individual phases:")
    print("  - python run_phase1_geometric_diagnostic.py")
    print("  - python run_phase2_geometric_transform_test.py")
    print("  - python run_phase3_geometric_validation.py")
    print("  - python run_phase4_geometric_optimization.py")
    print()
else:
    print("⚠️  ISSUES DETECTED")
    print()
    print("Some checks failed. Please review the messages above and:")
    print("  1. Install missing packages: pip install numpy scipy matplotlib")
    print("  2. Ensure all required files are present")
    print("  3. Check that src/ directory contains all modules")
    print()
    print("Re-run this script after fixing issues:")
    print("  python verify_geometric_research_setup.py")
    print()

print("=" * 70)
