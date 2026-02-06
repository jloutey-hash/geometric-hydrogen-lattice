"""
Comprehensive Test Runner for Discrete Polar Lattice Model
Runs all validation tests and generates a coverage report
"""

import subprocess
import sys
import time
from pathlib import Path

# Define all tests in order
TESTS = [
    # Core phases (1-7)
    ("Phase 1: Core Lattice", "tests/validate_phase1.py"),
    ("Phase 2: Operators & Hamiltonians", "tests/validate_phase2.py"),
    ("Phase 3: Angular Momentum", "tests/validate_phase3.py"),
    ("Phase 4: Eigenvalue Validation", "tests/validate_phase4.py"),
    ("Phase 5: Spherical Harmonics", "tests/validate_phase5.py"),
    ("Phase 6: Multi-particle Physics", "tests/validate_phase6.py"),
    ("Phase 7: Spin Physics", "tests/validate_phase7.py"),
    
    # Discovery phases (8-9)
    ("Phase 8: Geometric Constant Discovery", "tests/validate_phase8.py"),
    ("Phase 8: Full Convergence", "tests/validate_phase8_full_convergence.py"),
    ("Phase 9: Hydrogen Test", "tests/validate_phase9_hydrogen.py"),
]

def run_test(name, script):
    """Run a single test script and capture results"""
    print(f"\n{'='*70}")
    print(f"Running: {name}")
    print(f"Script: {script}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    try:
        # Set UTF-8 encoding for Windows
        env = {'PYTHONIOENCODING': 'utf-8'}
        result = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout per test
            env={**subprocess.os.environ, **env}
        )
        
        elapsed = time.time() - start_time
        
        # Check for success indicators
        success = (
            result.returncode == 0 and
            ('PASSED' in result.stdout or 'validation complete' in result.stdout)
        )
        
        return {
            'name': name,
            'script': script,
            'success': success,
            'elapsed': elapsed,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.TimeoutExpired:
        return {
            'name': name,
            'script': script,
            'success': False,
            'elapsed': time.time() - start_time,
            'returncode': -1,
            'stdout': '',
            'stderr': 'TIMEOUT: Test exceeded 5 minutes'
        }
    except Exception as e:
        return {
            'name': name,
            'script': script,
            'success': False,
            'elapsed': time.time() - start_time,
            'returncode': -1,
            'stdout': '',
            'stderr': f'ERROR: {str(e)}'
        }

def print_summary(results):
    """Print summary of all test results"""
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results if r['success'])
    failed = len(results) - passed
    total_time = sum(r['elapsed'] for r in results)
    
    print(f"\nTotal Tests: {len(results)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total Time: {total_time:.1f}s")
    
    print(f"\n{'Test':<40} {'Status':<10} {'Time':<10}")
    print("-"*70)
    
    for r in results:
        status = "✅ PASS" if r['success'] else "❌ FAIL"
        test_name = r['name'][:38]
        print(f"{test_name:<40} {status:<10} {r['elapsed']:>6.1f}s")
    
    if failed > 0:
        print("\n" + "="*70)
        print("FAILED TESTS - DETAILS")
        print("="*70)
        
        for r in results:
            if not r['success']:
                print(f"\n❌ {r['name']}")
                print(f"   Script: {r['script']}")
                print(f"   Return code: {r['returncode']}")
                if r['stderr']:
                    print(f"   Error:\n{r['stderr'][:500]}")

def main():
    """Run all tests and generate report"""
    print("╔" + "="*68 + "╗")
    print("║" + " "*18 + "COMPREHENSIVE TEST SUITE" + " "*26 + "║")
    print("║" + " "*14 + "Discrete Polar Lattice Model" + " "*25 + "║")
    print("╚" + "="*68 + "╝")
    
    # Check if we're in the right directory
    if not Path("tests").exists():
        print("\n❌ ERROR: 'tests' directory not found!")
        print("Please run this script from the project root directory.")
        return 1
    
    # Run all tests
    results = []
    for name, script in TESTS:
        if not Path(script).exists():
            print(f"\n⚠️  WARNING: Test script not found: {script}")
            results.append({
                'name': name,
                'script': script,
                'success': False,
                'elapsed': 0,
                'returncode': -1,
                'stdout': '',
                'stderr': 'Test script not found'
            })
            continue
        
        result = run_test(name, script)
        results.append(result)
        
        # Show immediate status
        status = "✅" if result['success'] else "❌"
        print(f"\n{status} {name}: {'PASSED' if result['success'] else 'FAILED'} ({result['elapsed']:.1f}s)")
    
    # Print summary
    print_summary(results)
    
    # Return exit code
    failed = sum(1 for r in results if not r['success'])
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
