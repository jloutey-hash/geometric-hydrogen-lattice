"""Quick test runner for Phase 32"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from experiments.phase32_numerov_solver import Phase32_NumerovSolver

print("Starting Phase 32...")
phase32 = Phase32_NumerovSolver()
results = phase32.run_full_analysis(ℓ=0, r_max=50.0, n_points=200)
print("\nPhase 32 complete!")
