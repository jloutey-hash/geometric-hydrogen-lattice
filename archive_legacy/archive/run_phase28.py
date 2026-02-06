"""Quick test runner for Phase 28"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from experiments.phase28_u1_wilson_loops import Phase28_U1WilsonLoops

# Run Phase 28
print("Starting Phase 28...")
phase28 = Phase28_U1WilsonLoops(n_max=6)
stats, coupling = phase28.run_full_analysis(initialization='random', amplitude=0.3)
print("\nPhase 28 complete!")
