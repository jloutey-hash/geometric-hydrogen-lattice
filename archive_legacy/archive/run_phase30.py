"""Quick test runner for Phase 30"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from experiments.phase30_su3_flavor_weights import Phase30_SU3FlavorWeights

print("Starting Phase 30...")
phase30 = Phase30_SU3FlavorWeights()
phase30.run_full_analysis()
print("\nPhase 30 complete!")
