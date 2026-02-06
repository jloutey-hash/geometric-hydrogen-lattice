"""Quick test runner for Phase 29"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from experiments.phase29_su2_u1_mixing import Phase29_MixedGaugeLinks

print("Starting Phase 29...")
phase29 = Phase29_MixedGaugeLinks(n_max=6)
results = phase29.run_full_analysis(n_w=21, u1_amplitude=0.3, su2_amplitude=0.1)
print("\nPhase 29 complete!")
