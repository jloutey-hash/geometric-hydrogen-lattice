"""
Deep Investigation: Combined Geometric Factors for Fine Structure

Since individual ratios showed >400% error, this explores:
- Products of quantities
- Complex ratios
- Power series
- Weighted combinations
- Transcendental functions
"""

import sys
import os

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.lattice import PolarLattice
from src.operators import LatticeOperators
from src.angular_momentum import AngularMomentumOperators
from src.spin import SpinOperators
from src.fine_structure_deep import DeepFineStructureExplorer, ALPHA_FINE


def main():
    print("="*70)
    print("DEEP INVESTIGATION: COMBINED GEOMETRIC FACTORS")
    print("="*70)
    print(f"\nTarget: alpha = {ALPHA_FINE:.10f} = 1/137.036")
    print("\nApproach: Since simple ratios failed (>400% error),")
    print("we now test combinations, products, and series.\n")
    
    # Use n_max=6 for initial exploration
    print("Building lattice (n_max=6)...")
    lattice = PolarLattice(n_max=6)
    
    print("Building operators...")
    operators = LatticeOperators(lattice)
    angular_momentum = AngularMomentumOperators(lattice)
    spin_ops = SpinOperators(lattice, operators)
    
    print("Initializing deep explorer...")
    explorer = DeepFineStructureExplorer(
        lattice=lattice,
        operators=operators,
        angular_momentum=angular_momentum,
        spin_ops=spin_ops
    )
    
    print("\n" + "="*70)
    print("STARTING DEEP EXPLORATION")
    print("="*70)
    
    # Run all deep explorations
    results = explorer.explore_all_deep(verbose=True)
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Text report
    results_file = os.path.join(results_dir, 'phase8_deep_investigation.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("DEEP INVESTIGATION: COMBINED GEOMETRIC FACTORS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Target: alpha = {ALPHA_FINE:.10f}\n")
        f.write(f"Lattice: n_max={lattice.n_max}, N={len(lattice.points)} points\n\n")
        
        synthesis = results['Synthesis']
        stats = synthesis['statistics']
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Total candidates: {stats['total_candidates']}\n")
        f.write(f"Within 1%:  {stats['within_1_percent']}\n")
        f.write(f"Within 5%:  {stats['within_5_percent']}\n")
        f.write(f"Within 10%: {stats['within_10_percent']}\n")
        f.write(f"Within 20%: {stats['within_20_percent']}\n\n")
        
        f.write("TOP 50 CANDIDATES\n")
        f.write("="*70 + "\n\n")
        
        for i, cand in enumerate(synthesis['best_candidates'][:50], 1):
            f.write(f"{i}. {cand['description']}\n")
            f.write(f"   Value: {cand['value']:.10f}\n")
            f.write(f"   alpha: {ALPHA_FINE:.10f}\n")
            f.write(f"   Error: {cand['relative_error']*100:.6f}%\n\n")
        
        # Category breakdown
        f.write("\n" + "="*70 + "\n")
        f.write("CATEGORY BREAKDOWN\n")
        f.write("="*70 + "\n\n")
        
        for cat_name, cat_result in results.items():
            if cat_name == 'Synthesis':
                continue
            
            f.write(f"\n{cat_name}\n")
            f.write("-"*70 + "\n")
            
            if 'candidates' in cat_result and cat_result['candidates']:
                best = min(cat_result['candidates'], key=lambda x: x['relative_error'])
                f.write(f"Best: {best['description']}\n")
                f.write(f"Value: {best['value']:.10f}\n")
                f.write(f"Error: {best['relative_error']*100:.6f}%\n")
                f.write(f"Count: {len(cat_result['candidates'])}\n")
                
                within_20 = sum(1 for c in cat_result['candidates'] if c['relative_error'] < 0.20)
                f.write(f"Within 20%: {within_20}\n")
    
    print(f"Text report: {results_file}")
    
    # Visualization
    print("\nGenerating visualization...")
    fig_path = os.path.join(results_dir, 'phase8_deep_investigation.png')
    explorer.plot_deep_results(save_path=fig_path)
    
    print("\n" + "="*70)
    print("DEEP INVESTIGATION COMPLETE")
    print("="*70)
    
    synthesis = results['Synthesis']
    stats = synthesis['statistics']
    
    print(f"\nEvaluated {stats['total_candidates']} combined expressions")
    print(f"Within 1%:  {stats['within_1_percent']}")
    print(f"Within 5%:  {stats['within_5_percent']}")
    print(f"Within 10%: {stats['within_10_percent']}")
    print(f"Within 20%: {stats['within_20_percent']}")
    
    if synthesis['best_candidates']:
        best = synthesis['best_candidates'][0]
        print(f"\nBEST CANDIDATE:")
        print(f"  {best['description']}")
        print(f"  Value: {best['value']:.10f}")
        print(f"  Error: {best['relative_error']*100:.4f}%")
        
        if best['relative_error'] < 0.20:
            print("\n  *** SIGNIFICANT RESULT: < 20% error! ***")
        elif best['relative_error'] < 0.50:
            print("\n  ** Promising: < 50% error **")
    
    print(f"\nFull report: {results_file}")
    print(f"Visualization: {fig_path}")
    
    # Recommendations
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    
    if stats['within_10_percent'] > 0:
        print("\n1. EXCELLENT! Found candidates within 10% of alpha")
        print("   --> Investigate theoretical basis for these combinations")
        print("   --> Test with larger n_max for convergence")
        print("   --> Publish findings")
    elif stats['within_20_percent'] > 0:
        print("\n1. PROMISING! Found candidates within 20% of alpha")
        print("   --> Refine these combinations")
        print("   --> Add correction terms")
        print("   --> Test convergence with n_max")
    else:
        best_error = synthesis['best_candidates'][0]['relative_error']
        print(f"\n1. Best result: {best_error*100:.1f}% error")
        print("   --> Try higher-order combinations (4+ factors)")
        print("   --> Explore continued fractions")
        print("   --> Test with larger lattices (n_max=10, 20)")
        print("   --> Consider non-algebraic functions")


if __name__ == '__main__':
    main()
