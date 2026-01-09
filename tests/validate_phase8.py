"""
Phase 8 Validation: Fine Structure Constant from Geometry

This script explores 10 different geometric approaches to identify
connections between the discrete polar lattice and the fine structure
constant alpha (approximately 1/137.036).
"""

import sys
import os

# Set UTF-8 encoding for output to handle special characters
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from src.lattice import PolarLattice
from src.operators import LatticeOperators
from src.angular_momentum import AngularMomentumOperators
from src.spin import SpinOperators
from src.fine_structure import FineStructureExplorer, ALPHA_FINE


def main():
    """Run Phase 8 validation."""
    print("="*70)
    print("PHASE 8: FINE STRUCTURE CONSTANT FROM GEOMETRY")
    print("="*70)
    print(f"\nTarget: alpha = 1/137.035999084 = {ALPHA_FINE:.10f}")
    print("\nThis exploration investigates 10 different geometric approaches")
    print("to determine if alpha emerges naturally from the lattice structure.\n")
    
    # Build lattice with n_max = 6 (good balance of size and speed)
    print("Building lattice with n_max = 6...")
    lattice = PolarLattice(n_max=6)
    print(f"Total points: {len(lattice.points)}")
    
    # Build operators
    print("\nBuilding operators...")
    operators = LatticeOperators(lattice)
    angular_momentum = AngularMomentumOperators(lattice)
    spin_ops = SpinOperators(lattice, operators)
    
    # Create explorer
    print("Initializing Fine Structure Explorer...")
    explorer = FineStructureExplorer(
        lattice=lattice,
        operators=operators,
        angular_momentum=angular_momentum,
        spin_ops=spin_ops
    )
    
    # Run all explorations
    print("\n" + "="*70)
    print("BEGINNING EXPLORATION")
    print("="*70)
    
    results = explorer.explore_all(verbose=True)
    
    # Save detailed results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save numerical results
    results_file = os.path.join(results_dir, 'phase8_fine_structure_results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("PHASE 8: FINE STRUCTURE CONSTANT EXPLORATION\n")
        f.write("="*70 + "\n\n")
        f.write(f"Target: alpha = {ALPHA_FINE:.10f}\n")
        f.write(f"Lattice: n_max = {lattice.n_max}, N = {len(lattice.points)} points\n\n")
        
        # Write synthesis
        synthesis = results['Synthesis']
        stats = synthesis['statistics']
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Total candidates evaluated: {stats['total_candidates']}\n")
        f.write(f"Within 1% of alpha: {stats['within_1_percent']}\n")
        f.write(f"Within 5% of alpha: {stats['within_5_percent']}\n")
        f.write(f"Within 10% of alpha: {stats['within_10_percent']}\n")
        f.write(f"Within 50% of alpha: {stats['within_50_percent']}\n\n")
        
        # Write top candidates
        f.write("TOP 30 CANDIDATES\n")
        f.write("="*70 + "\n\n")
        
        for i, candidate in enumerate(synthesis['best_candidates'][:30], 1):
            f.write(f"{i}. {candidate['description']}\n")
            f.write(f"   Track: {candidate['track']}\n")
            f.write(f"   Value: {candidate['value']:.10f}\n")
            f.write(f"   alpha = {ALPHA_FINE:.10f}\n")
            f.write(f"   Absolute deviation: {candidate['deviation_from_alpha']:.10f}\n")
            f.write(f"   Relative error: {candidate['relative_error']*100:.6f}%\n\n")
        
        # Write track-by-track summary
        f.write("\n" + "="*70 + "\n")
        f.write("TRACK-BY-TRACK SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        for track_name, track_result in results.items():
            if track_name == 'Synthesis':
                continue
            
            f.write(f"\n{track_name}\n")
            f.write("-"*70 + "\n")
            
            if 'status' in track_result:
                f.write(f"Status: {track_result['status']}\n")
                continue
            
            if 'candidates' not in track_result or not track_result['candidates']:
                f.write("No candidates found\n")
                continue
            
            # Best from this track
            best = min(track_result['candidates'], key=lambda x: x['relative_error'])
            f.write(f"Best candidate: {best['description']}\n")
            f.write(f"  Value: {best['value']:.10f}\n")
            f.write(f"  Relative error: {best['relative_error']*100:.6f}%\n")
            f.write(f"  Total candidates: {len(track_result['candidates'])}\n")
            
            # Count by accuracy
            within_10 = sum(1 for c in track_result['candidates'] if c['relative_error'] < 0.10)
            within_50 = sum(1 for c in track_result['candidates'] if c['relative_error'] < 0.50)
            f.write(f"  Within 10%: {within_10}\n")
            f.write(f"  Within 50%: {within_50}\n")
    
    print(f"Detailed results saved to: {results_file}")
    
    # Create visualization
    print("\nGenerating visualization...")
    fig_path = os.path.join(results_dir, 'phase8_fine_structure_analysis.png')
    explorer.plot_results(save_path=fig_path)
    
    print("\n" + "="*70)
    print("PHASE 8 VALIDATION COMPLETE")
    print("="*70)
    
    # Print key findings
    synthesis = results['Synthesis']
    stats = synthesis['statistics']
    
    print(f"\n✓ Evaluated {stats['total_candidates']} candidate expressions")
    print(f"✓ Found {stats['within_10_percent']} candidates within 10% of alpha")
    print(f"✓ Found {stats['within_50_percent']} candidates within 50% of alpha")
    
    if synthesis['best_candidates']:
        best = synthesis['best_candidates'][0]
        print(f"\nBest candidate:")
        print(f"   {best['description']}")
        print(f"   Value: {best['value']:.10f}")
        print(f"   Error: {best['relative_error']*100:.6f}%")
    
    print(f"\nFull results: {results_file}")
    print(f"Visualization: {fig_path}")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR FURTHER INVESTIGATION")
    print("="*70)
    
    # Find tracks with best results
    track_best = {}
    for track_name, track_result in results.items():
        if track_name == 'Synthesis' or 'candidates' not in track_result:
            continue
        if track_result['candidates']:
            best_error = min(c['relative_error'] for c in track_result['candidates'])
            track_best[track_name] = best_error
    
    # Sort by performance
    sorted_tracks = sorted(track_best.items(), key=lambda x: x[1])
    
    print("\nMost promising tracks (by best candidate):")
    for i, (track, error) in enumerate(sorted_tracks[:5], 1):
        print(f"{i}. {track}: {error*100:.2f}% error")
    
    print("\nNext steps:")
    print("1. Deeper analysis of top 3 tracks")
    print("2. Theoretical derivation for promising geometric ratios")
    print("3. Test with larger n_max to check convergence")
    print("4. Explore combinations of geometric factors")
    print("5. Connect to known physics (QED, Dirac equation, etc.)")


if __name__ == '__main__':
    main()
