"""
Run SU(3) impedance scan and export data without interactive plotting
"""

from su3_impedance import scan_representations, analyze_scaling, export_impedance_data, plot_impedance_scaling

print("="*80)
print("SU(3) SYMPLECTIC IMPEDANCE CALCULATION")
print("="*80)

# Scan representations
print("\nScanning representations...")
results = scan_representations(max_sum=4, verbose=False)

print(f"\nCalculated impedance for {len(results)} representations")

# Analyze scaling
print("\nAnalyzing scaling laws...")
analysis = analyze_scaling(results)

# Export data
print("\nExporting data to CSV...")
export_impedance_data(results, 'su3_impedance_data.csv')

# Create plots (non-interactive)
print("\nGenerating plots...")
plot_impedance_scaling(results, save_path='su3_impedance_scaling.png')

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print(f"Generated files:")
print(f"  - su3_impedance_data.csv")
print(f"  - su3_impedance_scaling.png")
