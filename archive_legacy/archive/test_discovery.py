from physics_discovery import PhysicsDiscovery

print("Testing Physics Discovery - Polar Quaternion Model")
print("=" * 60)

# Initialize
d = PhysicsDiscovery(max_n=30)

# Test Task 1
print("\nTask 1: Alpha Hunt")
results = d.hunt_alpha([5, 10, 20, 30])
print(f"\nFinal ratio: {results[-1]['ratio_S_V']:.6f}")

# Test Task 2
print("\nTask 2: Lamb Shift Hunt")
lamb = d.hunt_lamb_shift(max_n=10)
if 'delta_reach' in lamb:
    print(f"\nDelta reach (2p-2s): {lamb['delta_reach']:.6f}")

# Test Task 3
print("\nTask 3: Quaternion Setup")
qdata = d.prepare_spinor_lattice(max_n=5)
print(f"\nMean alignment: {qdata['mean_alignment']:.6f}")

# Generate report
d.generate_report("geometric_constants.txt")

print("\n" + "=" * 60)
print("Test complete!")
