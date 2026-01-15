"""
Simple test to create a fresh brain
"""
import sys
import os

print("Testing brain creation...")
print("Python:", sys.version)
print()

try:
    print("[1/5] Importing modules...")
    from circuit import NeuralCircuit
    print("  ✓ circuit imported")

    from brain_io import save_brain, load_brain
    print("  ✓ brain_io imported")

    print("\n[2/5] Creating new brain...")
    # Create default brain
    brain = NeuralCircuit(
        num_neurons=36,
        input_channels=64,
        dt=1.0,
        max_delay=5
    )
    print(f"  ✓ Brain created: {brain.num_neurons} neurons, {brain.input_channels} inputs")

    print("\n[3/5] Saving brain...")
    save_brain(brain, 'my_brain.pkl')
    print("  ✓ Brain saved to my_brain.pkl")

    print("\n[4/5] Loading brain...")
    loaded_brain = load_brain('my_brain.pkl')
    print(f"  ✓ Brain loaded: {loaded_brain.num_neurons} neurons")

    print("\n[5/5] Verification...")
    assert loaded_brain.num_neurons == brain.num_neurons
    assert loaded_brain.input_channels == brain.input_channels
    print("  ✓ Brain verified!")

    print("\n" + "="*50)
    print("✅ SUCCESS! Brain is ready!")
    print("="*50)
    print(f"\nBrain file: my_brain.pkl ({os.path.getsize('my_brain.pkl')} bytes)")
    print("\nYou can now run:")
    print("  launch_dashboard.bat")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

