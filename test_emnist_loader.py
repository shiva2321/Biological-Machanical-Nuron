"""
Quick test script to verify EMNIST dataset loader works correctly.
"""

print("="*70)
print("Testing EMNIST Dataset Loader")
print("="*70)

# Test 1: Import test
print("\n[Test 1] Testing imports...")
try:
    import numpy as np
    print("  ✓ NumPy imported")
    import torch
    print("  ✓ PyTorch imported")
    from dataset_loader import load_emnist_dataset
    print("  ✓ dataset_loader imported")
    print("✅ All imports successful!")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test 2: Load small dataset
print("\n[Test 2] Loading small EMNIST dataset...")
print("  Characters: A, B, C, X, Y, Z")
print("  Samples per character: 50")

try:
    dataset = load_emnist_dataset(
        split='train',
        characters=['A', 'B', 'C', 'X', 'Y', 'Z'],
        max_samples_per_class=50
    )

    print(f"\n✅ Dataset loaded successfully!")
    print(f"  Inputs shape: {dataset['inputs'].shape}")
    print(f"  Labels shape: {dataset['labels'].shape}")
    print(f"  Character map: {dataset['char_map']}")

    # Verify data
    print(f"\n[Test 3] Verifying data...")
    print(f"  Input range: [{dataset['inputs'].min():.2f}, {dataset['inputs'].max():.2f}]")
    print(f"  Label range: [{dataset['labels'].min()}, {dataset['labels'].max()}]")
    print(f"  Unique labels: {np.unique(dataset['labels'])}")

    # Sample check
    sample_idx = 0
    sample = dataset['inputs'][sample_idx].reshape(8, 8)
    label = dataset['labels'][sample_idx]
    char = dataset['char_map'][label]

    print(f"\n[Test 4] Sample visualization (index {sample_idx}):")
    print(f"  Character: {char}")
    print(f"  Label: {label}")
    print(f"  8x8 bitmap:")
    for row in sample:
        print("  ", "".join(["█" if x > 0.5 else " " for x in row]))

    print("\n✅ All tests passed!")
    print("\n" + "="*70)
    print("EMNIST Dataset Loader is working correctly!")
    print("You can now train your brain with real handwritten data!")
    print("="*70)

except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

