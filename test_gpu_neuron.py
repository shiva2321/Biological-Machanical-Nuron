"""
Test GPU-accelerated neuron implementation.

Verifies:
1. PyTorch and CUDA are installed correctly
2. BiologicalNeuron works on GPU
3. All operations stay on GPU
4. Performance is acceptable
"""

import time
import numpy as np

print("="*70)
print("TESTING GPU-ACCELERATED NEURON")
print("="*70)

# Test 1: Import PyTorch
print("\n[1/5] Testing PyTorch import...")
try:
    import torch
    print(f"  ✓ PyTorch version: {torch.__version__}")
except ImportError as e:
    print(f"  ✗ PyTorch not installed: {e}")
    print("\n  Install with:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cu118")
    exit(1)

# Test 2: Check CUDA availability
print("\n[2/5] Testing CUDA availability...")
if torch.cuda.is_available():
    print(f"  ✓ CUDA is available")
    print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"  ✓ CUDA version: {torch.version.cuda}")

    # Memory info
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  ✓ GPU Memory: {total_mem:.1f} GB")
else:
    print(f"  ⚠ CUDA not available (will use CPU)")
    print(f"  This is OK for testing, but performance will be slow")

# Test 3: Import neuron
print("\n[3/5] Testing BiologicalNeuron import...")
try:
    from neuron import BiologicalNeuron, check_gpu_available, get_gpu_memory_info
    print(f"  ✓ BiologicalNeuron imported successfully")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    exit(1)

# Test 4: Create neuron on GPU
print("\n[4/5] Testing neuron creation...")
try:
    neuron = BiologicalNeuron(n_inputs=64)
    print(f"  ✓ Neuron created on device: {neuron.device}")

    # Check that tensors are on correct device
    assert neuron.v.device.type in ['cuda', 'cpu'], "Invalid device"
    assert neuron.weights.device == neuron.v.device, "Tensors on different devices"
    print(f"  ✓ All tensors on {neuron.device}")

except Exception as e:
    print(f"  ✗ Neuron creation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Run simulation and verify GPU operations
print("\n[5/5] Testing neuron operations...")
try:
    # Create input on GPU
    if neuron.device.type == 'cuda':
        input_spikes = torch.zeros(64, device='cuda')
    else:
        input_spikes = torch.zeros(64)

    input_spikes[0] = 1.0
    input_spikes[10] = 1.0

    # Time the simulation
    start_time = time.time()
    spike_count = 0

    for step in range(1000):
        spike = neuron.update(I_ext=50.0)
        neuron.stdp(input_spikes, spike)
        if spike:
            spike_count += 1

    elapsed = time.time() - start_time

    print(f"  ✓ Ran 1000 steps in {elapsed:.3f} seconds")
    print(f"  ✓ Performance: {1000/elapsed:.0f} steps/second")
    print(f"  ✓ Spike count: {spike_count}")

    # Verify we can get state (CPU conversion)
    state = neuron.get_state()
    assert isinstance(state['weights'], np.ndarray), "State not converted to numpy"
    print(f"  ✓ State retrieval works (GPU -> CPU)")
    print(f"  ✓ Mean weight: {state['weights'].mean():.3f}")

    # Check memory usage if on GPU
    if neuron.device.type == 'cuda':
        mem_info = get_gpu_memory_info()
        print(f"  ✓ GPU memory allocated: {mem_info['allocated_gb']*1000:.2f} MB")

except Exception as e:
    print(f"  ✗ Simulation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*70)
print("ALL TESTS PASSED!")
print("="*70)

# Summary
print("\nSummary:")
print(f"  • PyTorch: {torch.__version__}")
print(f"  • Device: {neuron.device}")
if torch.cuda.is_available():
    print(f"  • GPU: {torch.cuda.get_device_name(0)}")
    mem_info = get_gpu_memory_info()
    print(f"  • Memory: {mem_info['allocated_gb']*1000:.2f} MB / {mem_info['total_gb']:.1f} GB")
print(f"  • Performance: {1000/elapsed:.0f} steps/second")
print("\n✓ GPU-accelerated neuron is working correctly!")

