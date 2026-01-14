# NeuralCircuit - Quick Start Guide

## Installation

No installation needed - just ensure you have:
```bash
pip install numpy matplotlib
```

## 5-Minute Tutorial

### Step 1: Create a Circuit
```python
from circuit import NeuralCircuit
import numpy as np

# Create a network with 5 neurons and 3 input channels
circuit = NeuralCircuit(
    num_neurons=5,
    input_channels=3,
    dt=1.0  # 1ms time step
)

print(circuit)  # NeuralCircuit(neurons=5, inputs=3, ...)
```

### Step 2: Add Connections
```python
# Connect neurons in a chain: 0 → 1 → 2 → 3 → 4
circuit.connect_chain(weight=1.0, delay=1)

# Or add individual connections
circuit.connect(
    source_id=0,
    target_id=4,
    weight=0.5,
    delay=3  # 3ms delay
)

print(f"Connections: {circuit.get_num_connections()}")
```

### Step 3: Run Simulation
```python
# Simulate for 10 time steps
for t in range(10):
    # External inputs (3 channels)
    input_spikes = np.array([1.0, 0.5, 0.0])
    
    # Optional external currents per neuron
    I_ext = np.ones(5) * 10.0
    
    # Step the circuit
    output_spikes = circuit.step(
        input_spikes,
        I_ext=I_ext,
        learning=True
    )
    
    # Check which neurons fired
    if np.any(output_spikes):
        fired = np.where(output_spikes)[0]
        print(f"t={t}ms: Neuron(s) {fired} fired")
```

### Step 4: Add Competition
```python
# Enable lateral inhibition for winner-take-all
circuit.set_inhibition(strength=3.0)  # mV

# Now only the most excited neuron will fire
for t in range(20):
    inputs = np.random.rand(3)
    I_ext = np.ones(5) * 15.0
    outputs = circuit.step(inputs, I_ext=I_ext)
```

## Common Patterns

### Pattern 1: Feed-Forward Chain
```python
circuit = NeuralCircuit(num_neurons=5, input_channels=2)
circuit.connect_chain(weight=10.0, delay=1)

# Stimulate first neuron and watch propagation
for t in range(15):
    inputs = np.array([1.0, 0.0]) if t == 0 else np.array([0.0, 0.0])
    I_ext = np.array([20.0, 5.0, 5.0, 5.0, 5.0])
    outputs = circuit.step(inputs, I_ext=I_ext, learning=False)
    if np.any(outputs):
        print(f"t={t}: {np.where(outputs)[0]}")
```

### Pattern 2: Competitive Network
```python
circuit = NeuralCircuit(num_neurons=10, input_channels=10)
circuit.set_inhibition(strength=4.0)

# Set identity weights (each neuron responds to one input)
for i in range(10):
    weights = np.zeros(10)
    weights[i] = 1.0
    circuit.set_weights(i, weights)

# Present varied inputs
inputs = np.array([0.3, 0.5, 0.9, 0.4, 0.6, 0.2, 0.7, 0.4, 0.5, 0.3])
I_ext = np.ones(10) * 15.0

spike_counts = np.zeros(10)
for t in range(30):
    outputs = circuit.step(inputs, I_ext=I_ext)
    spike_counts += outputs.astype(int)

print(f"Winner: Neuron {np.argmax(spike_counts)}")
```

### Pattern 3: Recurrent Network
```python
circuit = NeuralCircuit(num_neurons=20, input_channels=5, max_delay=5)

# Add random recurrent connections
for _ in range(50):
    i, j = np.random.randint(0, 20, size=2)
    if i != j:
        weight = np.random.randn() * 0.1
        delay = np.random.randint(1, 5)
        circuit.connect(i, j, weight, delay)

# Simulate
for t in range(100):
    inputs = np.random.rand(5) * 0.5
    outputs = circuit.step(inputs)
```

## Key Methods

### Connection Methods
```python
circuit.connect(source_id, target_id, weight, delay=1)
circuit.connect_chain(weight, delay=1, bidirectional=False)
circuit.connect_all_to_all(weight, delay=1, include_self=False)
circuit.connect_lateral_inhibition(weight=-1.0, delay=1)
```

### Simulation Methods
```python
circuit.step(input_spikes, I_ext=None, learning=True)
circuit.reset_state()  # Clear dynamics, keep weights
circuit.set_inhibition(strength)
```

### Inspection Methods
```python
circuit.get_states()              # All neuron states
circuit.get_weights(neuron_id)    # Input weights
circuit.get_connection_matrix()   # NxN connection matrix
circuit.get_num_connections()     # Total connections
circuit.summary()                 # Detailed summary
```

## Tips & Tricks

### Tip 1: Neuron Parameters
Pass custom parameters to all neurons:
```python
circuit = NeuralCircuit(
    num_neurons=10,
    input_channels=5,
    neuron_params={
        'tau_m': 15.0,
        'theta_base': -55.0,
        'a_plus': 0.05,
        'a_minus': 0.03
    }
)
```

### Tip 2: Debugging
Check network structure:
```python
print(circuit.summary())
# Shows neurons, connections, delays, inhibition

conn_matrix = circuit.get_connection_matrix()
print(conn_matrix)
# Shows weight from neuron i to neuron j
```

### Tip 3: Reset Between Trials
```python
# Reset dynamics but keep learned weights
circuit.reset_state()

# Or create new circuit for fresh start
circuit = NeuralCircuit(...)
```

### Tip 4: External Currents
Use `I_ext` to control excitability:
```python
# Boost specific neurons
I_ext = np.zeros(num_neurons)
I_ext[target_neurons] = 20.0  # mV

outputs = circuit.step(inputs, I_ext=I_ext)
```

## Examples

### Run Tests
```bash
python test_circuit.py
# Runs comprehensive tests of all features
```

### Run Visual Demos
```bash
python demo_circuit.py
# Creates plots showing:
# 1. Spike propagation with delays
# 2. Winner-take-all competition
```

## Troubleshooting

### Problem: No Spikes
**Solution**: Increase excitability
```python
# Lower threshold
neuron_params = {'theta_base': -55.0}  # or -60.0

# Increase baseline current
I_ext = np.ones(num_neurons) * 20.0

# Increase connection weights
circuit.connect(0, 1, weight=15.0, delay=1)
```

### Problem: Too Many Spikes
**Solution**: Reduce excitability
```python
# Raise threshold
neuron_params = {'theta_base': -50.0}

# Add lateral inhibition
circuit.set_inhibition(strength=5.0)

# Reduce connection weights
circuit.connect(0, 1, weight=0.5, delay=1)
```

### Problem: Spikes Don't Propagate
**Solution**: Check delays and weights
```python
# Verify connections exist
print(f"Connections: {circuit.get_num_connections()}")
print(circuit.get_connection_matrix())

# Increase connection strength
circuit.connect(0, 1, weight=10.0, delay=1)  # Higher weight

# Check max_delay setting
circuit = NeuralCircuit(num_neurons=5, input_channels=3, max_delay=10)
```

## Next Steps

1. **Read Full Documentation**: `CIRCUIT_README.md`
2. **Study Examples**: `test_circuit.py`, `demo_circuit.py`
3. **Build Your Network**: Start with simple patterns, add complexity
4. **Tune Parameters**: Adjust thresholds, weights, inhibition
5. **Visualize Results**: Plot spike rasters, membrane potentials

## Quick Reference Card

```python
# CREATE
circuit = NeuralCircuit(num_neurons, input_channels, dt=1.0, max_delay=10)

# CONNECT
circuit.connect(source, target, weight, delay)
circuit.connect_chain(weight, delay)
circuit.connect_all_to_all(weight, delay)
circuit.set_inhibition(strength)

# SIMULATE
outputs = circuit.step(inputs, I_ext, learning=True)

# INSPECT
circuit.summary()
circuit.get_connection_matrix()
circuit.get_weights(neuron_id)
circuit.get_states()

# MANAGE
circuit.reset_state()
circuit.set_weights(neuron_id, weights)
```

## Support

For detailed documentation, see:
- `CIRCUIT_README.md` - Full API and architecture
- `CIRCUIT_SUMMARY.md` - Implementation overview
- `test_circuit.py` - Working examples

---

**Ready to build neural networks!** 🧠⚡

