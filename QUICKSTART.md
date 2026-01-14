# Nuron Quick Reference

## 📁 Directory Structure

```
Nuron/
├── neuron.py              # BiologicalNeuron class
├── circuit.py             # NeuralCircuit infrastructure
├── README.md              # Main documentation
├── experiments/           # All experiments
├── tests/                 # Unit tests
├── docs/                  # Detailed documentation
└── outputs/               # Generated visualizations
```

## 🚀 Quick Commands

```bash
# Run experiments
python experiments/visual_experiment.py      # Pattern detection
python experiments/pavlov_experiment.py      # Classical conditioning
python experiments/sequence_experiment.py    # Sequence detection
python experiments/demo_circuit.py           # Circuit demos

# Run tests
python tests/test_neuron.py                  # Test BiologicalNeuron
python tests/test_circuit.py                 # Test NeuralCircuit
```

## 🔬 Core API

### BiologicalNeuron
```python
from neuron import BiologicalNeuron

neuron = BiologicalNeuron(n_inputs=10, tau_m=20.0, theta_base=-55.0)
spike = neuron.step(input_spikes, I_ext=10.0, learning=True)
neuron.reset_state()  # Reset dynamics, keep weights
```

### NeuralCircuit
```python
from circuit import NeuralCircuit

circuit = NeuralCircuit(num_neurons=10, input_channels=5, max_delay=10)
circuit.connect(source=0, target=1, weight=1.0, delay=2)
circuit.set_inhibition(strength=3.0)
outputs = circuit.step(input_spikes, I_ext=None, learning=True)
```

## 📊 Experiments

| Experiment | Concept | Success Rate |
|------------|---------|--------------|
| Visual | Pattern detection | 70-80% |
| Pavlov | Classical conditioning | 2/3 criteria |
| Sequence | Temporal selectivity | 100% |

## 📚 Documentation

- `README.md` - Main overview
- `docs/CIRCUIT_README.md` - Circuit API
- `docs/CIRCUIT_QUICKSTART.md` - Circuit guide
- `docs/SEQUENCE_TUNING_SUCCESS.md` - Sequence details
- `docs/VISUAL_EXPERIMENT_GUIDE.md` - Pattern detection
- `docs/PAVLOV_SUMMARY.md` - Classical conditioning

## 🎯 Key Parameters

### For Reliable Firing
```python
theta_base = -65.0     # Lower = easier firing
tau_m = 20.0          # Higher = longer integration
input_scale = 80.0    # Higher = stronger input
I_ext = 7.0           # Baseline excitability
```

### For Learning
```python
tau_trace = 20.0      # STDP window
a_plus = 0.01         # Potentiation rate
a_minus = 0.01        # Depression rate
weight_max = 1.0      # Weight ceiling
```

## ⚡ Tips

1. **No spikes?** → Lower threshold, increase I_ext, increase input_scale
2. **Too many spikes?** → Raise threshold, decrease I_ext, add inhibition
3. **Slow learning?** → Increase a_plus/a_minus
4. **Unstable weights?** → Check weight_min/weight_max bounds

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| No spikes | `theta_base = -65.0`, `I_ext = 10.0` |
| Spontaneous firing | `I_ext = 0.0`, `theta_base = -55.0` |
| Poor learning | `a_plus = 0.05`, `tau_trace = 40.0` |
| Weight explosion | Set `weight_max = 1.0` |

## 📦 Dependencies

```bash
pip install numpy matplotlib
```

## ✅ Quick Test

```python
from neuron import BiologicalNeuron
import numpy as np

neuron = BiologicalNeuron(n_inputs=5)
for t in range(100):
    inputs = np.ones(5) if t == 10 else np.zeros(5)
    spike = neuron.step(inputs * 20.0, I_ext=10.0)
    if spike:
        print(f"Spike at t={t}ms")  # Should fire around t=10
```

## 🎓 Learn More

1. Read `README.md` for overview
2. Run `visual_experiment.py` to see pattern detection
3. Check `docs/CIRCUIT_QUICKSTART.md` for circuit guide
4. Explore `experiments/` for more examples

---

**Quick Start**: `python experiments/visual_experiment.py`

