# Nuron: Biologically-Inspired Spiking Neural Network Framework

A Python implementation of biologically plausible spiking neural networks featuring Leaky Integrate-and-Fire (LIF) neurons, Spike-Timing-Dependent Plasticity (STDP), and temporal sequence detection.

## 🧠 Overview

Nuron provides a complete framework for building and experimenting with spiking neural networks that mimic biological neural computation. The framework includes:

- **BiologicalNeuron**: LIF neuron with adaptation and STDP learning
- **NeuralCircuit**: Network infrastructure with axonal delays and lateral inhibition
- **Experiments**: Demonstrations of pattern detection, classical conditioning, and sequence recognition

## 🚀 Quick Start

### Installation

```bash
# Clone or download the repository
cd Nuron

# Create virtual environment (optional but recommended)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install numpy matplotlib
```

### Run Experiments

```bash
# Pattern detection experiment
python experiments/visual_experiment.py

# Classical conditioning (Pavlov's experiment)
python experiments/pavlov_experiment.py

# Temporal sequence detection ("passcode lock")
python experiments/sequence_experiment.py

# Circuit demonstrations
python experiments/demo_circuit.py
```

## 📁 Project Structure

```
Nuron/
│
├── neuron.py                    # Core BiologicalNeuron class
├── circuit.py                   # NeuralCircuit infrastructure
├── README.md                    # This file
│
├── experiments/                 # Experiment demonstrations
│   ├── visual_experiment.py    # Pattern detection in noise
│   ├── pavlov_experiment.py    # Classical conditioning
│   ├── sequence_experiment.py  # Temporal sequence detection
│   └── demo_circuit.py         # Circuit capabilities demo
│
├── tests/                       # Unit and integration tests
│   ├── test_neuron.py          # BiologicalNeuron tests
│   └── test_circuit.py         # NeuralCircuit tests
│
├── docs/                        # Detailed documentation
│   ├── CIRCUIT_README.md       # Circuit API reference
│   ├── CIRCUIT_QUICKSTART.md   # Circuit quick guide
│   ├── SEQUENCE_TUNING_SUCCESS.md  # Sequence experiment details
│   └── ...                     # Additional documentation
│
└── outputs/                     # Generated visualizations
    ├── sequence_experiment_results.png
    ├── circuit_demo_propagation.png
    └── ...
```

## 🔬 Core Components

### BiologicalNeuron (`neuron.py`)

Implements a biologically plausible spiking neuron with:

- **LIF Dynamics**: Membrane potential with leak
- **Adaptation**: Activity-dependent threshold increase
- **STDP Learning**: Hebbian plasticity ("cells that fire together, wire together")
- **Eligibility Traces**: Temporal credit assignment window

**Key Features:**
- Configurable time constants (τ_m, τ_u, τ_theta, τ_trace)
- Dynamic threshold with homeostatic regulation
- Weight bounds for stability
- State management and reset

### NeuralCircuit (`circuit.py`)

Network infrastructure for managing neuron populations:

- **Connectivity**: Flexible connection patterns with delays
- **Spike Buffering**: Efficient axonal delay handling
- **Lateral Inhibition**: Winner-take-all competition
- **Network Dynamics**: Coordinated multi-neuron simulation

**Key Features:**
- Arbitrary connection topologies
- Axonal transmission delays (0-max_delay ms)
- Lateral inhibition mechanism
- Pre-built patterns (chain, all-to-all, etc.)

## 🧪 Experiments

### 1. Visual Pattern Detection
**File**: `experiments/visual_experiment.py`

Demonstrates unsupervised learning of hidden patterns in noise.

- **Setup**: 20 input channels, 2% Poisson noise
- **Hidden Pattern**: Inputs [0, 5, 10, 15] fire together every 100ms
- **Result**: Pattern weights strengthen (0.2 → 0.9+), noise weights stay low
- **Concept**: Feature detection without supervision

```bash
python experiments/visual_experiment.py
```

### 2. Classical Conditioning (Pavlov)
**File**: `experiments/pavlov_experiment.py`

Temporal associative learning - predicting future events.

- **Setup**: Bell (CS) → Food (US) with 20ms gap
- **Training**: 100 trials of Bell→Food pairing
- **Result**: Bell weight increases (0.2 → 1.0), Bell alone triggers response
- **Concept**: Predictive coding, temporal association

```bash
python experiments/pavlov_experiment.py
```

### 3. Temporal Sequence Detection
**File**: `experiments/sequence_experiment.py`

"Passcode lock" circuit - only responds to correct sequence.

- **Setup**: 4-neuron bucket brigade with 20ms delays
- **Correct Sequence**: Input 0 → 2 → 1 (with 20ms gaps)
- **Result**: Output fires ONLY for correct sequence
- **Concept**: Sequence recognition, temporal selectivity

```bash
python experiments/sequence_experiment.py
```

### 4. Circuit Demonstrations
**File**: `experiments/demo_circuit.py`

Visual demos of circuit capabilities.

- **Demo 1**: Spike propagation with axonal delays
- **Demo 2**: Winner-take-all with lateral inhibition
- **Output**: Publication-quality visualizations

```bash
python experiments/demo_circuit.py
```

## 🧮 Usage Examples

### Basic Neuron

```python
from neuron import BiologicalNeuron
import numpy as np

# Create neuron
neuron = BiologicalNeuron(
    n_inputs=5,
    tau_m=20.0,
    theta_base=-55.0
)

# Simulate
for t in range(100):
    inputs = np.random.rand(5)
    spike = neuron.step(inputs, I_ext=10.0, learning=True)
    if spike:
        print(f"Spike at t={t}ms")
```

### Neural Circuit

```python
from circuit import NeuralCircuit
import numpy as np

# Create circuit
circuit = NeuralCircuit(
    num_neurons=10,
    input_channels=5,
    max_delay=10
)

# Add connections
circuit.connect_chain(weight=1.0, delay=2)
circuit.set_inhibition(strength=3.0)

# Simulate
for t in range(100):
    inputs = np.random.rand(5)
    outputs = circuit.step(inputs)
    print(f"t={t}: {np.where(outputs)[0]}")
```

## 📊 Key Results

### Pattern Detection
- **Success Rate**: 70-80% (pattern weights > noise weights)
- **Separation**: 0.1-0.3 weight difference
- **Robustness**: Works with 2% continuous noise

### Classical Conditioning
- **Learning**: Bell weight 0.2 → 1.0 in ~100 trials
- **Criteria Met**: 2/3 (weight increase ✓, response ✓, timing ~)
- **Mechanism**: STDP-based temporal association

### Sequence Detection
- **Selectivity**: 100% (only correct sequence triggers output)
- **Timing**: Precise cascade (10ms → 30ms → 50ms → 51ms)
- **Robustness**: Rejects wrong timing and wrong order

## 🔧 Configuration

### Neuron Parameters

```python
BiologicalNeuron(
    n_inputs=10,           # Number of input synapses
    tau_m=20.0,           # Membrane time constant (ms)
    tau_trace=20.0,       # STDP trace decay (ms)
    theta_base=-55.0,     # Firing threshold (mV)
    a_plus=0.01,          # STDP potentiation rate
    a_minus=0.01,         # STDP depression rate
    weight_max=1.0        # Maximum synaptic weight
)
```

### Circuit Parameters

```python
NeuralCircuit(
    num_neurons=10,       # Population size
    input_channels=5,     # External inputs
    dt=1.0,              # Time step (ms)
    max_delay=10,        # Maximum axonal delay (ms)
    neuron_params={...}  # Shared neuron configuration
)
```

## 📚 Documentation

### Academic Papers
- **Research Paper**: `RESEARCH_PAPER.md` (18-page full academic paper)
- **Executive Summary**: `EXECUTIVE_SUMMARY.md` (6-page condensed version)
- **Presentation**: `PRESENTATION_OUTLINE.md` (Talk slides for professors/peers)

### Technical Documentation
Detailed documentation available in `docs/`:

- **Circuit API**: `docs/CIRCUIT_README.md`
- **Quick Start**: `docs/CIRCUIT_QUICKSTART.md`
- **Sequence Tuning**: `docs/SEQUENCE_TUNING_SUCCESS.md`
- **Visual Experiment**: `docs/VISUAL_EXPERIMENT_GUIDE.md`
- **Pavlov Experiment**: `docs/PAVLOV_SUMMARY.md`

## 🧪 Testing

Run unit tests:

```bash
# Test BiologicalNeuron
python tests/test_neuron.py

# Test NeuralCircuit
python tests/test_circuit.py
```

## 🎓 Educational Value

This framework teaches:

- **Spiking Neural Networks**: Event-driven computation
- **STDP Learning**: Hebbian plasticity mechanisms
- **Temporal Coding**: Time as information
- **Coincidence Detection**: AND-like neural computation
- **Sequence Processing**: Temporal pattern recognition

**Biological Concepts**:
- Leaky integration (membrane dynamics)
- Adaptation (activity-dependent changes)
- Synaptic plasticity (learning)
- Axonal delays (timing diversity)
- Lateral inhibition (competition)

## 🔬 Scientific Background

### LIF Neuron Model
```
dv/dt = (-v + v_rest + I_syn + I_ext - u) / τ_m
du/dt = -u / τ_u
```

### STDP Learning Rule
```
Δw = A+ * trace_pre  (if post-synaptic spike)
Δw = -A- * trace_post (if pre-synaptic spike)
```

**Key Principle**: "Cells that fire together, wire together"

## 🚧 Known Limitations

1. **Parameter Sensitivity**: LIF neurons require careful tuning
2. **Computation Speed**: Pure Python (not optimized for large networks)
3. **Learning Scope**: STDP only on external inputs (not internal connections)
4. **Network Size**: Tested up to ~100 neurons

## 🛠️ Future Enhancements

- **GPU Acceleration**: CUDA/PyTorch implementation
- **STDP on Internal Connections**: Full network plasticity
- **Structural Plasticity**: Dynamic connection creation/pruning
- **Multi-Layer Support**: Deep spiking networks
- **Neuromorphic Hardware**: Export to SpiNNaker/Loihi

## 📖 References

**LIF Model**:
- Gerstner & Kistler (2002). *Spiking Neuron Models*

**STDP**:
- Bi & Poo (1998). *Synaptic Modifications in Cultured Hippocampal Neurons*

**Sequence Learning**:
- Hopfield & Brody (2001). *What is a Moment? Temporal Coding in Networks*

## 📄 License

This project is provided as-is for educational and research purposes.

## 👥 Contributing

Feel free to:
- Report issues
- Suggest improvements
- Add new experiments
- Optimize performance

## 🎯 Project Status

**Version**: 1.0  
**Date**: January 2026  
**Status**: ✅ Production Ready

**Completed Features**:
- ✅ BiologicalNeuron with LIF + STDP
- ✅ NeuralCircuit infrastructure
- ✅ Pattern detection experiment
- ✅ Classical conditioning experiment
- ✅ Sequence detection experiment
- ✅ Comprehensive testing
- ✅ Complete documentation

## 📬 Contact

For questions or collaboration: See documentation in `docs/`

---

**Built with 🧠 for understanding biological neural computation**

*Nuron - Where neuroscience meets code*

