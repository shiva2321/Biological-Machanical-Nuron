# Nuron Framework - Complete Documentation

**Last Updated**: January 14, 2026  
**Version**: 1.0  
**Status**: Production-Ready

---

## 📚 Table of Contents

1. [Quick Start](#quick-start)
2. [Core Components](#core-components)
3. [Web Interface](#web-interface)
4. [Training System](#training-system)
5. [Experiments](#experiments)
6. [API Reference](#api-reference)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation

```bash
cd Nuron
pip install -r requirements.txt
```

### Run Web App (Easiest)

```bash
streamlit run web_app.py
# Or double-click: run_web_app.bat (Windows)
```

Opens at http://localhost:8501 with 4 tabs:
- 🔍 **Weight Visualizer** - Interactive weight matrix heatmap
- ✏️ **Draw & Predict** - Draw 5×5 patterns, test predictions
- 🎓 **Training Mode** - Live training with real-time charts
- 📈 **Brain Info** - Architecture and statistics

### Run Experiments

```bash
python experiments/visual_experiment.py    # Pattern detection
python experiments/pavlov_experiment.py    # Classical conditioning
python experiments/sequence_experiment.py  # Temporal sequences
```

### Smart Training (Python API)

```python
from brain_io import load_brain
from smart_trainer import train_reader_smart

brain = load_brain('my_brain.pkl')

# Auto-tunes parameters, expands dataset, saves on improvements
for status in train_reader_smart(brain, target_acc=0.85):
    print(f"Epoch {status['epoch']}: {status['accuracy']*100:.1f}%")
```

---

## Core Components

### 1. BiologicalNeuron (`neuron.py`)

**Leaky Integrate-and-Fire (LIF) neuron with STDP learning**

```python
from neuron import BiologicalNeuron

neuron = BiologicalNeuron(
    n_inputs=25,           # Number of input synapses
    tau_m=20.0,           # Membrane time constant (ms)
    tau_u=30.0,           # Adaptation time constant
    tau_theta=50.0,       # Threshold adaptation time
    tau_trace=20.0,       # STDP trace time constant
    dt=1.0,               # Time step (ms)
    a_plus=0.05,          # LTP learning rate
    a_minus=0.05,         # LTD learning rate
    weight_max=10.0       # Maximum weight
)
```

**Key Methods**:
- `update(I_ext)` - Update membrane potential, return spike
- `stdp(input_spikes, output_spike)` - Apply STDP learning
- `reset()` - Reset state variables

**Dynamics**:
- Membrane potential: `dv/dt = (-v + I) / tau_m`
- Adaptation current: `du/dt = -u / tau_u`
- Dynamic threshold: `dθ/dt = -θ / tau_theta`
- Spike condition: `v > theta_base + u + theta`

### 2. NeuralCircuit (`circuit.py`)

**Network infrastructure with delays and inhibition**

```python
from circuit import NeuralCircuit

circuit = NeuralCircuit(
    num_neurons=16,
    input_channels=64,
    dt=1.0,
    max_delay=10,
    tau_m=20.0,
    theta_base=-65.0,
    weight_max=10.0
)
```

**Key Methods**:
- `step(input_spikes, I_ext, learning=True)` - Forward pass
- `connect(source, target, weight, delay)` - Add connection
- `set_inhibition(strength)` - Lateral inhibition
- `get_weights(neuron_idx)` - Get synaptic weights

**Features**:
- Axonal delays (spike buffer)
- Lateral inhibition (winner-take-all)
- Connection management
- Learning control

### 3. NeuroGym (`neuro_gym.py`)

**Training framework with supervised learning**

```python
from neuro_gym import NeuroGym

gym = NeuroGym(
    circuit=circuit,
    task_data={'inputs': X, 'labels': y},
    input_scale=120.0,
    teacher_current=200.0,
    baseline_current=15.0
)

# Train until convergence
final_acc = gym.train_until_converged(
    target_acc=0.85,
    max_epochs=100
)
```

**Key Methods**:
- `train_step(mode='supervised')` - Single training step
- `train_epoch()` - Full epoch through dataset
- `evaluate(verbose=True)` - Test accuracy
- `train_until_converged()` - Auto-train to target

**Modes**:
- **Supervised**: Teacher forcing (strong current to correct neuron)
- **Unsupervised**: STDP only (no teacher signal)

### 4. Brain I/O (`brain_io.py`)

**Persistence and neurogenesis**

```python
from brain_io import load_brain, save_brain, grow_brain

# Load or create
brain = load_brain('my_brain.pkl')  # Creates if doesn't exist

# Save
save_brain(brain, 'my_brain.pkl')

# Grow (add capacity)
grow_brain(brain, new_inputs=10, new_neurons=5)
```

**Functions**:
- `load_brain(filename)` - Load or create default brain
- `save_brain(circuit, filename)` - Serialize to disk
- `grow_brain(circuit, new_inputs, new_neurons)` - Neurogenesis
- `get_brain_info(circuit)` - Statistics dictionary

**Default Brain**:
- 16 neurons (3 reader, 4 hunter, 9 available)
- 64 input channels (25 reader, 4 hunter, 35 available)
- Optimized parameters (tau_m=20, theta_base=-65, weight_max=10)

### 5. Smart Trainer (`smart_trainer.py`)

**Auto-tuning training with generator-based progress**

```python
from smart_trainer import train_reader_smart, train_smart

# Task-specific helper
for status in train_reader_smart(brain, target_acc=0.85, max_epochs=50):
    print(f"{status['epoch']}: {status['accuracy']*100:.1f}% - {status['action']}")

# Generic smart trainer
for status in train_smart(circuit, 'MyTask', task_data, ...):
    # Update UI, save logs, etc.
    pass
```

**Auto-Tuning Features**:
- 🚨 **Silent brain detection**: Doubles input_scale if no spikes
- 📚 **Dataset expansion**: Adds 500 samples if accuracy plateaus
- 💾 **Auto-save**: Saves brain after every improvement
- ⚡ **Live progress**: Yields status dict every epoch

**Status Dictionary**:
```python
{
    'epoch': 15,
    'accuracy': 0.75,
    'best_accuracy': 0.78,
    'loss': 0.25,
    'action': 'Brain saved (improvement)',
    'input_scale': 240.0,
    'dataset_size': 1500,
    'total_spikes': 45,
    'progress': 0.88  # 0.0 to 1.0
}
```

### 6. Lessons (`lessons.py`)

**Modular training tasks with channel mapping**

```python
from lessons import train_reader, train_hunter

# Reader: Channels 0-24 → Neurons 0-2 (A, B, C)
train_reader(brain, num_samples=1000, target_acc=0.85)

# Hunter: Channels 25-28 → Neurons 3-6 (N, S, E, W)
train_hunter(brain, num_episodes=50)
```

**Available Lessons**:
- **train_reader**: 5×5 character recognition (A, B, C)
- **train_hunter**: Sensory-motor navigation
- Easy to add custom lessons

### 7. Dashboard (`dashboard.py`)

**Command-line control center**

```bash
python dashboard.py
```

**Menu Options**:
1. Status Report (neurons, connections, age)
2. Visualize Memory (weight heatmap PNG)
3. Attend Class: Reading
4. Attend Class: Hunting
5. List All Lessons
6. Grow Brain (neurogenesis)
7. Save & Exit
8. Exit Without Saving

---

## Web Interface

### Running the Web App

```bash
# CORRECT way
streamlit run web_app.py

# WRONG way (causes warnings)
python web_app.py  # Don't do this!
```

**Windows Shortcut**: Double-click `run_web_app.bat`

### Tab 1: Weight Visualizer 🔍

**Interactive Plotly heatmap of all weights**

- Zoom, pan, hover for details
- Color-coded regions:
  - 🔴 Red box: Reader (channels 0-24, neurons 0-2)
  - 🔵 Blue box: Hunter (channels 25-28, neurons 3-6)
- Matrix statistics (size, sparsity, non-zero count)

### Tab 2: Draw & Predict ✏️

**Real-time pattern testing**

1. Click checkboxes to draw 5×5 pattern
2. Quick fill buttons: Letter A, B, C, Clear
3. See prediction instantly:
   - Which neurons fired
   - Decoded letter (A/B/C)
   - Voltage bar chart (all neurons)

### Tab 3: Training Mode 🎓

**Live training visualization**

1. Configure:
   - Task: Reader (Character Recognition)
   - Target accuracy: 0.85 recommended
   - Max epochs: 50 default
   - Dataset size: 1000 samples

2. Click "🚀 Start Training"

3. Watch live updates:
   - Epoch, accuracy, loss metrics
   - Progress bar
   - Accuracy chart (green, growing)
   - Loss chart (red, decreasing)
   - Weight heatmap (evolving every epoch)
   - Smart trainer actions (boosts, expansions, saves)

**Smart Trainer Actions**:
- 🚨 Red: Silent brain → Boosting sensitivity
- 📚 Yellow: Plateau → Expanding dataset
- 💾 Green: Improvement → Brain saved
- 🎉 Green: Converged!

### Tab 4: Brain Info 📈

**Architecture and statistics**

- Neuron/connection counts
- Weight distribution histogram
- Neuron parameters display
- Coverage metrics

---

## Training System

### Channel Architecture

The brain uses **channel mapping** for multi-task learning:

```
Input Channels (64 total):
  0-24:  Reader region (5×5 = 25 channels)
  25-28: Hunter region (4 sensors)
  29-63: Available (35 channels)

Output Neurons (16 total):
  0-2:   Reader outputs (A, B, C)
  3-6:   Hunter outputs (N, S, E, W)
  7-15:  Available (9 neurons)
```

**Benefits**:
- Multiple tasks in one brain
- No interference between tasks
- Continual learning without forgetting
- Easy to add new tasks

### Training Workflow

**1. Using Web App (No Code)**:
```bash
streamlit run web_app.py
# Go to Training Mode tab
# Configure and click Start Training
```

**2. Using Smart Trainer (Python)**:
```python
from brain_io import load_brain
from smart_trainer import train_reader_smart

brain = load_brain('my_brain.pkl')

for status in train_reader_smart(brain, target_acc=0.85):
    print(f"Epoch {status['epoch']}: {status['accuracy']*100:.1f}%")
    if status['action']:
        print(f"  → {status['action']}")
```

**3. Using NeuroGym Directly**:
```python
from neuro_gym import NeuroGym

gym = NeuroGym(circuit, task_data)
final_acc = gym.train_until_converged(target_acc=0.85)
```

**4. Using Dashboard (Interactive CLI)**:
```bash
python dashboard.py
# Select option 3 or 4 to train
```

### Creating Custom Tasks

```python
import numpy as np

def generate_my_task_samples(num_samples):
    """Generate samples for custom task."""
    inputs = []
    labels = []
    
    for _ in range(num_samples):
        # Your task logic
        pattern = np.random.rand(10)  # 10-element pattern
        
        # Pad to 64 channels (use channels 29-38)
        padded = np.zeros(64)
        padded[29:39] = pattern
        
        label = ...  # Your label logic
        
        inputs.append(padded)
        labels.append(label)
    
    return {'inputs': np.array(inputs), 'labels': np.array(labels)}

# Train
from smart_trainer import train_smart

task_data = generate_my_task_samples(1000)
for status in train_smart(brain, 'MyTask', task_data, target_acc=0.80):
    print(f"{status['epoch']}: {status['accuracy']*100:.1f}%")
```

---

## Experiments

### 1. Visual Experiment - Pattern Detection

**File**: `experiments/visual_experiment.py`

**Purpose**: Unsupervised pattern learning

**Setup**:
- 20 input channels
- 2000ms simulation
- Noise: 2% Poisson spikes
- Signal: 4 channels fire together every 100ms

**Result**: 70-80% pattern separation (Green lines diverge from gray)

**Run**: `python experiments/visual_experiment.py`

### 2. Pavlov Experiment - Classical Conditioning

**File**: `experiments/pavlov_experiment.py`

**Purpose**: Temporal association learning

**Setup**:
- 2 inputs (Bell, Food)
- 100 trials: Bell at t=10ms, Food at t=30ms
- Test: Bell only

**Result**: 500% weight increase, spike time migrates earlier

**Run**: `python experiments/pavlov_experiment.py`

### 3. Sequence Experiment - Temporal Sequence Detection

**File**: `experiments/sequence_experiment.py`

**Purpose**: Precise temporal sequence recognition

**Setup**:
- "Passcode lock" circuit
- Correct: Input 0 → Input 2 → Input 1 (20ms gaps)
- Incorrect: Different order or timing

**Result**: 100% selectivity (only correct sequence fires output)

**Run**: `python experiments/sequence_experiment.py`

### 4. Hunter Experiment - Sensory-Motor Learning

**File**: `experiments/hunter_experiment.py`

**Purpose**: Navigation through sensor-motor associations

**Setup**:
- 10×10 grid with agent and food
- Training: Teacher forcing (stimulate correct sensor+motor)
- Testing: Autonomous navigation

**Result**: ~50% navigation accuracy after 50 training steps

**Run**: `python experiments/hunter_experiment.py`

### 5. Trader Experiment - Time-Series Prediction

**File**: `experiments/trader_experiment.py`

**Purpose**: Continuous signal forecasting using population coding

**Setup**:
- 20 neurons represent values from -1 to +1
- Sine wave prediction
- Associative chain learning

**Result**: Phase-locked tracking of sine wave

**Run**: `python experiments/trader_experiment.py`

### 6. Reader Experiment - Character Recognition

**File**: `experiments/reader_experiment.py`

**Purpose**: Visual pattern recognition with winner-take-all

**Setup**:
- 5×5 binary patterns (A, B, C)
- 1000 noisy samples (5% pixel flips)
- 3 output neurons with lateral inhibition

**Result**: ~33% accuracy baseline, improves with training

**Run**: `python experiments/reader_experiment.py`

---

## API Reference

### Neuron Parameters

```python
BiologicalNeuron(
    n_inputs=25,          # Number of synapses
    tau_m=20.0,          # Membrane time constant (ms)
    tau_u=30.0,          # Adaptation time constant (ms)
    tau_theta=50.0,      # Threshold adaptation time (ms)
    tau_trace=20.0,      # STDP trace time constant (ms)
    v_rest=-70.0,        # Resting potential (mV)
    v_reset=-75.0,       # Reset potential (mV)
    theta_base=-55.0,    # Base threshold (mV)
    u_increment=2.0,     # Adaptation increment
    theta_increment=1.0, # Threshold increment
    dt=1.0,              # Time step (ms)
    a_plus=0.05,         # LTP rate
    a_minus=0.05,        # LTD rate
    weight_min=0.0,      # Minimum weight
    weight_max=1.0       # Maximum weight
)
```

### Circuit Parameters

```python
NeuralCircuit(
    num_neurons=16,       # Number of neurons
    input_channels=64,    # Input dimensionality
    dt=1.0,              # Time step (ms)
    max_delay=10,        # Maximum axonal delay (ms)
    tau_m=20.0,          # Membrane time constant
    theta_base=-65.0,    # Base threshold (lower = easier to fire)
    weight_max=10.0      # Maximum weight (higher = stronger)
)
```

### Training Parameters

```python
NeuroGym(
    circuit=circuit,
    task_data={'inputs': X, 'labels': y},
    input_scale=120.0,        # Input signal strength
    teacher_current=200.0,    # Teacher forcing strength
    baseline_current=15.0     # Baseline excitability
)
```

**Parameter Tuning**:
- **input_scale**: Higher = stronger input, easier to fire
- **teacher_current**: Higher = stronger teaching signal
- **baseline_current**: Higher = more spontaneous activity
- **theta_base**: Lower = easier to fire (e.g., -65 vs -55)
- **tau_m**: Higher = slower membrane dynamics, holds charge longer
- **weight_max**: Higher = stronger synapses possible

---

## Troubleshooting

### Web App Issues

**Problem**: ScriptRunContext warnings
```
Thread 'MainThread': missing ScriptRunContext!
```
**Solution**: Use `streamlit run web_app.py`, not `python web_app.py`

**Problem**: Port 8501 already in use
```bash
streamlit run web_app.py --server.port 8502
```

**Problem**: ValueError: Expected 64 input channels, got 25
**Solution**: Already fixed in `smart_trainer.py` (inputs padded to 64)

### Training Issues

**Problem**: Brain is silent (no neurons fire)
**Solution**: Smart trainer auto-detects and boosts input_scale

**Problem**: Accuracy stuck at 33% (random guessing)
**Solution**: Smart trainer auto-expands dataset after 5 epochs without improvement

**Problem**: Training too slow
**Solution**: 
- Reduce max_epochs
- Reduce dataset size
- Increase teacher_current

### Import Errors

**Problem**: ModuleNotFoundError
```bash
pip install -r requirements.txt
```

**Dependencies**:
- numpy>=1.20.0
- matplotlib>=3.3.0
- streamlit>=1.28.0
- plotly>=5.14.0

---

## Best Practices

### 1. Always Save After Training
```python
from brain_io import save_brain
save_brain(brain, 'my_brain.pkl')
```
Smart trainer does this automatically!

### 2. Use Channel Mapping
Don't overlap task regions:
- Reader: 0-24
- Hunter: 25-28
- Your task: 29+ (choose unused channels)

### 3. Monitor Training
Use smart_trainer for automatic monitoring and adjustment.

### 4. Start Small
- Begin with small datasets (100-1000 samples)
- Low target accuracy initially (0.7-0.8)
- Increase gradually

### 5. Visualize Results
- Use web app to inspect weights
- Check outputs/ folder for experiment plots
- Monitor training curves

---

## File Structure

```
Nuron/
├── Core Framework
│   ├── neuron.py           (LIF neuron + STDP)
│   ├── circuit.py          (Network infrastructure)
│   ├── neuro_gym.py        (Training framework)
│   ├── brain_io.py         (Persistence)
│   ├── lessons.py          (Training tasks)
│   ├── smart_trainer.py    (Auto-tuning training)
│   ├── dashboard.py        (CLI interface)
│   └── web_app.py          (Web interface)
│
├── Experiments
│   ├── visual_experiment.py
│   ├── pavlov_experiment.py
│   ├── sequence_experiment.py
│   ├── hunter_experiment.py
│   ├── trader_experiment.py
│   └── reader_experiment.py
│
├── Utilities
│   ├── requirements.txt
│   ├── run_web_app.bat     (Windows launcher)
│   └── .gitignore
│
├── Documentation
│   ├── README.md           (Quick overview)
│   ├── DOCUMENTATION.md    (This file - complete docs)
│   └── RESEARCH.md         (Academic content)
│
└── Runtime
    ├── my_brain.pkl        (Saved brain state)
    ├── my_brain_age.txt    (Training sessions)
    └── outputs/            (Generated plots)
```

---

## Quick Commands Reference

```bash
# Web Interface
streamlit run web_app.py

# Dashboard
python dashboard.py

# Experiments
python experiments/visual_experiment.py
python experiments/pavlov_experiment.py
python experiments/sequence_experiment.py

# Smart Training (Python)
python -c "from brain_io import load_brain; from smart_trainer import train_reader_smart; brain = load_brain('my_brain.pkl'); list(train_reader_smart(brain))"

# Install/Update
pip install -r requirements.txt
pip install --upgrade streamlit plotly
```

---

**For academic/research content, see RESEARCH.md**  
**For quick start, see README.md**

---

*Last updated: January 14, 2026*  
*Nuron Framework v1.0 - Production Ready*

