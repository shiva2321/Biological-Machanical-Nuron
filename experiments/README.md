# Experiments

This folder contains demonstration experiments showcasing the capabilities of the Nuron framework.

## 📁 Contents

### 1. visual_experiment.py
**Pattern Detection in Noise**

Demonstrates unsupervised learning of hidden temporal patterns.

- **Input**: 20 channels with 2% Poisson noise
- **Hidden Pattern**: Channels [0, 5, 10, 15] fire together every 100ms
- **Learning**: STDP strengthens pattern weights, ignores noise
- **Output**: Visualization showing weight separation

**Run**:
```bash
python visual_experiment.py
```

**Expected Result**: Pattern weights (green) rise to ~0.9, noise weights (gray) stay at ~0.6

---

### 2. pavlov_experiment.py
**Classical Conditioning**

Temporal associative learning - Pavlov's Bell & Food experiment.

- **Setup**: Bell (CS) at t=10ms → Food (US) at t=30ms
- **Training**: 100 trials of temporal pairing
- **Learning**: Bell weight increases from 0.2 to 1.0
- **Test**: Bell alone triggers response

**Run**:
```bash
python pavlov_experiment.py
```

**Expected Result**: Bell weight reaches >0.8, bell-only test triggers spike

---

### 3. sequence_experiment.py
**Temporal Sequence Detection ("Passcode Lock")**

Circuit that only responds to correct sequence with correct timing.

- **Architecture**: 4-neuron bucket brigade with 20ms delays
- **Correct Sequence**: Input 0 → 2 → 1 (at t=10, 30, 50)
- **Mechanism**: Sub-threshold summation + axonal delays
- **Selectivity**: Output fires ONLY for correct sequence

**Run**:
```bash
python sequence_experiment.py
```

**Expected Result**:
- Trial 1 (correct): Output fires ✓
- Trial 2 (wrong timing): No output ✓
- Trial 3 (wrong order): No output ✓

---

### 4. demo_circuit.py
**Circuit Infrastructure Demonstrations**

Visual demonstrations of circuit capabilities.

- **Demo 1**: Spike propagation through chain with delays
- **Demo 2**: Winner-take-all with lateral inhibition
- **Output**: Publication-quality visualizations saved to `outputs/`

**Run**:
```bash
python demo_circuit.py
```

**Expected Output**: Two PNG files showing propagation and competition

---

## 🎯 Learning Objectives

### Visual Experiment
- Unsupervised learning
- Feature detection
- Noise robustness
- STDP in action

### Pavlov Experiment
- Temporal association
- Predictive coding
- Classical conditioning
- Anticipatory responses

### Sequence Experiment
- Temporal selectivity
- Coincidence detection
- Bucket brigade architecture
- Sequence recognition

### Circuit Demo
- Axonal delays
- Lateral inhibition
- Winner-take-all
- Network dynamics

---

## 📊 Quick Comparison

| Experiment | Duration | Neurons | Inputs | Key Concept |
|------------|----------|---------|--------|-------------|
| Visual | 4000ms | 1 | 20 | Pattern detection |
| Pavlov | 100 trials | 2 | 2 | Temporal association |
| Sequence | 100ms × 3 | 4 | 3 | Sequence selectivity |
| Demo | Various | 5-10 | Various | Infrastructure |

---

## 🔧 Customization

All experiments can be customized by editing parameters at the top of each file:

### Visual Experiment
```python
n_inputs = 20              # Number of input channels
pattern = [0, 5, 10, 15]   # Which inputs are pattern
duration_ms = 4000         # Simulation time
```

### Pavlov Experiment
```python
n_trials = 100             # Number of training trials
bell_time = 10             # CS timing (ms)
food_time = 30             # US timing (ms)
```

### Sequence Experiment
```python
input_times = {            # Define input timing
    0: [10],               # Input 0 at t=10
    2: [30],               # Input 2 at t=30
    1: [50]                # Input 1 at t=50
}
```

---

## 📈 Expected Performance

### Visual Experiment
- **Success**: Pattern weights > 0.8, noise weights < 0.7
- **Separation**: 0.1-0.3 difference
- **Reliability**: 70-80% of runs

### Pavlov Experiment
- **Weight Change**: 0.2 → 1.0
- **Criteria Met**: 2/3 (weight ✓, response ✓, timing ~)
- **Reliability**: Consistent

### Sequence Experiment
- **Selectivity**: 100% (only correct sequence)
- **Timing**: Precise (±1ms)
- **Reliability**: Perfect with tuned parameters

---

## 🐛 Troubleshooting

### Visual Experiment
- **No learning**: Increase `a_plus`, decrease noise probability
- **All weights high**: Reduce `I_ext`, increase threshold

### Pavlov Experiment
- **No spikes**: Increase `input_spikes_scaled`, decrease `theta_base`
- **Too early spikes**: Reduce `I_ext`

### Sequence Experiment
- **No cascade**: Increase `input_spikes_scaled` to 80.0
- **Output doesn't fire**: Set `I_ext[3] = 7.0`

---

## 📚 Learn More

- See `../docs/` for detailed documentation
- Check `../README.md` for framework overview
- Run `../tests/` to verify installation

---

**Quick Start**: Run `python visual_experiment.py` to see pattern detection in action!

