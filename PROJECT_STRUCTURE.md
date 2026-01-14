# Nuron Project Structure

```
Nuron/
│
├── 📄 README.md                    # Main project documentation
├── 📄 QUICKSTART.md                # Quick reference guide
├── 📄 requirements.txt             # Python dependencies
│
├── 🧠 neuron.py                    # Core BiologicalNeuron class (263 lines)
├── 🔌 circuit.py                   # NeuralCircuit infrastructure (580+ lines)
│
├── 🧪 experiments/                 # Demonstration experiments
│   ├── README.md                   # Experiments documentation
│   ├── visual_experiment.py        # Pattern detection in noise
│   ├── pavlov_experiment.py        # Classical conditioning (Pavlov)
│   ├── sequence_experiment.py      # Temporal sequence detection
│   └── demo_circuit.py             # Circuit capabilities demo
│
├── 🧬 tests/                       # Unit and integration tests
│   ├── test_neuron.py              # BiologicalNeuron tests
│   └── test_circuit.py             # NeuralCircuit tests
│
├── 📚 docs/                        # Detailed documentation
│   ├── CIRCUIT_README.md           # Circuit API reference
│   ├── CIRCUIT_QUICKSTART.md       # Circuit quick guide
│   ├── CIRCUIT_SUMMARY.md          # Circuit implementation summary
│   ├── SEQUENCE_TUNING_SUCCESS.md  # Sequence experiment tuning guide
│   ├── SEQUENCE_EXPERIMENT_SUMMARY.md  # Sequence architecture details
│   ├── PAVLOV_SUMMARY.md           # Pavlov experiment summary
│   ├── PAVLOV_EXPERIMENT_README.md # Pavlov detailed docs
│   ├── VISUAL_EXPERIMENT_GUIDE.md  # Visual experiment complete guide
│   ├── VISUAL_EXPERIMENT_README.md # Visual experiment technical docs
│   ├── VISUAL_EXPERIMENT_SUMMARY.md # Visual experiment summary
│   └── IMPLEMENTATION_SUMMARY.md   # Overall implementation notes
│
├── 📊 outputs/                     # Generated visualizations
│   ├── sequence_experiment_results.png        # Sequence detection results
│   ├── circuit_demo_propagation.png          # Delay propagation demo
│   └── circuit_demo_winner_take_all.png      # Competition demo
│
├── 🔧 .venv/                       # Python virtual environment (optional)
├── 💾 __pycache__/                 # Python bytecode cache
└── 🛠️ .idea/                       # IDE configuration (optional)
```

## 📊 File Statistics

### Core Implementation
- **neuron.py**: 263 lines - BiologicalNeuron with LIF + STDP
- **circuit.py**: 580+ lines - Network infrastructure

### Experiments (4 files)
- **visual_experiment.py**: 300+ lines - Pattern detection
- **pavlov_experiment.py**: 400+ lines - Classical conditioning
- **sequence_experiment.py**: 413 lines - Sequence detection
- **demo_circuit.py**: 240+ lines - Circuit demonstrations

### Tests (2 files)
- **test_neuron.py**: ~100 lines - Neuron unit tests
- **test_circuit.py**: 300+ lines - Circuit integration tests

### Documentation (13 files)
- **README.md**: Comprehensive project overview
- **QUICKSTART.md**: Quick reference
- **docs/**: 11 detailed documentation files

### Total Lines of Code: ~2,500+ lines

## 🎯 Key Entry Points

### For Users
1. **Start here**: `README.md`
2. **Quick ref**: `QUICKSTART.md`
3. **Run demo**: `python experiments/visual_experiment.py`

### For Developers
1. **Core classes**: `neuron.py`, `circuit.py`
2. **Tests**: `tests/test_neuron.py`, `tests/test_circuit.py`
3. **API docs**: `docs/CIRCUIT_README.md`

### For Learners
1. **Simple demo**: `experiments/visual_experiment.py`
2. **Concepts**: `docs/VISUAL_EXPERIMENT_GUIDE.md`
3. **Circuit guide**: `docs/CIRCUIT_QUICKSTART.md`

## 🧹 Cleanup Summary

### ✅ Organized
- Created `experiments/`, `tests/`, `docs/`, `outputs/` folders
- Moved 4 experiment files to `experiments/`
- Moved 2 test files to `tests/`
- Moved 11 documentation files to `docs/`
- Moved 3 PNG files to `outputs/`

### ❌ Deleted (Development Artifacts)
- `debug_new_stdp.py`
- `debug_stdp_detail.py`
- `debug_trace.py`
- `demo_neuron.py`
- `final_validation.py`
- `main.py`
- `test_input_stops.py`
- `test_sparse_spikes.py`
- `test_stdp_fix.py`
- `validate_fix.py`

### ✨ Created
- `README.md` - Comprehensive project documentation
- `QUICKSTART.md` - Quick reference guide
- `requirements.txt` - Python dependencies
- `experiments/README.md` - Experiments documentation
- `PROJECT_STRUCTURE.md` - This file

## 📦 Result

**Before**: 30+ files scattered in root directory  
**After**: Clean structure with 4 organized folders

**Root directory now contains**:
- 2 core modules (`neuron.py`, `circuit.py`)
- 3 documentation files (`README.md`, `QUICKSTART.md`, `requirements.txt`)
- 4 organized folders (`experiments/`, `tests/`, `docs/`, `outputs/`)

**Total reduction**: 30+ files → 9 items in root (78% cleaner)

---

**The directory is now organized, professional, and easy to navigate! 🎉**

