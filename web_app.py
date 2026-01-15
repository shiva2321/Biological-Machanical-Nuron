"""
Web Dashboard (Mission Control): Streamlit Interface for Nuron Framework

A comprehensive visual dashboard for training and testing neural networks with:
- Live training monitoring with real-time charts
- Interactive testing with 8x8 drawing board
- Brain statistics and hardware info
- Weight matrix visualization

Run with: streamlit run web_app.py
"""

import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import sys
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brain_io import load_brain, save_brain, get_brain_info
from lessons import train_reader, train_hunter, train_alphabet, train_digits, train_character_recognition
from data_factory import get_character_bitmap, visualize_character
from circuit import NeuralCircuit
from data_factory import generate_dataset

# Try to import torch for GPU detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================================
# Configuration
# ============================================================================

st.set_page_config(
    page_title="Nuron Mission Control",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

BRAIN_FILE = 'my_brain.pkl'
BRAIN_AGE_FILE = 'my_brain_age.txt'


# ============================================================================
# Helper Functions
# ============================================================================

def get_brain_age():
    """Get the number of training sessions."""
    if os.path.exists(BRAIN_AGE_FILE):
        try:
            with open(BRAIN_AGE_FILE, 'r') as f:
                return int(f.read().strip())
        except:
            return 0
    return 0


def increment_brain_age():
    """Increment brain age counter."""
    age = get_brain_age() + 1
    with open(BRAIN_AGE_FILE, 'w') as f:
        f.write(str(age))


def get_hardware_info():
    """Get CPU/GPU hardware information."""
    info = {
        'cpu': 'Available',
        'gpu': 'Not Available',
        'gpu_name': 'N/A',
        'gpu_memory': 'N/A'
    }

    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            info['gpu'] = 'Available'
            info['gpu_name'] = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory
            info['gpu_memory'] = f"{total_memory / 1024**3:.1f} GB"

    return info


def build_weight_matrix(circuit):
    """Build complete weight matrix for visualization."""
    num_neurons = circuit.num_neurons
    num_inputs = circuit.input_channels

    weight_matrix = np.zeros((num_neurons, num_inputs))
    for i in range(num_neurons):
        weights = circuit.get_weights(i)
        if len(weights) > 0:
            weight_matrix[i, :len(weights)] = weights

    return weight_matrix


def predict_from_grid(circuit, grid_8x8, char_map=None):
    """
    Predict character from 8x8 grid.

    Args:
        circuit: NeuralCircuit
        grid_8x8: 8x8 numpy array
        char_map: Optional character mapping (label_idx -> character)

    Returns:
        Tuple of (predicted_neuron_id, predicted_char, all_voltages)
    """
    # Flatten and pad to 64 channels
    flat_grid = grid_8x8.flatten()

    # Scale input
    input_spikes = flat_grid * 120.0

    # Baseline current
    I_ext = np.ones(circuit.num_neurons) * 15.0

    # Forward pass (no learning)
    output_spikes = circuit.step(
        input_spikes=input_spikes,
        I_ext=I_ext,
        learning=False
    )

    # Get voltages
    voltages = np.array([circuit.neurons[i].v for i in range(circuit.num_neurons)])

    # Find highest voltage (most activated neuron)
    predicted_idx = np.argmax(voltages)

    # Map to character if mapping provided
    predicted_char = None
    if char_map and predicted_idx in char_map:
        predicted_char = char_map[predicted_idx]

    return predicted_idx, predicted_char, voltages


# ============================================================================
# Streamlit App
# ============================================================================

def main():
    """Main Streamlit application."""

    # Title
    st.title("🧠 Nuron Mission Control")
    st.markdown("*Real-time neural network training and testing dashboard*")
    st.markdown("---")

    # Load brain
    try:
        if 'brain' not in st.session_state:
            st.session_state.brain = load_brain(BRAIN_FILE)
            st.session_state.brain_loaded = True
    except Exception as e:
        st.error(f"❌ Error loading brain: {e}")
        st.info("Creating new brain...")
        st.session_state.brain = load_brain(BRAIN_FILE)
        st.session_state.brain_loaded = True

    brain = st.session_state.brain

    # ========== Sidebar: Brain Status & Hardware Info ==========
    with st.sidebar:
        st.header("📊 Brain Status")

        # Get brain info
        info = get_brain_info(brain)
        age = get_brain_age()

        # Brain metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Neurons", info['num_neurons'])
            st.metric("Inputs", info['input_channels'])
        with col2:
            st.metric("Weights", info['total_weights'])
            st.metric("Age", f"{age} sessions")

        # Weight statistics
        st.subheader("Weight Stats")
        st.write(f"**Range**: [{info['weight_stats']['min']:.2f}, {info['weight_stats']['max']:.2f}]")
        st.write(f"**Mean**: {info['weight_stats']['mean']:.3f}")
        st.write(f"**Std**: {info['weight_stats']['std']:.3f}")

        st.markdown("---")

        # Hardware info
        st.subheader("💻 Hardware Info")
        hw_info = get_hardware_info()

        st.write(f"**CPU**: {hw_info['cpu']}")
        st.write(f"**GPU**: {hw_info['gpu']}")
        if hw_info['gpu'] == 'Available':
            st.write(f"**GPU Name**: {hw_info['gpu_name']}")
            st.write(f"**GPU Memory**: {hw_info['gpu_memory']}")

        st.markdown("---")

        # File info
        if os.path.exists(BRAIN_FILE):
            file_size = os.path.getsize(BRAIN_FILE) / 1024
            st.caption(f"📁 Brain file: {file_size:.2f} KB")

    # ========== Main Content: Tabs ==========
    tab1, tab2 = st.tabs(["🎓 Training", "🧪 Testing"])

    # ========== Tab 1: Training ==========
    with tab1:
        st.header("🎓 Relentless Training")
        st.markdown("Train the brain with auto-tuning until mastery is achieved!")

        # Training configuration
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Training Configuration")

            # Task selection
            task_type = st.selectbox(
                "Select Task",
                ["Alphabet (A-Z)", "Digits (0-9)", "Custom (A,B,C)"]
            )

            # Parameters
            target_acc = st.slider("Target Accuracy", 0.5, 0.95, 0.75, 0.05)
            dataset_size = st.number_input("Dataset Size", 500, 5000, 1000, 100)

        with col2:
            st.subheader("Status")
            if 'training_running' in st.session_state and st.session_state.training_running:
                st.info("🟢 Training in progress...")
            else:
                st.success("🟡 Ready to train")

        # Start training button
        if st.button("🚀 Start Relentless Training", type="primary", use_container_width=True):
            st.session_state.training_running = True
            st.session_state.training_history = {
                'epochs': [],
                'accuracy': [],
                'loss': [],
                'best_accuracy': []
            }

            # Create placeholder containers
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)

            with metrics_col1:
                epoch_metric = st.empty()
            with metrics_col2:
                acc_metric = st.empty()
            with metrics_col3:
                best_acc_metric = st.empty()
            with metrics_col4:
                loss_metric = st.empty()

            progress_bar = st.progress(0)
            action_box = st.empty()

            # Charts containers
            st.markdown("### 📊 Training Progress")
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                acc_chart_container = st.empty()
            with chart_col2:
                loss_chart_container = st.empty()

            st.markdown("### 🔍 Weight Matrix Evolution")
            weight_heatmap_container = st.empty()

            # Select training function
            if task_type == "Alphabet (A-Z)":
                trainer = train_alphabet(brain, target_acc=target_acc, dataset_size=dataset_size)
            elif task_type == "Digits (0-9)":
                trainer = train_digits(brain, target_acc=target_acc, dataset_size=dataset_size)
            else:  # Custom
                trainer = train_character_recognition(brain, chars=['A', 'B', 'C'],
                                                      target_acc=target_acc, dataset_size=dataset_size)

            # Run training
            try:
                for status in trainer:
                    # Update metrics
                    epoch_metric.metric("Epoch", status['epoch'])
                    acc_metric.metric("Accuracy", f"{status['accuracy']*100:.1f}%")
                    best_acc_metric.metric("Best", f"{status['best_accuracy']*100:.1f}%")
                    loss_metric.metric("Loss", f"{status['loss']:.3f}")

                    # Update progress bar
                    progress_bar.progress(status['progress'])

                    # Show action if any
                    if status.get('action'):
                        if '🚨' in status['action']:
                            action_box.error(status['action'])
                        elif '⏸️' in status['action']:
                            action_box.warning(status['action'])
                        elif '💾' in status['action']:
                            action_box.success(status['action'])
                        elif '🎉' in status['action']:
                            action_box.success(status['action'])
                        else:
                            action_box.info(status['action'])

                    # Update history
                    st.session_state.training_history['epochs'].append(status['epoch'])
                    st.session_state.training_history['accuracy'].append(status['accuracy'])
                    st.session_state.training_history['loss'].append(status['loss'])
                    st.session_state.training_history['best_accuracy'].append(status['best_accuracy'])

                    # Update accuracy chart
                    acc_df = pd.DataFrame({
                        'Epoch': st.session_state.training_history['epochs'],
                        'Accuracy': [a*100 for a in st.session_state.training_history['accuracy']],
                        'Best': [a*100 for a in st.session_state.training_history['best_accuracy']]
                    })

                    fig_acc = go.Figure()
                    fig_acc.add_trace(go.Scatter(x=acc_df['Epoch'], y=acc_df['Accuracy'],
                                                mode='lines+markers', name='Current',
                                                line=dict(color='blue', width=2)))
                    fig_acc.add_trace(go.Scatter(x=acc_df['Epoch'], y=acc_df['Best'],
                                                mode='lines', name='Best',
                                                line=dict(color='green', width=2, dash='dash')))
                    fig_acc.update_layout(title="Accuracy Over Time",
                                        xaxis_title="Epoch",
                                        yaxis_title="Accuracy (%)",
                                        height=300)
                    acc_chart_container.plotly_chart(fig_acc, use_container_width=True)

                    # Update loss chart
                    loss_df = pd.DataFrame({
                        'Epoch': st.session_state.training_history['epochs'],
                        'Loss': st.session_state.training_history['loss']
                    })

                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(x=loss_df['Epoch'], y=loss_df['Loss'],
                                                 mode='lines+markers', name='Loss',
                                                 line=dict(color='red', width=2)))
                    fig_loss.update_layout(title="Loss Over Time",
                                         xaxis_title="Epoch",
                                         yaxis_title="Loss",
                                         height=300)
                    loss_chart_container.plotly_chart(fig_loss, use_container_width=True)

                    # Update weight heatmap every 5 epochs
                    if status['epoch'] % 5 == 0 or status['progress'] >= 1.0:
                        weight_matrix = build_weight_matrix(brain)

                        # Limit display to first 64 inputs for performance
                        display_matrix = weight_matrix[:, :64]

                        fig_weights = px.imshow(
                            display_matrix,
                            labels=dict(x="Input Channel", y="Neuron", color="Weight"),
                            x=[f"In{i}" for i in range(64)],
                            y=[f"N{i}" for i in range(brain.num_neurons)],
                            color_continuous_scale='Viridis',
                            title=f"Weight Matrix at Epoch {status['epoch']}"
                        )
                        fig_weights.update_layout(height=400)
                        weight_heatmap_container.plotly_chart(fig_weights, use_container_width=True)

                    # Small delay for visualization
                    time.sleep(0.05)

                # Training complete
                st.session_state.training_running = False
                increment_brain_age()

                st.success("✅ Training completed successfully!")
                st.balloons()

                # Show final statistics
                st.markdown("### 📈 Final Statistics")
                final_col1, final_col2, final_col3, final_col4 = st.columns(4)

                with final_col1:
                    st.metric("Total Epochs", status['epoch'])
                with final_col2:
                    st.metric("Final Accuracy", f"{status['accuracy']*100:.1f}%")
                with final_col3:
                    st.metric("Best Accuracy", f"{status['best_accuracy']*100:.1f}%")
                with final_col4:
                    if 'log_file' in status:
                        st.caption(f"Log: {os.path.basename(status['log_file'])}")

            except Exception as e:
                st.error(f"❌ Training error: {e}")
                st.session_state.training_running = False
                import traceback
                st.code(traceback.format_exc())

        # Show instructions if not training
        if 'training_running' not in st.session_state or not st.session_state.training_running:
            st.markdown("---")
            st.markdown("### 📝 How to Use")
            st.markdown("""
            1. **Select a task** (Alphabet, Digits, or Custom)
            2. **Set target accuracy** (0.75 recommended for full alphabet)
            3. **Configure dataset size**
            4. **Click "Start Relentless Training"**
            
            **Features**:
            - 🚨 Auto-detects silent brain and boosts sensitivity
            - ⏸️ Detects stalled learning and increases guidance
            - 💾 Auto-saves brain on improvements
            - 📊 Live charts update in real-time
            - 🔍 Weight matrix evolution every 5 epochs
            - 📝 Complete CSV logging in outputs/logs/
            """)

    # ========== Tab 2: Testing ==========
    with tab2:
        st.header("🧪 Interactive Testing")
        st.markdown("Draw an 8×8 character and test the brain's prediction!")

        # Initialize grid in session state
        if 'test_grid' not in st.session_state:
            st.session_state.test_grid = np.zeros((8, 8), dtype=int)

        # Drawing board
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("✏️ Drawing Board (8×8)")

            # Create 8x8 grid of buttons
            for i in range(8):
                cols = st.columns(8)
                for j in range(8):
                    with cols[j]:
                        # Button label (empty or filled square)
                        label = "⬛" if st.session_state.test_grid[i, j] == 1 else "⬜"

                        # Toggle button
                        if st.button(label, key=f"cell_{i}_{j}", use_container_width=True):
                            st.session_state.test_grid[i, j] = 1 - st.session_state.test_grid[i, j]
                            st.rerun()

            # Control buttons
            st.markdown("")
            btn_col1, btn_col2, btn_col3 = st.columns(3)

            with btn_col1:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.test_grid = np.zeros((8, 8), dtype=int)
                    st.rerun()

            with btn_col2:
                if st.button("🔄 Invert", use_container_width=True):
                    st.session_state.test_grid = 1 - st.session_state.test_grid
                    st.rerun()

            with btn_col3:
                if st.button("🎲 Random", use_container_width=True):
                    st.session_state.test_grid = np.random.randint(0, 2, size=(8, 8))
                    st.rerun()

        with col2:
            st.subheader("🔮 Prediction")

            # Predict button
            if st.button("🚀 Predict Character", type="primary", use_container_width=True):
                # Predict
                predicted_id, predicted_char, voltages = predict_from_grid(
                    brain,
                    st.session_state.test_grid
                )

                # Store results
                st.session_state.last_prediction = {
                    'neuron_id': predicted_id,
                    'char': predicted_char,
                    'voltages': voltages
                }

            # Show prediction results
            if 'last_prediction' in st.session_state:
                pred = st.session_state.last_prediction

                st.markdown("### 📊 Results")

                # Show predicted neuron
                st.metric("Predicted Neuron", f"Neuron {pred['neuron_id']}")

                # Show character if available
                if pred['char']:
                    st.markdown(f"### Predicted Character: **{pred['char']}**")

                # Show voltage bar chart
                st.markdown("#### Neuron Voltages")

                voltage_df = pd.DataFrame({
                    'Neuron': [f"N{i}" for i in range(len(pred['voltages']))],
                    'Voltage': pred['voltages']
                })

                fig_voltages = go.Figure()
                fig_voltages.add_trace(go.Bar(
                    x=voltage_df['Neuron'],
                    y=voltage_df['Voltage'],
                    marker_color=['red' if i == pred['neuron_id'] else 'lightblue'
                                for i in range(len(pred['voltages']))]
                ))
                fig_voltages.update_layout(
                    title="Neuron Activation Levels",
                    xaxis_title="Neuron",
                    yaxis_title="Voltage (mV)",
                    height=300
                )
                st.plotly_chart(fig_voltages, use_container_width=True)

        # Quick load templates
        st.markdown("---")
        st.subheader("📝 Quick Load Templates")
        st.markdown("Load perfect character templates for testing:")

        # Character selector
        template_chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

        # Create buttons in rows
        num_cols = 9
        for row_start in range(0, len(template_chars), num_cols):
            cols = st.columns(num_cols)
            for idx, char in enumerate(template_chars[row_start:row_start+num_cols]):
                with cols[idx]:
                    if st.button(char, key=f"template_{char}", use_container_width=True):
                        # Load template
                        try:
                            bitmap = get_character_bitmap(char)
                            st.session_state.test_grid = bitmap.reshape(8, 8)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error loading {char}: {e}")

    # Footer
    st.markdown("---")
    st.caption("🧠 Nuron Framework - Mission Control Dashboard | Built with Streamlit")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    main()
