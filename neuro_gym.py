"""
NeuroGym: Training Harness for Neural Circuits

A persistent training environment that trains spiking neural circuits until
they master a task. Provides supervised learning with teacher forcing,
evaluation, and convergence monitoring.

Key Features:
- Supervised training with teacher forcing
- Automatic convergence detection
- Progress monitoring with metrics
- Flexible task specification
- STDP-based learning
"""

import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit import NeuralCircuit


class NeuroGym:
    """
    Training harness for Neural Circuits.

    Provides a complete training framework including:
    - Supervised learning with teacher forcing
    - Evaluation on test/validation data
    - Automatic convergence detection
    - Progress monitoring and reporting

    Example:
        ```python
        # Create task data
        task_data = {
            'inputs': np.array([[...], [...]]),  # Shape: (N, input_dim)
            'labels': np.array([0, 1, ...])      # Shape: (N,)
        }

        # Create gym
        gym = NeuroGym(circuit, task_data)

        # Train until convergence
        final_acc = gym.train_until_converged(target_acc=0.9)
        ```
    """

    def __init__(
        self,
        circuit: NeuralCircuit,
        task_data: Dict[str, np.ndarray],
        input_scale: float = 100.0,
        teacher_current: float = 200.0,
        baseline_current: float = 10.0
    ):
        """
        Initialize training environment.

        Args:
            circuit: NeuralCircuit to train
            task_data: Dictionary with 'inputs' and 'labels'
                      inputs: Shape (N_samples, input_channels)
                      labels: Shape (N_samples,) with integer class labels
            input_scale: Scaling factor for input signals
            teacher_current: External current for teacher forcing (mV)
            baseline_current: Baseline excitability current (mV)
        """
        self.circuit = circuit

        # Validate and store task data
        if 'inputs' not in task_data or 'labels' not in task_data:
            raise ValueError("task_data must contain 'inputs' and 'labels' keys")

        self.inputs = np.array(task_data['inputs'])
        self.labels = np.array(task_data['labels'])

        if len(self.inputs) != len(self.labels):
            raise ValueError(f"Inputs ({len(self.inputs)}) and labels ({len(self.labels)}) "
                           f"must have same length")

        self.num_samples = len(self.inputs)
        self.num_classes = len(np.unique(self.labels))

        # Training parameters
        self.input_scale = input_scale
        self.teacher_current = teacher_current
        self.baseline_current = baseline_current

        # Training history
        self.history = {
            'epoch': [],
            'accuracy': [],
            'loss': []
        }

        self.current_epoch = 0

        print(f"NeuroGym initialized:")
        print(f"  Samples: {self.num_samples}")
        print(f"  Classes: {self.num_classes}")
        print(f"  Input dim: {self.inputs.shape[1]}")
        print(f"  Circuit neurons: {circuit.num_neurons}")
        print(f"  Input channels: {circuit.input_channels}")
        print()

    def train_step(self, mode='supervised') -> Dict[str, float]:
        """
        Perform one training step (single sample).

        Process:
        1. Select random sample
        2. Present input to circuit
        3. If supervised: Apply teacher forcing (stimulate correct output)
        4. Allow STDP to update weights
        5. Return metrics

        Args:
            mode: 'supervised' (with teacher forcing) or 'unsupervised'

        Returns:
            Dictionary with step metrics (correct, output_neuron, etc.)
        """
        # Select random sample
        idx = np.random.randint(0, self.num_samples)
        input_pattern = self.inputs[idx]
        target_label = self.labels[idx]

        # Scale input
        input_spikes = input_pattern * self.input_scale

        # Prepare external current
        I_ext = np.ones(self.circuit.num_neurons) * self.baseline_current

        # Teacher forcing: Strongly stimulate correct output neuron
        if mode == 'supervised':
            if target_label < self.circuit.num_neurons:
                I_ext[target_label] = self.teacher_current

        # Forward pass through circuit
        output_spikes = self.circuit.step(
            input_spikes=input_spikes,
            I_ext=I_ext,
            learning=(mode == 'supervised')  # Only learn in supervised mode
        )

        # Determine predicted class (first firing neuron)
        fired_neurons = np.where(output_spikes)[0]
        if len(fired_neurons) > 0:
            predicted_label = fired_neurons[0]
            correct = (predicted_label == target_label)
        else:
            predicted_label = -1  # No neuron fired
            correct = False

        # Calculate loss (0 if correct, 1 if wrong)
        loss = 0.0 if correct else 1.0

        return {
            'correct': correct,
            'predicted': predicted_label,
            'target': target_label,
            'loss': loss,
            'num_fired': len(fired_neurons)
        }

    def evaluate(self, verbose: bool = False) -> Tuple[float, Dict]:
        """
        Evaluate circuit on all samples without teacher forcing.

        Tests learned behavior by presenting inputs and checking
        which output neurons fire (no supervision).

        Args:
            verbose: If True, print per-sample results

        Returns:
            accuracy: Fraction of correct predictions
            metrics: Dictionary with detailed metrics
        """
        correct_count = 0
        predictions = []
        num_fired_list = []

        for idx in range(self.num_samples):
            input_pattern = self.inputs[idx]
            target_label = self.labels[idx]

            # Scale input
            input_spikes = input_pattern * self.input_scale

            # Baseline current only (no teacher forcing)
            I_ext = np.ones(self.circuit.num_neurons) * self.baseline_current

            # Forward pass (no learning)
            output_spikes = self.circuit.step(
                input_spikes=input_spikes,
                I_ext=I_ext,
                learning=False
            )

            # Get prediction
            fired_neurons = np.where(output_spikes)[0]
            if len(fired_neurons) > 0:
                predicted_label = fired_neurons[0]
                correct = (predicted_label == target_label)
            else:
                predicted_label = -1
                correct = False

            if correct:
                correct_count += 1

            predictions.append(predicted_label)
            num_fired_list.append(len(fired_neurons))

            if verbose:
                status = "✓" if correct else "✗"
                print(f"  Sample {idx:2d}: Input={input_pattern[:3]}..., "
                      f"Target={target_label}, Predicted={predicted_label} {status}")

        accuracy = correct_count / self.num_samples

        metrics = {
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_samples': self.num_samples,
            'predictions': np.array(predictions),
            'avg_fired': np.mean(num_fired_list)
        }

        return accuracy, metrics

    def train_epoch(self, num_steps: int = None) -> Dict[str, float]:
        """
        Train for one epoch (multiple steps).

        Args:
            num_steps: Number of training steps. If None, uses num_samples.

        Returns:
            Epoch metrics (average accuracy, loss)
        """
        if num_steps is None:
            num_steps = self.num_samples

        epoch_correct = 0
        epoch_loss = 0.0

        for _ in range(num_steps):
            step_metrics = self.train_step(mode='supervised')
            if step_metrics['correct']:
                epoch_correct += 1
            epoch_loss += step_metrics['loss']

        avg_accuracy = epoch_correct / num_steps
        avg_loss = epoch_loss / num_steps

        return {
            'accuracy': avg_accuracy,
            'loss': avg_loss
        }

    def train_until_converged(
        self,
        target_acc: float = 0.9,
        max_epochs: int = 1000,
        eval_every: int = 1,
        early_stop_patience: int = 50,
        verbose: bool = True
    ) -> float:
        """
        Train circuit until it reaches target accuracy or max epochs.

        Training loop:
        1. Train for one epoch (num_samples steps)
        2. Evaluate on all samples
        3. Check convergence
        4. Repeat until converged or max epochs

        Args:
            target_acc: Target accuracy to reach (0.0 to 1.0)
            max_epochs: Maximum number of epochs
            eval_every: Evaluate every N epochs
            early_stop_patience: Stop if no improvement for N epochs
            verbose: Print progress

        Returns:
            Final accuracy achieved
        """
        if verbose:
            print("="*70)
            print("TRAINING UNTIL CONVERGENCE")
            print("="*70)
            print(f"Target Accuracy: {target_acc*100:.1f}%")
            print(f"Max Epochs: {max_epochs}")
            print(f"Samples per Epoch: {self.num_samples}")
            print()

        best_accuracy = 0.0
        epochs_without_improvement = 0
        converged = False

        for epoch in range(max_epochs):
            self.current_epoch = epoch + 1

            # Train for one epoch
            train_metrics = self.train_epoch()

            # Evaluate periodically
            if (epoch + 1) % eval_every == 0:
                eval_acc, eval_metrics = self.evaluate(verbose=False)

                # Store history
                self.history['epoch'].append(self.current_epoch)
                self.history['accuracy'].append(eval_acc)
                self.history['loss'].append(train_metrics['loss'])

                # Check for improvement
                if eval_acc > best_accuracy:
                    best_accuracy = eval_acc
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += eval_every

                # Progress reporting
                if verbose:
                    # Create progress bar
                    progress = eval_acc / target_acc
                    bar_length = 30
                    filled = int(bar_length * min(progress, 1.0))
                    bar = '█' * filled + '░' * (bar_length - filled)

                    print(f"Epoch {self.current_epoch:4d}: "
                          f"[{bar}] "
                          f"Acc={eval_acc*100:5.1f}% "
                          f"Loss={train_metrics['loss']:.3f} "
                          f"(Best={best_accuracy*100:.1f}%)")

                # Check convergence
                if eval_acc >= target_acc:
                    converged = True
                    if verbose:
                        print()
                        print("="*70)
                        print(f"✓ CONVERGED! Reached target accuracy {target_acc*100:.1f}%")
                        print(f"  Final Accuracy: {eval_acc*100:.1f}%")
                        print(f"  Epochs: {self.current_epoch}")
                        print("="*70)
                    break

                # Check early stopping
                if epochs_without_improvement >= early_stop_patience:
                    if verbose:
                        print()
                        print("="*70)
                        print(f"⚠ EARLY STOPPING (no improvement for {early_stop_patience} epochs)")
                        print(f"  Best Accuracy: {best_accuracy*100:.1f}%")
                        print(f"  Epochs: {self.current_epoch}")
                        print("="*70)
                    break

        # Final evaluation
        final_acc, final_metrics = self.evaluate(verbose=False)

        if verbose and not converged:
            print()
            print("="*70)
            print(f"✗ DID NOT CONVERGE (reached max epochs)")
            print(f"  Final Accuracy: {final_acc*100:.1f}%")
            print(f"  Target Accuracy: {target_acc*100:.1f}%")
            print(f"  Epochs: {self.current_epoch}")
            print("="*70)

        return final_acc

    def get_history(self) -> Dict[str, List]:
        """
        Get training history.

        Returns:
            Dictionary with lists of epoch, accuracy, and loss values
        """
        return self.history.copy()

    def reset_history(self):
        """Reset training history."""
        self.history = {
            'epoch': [],
            'accuracy': [],
            'loss': []
        }
        self.current_epoch = 0

    def save_state(self, filename: str):
        """
        Save training state (history and circuit weights).

        Args:
            filename: Path to save file
        """
        state = {
            'history': self.history,
            'current_epoch': self.current_epoch,
            'weights': [self.circuit.get_weights(i)
                       for i in range(self.circuit.num_neurons)]
        }
        np.save(filename, state, allow_pickle=True)
        print(f"State saved to {filename}")

    def load_state(self, filename: str):
        """
        Load training state (history and circuit weights).

        Args:
            filename: Path to load file
        """
        state = np.load(filename, allow_pickle=True).item()
        self.history = state['history']
        self.current_epoch = state['current_epoch']

        # Restore weights
        for i, weights in enumerate(state['weights']):
            if i < self.circuit.num_neurons:
                self.circuit.set_weights(i, weights)

        print(f"State loaded from {filename}")
        print(f"  Resumed at epoch {self.current_epoch}")
        print(f"  Best accuracy: {max(self.history['accuracy'])*100:.1f}%")


# ============================================================================
# Example Usage and Testing
# ============================================================================

def create_simple_xor_task():
    """
    Create simple XOR-like task for demonstration.

    Returns:
        task_data: Dictionary with inputs and labels
    """
    # 4 samples, 2 inputs, 2 classes
    inputs = np.array([
        [0.0, 0.0],  # Class 0
        [0.0, 1.0],  # Class 1
        [1.0, 0.0],  # Class 1
        [1.0, 1.0]   # Class 0
    ])

    labels = np.array([0, 1, 1, 0])

    return {'inputs': inputs, 'labels': labels}


def create_pattern_recognition_task():
    """
    Create pattern recognition task.

    Returns:
        task_data: Dictionary with inputs and labels
    """
    # 6 samples, 4 inputs, 2 classes
    # Pattern A: [1,1,0,0] and similar
    # Pattern B: [0,0,1,1] and similar

    inputs = np.array([
        [1.0, 1.0, 0.0, 0.0],  # Class 0 (Pattern A)
        [1.0, 0.9, 0.1, 0.0],  # Class 0 (Pattern A variant)
        [0.9, 1.0, 0.0, 0.1],  # Class 0 (Pattern A variant)
        [0.0, 0.0, 1.0, 1.0],  # Class 1 (Pattern B)
        [0.1, 0.0, 0.9, 1.0],  # Class 1 (Pattern B variant)
        [0.0, 0.1, 1.0, 0.9],  # Class 1 (Pattern B variant)
    ])

    labels = np.array([0, 0, 0, 1, 1, 1])

    return {'inputs': inputs, 'labels': labels}


def demo_neuro_gym():
    """
    Demonstration of NeuroGym training framework.
    """
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*22 + "NEUROGYM DEMO" + " "*32 + "║")
    print("║" + " "*17 + "Training Harness for Neural Circuits" + " "*15 + "║")
    print("╚" + "═"*68 + "╝")
    print()

    # Create task
    print("Creating Pattern Recognition Task...")
    task_data = create_pattern_recognition_task()
    print(f"  Inputs shape: {task_data['inputs'].shape}")
    print(f"  Labels shape: {task_data['labels'].shape}")
    print(f"  Classes: {len(np.unique(task_data['labels']))}")
    print()

    # Create circuit
    print("Creating Neural Circuit...")
    circuit = NeuralCircuit(
        num_neurons=2,  # 2 output neurons for 2 classes
        input_channels=4,  # 4 input features
        dt=1.0,
        max_delay=5,
        neuron_params={
            'tau_m': 20.0,
            'tau_trace': 20.0,
            'theta_base': -65.0,
            'v_rest': -70.0,
            'v_reset': -75.0,
            'u_increment': 0.1,  # Reduced adaptation
            'theta_increment': 0.05,  # Reduced threshold adaptation
            'weight_max': 10.0
        }
    )

    # Initialize weights to small random values
    for i in range(circuit.num_neurons):
        circuit.set_weights(i, np.random.rand(4) * 0.1)

    print(f"  Neurons: {circuit.num_neurons}")
    print(f"  Input channels: {circuit.input_channels}")
    print()

    # Create training environment
    print("Initializing NeuroGym...")
    gym = NeuroGym(
        circuit=circuit,
        task_data=task_data,
        input_scale=150.0,  # Strong input
        teacher_current=250.0,  # Strong teacher forcing
        baseline_current=20.0  # Higher baseline
    )

    # Initial evaluation
    print("Initial Performance:")
    init_acc, init_metrics = gym.evaluate(verbose=True)
    print(f"  Accuracy: {init_acc*100:.1f}%")
    print()

    # Train until convergence
    final_acc = gym.train_until_converged(
        target_acc=0.8,
        max_epochs=200,
        eval_every=10,
        early_stop_patience=50,
        verbose=True
    )

    # Final evaluation
    print("\nFinal Performance:")
    final_acc, final_metrics = gym.evaluate(verbose=True)
    print(f"  Accuracy: {final_acc*100:.1f}%")
    print(f"  Average neurons fired: {final_metrics['avg_fired']:.1f}")
    print()

    # Show training history
    history = gym.get_history()
    if len(history['epoch']) > 0:
        print("Training History:")
        print(f"  Epochs: {history['epoch']}")
        print(f"  Accuracy: {[f'{a*100:.1f}%' for a in history['accuracy']]}")
        print()

    print("="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print()
    print("NeuroGym provides:")
    print("  ✓ Supervised learning with teacher forcing")
    print("  ✓ Automatic convergence detection")
    print("  ✓ Progress monitoring with progress bars")
    print("  ✓ Training history tracking")
    print("  ✓ Early stopping to prevent overfitting")
    print("="*70)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Run demonstration
    demo_neuro_gym()

