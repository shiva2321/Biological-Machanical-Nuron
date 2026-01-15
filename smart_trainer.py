"""
Smart Relentless Trainer: Training Engine That Refuses to Quit

Trains neural networks until mastery is achieved with:
- Relentless training loop (never gives up)
- Auto-tuning (detects and fixes silent brain, stalled learning)
- Complete logging (CSV logs of every epoch)
- Live progress updates (generator-based for UI)
- Automatic brain persistence (saves on improvements)

This trainer WILL NOT STOP until the target accuracy is reached!
"""

import numpy as np
import os
import sys
import csv
import time
from datetime import datetime
from typing import Generator, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from circuit import NeuralCircuit
from neuro_gym import NeuroGym
from brain_io import save_brain


# ============================================================================
# Training Logger
# ============================================================================

class TrainingLogger:
    """
    Logs every epoch's metrics to CSV file.

    Creates timestamped log files in outputs/logs/ directory with:
    - Epoch number
    - Accuracy
    - Loss
    - Input scale
    - Teacher current
    - Baseline current
    - Total spikes
    - Actions taken
    - Timestamp
    """

    def __init__(self, task_name: str, log_dir: str = 'outputs/logs'):
        """
        Initialize training logger.

        Args:
            task_name: Name of training task (for filename)
            log_dir: Directory to save logs
        """
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Create timestamped filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_task_name = "".join(c if c.isalnum() or c in ('-', '_') else '_'
                                 for c in task_name)
        filename = f"training_{safe_task_name}_{timestamp}.csv"
        self.filepath = os.path.join(log_dir, filename)

        # CSV fieldnames
        self.fieldnames = [
            'epoch',
            'accuracy',
            'best_accuracy',
            'loss',
            'input_scale',
            'teacher_current',
            'baseline_current',
            'total_spikes',
            'avg_spikes',
            'action',
            'timestamp'
        ]

        # Create and initialize CSV file with utf-8 encoding
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

        print(f"[*] Training log: {self.filepath}")

    def log_epoch(self,
                  epoch: int,
                  accuracy: float,
                  best_accuracy: float,
                  loss: float,
                  input_scale: float,
                  teacher_current: float,
                  baseline_current: float,
                  total_spikes: int,
                  avg_spikes: float,
                  action: str = ""):
        """
        Log a single epoch's metrics.

        Args:
            epoch: Current epoch number
            accuracy: Current accuracy
            best_accuracy: Best accuracy so far
            loss: Training loss
            input_scale: Current input scaling
            teacher_current: Teacher forcing current
            baseline_current: Baseline excitability
            total_spikes: Total spikes in test batch
            avg_spikes: Average spikes per neuron
            action: Any action taken this epoch
        """
        row = {
            'epoch': epoch,
            'accuracy': f"{accuracy:.4f}",
            'best_accuracy': f"{best_accuracy:.4f}",
            'loss': f"{loss:.4f}",
            'input_scale': f"{input_scale:.2f}",
            'teacher_current': f"{teacher_current:.2f}",
            'baseline_current': f"{baseline_current:.2f}",
            'total_spikes': total_spikes,
            'avg_spikes': f"{avg_spikes:.4f}",
            'action': action,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(self.filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

    def get_filepath(self) -> str:
        """Get the full path to the log file."""
        return self.filepath


# ============================================================================
# Relentless Training Engine
# ============================================================================

def train_until_mastery(
    circuit: NeuralCircuit,
    task_data: Dict[str, np.ndarray],
    target_acc: float = 0.90,
    task_name: str = "Task",
    initial_input_scale: float = 120.0,
    initial_teacher_current: float = 200.0,
    initial_baseline_current: float = 15.0,
    brain_save_path: str = 'my_brain.pkl',
    silent_brain_threshold: float = 0.1,
    stall_patience: int = 20,
    scale_boost_factor: float = 1.5,
    teacher_boost_factor: float = 1.2,
    log_dir: str = 'outputs/logs'
) -> Generator[Dict[str, Any], None, float]:
    """
    Train relentlessly until target accuracy is achieved.

    This function WILL NOT STOP until mastery is achieved!

    Features:
    - Infinite training loop (while True)
    - Auto-tunes parameters if brain is silent or stalled
    - Logs every epoch to CSV
    - Saves brain on improvements
    - Yields status for live UI updates

    Auto-Tuning Logic:
    1. Silent Brain (avg_spikes < threshold): Boost input_scale
    2. Stalled Learning (no improvement for N epochs): Boost teacher_current
    3. Best Accuracy: Save brain immediately

    Args:
        circuit: NeuralCircuit to train
        task_data: Dictionary with 'inputs' and 'labels'
        target_acc: Target accuracy (training stops when reached)
        task_name: Name for logging
        initial_input_scale: Starting input sensitivity
        initial_teacher_current: Starting teacher forcing strength
        initial_baseline_current: Starting baseline excitability
        brain_save_path: Where to save brain
        silent_brain_threshold: Avg spikes below this = silent brain
        stall_patience: Epochs without improvement before boosting teacher
        scale_boost_factor: Multiply input_scale by this when silent
        teacher_boost_factor: Multiply teacher_current by this when stalled
        log_dir: Directory for CSV logs

    Yields:
        Status dictionary with current metrics and actions

    Returns:
        Final accuracy achieved

    Example:
        ```python
        for status in train_until_mastery(brain, data, target_acc=0.85):
            print(f"Epoch {status['epoch']}: {status['accuracy']*100:.1f}%")
            if status['action']:
                print(f"  → {status['action']}")
        ```
    """
    # Initialize logger
    logger = TrainingLogger(task_name, log_dir)

    # Training parameters (mutable - will be auto-tuned)
    input_scale = initial_input_scale
    teacher_current = initial_teacher_current
    baseline_current = initial_baseline_current

    # Tracking variables
    best_accuracy = 0.0
    epochs_since_improvement = 0
    epoch = 0

    # Create NeuroGym
    gym = NeuroGym(
        circuit=circuit,
        task_data=task_data,
        input_scale=input_scale,
        teacher_current=teacher_current,
        baseline_current=baseline_current
    )

    # Initial status
    yield {
        'epoch': 0,
        'accuracy': 0.0,
        'best_accuracy': 0.0,
        'loss': 1.0,
        'action': f'[*] Starting relentless training for: {task_name}',
        'input_scale': input_scale,
        'teacher_current': teacher_current,
        'baseline_current': baseline_current,
        'total_spikes': 0,
        'avg_spikes': 0.0,
        'progress': 0.0,
        'log_file': logger.get_filepath()
    }

    # ========== RELENTLESS TRAINING LOOP ==========
    # This loop WILL NOT STOP until target accuracy is reached!

    while True:  # INFINITE LOOP - only breaks when target_acc reached
        epoch += 1

        # Train one epoch
        train_metrics = gym.train_epoch()

        # Evaluate
        eval_acc, eval_metrics = gym.evaluate(verbose=False)

        # Count spikes in test batch
        total_spikes = 0
        test_samples = min(10, len(task_data['labels']))

        for i in range(test_samples):
            input_pattern = task_data['inputs'][i]
            input_spikes = input_pattern * input_scale
            I_ext = np.ones(circuit.num_neurons) * baseline_current

            output_spikes = circuit.step(
                input_spikes=input_spikes,
                I_ext=I_ext,
                learning=False
            )
            total_spikes += np.sum(output_spikes)

        avg_spikes = total_spikes / (test_samples * circuit.num_neurons)

        # Initialize action for this epoch
        action_taken = None
        parameter_changed = False

        # ========== AUTO-TUNING LOGIC ==========

        # 1. SILENT BRAIN DETECTION
        if avg_spikes < silent_brain_threshold:
            old_scale = input_scale
            input_scale *= scale_boost_factor

            # Recreate gym with new parameters
            gym = NeuroGym(
                circuit=circuit,
                task_data=task_data,
                input_scale=input_scale,
                teacher_current=teacher_current,
                baseline_current=baseline_current
            )

            action_taken = (f"[!] SILENT BRAIN! Avg spikes: {avg_spikes:.4f} < {silent_brain_threshold} "
                          f"-> Boosted input_scale: {old_scale:.1f} -> {input_scale:.1f}")
            parameter_changed = True
            print(f"\n{action_taken}")

        # 2. STALLED LEARNING DETECTION
        elif epochs_since_improvement >= stall_patience:
            old_teacher = teacher_current
            teacher_current *= teacher_boost_factor

            # Recreate gym with new parameters
            gym = NeuroGym(
                circuit=circuit,
                task_data=task_data,
                input_scale=input_scale,
                teacher_current=teacher_current,
                baseline_current=baseline_current
            )

            action_taken = (f"[STALLED] No improvement for {stall_patience} epochs "
                          f"-> Boosted teacher_current: {old_teacher:.1f} -> {teacher_current:.1f}")
            parameter_changed = True
            epochs_since_improvement = 0  # Reset counter after intervention
            print(f"\n{action_taken}")

        # 3. IMPROVEMENT DETECTION - SAVE BRAIN
        if eval_acc > best_accuracy:
            best_accuracy = eval_acc
            epochs_since_improvement = 0

            # Save brain immediately
            try:
                save_brain(circuit, brain_save_path)
                if action_taken:
                    action_taken += " | [SAVED] Brain saved (new best!)"
                else:
                    action_taken = f"[SAVED] Brain saved! New best: {best_accuracy*100:.2f}%"
            except Exception as e:
                if action_taken:
                    action_taken += f" | [ERROR] Save failed: {e}"
                else:
                    action_taken = f"[ERROR] Save failed: {e}"
        else:
            epochs_since_improvement += 1

        # Calculate progress
        progress = min(1.0, eval_acc / target_acc) if target_acc > 0 else 0.0

        # Log to CSV
        logger.log_epoch(
            epoch=epoch,
            accuracy=eval_acc,
            best_accuracy=best_accuracy,
            loss=train_metrics['loss'],
            input_scale=input_scale,
            teacher_current=teacher_current,
            baseline_current=baseline_current,
            total_spikes=int(total_spikes),
            avg_spikes=avg_spikes,
            action=action_taken if action_taken else ""
        )

        # Yield status update
        status = {
            'epoch': epoch,
            'accuracy': eval_acc,
            'best_accuracy': best_accuracy,
            'loss': train_metrics['loss'],
            'action': action_taken,
            'input_scale': input_scale,
            'teacher_current': teacher_current,
            'baseline_current': baseline_current,
            'total_spikes': int(total_spikes),
            'avg_spikes': avg_spikes,
            'progress': progress,
            'log_file': logger.get_filepath()
        }

        yield status

        # ========== STOP CONDITION ==========
        # Only way to break the infinite loop: achieve target accuracy!
        if eval_acc >= target_acc:
            final_action = f"[SUCCESS] MASTERY ACHIEVED! Target {target_acc*100:.1f}% reached after {epoch} epochs!"

            # Final log entry
            logger.log_epoch(
                epoch=epoch,
                accuracy=eval_acc,
                best_accuracy=best_accuracy,
                loss=train_metrics['loss'],
                input_scale=input_scale,
                teacher_current=teacher_current,
                baseline_current=baseline_current,
                total_spikes=int(total_spikes),
                avg_spikes=avg_spikes,
                action=final_action
            )

            # Final yield
            yield {
                'epoch': epoch,
                'accuracy': eval_acc,
                'best_accuracy': best_accuracy,
                'loss': train_metrics['loss'],
                'action': final_action,
                'input_scale': input_scale,
                'teacher_current': teacher_current,
                'baseline_current': baseline_current,
                'total_spikes': int(total_spikes),
                'avg_spikes': avg_spikes,
                'progress': 1.0,
                'log_file': logger.get_filepath()
            }

            print(f"\n{'='*70}")
            print(f"{final_action}")
            print(f"{'='*70}")
            print(f"Final accuracy: {eval_acc*100:.2f}%")
            print(f"Best accuracy: {best_accuracy*100:.2f}%")
            print(f"Total epochs: {epoch}")
            print(f"Final input_scale: {input_scale:.1f}")
            print(f"Final teacher_current: {teacher_current:.1f}")
            print(f"Log saved: {logger.get_filepath()}")
            print(f"{'='*70}")

            return eval_acc  # SUCCESS!

        # If we haven't reached target, the loop continues...
        # Progress update every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch:4d}: Acc={eval_acc*100:5.1f}% "
                  f"(Best={best_accuracy*100:5.1f}%) "
                  f"Target={target_acc*100:5.1f}% "
                  f"[{int(progress*100):3d}%]")


# ============================================================================
# Convenience Wrappers for Common Tasks
# ============================================================================

def train_reader_relentless(
    circuit: NeuralCircuit,
    task_data: Dict[str, np.ndarray] = None,
    target_acc: float = 0.85,
    **kwargs
) -> Generator[Dict[str, Any], None, float]:
    """
    Relentless training for character recognition.

    Args:
        circuit: NeuralCircuit to train
        task_data: Optional dataset (generates if None)
        target_acc: Target accuracy
        **kwargs: Additional args for train_until_mastery

    Yields:
        Status updates

    Returns:
        Final accuracy
    """
    # Generate data if not provided
    if task_data is None:
        from data_factory import generate_alphabet_dataset
        task_data = generate_alphabet_dataset(size=2600)
        print(f"[DATA] Generated alphabet dataset: {len(task_data['labels'])} samples")

    # Train relentlessly
    yield from train_until_mastery(
        circuit=circuit,
        task_data=task_data,
        target_acc=target_acc,
        task_name='Reader (Character Recognition)',
        **kwargs
    )


def train_digits_relentless(
    circuit: NeuralCircuit,
    task_data: Dict[str, np.ndarray] = None,
    target_acc: float = 0.85,
    **kwargs
) -> Generator[Dict[str, Any], None, float]:
    """
    Relentless training for digit recognition.

    Args:
        circuit: NeuralCircuit to train
        task_data: Optional dataset (generates if None)
        target_acc: Target accuracy
        **kwargs: Additional args for train_until_mastery

    Yields:
        Status updates

    Returns:
        Final accuracy
    """
    # Generate data if not provided
    if task_data is None:
        from data_factory import generate_digits_dataset
        task_data = generate_digits_dataset(size=1000)
        print(f"[DATA] Generated digits dataset: {len(task_data['labels'])} samples")

    # Train relentlessly
    yield from train_until_mastery(
        circuit=circuit,
        task_data=task_data,
        target_acc=target_acc,
        task_name='Digits (0-9 Recognition)',
        **kwargs
    )


# ============================================================================
# Demo and Testing
# ============================================================================

def demo_relentless_trainer():
    """
    Demonstration of relentless training.
    """
    print("\n" + "="*70)
    print("RELENTLESS TRAINER DEMO")
    print("Training that NEVER gives up until mastery is achieved!")
    print("="*70)
    print()

    # Load brain
    from brain_io import load_brain
    from data_factory import generate_dataset

    print("Loading brain...")
    brain = load_brain('my_brain.pkl')
    print(f"  Loaded: {brain.num_neurons} neurons, {brain.input_channels} inputs")
    print()

    # Generate small dataset for demo
    print("Generating dataset...")
    data = generate_dataset(['A', 'B', 'C'], size=300)
    print(f"  Generated: {len(data['labels'])} samples")
    print(f"  Classes: {data['char_map']}")
    print()

    # Train relentlessly
    print("Starting relentless training...")
    print("This will NOT stop until 70% accuracy is reached!")
    print("="*70)
    print()

    for status in train_until_mastery(
        brain,
        data,
        target_acc=0.70,
        task_name='ABC_Demo'
    ):
        # Just yield - status updates are printed internally
        pass

    print()
    print("Demo complete!")


if __name__ == "__main__":
    # Set random seed
    np.random.seed(42)

    # Run demo
    demo_relentless_trainer()
