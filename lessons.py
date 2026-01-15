"""
Lessons: Modular Training Tasks for Neural Circuits

Provides reusable training functions that map specific tasks to brain regions.
Each lesson teaches the circuit a different skill by utilizing specific
input/output channel mappings.

Lessons:
- train_reader: Character recognition (A, B, C) using channels 0-24 → 0-2
- train_hunter: Sensory-motor navigation using channels 25-28 → 3-6

Key Concept: Channel Mapping
Different brain regions handle different tasks by mapping to specific
input and output channels, similar to cortical specialization in biology.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit import NeuralCircuit
from neuro_gym import NeuroGym


# ============================================================================
# Lesson 1: Character Recognition (Reader)
# ============================================================================

def create_letter_patterns():
    """
    Create 5x5 binary patterns for letters A, B, C.

    Returns:
        Dictionary mapping letter names to 5x5 arrays
    """
    patterns = {}

    # Letter A: Pointy top, horizontal bar, legs at bottom
    patterns['A'] = np.array([
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1]
    ])

    # Letter B: Flat top/bottom, middle bar, rounded right side
    patterns['B'] = np.array([
        [1, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 0]
    ])

    # Letter C: Top/bottom horizontal bars, left vertical bar
    patterns['C'] = np.array([
        [0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1]
    ])

    return patterns


def add_noise(pattern, noise_level=0.05):
    """
    Add noise to pattern by randomly flipping pixels.

    Args:
        pattern: 5x5 binary array
        noise_level: Probability of flipping each pixel

    Returns:
        Noisy pattern
    """
    noisy = pattern.copy()
    flip_mask = np.random.rand(*pattern.shape) < noise_level
    noisy[flip_mask] = 1 - noisy[flip_mask]
    return noisy


def generate_reader_dataset(num_samples=1000, noise_level=0.05):
    """
    Generate dataset for character recognition task.

    Args:
        num_samples: Number of training samples
        noise_level: Probability of pixel flip

    Returns:
        task_data: Dictionary with 'inputs' and 'labels'
    """
    patterns = create_letter_patterns()
    letters = ['A', 'B', 'C']

    inputs = []
    labels = []

    for _ in range(num_samples):
        # Randomly select letter
        letter = np.random.choice(letters)
        label = letters.index(letter)

        # Get pattern and add noise
        pattern = patterns[letter]
        noisy_pattern = add_noise(pattern, noise_level)

        # Flatten to 1D (25 values)
        flat_pattern = noisy_pattern.flatten()

        inputs.append(flat_pattern)
        labels.append(label)

    return {
        'inputs': np.array(inputs),
        'labels': np.array(labels)
    }


def visualize_reader_receptive_fields(circuit, output_neurons):
    """
    Visualize receptive fields for reader neurons.

    Args:
        circuit: Trained NeuralCircuit
        output_neurons: List of neuron IDs to visualize (e.g., [0, 1, 2])
    """
    fig, axes = plt.subplots(1, len(output_neurons), figsize=(12, 4))
    if len(output_neurons) == 1:
        axes = [axes]

    fig.suptitle('Learned Character Receptive Fields (Reader Task)',
                 fontsize=14, fontweight='bold')

    letters = ['A', 'B', 'C']

    for idx, neuron_id in enumerate(output_neurons):
        ax = axes[idx]

        # Get weights (25 values for reader task)
        weights = circuit.get_weights(neuron_id)

        # Reshape to 5x5
        if len(weights) >= 25:
            receptive_field = weights[:25].reshape(5, 5)
        else:
            receptive_field = np.zeros((5, 5))

        # Plot heatmap
        im = ax.imshow(receptive_field, cmap='RdBu_r', interpolation='nearest',
                      vmin=0, vmax=np.max(weights) if np.max(weights) > 0 else 1)

        letter = letters[idx] if idx < len(letters) else '?'
        ax.set_title(f"Neuron {neuron_id}: Letter {letter}",
                    fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

        # Add grid
        for x in range(6):
            ax.axhline(x - 0.5, color='black', linewidth=0.5, alpha=0.3)
            ax.axvline(x - 0.5, color='black', linewidth=0.5, alpha=0.3)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Create outputs directory if needed
    os.makedirs('outputs', exist_ok=True)

    plt.savefig('outputs/lesson_reader_receptive_fields.png', dpi=150, bbox_inches='tight')
    print("  Receptive fields saved to 'outputs/lesson_reader_receptive_fields.png'")
    plt.close()


def train_reader(
    circuit: NeuralCircuit,
    input_channels: tuple = (0, 24),  # Channels 0-24 (25 channels)
    output_neurons: tuple = (0, 2),    # Neurons 0-2 (3 neurons)
    num_samples: int = 1000,
    noise_level: float = 0.05,
    target_acc: float = 0.95,
    max_epochs: int = 100,
    verbose: bool = True
) -> float:
    """
    Train character recognition task (A, B, C).

    This lesson teaches the circuit to recognize 5x5 pixel patterns of letters.

    Channel Mapping:
    - Inputs: Channels 0-24 (flattened 5x5 grid)
    - Outputs: Neurons 0-2 (A, B, C respectively)

    Args:
        circuit: NeuralCircuit to train
        input_channels: Tuple (start, end) of input channels to use
        output_neurons: Tuple (start, end) of output neurons to use
        num_samples: Number of training samples
        noise_level: Pixel flip probability
        target_acc: Target accuracy for convergence
        max_epochs: Maximum training epochs
        verbose: Print progress

    Returns:
        Final accuracy achieved

    Example:
        ```python
        from brain_io import load_brain
        from lessons import train_reader

        brain = load_brain('my_brain.pkl')
        accuracy = train_reader(brain)
        print(f"Reader accuracy: {accuracy*100:.1f}%")
        ```
    """
    if verbose:
        print("\n" + "="*70)
        print("LESSON 1: CHARACTER RECOGNITION (READER)")
        print("="*70)
        print(f"Task: Recognize letters A, B, C from 5x5 pixel patterns")
        print(f"Input channels: {input_channels[0]}-{input_channels[1]} (25 channels)")
        print(f"Output neurons: {output_neurons[0]}-{output_neurons[1]} (3 neurons)")
        print()

    # Verify circuit has required capacity
    input_start, input_end = input_channels
    output_start, output_end = output_neurons
    required_inputs = input_end - input_start + 1
    required_outputs = output_end - output_start + 1

    if circuit.input_channels < required_inputs:
        raise ValueError(f"Circuit has only {circuit.input_channels} input channels, "
                        f"but reader task needs {required_inputs}")

    if circuit.num_neurons < required_outputs:
        raise ValueError(f"Circuit has only {circuit.num_neurons} neurons, "
                        f"but reader task needs {required_outputs}")

    # Generate dataset
    if verbose:
        print("Generating training dataset...")

    task_data = generate_reader_dataset(num_samples, noise_level)

    if verbose:
        print(f"  Samples: {num_samples}")
        print(f"  Noise level: {noise_level*100:.0f}%")
        print(f"  Input shape: {task_data['inputs'].shape}")
        print()

    # Create mapped dataset (pad inputs to match circuit channels)
    mapped_inputs = np.zeros((num_samples, circuit.input_channels))
    mapped_inputs[:, input_start:input_end+1] = task_data['inputs']

    # Adjust labels to match output neuron mapping
    mapped_labels = task_data['labels'] + output_start

    mapped_task_data = {
        'inputs': mapped_inputs,
        'labels': mapped_labels
    }

    # Initialize NeuroGym
    if verbose:
        print("Initializing NeuroGym trainer...")

    gym = NeuroGym(
        circuit=circuit,
        task_data=mapped_task_data,
        input_scale=120.0,      # Moderate input strength
        teacher_current=200.0,  # Strong teacher forcing
        baseline_current=15.0   # Moderate baseline
    )

    # Train until convergence
    if verbose:
        print("Training...")
        print()

    final_acc = gym.train_until_converged(
        target_acc=target_acc,
        max_epochs=max_epochs,
        eval_every=5,
        early_stop_patience=20,
        verbose=verbose
    )

    # Visualize receptive fields
    if verbose:
        print("\nVisualizing learned receptive fields...")

    output_neuron_ids = list(range(output_start, output_end + 1))
    visualize_reader_receptive_fields(circuit, output_neuron_ids)

    if verbose:
        print("\n" + "="*70)
        print("READER LESSON COMPLETE")
        print("="*70)
        print(f"Final Accuracy: {final_acc*100:.1f}%")
        print(f"Trained neurons: {output_neuron_ids}")
        print(f"Input channels: {input_start}-{input_end}")
        print("="*70)

    return final_acc


# ============================================================================
# Lesson 2: Sensory-Motor Navigation (Hunter)
# ============================================================================

class GridWorld:
    """Simple grid world for hunter navigation task."""

    def __init__(self, size=10):
        self.size = size
        self.agent_pos = [size // 2, size // 2]
        self.food_pos = None
        self.spawn_food()

    def spawn_food(self):
        """Spawn food at random location."""
        import random
        while True:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            if [x, y] != self.agent_pos:
                self.food_pos = [x, y]
                break

    def get_sensor_activations(self):
        """Get which direction sensors should activate (N, S, E, W)."""
        dx = self.food_pos[0] - self.agent_pos[0]
        dy = self.food_pos[1] - self.agent_pos[1]

        sensors = [0, 0, 0, 0]  # N, S, E, W

        if dy < 0:  # North
            sensors[0] = 1
        elif dy > 0:  # South
            sensors[1] = 1

        if dx > 0:  # East
            sensors[2] = 1
        elif dx < 0:  # West
            sensors[3] = 1

        return sensors

    def move_agent(self, motor_id):
        """Move agent based on motor action."""
        if motor_id == 0:  # North
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif motor_id == 1:  # South
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        elif motor_id == 2:  # East
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        elif motor_id == 3:  # West
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)

    def distance_to_food(self):
        """Manhattan distance to food."""
        return abs(self.food_pos[0] - self.agent_pos[0]) + \
               abs(self.food_pos[1] - self.agent_pos[1])


def train_hunter(
    circuit: NeuralCircuit,
    sensor_channels: tuple = (25, 28),  # Channels 25-28 (4 sensors)
    motor_neurons: tuple = (3, 6),      # Neurons 3-6 (4 motors)
    num_steps: int = 500,
    verbose: bool = True
) -> float:
    """
    Train sensory-motor navigation task (Hunter).

    This lesson teaches the circuit to navigate toward food using directional
    sensors and motor actions.

    Channel Mapping:
    - Inputs: Channels 25-28 (N, S, E, W sensors)
    - Outputs: Neurons 3-6 (N, S, E, W motors)

    Args:
        circuit: NeuralCircuit to train
        sensor_channels: Tuple (start, end) of sensor input channels
        motor_neurons: Tuple (start, end) of motor output neurons
        num_steps: Number of training steps
        verbose: Print progress

    Returns:
        Final accuracy (fraction of correct movements)

    Example:
        ```python
        from brain_io import load_brain
        from lessons import train_hunter

        brain = load_brain('my_brain.pkl')
        accuracy = train_hunter(brain)
        print(f"Hunter accuracy: {accuracy*100:.1f}%")
        ```
    """
    if verbose:
        print("\n" + "="*70)
        print("LESSON 2: SENSORY-MOTOR NAVIGATION (HUNTER)")
        print("="*70)
        print(f"Task: Navigate toward food using directional sensors")
        print(f"Sensor channels: {sensor_channels[0]}-{sensor_channels[1]} (4 channels)")
        print(f"Motor neurons: {motor_neurons[0]}-{motor_neurons[1]} (4 neurons)")
        print()

    # Verify circuit capacity
    sensor_start, sensor_end = sensor_channels
    motor_start, motor_end = motor_neurons
    required_sensors = sensor_end - sensor_start + 1
    required_motors = motor_end - motor_start + 1

    if circuit.input_channels < sensor_end + 1:
        raise ValueError(f"Circuit has only {circuit.input_channels} input channels, "
                        f"but hunter needs channel {sensor_end}")

    if circuit.num_neurons < motor_end + 1:
        raise ValueError(f"Circuit has only {circuit.num_neurons} neurons, "
                        f"but hunter needs neuron {motor_end}")

    # Create environment
    world = GridWorld(size=10)

    # Training phase (Teacher Forcing)
    if verbose:
        print("Training Phase: Teacher Forcing")
        print("-" * 70)

    correct_moves = 0
    total_moves = 0

    for step in range(num_steps):
        # Spawn new food
        world.spawn_food()

        # Get sensor activations
        sensors = world.get_sensor_activations()

        # Create input pattern (map sensors to correct channels)
        input_pattern = np.zeros(circuit.input_channels)
        for i, sensor_val in enumerate(sensors):
            input_pattern[sensor_start + i] = sensor_val

        # Scale input
        input_spikes = input_pattern * 120.0

        # Teacher forcing: Stimulate correct motor neurons
        I_ext = np.ones(circuit.num_neurons) * 15.0
        for i, sensor_val in enumerate(sensors):
            if sensor_val == 1:
                motor_id = motor_start + i
                I_ext[motor_id] = 200.0  # Force correct motor

        # Forward pass with STDP
        output_spikes = circuit.step(
            input_spikes=input_spikes,
            I_ext=I_ext,
            learning=True
        )

        # Check if correct motor fired
        motor_spikes = output_spikes[motor_start:motor_end+1]
        if np.any(motor_spikes):
            fired_motor = np.where(motor_spikes)[0][0]
            expected_motors = [i for i, s in enumerate(sensors) if s == 1]
            if fired_motor in expected_motors:
                correct_moves += 1
            total_moves += 1

        # Progress
        if verbose and (step + 1) % 100 == 0:
            acc = correct_moves / total_moves if total_moves > 0 else 0.0
            print(f"  Step {step+1:4d}: Accuracy so far: {acc*100:.1f}%")

    training_acc = correct_moves / total_moves if total_moves > 0 else 0.0

    if verbose:
        print(f"\nTraining complete!")
        print(f"  Training accuracy: {training_acc*100:.1f}%")
        print()

    # Testing phase (Autonomous)
    if verbose:
        print("Testing Phase: Autonomous Navigation")
        print("-" * 70)

    test_correct = 0
    test_total = 0
    num_test = 50

    for test_step in range(num_test):
        # Reset and spawn food
        world.agent_pos = [world.size // 2, world.size // 2]
        world.spawn_food()

        initial_distance = world.distance_to_food()

        # Get sensors
        sensors = world.get_sensor_activations()

        # Create input (no teacher forcing)
        input_pattern = np.zeros(circuit.input_channels)
        for i, sensor_val in enumerate(sensors):
            input_pattern[sensor_start + i] = sensor_val

        input_spikes = input_pattern * 120.0
        I_ext = np.ones(circuit.num_neurons) * 15.0

        # Forward pass (no learning)
        output_spikes = circuit.step(
            input_spikes=input_spikes,
            I_ext=I_ext,
            learning=False
        )

        # Check motor response
        motor_spikes = output_spikes[motor_start:motor_end+1]
        if np.any(motor_spikes):
            fired_motor = np.where(motor_spikes)[0][0]
            world.move_agent(fired_motor)

            final_distance = world.distance_to_food()

            if final_distance < initial_distance:
                test_correct += 1
                result = "✓"
            else:
                result = "✗"

            test_total += 1

            if verbose and test_step < 10:
                sensor_labels = ['N', 'S', 'E', 'W']
                active = [sensor_labels[i] for i, s in enumerate(sensors) if s == 1]
                motor = sensor_labels[fired_motor]
                print(f"  Test {test_step+1:2d}: Sensor {active} → Motor {motor}, "
                      f"Dist {initial_distance}→{final_distance} {result}")

    test_acc = test_correct / test_total if test_total > 0 else 0.0

    if verbose:
        print(f"\nTesting complete!")
        print(f"  Test accuracy: {test_acc*100:.1f}% ({test_correct}/{test_total})")
        print()
        print("="*70)
        print("HUNTER LESSON COMPLETE")
        print("="*70)
        print(f"Training Accuracy: {training_acc*100:.1f}%")
        print(f"Test Accuracy: {test_acc*100:.1f}%")
        print(f"Trained neurons: {list(range(motor_start, motor_end+1))}")
        print(f"Sensor channels: {sensor_start}-{sensor_end}")
        print("="*70)

    return test_acc


# ============================================================================
# Lesson Management
# ============================================================================

def list_available_lessons():
    """Print information about available training lessons."""
    print("\n" + "="*70)
    print("AVAILABLE TRAINING LESSONS")
    print("="*70)

    print("\n1. READER (Character Recognition)")
    print("   Task: Recognize letters A, B, C from 5x5 pixel patterns")
    print("   Inputs: Channels 0-24 (25 channels)")
    print("   Outputs: Neurons 0-2 (3 neurons)")
    print("   Usage: train_reader(circuit)")

    print("\n2. HUNTER (Sensory-Motor Navigation)")
    print("   Task: Navigate toward food using directional sensors")
    print("   Inputs: Channels 25-28 (4 channels)")
    print("   Outputs: Neurons 3-6 (4 neurons)")
    print("   Usage: train_hunter(circuit)")

    print("\n" + "="*70)
    print("Total Requirements: 29 input channels, 7 output neurons")
    print("Default brain (64 inputs, 16 neurons) can handle both lessons!")
    print("="*70)


# ============================================================================
# Demo
# ============================================================================

def demo_lessons():
    """Demonstration of modular lesson system."""
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*24 + "LESSONS DEMO" + " "*32 + "║")
    print("║" + " "*18 + "Modular Training Tasks" + " "*28 + "║")
    print("╚" + "═"*68 + "╝")

    # Show available lessons
    list_available_lessons()

    # Load or create brain
    print("\nLoading brain...")
    from brain_io import load_brain, save_brain

    brain = load_brain('demo_lessons_brain.pkl')

    # Train reader lesson
    print("\nTraining READER lesson...")
    reader_acc = train_reader(brain, target_acc=0.80, max_epochs=50)

    # Save progress
    save_brain(brain, 'demo_lessons_brain.pkl')

    # Train hunter lesson
    print("\nTraining HUNTER lesson...")
    hunter_acc = train_hunter(brain, num_steps=200)

    # Save final brain
    save_brain(brain, 'demo_lessons_brain.pkl')

    # Summary
    print("\n" + "="*70)
    print("LESSONS DEMO COMPLETE")
    print("="*70)
    print(f"Reader accuracy: {reader_acc*100:.1f}%")
    print(f"Hunter accuracy: {hunter_acc*100:.1f}%")
    print(f"\nBrain trained on multiple tasks using different brain regions!")
    print("="*70)

    # Cleanup
    if os.path.exists('demo_lessons_brain.pkl'):
        os.remove('demo_lessons_brain.pkl')
        print("\nCleaned up demo files.")


if __name__ == "__main__":
    # Set random seed
    np.random.seed(42)

    # Run demo
    demo_lessons()

def train_alphabet(
    circuit: NeuralCircuit,
    target_acc: float = 0.85,
    dataset_size: int = 1000,
    **kwargs
):
    """
    Train the circuit to recognize the full alphabet (A-Z).
    This is a wrapper around the smart trainer.
    """
    from smart_trainer import train_reader_relentless
    from data_factory import generate_alphabet_dataset
    
    # Generate dataset
    task_data = generate_alphabet_dataset(size=dataset_size)
    
    # Train relentlessly
    yield from train_reader_relentless(
        circuit=circuit,
        task_data=task_data,
        target_acc=target_acc,
        **kwargs
    )

def train_digits(
    circuit: NeuralCircuit,
    target_acc: float = 0.85,
    dataset_size: int = 1000,
    **kwargs
):
    """
    Train the circuit to recognize digits (0-9).
    This is a wrapper around the smart trainer.
    """
    from smart_trainer import train_digits_relentless
    from data_factory import generate_digits_dataset
    
    # Generate dataset
    task_data = generate_digits_dataset(size=dataset_size)
    
    # Train relentlessly
    yield from train_digits_relentless(
        circuit=circuit,
        task_data=task_data,
        target_acc=target_acc,
        **kwargs
    )

def train_character_recognition(
    circuit: NeuralCircuit,
    chars: list = ['A', 'B', 'C'],
    target_acc: float = 0.85,
    dataset_size: int = 1000,
    **kwargs
):
    """
    Train the circuit to recognize a custom set of characters.
    This is a wrapper around the smart trainer.
    """
    from smart_trainer import train_until_mastery
    from data_factory import generate_dataset
    
    # Generate dataset
    task_data = generate_dataset(chars, size=dataset_size)
    
    # Train relentlessly
    yield from train_until_mastery(
        circuit=circuit,
        task_data=task_data,
        target_acc=target_acc,
        task_name=f"Custom_{''.join(chars)}",
        **kwargs
    )


def train_with_real_emnist(
    circuit: NeuralCircuit,
    characters: list = None,
    target_acc: float = 0.85,
    max_samples_per_class: int = 500,
    use_augmentation: bool = True,
    **kwargs
):
    """
    Train the circuit with REAL handwritten characters from EMNIST dataset.

    This fetches authentic handwritten character data from HuggingFace,
    providing much better real-world recognition capability.

    Args:
        circuit: NeuralCircuit to train
        characters: List of characters to train on (default: A-Z)
                   Examples: ['A','B','C'] or ['0','1','2','3','4','5','6','7','8','9']
        target_acc: Target accuracy (default: 0.85)
        max_samples_per_class: Samples per character (default: 500)
        use_augmentation: Add noise augmentation (default: True)
        **kwargs: Additional arguments for train_until_mastery

    Yields:
        Training status updates

    Example:
        >>> # Train on uppercase letters A-Z
        >>> for status in train_with_real_emnist(brain, characters=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')):
        ...     print(f"Epoch {status['epoch']}: {status['accuracy']:.2%}")

        >>> # Train on digits 0-9
        >>> for status in train_with_real_emnist(brain, characters=list('0123456789')):
        ...     print(f"Epoch {status['epoch']}: {status['accuracy']:.2%}")
    """
    from smart_trainer import train_until_mastery
    from dataset_loader import load_emnist_dataset, augment_dataset

    # Default to uppercase alphabet if not specified
    if characters is None:
        characters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    # Load real EMNIST dataset
    print(f"\n{'='*70}")
    print("TRAINING WITH REAL EMNIST HANDWRITTEN DATA")
    print(f"{'='*70}")
    print(f"Characters: {characters}")
    print(f"Samples per character: {max_samples_per_class}")
    print(f"Target accuracy: {target_acc:.1%}")
    print()

    dataset = load_emnist_dataset(
        split='train',
        characters=characters,
        max_samples_per_class=max_samples_per_class
    )

    inputs = dataset['inputs']
    labels = dataset['labels']
    char_map = dataset['char_map']
    label_map = dataset['label_map']

    # Apply augmentation if requested
    if use_augmentation:
        print("[*] Applying data augmentation (noise injection)...")
        inputs, labels = augment_dataset(inputs, labels, noise_level=0.08)

    # Shuffle data
    indices = np.random.permutation(len(inputs))
    inputs = inputs[indices]
    labels = labels[indices]

    # Prepare task data
    task_data = {
        'inputs': inputs,
        'labels': labels,
        'char_map': char_map,
        'label_map': label_map
    }

    task_name = f"EMNIST_{''.join(characters[:3])}{'...' if len(characters) > 3 else ''}"

    print(f"\n[*] Starting training with {len(inputs)} real handwritten samples!")
    print(f"[*] Task: {task_name}\n")

    # Train relentlessly with the smart trainer
    yield from train_until_mastery(
        circuit=circuit,
        task_data=task_data,
        target_acc=target_acc,
        task_name=task_name,
        **kwargs
    )


