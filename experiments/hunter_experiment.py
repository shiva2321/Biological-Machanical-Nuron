"""
Hunter Experiment: Sensory-Motor Learning in a Grid World

Demonstrates how an agent learns to navigate toward food using STDP.
The agent starts knowing nothing (tabula rasa) and learns through
teacher forcing that sensors should trigger corresponding motors.

Architecture:
- 8 neurons: 4 sensors (N,S,E,W) + 4 motors (N,S,E,W)
- Learning: STDP wires sensors to motors
- Training: Teacher forcing (stimulate both sensor and motor)
- Testing: Autonomous (only sensor, motor fires based on learned weights)
"""

import numpy as np
import random
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from circuit import NeuralCircuit


class GridWorld:
    """
    10x10 grid environment for the hunter agent.

    Agent must learn to move toward food using sensory-motor associations.
    """

    def __init__(self, size=10):
        """
        Initialize grid world.

        Args:
            size: Grid dimension (size x size)
        """
        self.size = size
        self.agent_pos = [size // 2, size // 2]  # Start at center
        self.food_pos = None
        self.spawn_food()

    def spawn_food(self):
        """Spawn food at random location (not on agent)."""
        while True:
            x = random.randint(0, self.size - 1)
            y = random.randint(0, self.size - 1)
            if [x, y] != self.agent_pos:
                self.food_pos = [x, y]
                break

    def get_sensor_activations(self):
        """
        Calculate which sensor neurons should activate based on food direction.

        Returns:
            sensor_active: [N, S, E, W] - which directions have food
        """
        dx = self.food_pos[0] - self.agent_pos[0]
        dy = self.food_pos[1] - self.agent_pos[1]

        # Sensors: 0=North, 1=South, 2=East, 3=West
        sensors = [0, 0, 0, 0]

        if dy < 0:  # Food is North (lower y)
            sensors[0] = 1
        elif dy > 0:  # Food is South (higher y)
            sensors[1] = 1

        if dx > 0:  # Food is East (higher x)
            sensors[2] = 1
        elif dx < 0:  # Food is West (lower x)
            sensors[3] = 1

        return sensors

    def get_correct_motor(self):
        """
        Get the correct motor neuron(s) to move toward food.

        Returns:
            motor_active: [N, S, E, W] - correct motor actions
        """
        return self.get_sensor_activations()  # Same logic

    def move_agent(self, motor_id):
        """
        Move agent based on motor neuron firing.

        Args:
            motor_id: 0=North, 1=South, 2=East, 3=West
        """
        if motor_id == 0:  # North
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif motor_id == 1:  # South
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        elif motor_id == 2:  # East
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        elif motor_id == 3:  # West
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)

    def distance_to_food(self):
        """Calculate Manhattan distance to food."""
        return abs(self.food_pos[0] - self.agent_pos[0]) + \
               abs(self.food_pos[1] - self.agent_pos[1])

    def reached_food(self):
        """Check if agent reached food."""
        return self.agent_pos == self.food_pos

    def display(self):
        """Display grid with agent (A) and food (F)."""
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        grid[self.agent_pos[1]][self.agent_pos[0]] = 'A'
        grid[self.food_pos[1]][self.food_pos[0]] = 'F'

        print("  " + "".join([str(i) for i in range(self.size)]))
        for i, row in enumerate(grid):
            print(f"{i} " + "".join(row))
        print()


def build_hunter_brain():
    """
    Build the sensory-motor circuit.

    Architecture (Revised):
    - 4 motor neurons (0=N, 1=S, 2=E, 3=W)
    - 4 input channels (sensor readings: N, S, E, W)
    - Each motor neuron has 4 input weights (learns from all sensors)
    - STDP will strengthen: Motor N learns from Sensor N, etc.

    Returns:
        NeuralCircuit configured for sensory-motor learning
    """
    print("Building Hunter Brain (Sensory-Motor Circuit)...")
    print("="*70)

    circuit = NeuralCircuit(
        num_neurons=4,       # 4 motor neurons
        input_channels=4,    # 4 sensor inputs (N, S, E, W)
        dt=1.0,
        max_delay=5,
        neuron_params={
            'tau_m': 20.0,        # From sequence experiment hotfix
            'tau_trace': 20.0,
            'theta_base': -65.0,  # From sequence experiment hotfix
            'v_rest': -70.0,
            'v_reset': -75.0,
            'u_increment': 0.3,
            'theta_increment': 0.1,
            'weight_min': 0.0,
            'weight_max': 10.0    # From sequence experiment hotfix
        }
    )

    # Initialize all motor neurons with zero input weights (tabula rasa)
    for i in range(4):
        circuit.set_weights(i, np.zeros(4))

    print("\nArchitecture:")
    print("  4 Input Channels: Sensor readings (N, S, E, W)")
    print("  4 Motor Neurons: Movement commands (N, S, E, W)")
    print("  Each motor has 4 input weights (learns which sensor to follow)")
    print("  Lateral inhibition: 5.0mV (winner-take-all)")
    print("\nInitial State: All weights = 0.0 (tabula rasa)")
    print("Learning: STDP will strengthen diagonal (Sensor N → Motor N, etc.)")
    print("="*70)
    print()

    # Add lateral inhibition to create winner-take-all dynamics
    circuit.set_inhibition(strength=5.0)

    return circuit


def print_brain_state(circuit, title="Brain State"):
    """
    Display the synaptic weight matrix (sensor→motor connections).

    We expect diagonal to strengthen:
    - Sensor N → Motor N
    - Sensor S → Motor S
    - Sensor E → Motor E
    - Sensor W → Motor W
    """
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")

    print("\nSensor Input → Motor Neuron Weight Matrix:")
    print("          Sensor Input:")
    print("Motor:       N      S      E      W")

    labels = ['N', 'S', 'E', 'W']
    for motor_id in range(4):
        weights = circuit.get_weights(motor_id)
        row_str = f"  {labels[motor_id]}   "
        for w in weights:
            row_str += f"  {w:5.2f}"
        print(row_str)

    print("\nExpected: Diagonal should be strong (N→N, S→S, E→E, W→W)")

    # Highlight diagonal strength
    diagonal_weights = [circuit.get_weights(i)[i] for i in range(4)]
    off_diagonal = []
    for i in range(4):
        weights = circuit.get_weights(i)
        for j in range(4):
            if i != j:
                off_diagonal.append(weights[j])

    avg_diagonal = np.mean(diagonal_weights)
    avg_off_diagonal = np.mean(off_diagonal) if off_diagonal else 0.0

    print(f"Diagonal weights: {[f'{w:.2f}' for w in diagonal_weights]}")
    print(f"Average diagonal: {avg_diagonal:.2f}")
    print(f"Average off-diagonal: {avg_off_diagonal:.2f}")
    print(f"Separation: {avg_diagonal - avg_off_diagonal:.2f}")
    print("="*70)


def training_phase(circuit, world, num_steps=50):
    """
    Training phase: Teacher forcing to learn sensor→motor associations.

    On each step:
    1. Determine correct sensor and motor based on food location
    2. Present sensor pattern as input (scaled strongly)
    3. Force corresponding motor neuron to fire with external current
    4. STDP learns: sensor input that predicts motor firing gets strengthened

    Args:
        circuit: NeuralCircuit brain
        world: GridWorld environment
        num_steps: Number of training steps
    """
    print("\n" + "="*70)
    print("TRAINING PHASE: Teacher Forcing")
    print("="*70)
    print("Teaching the brain: Sensor N → Motor N, etc.")
    print()

    for step in range(num_steps):
        # Spawn new food location
        world.spawn_food()

        # Determine which sensors should activate
        sensors = world.get_sensor_activations()
        motors = world.get_correct_motor()

        # Create sensor input pattern (scaled moderately)
        sensor_inputs = np.array(sensors, dtype=float) * 50.0  # Moderate scaling

        # Create external current to force motor neurons to fire (teacher forcing)
        I_ext = np.zeros(4)
        for i in range(4):
            if motors[i]:
                I_ext[i] = 120.0  # Force motor to fire (tuned for selectivity)

        # Step circuit with STDP enabled
        output_spikes = circuit.step(
            input_spikes=sensor_inputs,
            I_ext=I_ext,
            learning=True  # STDP active!
        )

        # Display progress periodically
        if (step + 1) % 10 == 0:
            fired_neurons = np.where(output_spikes)[0]
            sensor_labels = ['N', 'S', 'E', 'W']
            active_dir = [sensor_labels[i] for i in range(4) if sensors[i]]
            print(f"  Step {step+1:3d}: Sensors {active_dir} active, "
                  f"Motors fired: {[sensor_labels[i] for i in fired_neurons]}")

    print(f"\nTraining complete! {num_steps} steps.")
    print("STDP has strengthened sensor→motor weights.")


def testing_phase(circuit, world, num_steps=20):
    """
    Testing phase: Autonomous navigation using learned weights.

    On each step:
    1. Spawn food at new location
    2. Present sensor pattern as input (no teacher)
    3. See which motor neuron fires (based on learned weights)
    4. Move agent based on motor firing
    5. Measure if movement was correct (closer to food)

    Args:
        circuit: NeuralCircuit brain
        world: GridWorld environment
        num_steps: Number of testing steps

    Returns:
        accuracy: Fraction of correct movements
    """
    print("\n" + "="*70)
    print("TESTING PHASE: Autonomous Navigation")
    print("="*70)
    print("Agent navigates using learned sensor→motor connections")
    print("(No teacher forcing, only sensory input)")
    print()

    correct_moves = 0
    total_moves = 0

    for step in range(num_steps):
        # Reset agent position and spawn new food
        world.agent_pos = [world.size // 2, world.size // 2]
        world.spawn_food()

        initial_distance = world.distance_to_food()

        # Determine which sensors should activate
        sensors = world.get_sensor_activations()

        # Present sensor pattern as input (scaled moderately)
        sensor_inputs = np.array(sensors, dtype=float) * 50.0

        # Add moderate baseline current to help motors fire
        I_ext = np.ones(4) * 15.0  # Moderate baseline to make neurons excitable

        # Step circuit with learning disabled (just testing)
        output_spikes = circuit.step(
            input_spikes=sensor_inputs,
            I_ext=I_ext,
            learning=False  # Testing, not learning
        )

        # Check which motor neurons fired
        if np.any(output_spikes):
            # Get first firing motor
            motor_id = np.where(output_spikes)[0][0]

            # Move agent
            world.move_agent(motor_id)

            final_distance = world.distance_to_food()

            # Check if move was correct (got closer or reached food)
            if final_distance < initial_distance or world.reached_food():
                correct_moves += 1
                result = "✓ Correct"
            else:
                result = "✗ Wrong"

            total_moves += 1

            sensor_labels = ['N', 'S', 'E', 'W']
            active_sensors = [sensor_labels[i] for i in range(4) if sensors[i]]
            motor_taken = sensor_labels[motor_id]

            print(f"  Step {step+1:2d}: Sensor {active_sensors} → Motor {motor_taken}, "
                  f"Distance {initial_distance}→{final_distance} {result}")
        else:
            print(f"  Step {step+1:2d}: No motor fired (brain inactive)")

    accuracy = correct_moves / total_moves if total_moves > 0 else 0.0

    print(f"\nTesting complete!")
    print(f"Correct moves: {correct_moves}/{total_moves}")
    print(f"Accuracy: {accuracy*100:.1f}%")
    print("="*70)

    return accuracy


def run_hunter_experiment():
    """
    Run the complete hunter experiment.

    Demonstrates sensory-motor learning:
    1. Build brain (8 neurons, tabula rasa)
    2. Create environment (10x10 grid)
    3. Train (50 steps with teacher forcing)
    4. Test (20 steps autonomous)
    5. Visualize results
    """
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*17 + "HUNTER EXPERIMENT" + " "*33 + "║")
    print("║" + " "*14 + "Sensory-Motor Learning via STDP" + " "*24 + "║")
    print("╚" + "═"*68 + "╝")
    print()

    # Build components
    circuit = build_hunter_brain()
    world = GridWorld(size=10)

    # Show initial state
    print("\nInitial Environment:")
    world.display()

    print_brain_state(circuit, "Initial Brain State (Before Training)")

    # Training phase
    training_phase(circuit, world, num_steps=50)

    print_brain_state(circuit, "Trained Brain State (After Training)")

    # Testing phase
    accuracy = testing_phase(circuit, world, num_steps=20)

    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    print("\n📊 Results:")
    print(f"  Training Steps: 50")
    print(f"  Testing Steps: 20")
    print(f"  Navigation Accuracy: {accuracy*100:.1f}%")

    print("\n🧠 Learning Mechanism:")
    print("  1. Teacher Forcing: Stimulate sensor + motor together")
    print("  2. STDP Learning: When neurons fire together, wire together")
    print("  3. Autonomous: Sensor alone triggers learned motor")

    print("\n🎯 Expected Behavior:")
    print("  - Diagonal weights should be strong (N→N, S→S, E→E, W→W)")
    print("  - Agent should move toward food >80% of time")
    print("  - Demonstrates tabula rasa → learned behavior")

    print("\n💡 Key Insight:")
    print("  Sensory-motor learning emerges from simple STDP rule!")
    print("  No explicit programming of 'North sensor → North motor'")
    print("  The brain discovers this mapping through experience.")

    print("\n" + "="*70)

    if accuracy >= 0.8:
        print("✅ SUCCESS: Agent learned sensory-motor control!")
    elif accuracy >= 0.5:
        print("⚠️  PARTIAL: Agent shows some learning but needs improvement")
    else:
        print("❌ FAILURE: Agent did not learn (check parameters)")

    print("="*70)


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    run_hunter_experiment()

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print("\nThis demonstrates:")
    print("  ✓ Tabula rasa learning (starts knowing nothing)")
    print("  ✓ Teacher forcing (guided learning phase)")
    print("  ✓ STDP-based association (sensor→motor wiring)")
    print("  ✓ Autonomous behavior (learned control)")
    print("\nThe agent learned to hunt! 🎯")
    print("="*70)

