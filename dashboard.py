"""
Dashboard: Command-Line Control Center for Neural Circuit Brain

A comprehensive interface for managing, training, and visualizing the brain.
Provides menu-driven access to all brain management functions including
training, visualization, neurogenesis, and persistence.

Features:
- Load/save brain state
- Visualize weight matrix with region annotations
- Train on multiple lessons (reader, hunter)
- Grow brain capacity (neurogenesis)
- Status reporting and monitoring
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from brain_io import load_brain, save_brain, grow_brain, get_brain_info, print_brain_summary
from lessons import train_reader, train_hunter, list_available_lessons


# ============================================================================
# Global Configuration
# ============================================================================

BRAIN_FILE = 'my_brain.pkl'
BRAIN_AGE_FILE = 'my_brain_age.txt'  # Track brain age/training sessions


# ============================================================================
# Brain Age Tracking
# ============================================================================

def get_brain_age():
    """
    Get the number of training sessions the brain has completed.

    Returns:
        Age in sessions
    """
    if os.path.exists(BRAIN_AGE_FILE):
        with open(BRAIN_AGE_FILE, 'r') as f:
            return int(f.read().strip())
    return 0


def increment_brain_age():
    """Increment brain age (training session counter)."""
    age = get_brain_age() + 1
    with open(BRAIN_AGE_FILE, 'w') as f:
        f.write(str(age))
    return age


def reset_brain_age():
    """Reset brain age to 0."""
    with open(BRAIN_AGE_FILE, 'w') as f:
        f.write('0')


# ============================================================================
# Visualization
# ============================================================================

def visualize_brain_memory(circuit):
    """
    Create heatmap visualization of the brain's weight matrix with region annotations.

    Shows the complete weight matrix where each row is a neuron and each column
    is an input channel. Annotates regions used by different lessons.

    Args:
        circuit: NeuralCircuit to visualize
    """
    print("\n" + "="*70)
    print("VISUALIZING BRAIN MEMORY")
    print("="*70)

    # Build weight matrix
    num_neurons = circuit.num_neurons
    num_inputs = circuit.input_channels

    weight_matrix = np.zeros((num_neurons, num_inputs))
    for i in range(num_neurons):
        weights = circuit.get_weights(i)
        if len(weights) > 0:
            weight_matrix[i, :len(weights)] = weights

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))

    # Plot heatmap
    im = ax.imshow(weight_matrix, cmap='viridis', aspect='auto',
                   interpolation='nearest')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Synaptic Weight', rotation=270, labelpad=20, fontsize=12)

    # Labels
    ax.set_xlabel('Input Channels', fontsize=14, fontweight='bold')
    ax.set_ylabel('Output Neurons', fontsize=14, fontweight='bold')
    ax.set_title('Brain Weight Matrix - Memory Landscape',
                 fontsize=16, fontweight='bold', pad=20)

    # Annotate regions
    # Reader Region: Channels 0-24, Neurons 0-2
    ax.add_patch(plt.Rectangle((0, 0), 25, 3, fill=False,
                               edgecolor='red', linewidth=3, linestyle='--'))
    ax.text(12, -1.5, 'READER REGION', ha='center', va='top',
           fontsize=11, fontweight='bold', color='red',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                    edgecolor='red', linewidth=2))

    # Hunter Region: Channels 25-28, Neurons 3-6
    if num_inputs >= 29 and num_neurons >= 7:
        ax.add_patch(plt.Rectangle((25, 3), 4, 4, fill=False,
                                   edgecolor='blue', linewidth=3, linestyle='--'))
        ax.text(27, 2.3, 'HUNTER', ha='center', va='top',
               fontsize=10, fontweight='bold', color='blue',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                        edgecolor='blue', linewidth=2))

    # Available region annotation
    if num_neurons > 7:
        ax.text(num_inputs/2, num_neurons-1, 'AVAILABLE CAPACITY',
               ha='center', va='center', fontsize=12, fontweight='bold',
               color='gray', alpha=0.3)

    # Grid
    ax.set_xticks(np.arange(0, num_inputs, 5))
    ax.set_yticks(np.arange(num_neurons))
    ax.grid(True, which='both', color='gray', linewidth=0.3, alpha=0.3)

    # Statistics
    stats_text = f"Neurons: {num_neurons} | Inputs: {num_inputs}\n"
    stats_text += f"Total Weights: {num_neurons * num_inputs}\n"
    stats_text += f"Weight Range: [{np.min(weight_matrix):.2f}, {np.max(weight_matrix):.2f}]\n"
    stats_text += f"Mean Weight: {np.mean(weight_matrix):.2f}"

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save
    os.makedirs('outputs', exist_ok=True)
    filename = 'outputs/brain_dashboard_memory.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Memory visualization saved to '{filename}'")

    # Show
    plt.show()
    plt.close()

    print("="*70)


# ============================================================================
# Menu Functions
# ============================================================================

def show_status_report(circuit):
    """
    Display comprehensive status report about the brain.

    Args:
        circuit: NeuralCircuit to report on
    """
    print("\n" + "="*70)
    print("BRAIN STATUS REPORT")
    print("="*70)

    # Get info
    info = get_brain_info(circuit)
    age = get_brain_age()

    # File info
    file_size = os.path.getsize(BRAIN_FILE) / 1024 if os.path.exists(BRAIN_FILE) else 0

    print(f"\n📁 File Information:")
    print(f"  Location: {BRAIN_FILE}")
    print(f"  Size: {file_size:.2f} KB")
    print(f"  Age: {age} training sessions")

    print(f"\n🧠 Architecture:")
    print(f"  Total Neurons: {info['num_neurons']}")
    print(f"  Input Channels: {info['input_channels']}")
    print(f"  Total Capacity: {info['num_neurons'] * info['input_channels']} weights")
    print(f"  Internal Connections: {info['num_connections']}")

    print(f"\n⚖️  Weight Statistics:")
    print(f"  Total Weights: {info['total_weights']}")
    print(f"  Range: [{info['weight_stats']['min']:.3f}, {info['weight_stats']['max']:.3f}]")
    print(f"  Mean: {info['weight_stats']['mean']:.3f}")
    print(f"  Std Dev: {info['weight_stats']['std']:.3f}")

    print(f"\n🎓 Trained Regions:")
    print(f"  Reader: Channels 0-24 → Neurons 0-2")
    print(f"  Hunter: Channels 25-28 → Neurons 3-6")

    available_neurons = info['num_neurons'] - 7
    available_inputs = info['input_channels'] - 29
    print(f"\n💡 Available Capacity:")
    print(f"  Neurons: {available_neurons} (IDs 7-{info['num_neurons']-1})")
    print(f"  Inputs: {available_inputs} (Channels 29-{info['input_channels']-1})")

    print("\n" + "="*70)


def attend_reading_class(circuit):
    """
    Train the brain on character recognition (Reader lesson).

    Args:
        circuit: NeuralCircuit to train
    """
    print("\n" + "="*70)
    print("🎓 ATTENDING CLASS: CHARACTER RECOGNITION")
    print("="*70)

    try:
        accuracy = train_reader(
            circuit,
            input_channels=(0, 24),
            output_neurons=(0, 2),
            num_samples=1000,
            target_acc=0.90,
            max_epochs=100,
            verbose=True
        )

        # Increment age
        new_age = increment_brain_age()

        print("\n✅ Class completed!")
        print(f"Final accuracy: {accuracy*100:.1f}%")
        print(f"Brain age: {new_age} sessions")

    except Exception as e:
        print(f"\n❌ Error during training: {e}")


def attend_hunting_class(circuit):
    """
    Train the brain on sensory-motor navigation (Hunter lesson).

    Args:
        circuit: NeuralCircuit to train
    """
    print("\n" + "="*70)
    print("🎓 ATTENDING CLASS: SENSORY-MOTOR NAVIGATION")
    print("="*70)

    try:
        accuracy = train_hunter(
            circuit,
            sensor_channels=(25, 28),
            motor_neurons=(3, 6),
            num_steps=500,
            verbose=True
        )

        # Increment age
        new_age = increment_brain_age()

        print("\n✅ Class completed!")
        print(f"Test accuracy: {accuracy*100:.1f}%")
        print(f"Brain age: {new_age} sessions")

    except Exception as e:
        print(f"\n❌ Error during training: {e}")


def grow_brain_capacity(circuit):
    """
    Expand brain capacity through neurogenesis.

    Args:
        circuit: NeuralCircuit to expand

    Returns:
        Expanded circuit
    """
    print("\n" + "="*70)
    print("🌱 BRAIN GROWTH (NEUROGENESIS)")
    print("="*70)

    print("\nCurrent capacity:")
    print(f"  Neurons: {circuit.num_neurons}")
    print(f"  Input channels: {circuit.input_channels}")

    print("\nGrowth options:")
    print("  [1] Add 10 neurons (recommended)")
    print("  [2] Add 20 input channels")
    print("  [3] Add both (10 neurons + 20 channels)")
    print("  [4] Custom growth")
    print("  [5] Cancel")

    choice = input("\nSelect growth option (1-5): ").strip()

    if choice == '1':
        circuit = grow_brain(circuit, new_inputs=0, new_neurons=10)
    elif choice == '2':
        circuit = grow_brain(circuit, new_inputs=20, new_neurons=0)
    elif choice == '3':
        circuit = grow_brain(circuit, new_inputs=20, new_neurons=10)
    elif choice == '4':
        try:
            new_inputs = int(input("  Add how many input channels? "))
            new_neurons = int(input("  Add how many neurons? "))
            circuit = grow_brain(circuit, new_inputs=new_inputs, new_neurons=new_neurons)
        except ValueError:
            print("\n❌ Invalid input. Cancelled.")
            return circuit
    else:
        print("\nCancelled.")
        return circuit

    print("\n✅ Brain grown successfully!")
    print(f"New capacity: {circuit.num_neurons} neurons, {circuit.input_channels} inputs")

    return circuit


# ============================================================================
# Main Dashboard
# ============================================================================

def print_header():
    """Print dashboard header."""
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*19 + "NEURAL BRAIN DASHBOARD" + " "*27 + "║")
    print("║" + " "*16 + "Command Center & Control Panel" + " "*22 + "║")
    print("╚" + "═"*68 + "╝")


def print_menu():
    """Print main menu options."""
    print("\n" + "="*70)
    print("MAIN MENU")
    print("="*70)
    print("\n📊 Information:")
    print("  [1] Status Report - View brain architecture and statistics")
    print("  [2] Visualize Memory - Show weight matrix heatmap")

    print("\n🎓 Training:")
    print("  [3] Attend Class: Reading (A-C) - Character recognition")
    print("  [4] Attend Class: Hunting - Sensory-motor navigation")
    print("  [5] List Available Lessons - Show all training options")

    print("\n🌱 Management:")
    print("  [6] Grow Brain - Expand capacity (neurogenesis)")
    print("  [7] Save & Exit - Persist brain and quit")
    print("  [8] Exit without saving")

    print("\n" + "="*70)


def run_dashboard():
    """
    Main dashboard loop.

    Provides interactive menu for managing the brain including training,
    visualization, growth, and persistence.
    """
    print_header()

    print("\nInitializing...")
    print("Loading brain from disk...")

    # Load brain
    try:
        brain = load_brain(BRAIN_FILE)
        age = get_brain_age()
        print(f"✅ Brain loaded successfully! (Age: {age} sessions)")
    except Exception as e:
        print(f"❌ Error loading brain: {e}")
        print("Creating new brain...")
        brain = load_brain(BRAIN_FILE)
        reset_brain_age()
        age = 0

    # Quick summary
    print(f"\nQuick Summary:")
    print(f"  Neurons: {brain.num_neurons}")
    print(f"  Input Channels: {brain.input_channels}")
    print(f"  Age: {age} training sessions")

    # Main loop
    modified = False
    running = True

    while running:
        print_menu()

        choice = input("\nSelect option (1-8): ").strip()

        if choice == '1':
            # Status Report
            show_status_report(brain)

        elif choice == '2':
            # Visualize Memory
            visualize_brain_memory(brain)

        elif choice == '3':
            # Attend Reading Class
            attend_reading_class(brain)
            modified = True

        elif choice == '4':
            # Attend Hunting Class
            attend_hunting_class(brain)
            modified = True

        elif choice == '5':
            # List Available Lessons
            list_available_lessons()

        elif choice == '6':
            # Grow Brain
            brain = grow_brain_capacity(brain)
            modified = True

        elif choice == '7':
            # Save & Exit
            if modified:
                print("\nSaving brain...")
                save_brain(brain, BRAIN_FILE)
                print("✅ Brain saved successfully!")
            else:
                print("\nNo changes to save.")

            print("\n" + "="*70)
            print("DASHBOARD CLOSING")
            print("="*70)
            print("Thank you for using the Neural Brain Dashboard!")
            print(f"Brain sessions: {get_brain_age()}")
            print("="*70)
            running = False

        elif choice == '8':
            # Exit without saving
            if modified:
                confirm = input("\n⚠️  You have unsaved changes! Exit anyway? (y/n): ").strip().lower()
                if confirm != 'y':
                    continue

            print("\n" + "="*70)
            print("DASHBOARD CLOSING (unsaved)")
            print("="*70)
            print("Exiting without saving...")
            print("="*70)
            running = False

        else:
            print("\n❌ Invalid option. Please select 1-8.")

        # Pause before next menu
        if running and choice in ['1', '3', '4', '5', '6']:
            input("\nPress Enter to continue...")


def quick_status():
    """Quick status check without entering interactive mode."""
    print_header()
    print("\nQUICK STATUS CHECK")
    print("="*70)

    if not os.path.exists(BRAIN_FILE):
        print("❌ No brain file found!")
        print(f"   Expected: {BRAIN_FILE}")
        print("   Run dashboard.py to create a new brain.")
        return

    brain = load_brain(BRAIN_FILE)
    age = get_brain_age()

    info = get_brain_info(brain)

    print(f"\n✅ Brain Status:")
    print(f"   Neurons: {info['num_neurons']}")
    print(f"   Inputs: {info['input_channels']}")
    print(f"   Weights: {info['total_weights']}")
    print(f"   Age: {age} sessions")
    print(f"   Mean Weight: {info['weight_stats']['mean']:.3f}")

    print("\n" + "="*70)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys

    # Check for command-line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--status':
            # Quick status check
            quick_status()
        elif sys.argv[1] == '--help':
            print("\nNeural Brain Dashboard")
            print("="*70)
            print("Usage:")
            print("  python dashboard.py          - Run interactive dashboard")
            print("  python dashboard.py --status - Quick status check")
            print("  python dashboard.py --help   - Show this help")
            print("="*70)
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        # Run interactive dashboard
        try:
            run_dashboard()
        except KeyboardInterrupt:
            print("\n\n⚠️  Dashboard interrupted by user.")
            print("Exiting...")
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()

