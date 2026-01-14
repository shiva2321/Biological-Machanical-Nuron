"""
Pavlov Experiment: Classical Conditioning with STDP

Demonstrates how a BiologicalNeuron learns temporal associations through
classical conditioning. The neuron learns to predict a future event (Food)
based on an earlier warning signal (Bell).

Classical Conditioning Protocol:
- Bell (CS) at t=10ms -> Food (US) at t=30ms
- After training, Bell alone triggers anticipatory response
- Spike timing shifts earlier as learning progresses
"""

import numpy as np
import matplotlib.pyplot as plt
from neuron import BiologicalNeuron


def run_pavlov_experiment():
    """
    Run the classical conditioning experiment demonstrating temporal associative learning.
    """
    # ====================
    # Experiment Parameters
    # ====================
    n_inputs = 2
    n_trials = 100
    trial_duration_ms = 100
    dt = 1.0
    n_steps_per_trial = int(trial_duration_ms / dt)

    # Timing of stimuli
    bell_time = 10  # CS (Conditioned Stimulus) - Bell
    food_time = 30  # US (Unconditioned Stimulus) - Food

    # ====================
    # Initialize Neuron
    # ====================
    neuron = BiologicalNeuron(
        n_inputs=n_inputs,
        tau_m=10.0,           # Fast membrane dynamics
        tau_trace=40.0,       # Longer trace to bridge temporal gap
        dt=dt,
        a_plus=0.008,         # Very slow potentiation
        a_minus=0.004,        # Very slow depression
        weight_min=0.0,
        weight_max=1.0,
        v_rest=-70.0,
        v_reset=-75.0,
        theta_base=-62.0,     # Very low threshold (proven to work)
        u_increment=0.3,      # Minimal adaptation
        theta_increment=0.1   # Minimal threshold adaptation
    )

    # Set initial weights
    # Input 0 = Bell (weak, needs to be learned)
    # Input 1 = Food (strong, innately drives response)
    neuron.weights[0] = 0.2  # Bell starts weak
    neuron.weights[1] = 1.0  # Food starts strong

    print("="*70)
    print("PAVLOV'S CLASSICAL CONDITIONING EXPERIMENT")
    print("="*70)
    print(f"\nTraining Protocol:")
    print(f"  Trials: {n_trials}")
    print(f"  Trial Duration: {trial_duration_ms}ms")
    print(f"  Bell (CS) at: t={bell_time}ms")
    print(f"  Food (US) at: t={food_time}ms")
    print(f"  Time Gap: {food_time - bell_time}ms")
    print(f"\nInitial Weights:")
    print(f"  w_bell (Input 0): {neuron.weights[0]:.3f}")
    print(f"  w_food (Input 1): {neuron.weights[1]:.3f}")
    print()

    # ====================
    # Data Recording
    # ====================
    weight_bell_history = []
    weight_food_history = []
    spike_times_per_trial = []  # First spike time in each trial (if any)
    trial_numbers = []

    # ====================
    # Training Loop
    # ====================
    print("Running training trials...")

    for trial in range(n_trials):
        # Reset neuron state (but keep weights!)
        neuron.reset_state()

        # Track if neuron spiked this trial and when
        trial_spike_time = None

        # Run one trial
        for step in range(n_steps_per_trial):
            t = step * dt

            # Generate input spikes
            input_spikes = np.zeros(n_inputs)

            # Bell at t=10ms
            if t == bell_time:
                input_spikes[0] = 1.0

            # Food at t=30ms
            if t == food_time:
                input_spikes[1] = 1.0

            # Scale inputs to have physiological effect
            input_spikes_scaled = input_spikes * 25.0  # Strong PSPs (proven)

            # Update neuron with learning enabled
            # High baseline to ensure firing with Food input
            output_spike = neuron.step(input_spikes_scaled, I_ext=23.0, learning=True)

            # Record first spike time in this trial
            if output_spike and trial_spike_time is None:
                trial_spike_time = t

        # Record weights after this trial
        weight_bell_history.append(neuron.weights[0])
        weight_food_history.append(neuron.weights[1])

        # Record spike time if neuron fired
        if trial_spike_time is not None:
            spike_times_per_trial.append(trial_spike_time)
            trial_numbers.append(trial + 1)

        # Print progress every 20 trials
        if (trial + 1) % 20 == 0:
            avg_spike_time = np.mean(spike_times_per_trial[-20:]) if len(spike_times_per_trial[-20:]) > 0 else 0
            print(f"  Trial {trial + 1:3d}: w_bell={neuron.weights[0]:.3f}, "
                  f"w_food={neuron.weights[1]:.3f}, "
                  f"avg_spike_time={avg_spike_time:.1f}ms")

    print(f"\nTraining complete!")
    print(f"  Total trials with spikes: {len(spike_times_per_trial)}/{n_trials}")

    # ====================
    # Post-Training Test: Bell Only
    # ====================
    print("\n" + "="*70)
    print("POST-TRAINING TEST: Bell Only (No Food)")
    print("="*70)

    # Reset neuron state
    neuron.reset_state()

    test_spike_time = None
    test_spike_occurred = False

    # Run test trial with ONLY Bell
    for step in range(n_steps_per_trial):
        t = step * dt

        # Generate input spikes
        input_spikes = np.zeros(n_inputs)

        # Bell at t=10ms (NO FOOD!)
        if t == bell_time:
            input_spikes[0] = 1.0

        # Scale inputs
        input_spikes_scaled = input_spikes * 25.0

        # Update neuron (NO LEARNING in test)
        output_spike = neuron.step(input_spikes_scaled, I_ext=23.0, learning=False)

        # Record spike
        if output_spike and test_spike_time is None:
            test_spike_time = t
            test_spike_occurred = True

    print(f"\nTest Results:")
    print(f"  Final w_bell: {neuron.weights[0]:.3f}")
    print(f"  Final w_food: {neuron.weights[1]:.3f}")
    print(f"  Bell-only spike: {'YES' if test_spike_occurred else 'NO'}")
    if test_spike_occurred:
        print(f"  Spike time: {test_spike_time:.1f}ms (anticipatory response!)")

    # ====================
    # Visualization
    # ====================
    print("\nGenerating visualization...")

    # Convert to arrays
    weight_bell_history = np.array(weight_bell_history)
    weight_food_history = np.array(weight_food_history)
    spike_times_per_trial = np.array(spike_times_per_trial)
    trial_numbers = np.array(trial_numbers)

    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle("Pavlov's Classical Conditioning: Bell → Food Association",
                 fontsize=16, fontweight='bold')

    # ====================
    # Subplot 1: Reaction Time Shift
    # ====================
    ax1 = axes[0]

    if len(spike_times_per_trial) > 0:
        # Scatter plot of spike times vs trial number
        ax1.scatter(trial_numbers, spike_times_per_trial, c='blue', s=20, alpha=0.6,
                   label='Output Spike Time')

        # Add reference lines
        ax1.axhline(bell_time, color='green', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'Bell Time (t={bell_time}ms)')
        ax1.axhline(food_time, color='red', linestyle='--', linewidth=2,
                   alpha=0.7, label=f'Food Time (t={food_time}ms)')

        # Add trend line to show migration
        if len(trial_numbers) > 10:
            # Polynomial fit to show trend
            z = np.polyfit(trial_numbers, spike_times_per_trial, 2)
            p = np.poly1d(z)
            x_smooth = np.linspace(trial_numbers[0], trial_numbers[-1], 100)
            ax1.plot(x_smooth, p(x_smooth), 'orange', linewidth=2, alpha=0.8,
                    label='Trend (anticipation)')

    ax1.set_xlabel('Trial Number', fontsize=12)
    ax1.set_ylabel('Spike Time (ms)', fontsize=12)
    ax1.set_title('Reaction Time Migration: From Food-Response to Bell-Anticipation',
                 fontsize=13, fontweight='bold')
    ax1.set_xlim(0, n_trials + 1)
    ax1.set_ylim(0, trial_duration_ms)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Add annotation
    if len(spike_times_per_trial) > 0:
        early_avg = np.mean(spike_times_per_trial[:10]) if len(spike_times_per_trial[:10]) > 0 else 0
        late_avg = np.mean(spike_times_per_trial[-10:]) if len(spike_times_per_trial[-10:]) > 0 else 0
        shift = early_avg - late_avg
        ax1.text(0.02, 0.98, f'Early avg: {early_avg:.1f}ms\nLate avg: {late_avg:.1f}ms\nShift: {shift:.1f}ms',
                transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ====================
    # Subplot 2: Weight Evolution
    # ====================
    ax2 = axes[1]

    trial_axis = np.arange(1, n_trials + 1)

    # Plot weights
    ax2.plot(trial_axis, weight_bell_history, color='green', linewidth=2.5,
            label='w_bell (CS - Conditioned Stimulus)', marker='o', markersize=3,
            markevery=10, alpha=0.8)
    ax2.plot(trial_axis, weight_food_history, color='red', linewidth=2.5,
            label='w_food (US - Unconditioned Stimulus)', marker='s', markersize=3,
            markevery=10, alpha=0.8)

    # Add threshold line for success
    ax2.axhline(0.8, color='gray', linestyle=':', linewidth=1.5, alpha=0.5,
               label='Learning Threshold (0.8)')

    ax2.set_xlabel('Trial Number', fontsize=12)
    ax2.set_ylabel('Synaptic Weight', fontsize=12)
    ax2.set_title('Weight Evolution: Bell Association Strengthens via STDP',
                 fontsize=13, fontweight='bold')
    ax2.set_xlim(0, n_trials + 1)
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Add annotation showing weight change
    initial_bell = weight_bell_history[0]
    final_bell = weight_bell_history[-1]
    change_bell = final_bell - initial_bell
    ax2.text(0.98, 0.02, f'Bell weight: {initial_bell:.3f} → {final_bell:.3f}\nChange: +{change_bell:.3f}',
            transform=ax2.transAxes, fontsize=10, verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # Adjust layout
    plt.tight_layout(rect=(0, 0, 1, 0.97))

    # ====================
    # Final Statistics and Success Evaluation
    # ====================
    print("\n" + "="*70)
    print("EXPERIMENT RESULTS")
    print("="*70)

    print(f"\nWeight Changes:")
    print(f"  Initial w_bell: {initial_bell:.3f}")
    print(f"  Final w_bell:   {final_bell:.3f}")
    print(f"  Change:         +{change_bell:.3f}")
    print(f"  Initial w_food: {weight_food_history[0]:.3f}")
    print(f"  Final w_food:   {weight_food_history[-1]:.3f}")

    time_shift = 0.0  # Initialize
    if len(spike_times_per_trial) > 0:
        early_spikes = spike_times_per_trial[:20]
        late_spikes = spike_times_per_trial[-20:]
        early_avg = np.mean(early_spikes) if len(early_spikes) > 0 else 0
        late_avg = np.mean(late_spikes) if len(late_spikes) > 0 else 0
        time_shift = early_avg - late_avg

        print(f"\nReaction Time Shift:")
        print(f"  Early trials (1-20):   {early_avg:.1f}ms avg")
        print(f"  Late trials (81-100):  {late_avg:.1f}ms avg")
        print(f"  Time shift:            {time_shift:.1f}ms earlier")

    print(f"\nSuccess Criteria:")
    criteria_met = 0
    total_criteria = 3

    # Criterion 1: w_bell > 0.8
    criterion_1 = final_bell > 0.8
    print(f"  1. w_bell > 0.8:        {'✓ PASS' if criterion_1 else '✗ FAIL'} ({final_bell:.3f})")
    if criterion_1:
        criteria_met += 1

    # Criterion 2: Neuron fires in test (Bell only)
    criterion_2 = test_spike_occurred
    print(f"  2. Bell-only response:  {'✓ PASS' if criterion_2 else '✗ FAIL'}")
    if criterion_2:
        criteria_met += 1

    # Criterion 3: Spike time migrates earlier
    criterion_3 = False
    if len(spike_times_per_trial) > 0:
        criterion_3 = time_shift > 5.0  # At least 5ms shift
        print(f"  3. Time shift > 5ms:    {'✓ PASS' if criterion_3 else '✗ FAIL'} ({time_shift:.1f}ms)")
        if criterion_3:
            criteria_met += 1
    else:
        print(f"  3. Time shift > 5ms:    ✗ FAIL (no spikes)")

    print(f"\n{'='*70}")
    print(f"OVERALL: {criteria_met}/{total_criteria} criteria met")

    if criteria_met == total_criteria:
        print("✓✓✓ SUCCESS! Classical conditioning achieved!")
        print("    The neuron learned the Bell → Food association.")
        print("    Anticipatory response demonstrates temporal prediction.")
    elif criteria_met >= 2:
        print("✓✓ PARTIAL SUCCESS: Strong learning but not all criteria met.")
    else:
        print("✗ Learning incomplete. Consider adjusting parameters.")

    print("="*70)

    # Show plot
    plt.show()

    return {
        'weights_bell': weight_bell_history,
        'weights_food': weight_food_history,
        'spike_times': spike_times_per_trial,
        'trial_numbers': trial_numbers,
        'test_spike': test_spike_occurred,
        'test_spike_time': test_spike_time,
        'final_bell_weight': final_bell,
        'criteria_met': criteria_met
    }


if __name__ == "__main__":
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + " "*15 + "PAVLOV'S CLASSICAL CONDITIONING" + " "*22 + "║")
    print("║" + " "*10 + "Temporal Associative Learning with STDP" + " "*19 + "║")
    print("╚" + "═"*68 + "╝")
    print()

    # Run the experiment
    results = run_pavlov_experiment()

    print("\nExperiment complete! Close the plot window to exit.")

