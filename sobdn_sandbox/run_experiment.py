"""
SOBDN Sandbox -- Baseline Experiment: Embodied Food-Seeking

Seeds a small interneuron cloud around an embodied agent and lets the
world run under its own rules -- structural growth, mitosis, apoptosis,
reward-modulated plasticity -- with no external supervision. Nothing here
is hand-tuned per-run; the agent either finds food because the mechanism
in engine.py works, or it doesn't.

Tracks, every LOG_EVERY cycles:
  - population (total, and interneurons specifically)
  - mean energy (is the colony starving or thriving?)
  - number of synapses (is structure still growing?)
  - sensor->motor path length in hops (has a bridge formed at all?)
  - distance from agent to food (is anything actually being achieved?)
  - food eaten, hazard hits
  - wall-clock time per cycle (is this still fast enough to be usable?)

At the end, evaluates explicit pass/fail criteria and saves plots, in the
same spirit as experiments/pavlov_experiment.py elsewhere in this repo.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from engine import World

N_CYCLES = 4000  # population growth is uncapped by default and compounds --
                  # by cycle 4000 this already takes ~10 minutes (see
                  # FINDINGS.md). Pass max_pop=N to World(...) for a much
                  # faster, longer-horizon run once population isn't the
                  # variable you're studying (see learning_wall_experiment.py).
LOG_EVERY = 50
SEED = 1


def run(n_cycles: int = N_CYCLES, seed: int = SEED, verbose: bool = True):
    world = World(seed=seed)

    history = {
        "cycle": [], "population": [], "n_interneurons": [], "mean_energy": [],
        "n_edges": [], "dist_to_food": [], "food_eaten": [], "hazard_hits": [],
        "step_time_s": [], "path_len": [],
    }

    if verbose:
        print("=" * 70)
        print("SOBDN SANDBOX -- BASELINE EXPERIMENT: EMBODIED FOOD-SEEKING")
        print("=" * 70)
        print(f"Cycles: {n_cycles} | seed={seed} | grid={world.grid}^3")
        print(f"Seed population: {world.summary()}")
        print()

    wall_clock_start = time.perf_counter()
    for i in range(1, n_cycles + 1):
        stats = world.step()

        if i % LOG_EVERY == 0 or i == n_cycles:
            path_len = world.sensorimotor_path()
            history["cycle"].append(stats["cycle"])
            history["population"].append(stats["population"])
            history["n_interneurons"].append(stats["n_interneurons"])
            history["mean_energy"].append(stats["mean_energy"])
            history["n_edges"].append(stats["n_edges"])
            history["dist_to_food"].append(stats["dist_to_food"])
            history["food_eaten"].append(stats["food_eaten"])
            history["hazard_hits"].append(stats["hazard_hits"])
            history["step_time_s"].append(stats["step_time_s"])
            history["path_len"].append(path_len if path_len is not None else -1)

            if verbose and (i % (LOG_EVERY * 10) == 0 or i == n_cycles):
                print(f"cycle {i:6d} | pop={stats['population']:5d} (inter={stats['n_interneurons']:5d}) "
                      f"| mean_E={stats['mean_energy']:6.2f} | edges={stats['n_edges']:6d} "
                      f"| path={path_len} | dist_food={stats['dist_to_food']:6.2f} "
                      f"| eaten={stats['food_eaten']:3d} | hazard_hits={stats['hazard_hits']:3d} "
                      f"| step_ms={stats['step_time_s']*1000:.2f}")

    total_wall_clock = time.perf_counter() - wall_clock_start

    if verbose:
        print()
        print("=" * 70)
        print("RUN COMPLETE")
        print("=" * 70)
        print(f"Total wall-clock time: {total_wall_clock:.1f}s for {n_cycles} cycles "
              f"({n_cycles/total_wall_clock:.1f} cycles/s average)")

    return world, history, total_wall_clock


def evaluate(world: World, history: dict, total_wall_clock: float, n_cycles: int, verbose: bool = True):
    dist_series = np.array(history["dist_to_food"])
    pop_series = np.array(history["population"])
    path_series = np.array(history["path_len"])

    # Trend of distance-to-food over the back half of the run (ignores food
    # respawn discontinuities better than a full-series fit would).
    half = len(dist_series) // 2
    if half > 5:
        x = np.arange(half)
        slope = np.polyfit(x, dist_series[half:], 1)[0]
    else:
        slope = 0.0

    criteria = {
        "population_survived": pop_series[-1] > 5,
        "population_bounded": not world.max_pop_hit,
        "sensorimotor_bridge_formed": bool((path_series >= 0).any()),
        "distance_trending_down_or_food_eaten": bool(slope < -0.001 or world.food_eaten > 0),
        "stayed_usably_fast": (n_cycles / total_wall_clock) > 20,
    }

    if verbose:
        print("\nCriteria:")
        for name, passed in criteria.items():
            print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        print(f"\nFinal population: {pop_series[-1]} | food eaten: {world.food_eaten} | "
              f"hazard hits: {world.hazard_hits} | max_pop safety valve hit: {world.max_pop_hit}")
        print(f"Late-run distance-to-food trend: {slope:+.4f} per logged step "
              f"({'closing in' if slope < 0 else 'not improving'})")

    return criteria, slope


def plot(history: dict, out_dir: str = "outputs"):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("SOBDN Sandbox -- Baseline Food-Seeking Run", fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(history["cycle"], history["population"], label="total population", color="steelblue")
    ax.plot(history["cycle"], history["n_interneurons"], label="interneurons", color="steelblue", alpha=0.5, linestyle="--")
    ax.set_xlabel("cycle"); ax.set_ylabel("neuron count"); ax.set_title("Population over time")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(history["cycle"], history["dist_to_food"], color="darkorange")
    ax.set_xlabel("cycle"); ax.set_ylabel("distance to food (voxels)")
    ax.set_title(f"Agent distance to food (eaten={history['food_eaten'][-1]})")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    path = np.array(history["path_len"], dtype=float)
    path[path < 0] = np.nan
    ax.plot(history["cycle"], history["n_edges"], color="seagreen", label="synapse count")
    ax2 = ax.twinx()
    ax2.scatter(history["cycle"], path, color="crimson", s=8, label="sensor->motor hops")
    ax.set_xlabel("cycle"); ax.set_ylabel("synapses", color="seagreen"); ax2.set_ylabel("hops", color="crimson")
    ax.set_title("Connectome growth"); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(history["cycle"], np.array(history["step_time_s"]) * 1000, color="purple")
    ax.set_xlabel("cycle"); ax.set_ylabel("ms per logged cycle")
    ax.set_title("Per-cycle wall-clock cost"); ax.grid(alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    import os
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "baseline_run.png")
    plt.savefig(out_path, dpi=130)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    world, history, total_wall_clock = run()
    evaluate(world, history, total_wall_clock, N_CYCLES)
    plot(history)
