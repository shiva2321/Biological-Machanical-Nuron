"""
SOBDN Sandbox -- Learning Wall Experiment

The baseline run (run_experiment.py) shows population exploding roughly
exponentially and never reaching homeostasis within a few thousand cycles,
which confounds the more interesting question: given a *stable* population,
does reward-modulated structural growth ever actually learn to seek food,
or does it stay a random walk forever?

This holds population under a tight cap (max_pop=150) so per-cycle cost
stays cheap and we can afford a much longer horizon, then asks directly:
  - Does mean distance-to-food trend down across successive windows?
  - Does the food-eaten rate increase over time?
  - Do synapse weights on direct sensor->motor edges actually strengthen?
  - Does a sensor->motor path exist at all (any hop count), and is it
    *live* (does the path actually carry recent spiking activity), or
    just topologically present? (BFS existence alone is cheap to satisfy
    here since sensors/motors are body-anchored close together -- see
    FINDINGS.md -- so this tracks path_liveness(), not just hop count.)
  - Does the population's mean genome drift under selection at all?

Also supports a fixed_spontaneity override (see run_ablation() /
spontaneity_ablation.py) to test whether pinning the exploration rate
instead of leaving it heritable changes the outcome -- isolates the
tragedy-of-the-commons dynamic documented in FINDINGS.md from every
other variable.
"""

import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from engine import World

N_CYCLES = 20000
WINDOW = 500
SEED = 7
MAX_POP = 150


def direct_sensor_motor_weights(world: World):
    motor_set = set(world.motor_ids.values())
    weights = []
    for sid in world.sensor_ids:
        n = world.neurons.get(sid)
        if n is None:
            continue
        for tid, w in n.synapses.items():
            if tid in motor_set:
                weights.append(w)
    return weights


def run(seed: int = SEED, n_cycles: int = N_CYCLES, max_pop: int = MAX_POP,
        fixed_spontaneity=None, label: str = "evolvable-spontaneity", verbose: bool = True):
    world = World(seed=seed, n_seed_interneurons=40, max_pop=max_pop, fixed_spontaneity=fixed_spontaneity)

    if verbose:
        print("=" * 70)
        print(f"SOBDN SANDBOX -- LEARNING WALL EXPERIMENT [{label}]")
        print(f"(population capped at {max_pop}"
              + (f", spontaneity fixed at {fixed_spontaneity}" if fixed_spontaneity is not None
                 else ", spontaneity heritable/evolvable") + ")")
        print("=" * 70)
        print(f"Cycles: {n_cycles} | seed={seed}\n")

    window_dists = []
    history = defaultdict(list)
    food_last_window = 0
    t_start = time.perf_counter()

    for i in range(1, n_cycles + 1):
        stats = world.step()
        window_dists.append(stats["dist_to_food"])

        if i % WINDOW == 0:
            mean_dist = float(np.mean(window_dists))
            window_dists = []
            eaten_this_window = stats["food_eaten"] - food_last_window
            food_last_window = stats["food_eaten"]
            direct_w = direct_sensor_motor_weights(world)
            mean_direct_w = float(np.mean(direct_w)) if direct_w else 0.0
            mean_spontaneity = float(np.mean([n.genome["spontaneity"] for n in world.neurons.values()]))
            mean_v_thresh = float(np.mean([n.genome["v_thresh"] for n in world.neurons.values()]))
            path = world.sensorimotor_path()
            path_hops = (len(path) - 1) if path is not None else -1
            liveness = world.path_liveness(path)

            history["cycle"].append(i)
            history["mean_dist"].append(mean_dist)
            history["eaten_this_window"].append(eaten_this_window)
            history["cum_eaten"].append(stats["food_eaten"])
            history["population"].append(stats["population"])
            history["mean_direct_weight"].append(mean_direct_w)
            history["n_direct_edges"].append(len(direct_w))
            history["mean_spontaneity"].append(mean_spontaneity)
            history["mean_v_thresh"].append(mean_v_thresh)
            history["path_hops"].append(path_hops)
            history["path_liveness"].append(liveness)

            if verbose and (i // WINDOW) % 5 == 0:
                print(f"cycle {i:6d} | pop={stats['population']:4d} | mean_dist(window)={mean_dist:6.2f} "
                      f"| eaten(window)={eaten_this_window:2d} | eaten(total)={stats['food_eaten']:3d} "
                      f"| direct_edges={len(direct_w):2d} mean_w={mean_direct_w:.3f} "
                      f"| path_hops={path_hops} live={liveness:.3f} "
                      f"| mean_spontaneity={mean_spontaneity:.4f} mean_v_thresh={mean_v_thresh:.3f}")

    total_time = time.perf_counter() - t_start
    if verbose:
        print(f"\nTotal wall-clock: {total_time:.1f}s ({n_cycles/total_time:.1f} cycles/s average)")

    # First-half vs second-half comparison -- did anything actually improve?
    half = len(history["mean_dist"]) // 2
    first_half_dist = np.mean(history["mean_dist"][:half])
    second_half_dist = np.mean(history["mean_dist"][half:])
    first_half_eaten = sum(history["eaten_this_window"][:half])
    second_half_eaten = sum(history["eaten_this_window"][half:])
    frac_live = float(np.mean([1.0 if (h >= 0 and l > 0.01) else 0.0
                                for h, l in zip(history["path_hops"], history["path_liveness"])]))
    frac_path = float(np.mean([1.0 if h >= 0 else 0.0 for h in history["path_hops"]]))

    if verbose:
        print("\n" + "=" * 70)
        print(f"FIRST HALF vs SECOND HALF OF THE RUN [{label}]")
        print("=" * 70)
        print(f"Mean distance-to-food:  first half={first_half_dist:.2f}  second half={second_half_dist:.2f}  "
              f"({'improved' if second_half_dist < first_half_dist else 'did not improve'})")
        print(f"Food eaten:             first half={first_half_eaten}  second half={second_half_eaten}  "
              f"({'improved' if second_half_eaten > first_half_eaten else 'did not improve'})")
        print(f"Mean direct sensor->motor weight: "
              f"first={history['mean_direct_weight'][0]:.3f} -> last={history['mean_direct_weight'][-1]:.3f}")
        print(f"Mean spontaneity (genome): first={history['mean_spontaneity'][0]:.4f} -> "
              f"last={history['mean_spontaneity'][-1]:.4f}")
        print(f"Mean v_thresh (genome):    first={history['mean_v_thresh'][0]:.3f} -> "
              f"last={history['mean_v_thresh'][-1]:.3f}")
        print(f"Fraction of logged cycles with ANY sensor->motor path:  {frac_path*100:.1f}%")
        print(f"Fraction of logged cycles with a LIVE sensor->motor path: {frac_live*100:.1f}%")

    return world, dict(history), {
        "first_half_dist": first_half_dist, "second_half_dist": second_half_dist,
        "first_half_eaten": first_half_eaten, "second_half_eaten": second_half_eaten,
        "frac_path": frac_path, "frac_live": frac_live, "total_time": total_time,
    }


def plot(history, out_dir="outputs", filename="learning_wall.png", suptitle=None):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(suptitle or "SOBDN Sandbox -- Learning Wall Experiment (population capped)",
                 fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(history["cycle"], history["mean_dist"], color="darkorange")
    z = np.polyfit(history["cycle"], history["mean_dist"], 1)
    ax.plot(history["cycle"], np.poly1d(z)(history["cycle"]), "--", color="black", alpha=0.6,
            label=f"trend: {z[0]*1000:+.3f} / 1000 cycles")
    ax.set_xlabel("cycle"); ax.set_ylabel("mean distance to food (windowed)")
    ax.set_title("Distance-to-food over time"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(history["cycle"], history["cum_eaten"], color="seagreen")
    ax.set_xlabel("cycle"); ax.set_ylabel("cumulative food eaten")
    ax.set_title("Food eaten over time"); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    hops = np.array(history["path_hops"], dtype=float)
    hops[hops < 0] = np.nan
    liveness = np.array(history["path_liveness"])
    vmax = max(float(np.nanmax(liveness)) if liveness.size else 0.0, 0.05)
    sc = ax.scatter(history["cycle"], hops, c=liveness, cmap="RdYlGn", vmin=0, vmax=vmax, s=18)
    plt.colorbar(sc, ax=ax, label="path liveness (0=topological only)")
    ax.set_xlabel("cycle"); ax.set_ylabel("sensor->motor hops")
    ax.set_title("Does a path exist, and is it live?"); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(history["cycle"], history["mean_direct_weight"], color="crimson", label="mean weight")
    ax2 = ax.twinx()
    ax2.plot(history["cycle"], history["n_direct_edges"], color="steelblue", alpha=0.6, label="edge count")
    ax.set_xlabel("cycle"); ax.set_ylabel("mean weight", color="crimson")
    ax2.set_ylabel("# direct sensor->motor edges", color="steelblue")
    ax.set_title("Direct (1-hop) sensor->motor synapses"); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(history["cycle"], history["mean_spontaneity"], color="purple", label="mean spontaneity")
    ax2 = ax.twinx()
    ax2.plot(history["cycle"], history["mean_v_thresh"], color="teal", alpha=0.7, label="mean v_thresh")
    ax.set_xlabel("cycle"); ax.set_ylabel("mean spontaneity", color="purple")
    ax2.set_ylabel("mean v_thresh", color="teal")
    ax.set_title("Population-mean genome drift"); ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.plot(history["cycle"], history["population"], color="slategray")
    ax.set_xlabel("cycle"); ax.set_ylabel("population")
    ax.set_title("Population (should be flat at the cap)"); ax.grid(alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    import os
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=130)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    world, history, summary = run()
    plot(history)
