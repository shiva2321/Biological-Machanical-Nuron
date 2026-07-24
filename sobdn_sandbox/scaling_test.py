"""
SOBDN Sandbox -- Scaling / Performance Stress Test

The original design chat predicted its own limit: "If you run this with
10,000 neurons, Python will lag." This benchmarks exactly that claim,
and separately measures how much the spatial-hash neighbor search (added
here to fix an O(N^2) growth-search in the original sketch) actually buys.

Population is held fixed (World(freeze_population=True)) at each size so
the measurement isolates per-cycle cost from organic-growth confounds --
see run_experiment.py / FINDINGS.md for what happens when population is
left free to grow on its own.
"""

import time

import matplotlib.pyplot as plt
import numpy as np

from engine import World

SIZES_HASHED = [50, 100, 200, 400, 800, 1600]
SIZES_NAIVE = [50, 100, 200, 400, 800, 1600]   # matched sizes for a direct comparison
WARMUP_CYCLES = 20
TIMED_CYCLES = 15
USABLE_CYCLES_PER_SEC = 20.0  # below this, "interactive" use is no longer practical


def bench(n_neurons: int, use_spatial_hash: bool):
    world = World(seed=42, n_seed_interneurons=n_neurons, use_spatial_hash=use_spatial_hash,
                  freeze_population=True)
    for _ in range(WARMUP_CYCLES):
        world.step()
    t0 = time.perf_counter()
    for _ in range(TIMED_CYCLES):
        world.step()
    elapsed = time.perf_counter() - t0
    per_cycle = elapsed / TIMED_CYCLES
    return per_cycle, len(world.neurons), sum(len(x.synapses) for x in world.neurons.values())


def run():
    print("=" * 70)
    print("SOBDN SANDBOX -- SCALING / PERFORMANCE STRESS TEST")
    print("=" * 70)

    print("\n-- spatial-hash neighbor search --")
    hashed = []
    for n in SIZES_HASHED:
        per_cycle, pop, edges = bench(n, True)
        hashed.append(per_cycle)
        cps = 1.0 / per_cycle
        flag = "" if cps >= USABLE_CYCLES_PER_SEC else "  <-- below usability threshold"
        print(f"  n={pop:6d} edges={edges:7d} | {per_cycle*1000:9.2f} ms/cycle | {cps:8.1f} cycles/s{flag}")

    print("\n-- naive O(N^2) neighbor search (capped -- gets slow fast) --")
    naive = []
    naive_sizes_run = []
    for n in SIZES_NAIVE:
        per_cycle, pop, edges = bench(n, False)
        naive.append(per_cycle)
        naive_sizes_run.append(pop)
        cps = 1.0 / per_cycle
        flag = "" if cps >= USABLE_CYCLES_PER_SEC else "  <-- below usability threshold"
        print(f"  n={pop:6d} edges={edges:7d} | {per_cycle*1000:9.2f} ms/cycle | {cps:8.1f} cycles/s{flag}")

    print("\nSpeedup from spatial hashing at matched sizes:")
    for i, n in enumerate(SIZES_NAIVE):
        speedup = naive[i] / hashed[i]
        print(f"  n={n:6d}: naive {naive[i]*1000:8.2f} ms vs hashed {hashed[i]*1000:6.2f} ms "
              f"-> {speedup:5.1f}x faster")

    # where does each approach cross below the usability threshold?
    def first_below(sizes, times):
        for s, t in zip(sizes, times):
            if 1.0 / t < USABLE_CYCLES_PER_SEC:
                return s
        return None

    hash_wall = first_below(SIZES_HASHED, hashed)
    naive_wall = first_below(naive_sizes_run, naive)
    print(f"\nPopulation where spatial-hash drops below {USABLE_CYCLES_PER_SEC:.0f} cycles/s: "
          f"{hash_wall if hash_wall else 'not reached in tested range'}")
    print(f"Population where naive O(N^2) drops below {USABLE_CYCLES_PER_SEC:.0f} cycles/s: "
          f"{naive_wall if naive_wall else 'not reached in tested range'}")

    return SIZES_HASHED, hashed, naive_sizes_run, naive


def plot(sizes_hashed, hashed, sizes_naive, naive, out_dir="outputs"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("SOBDN Sandbox -- Scaling Wall", fontsize=15, fontweight="bold")

    ax = axes[0]
    ax.plot(sizes_hashed, np.array(hashed) * 1000, "o-", label="spatial hash", color="seagreen")
    ax.plot(sizes_naive, np.array(naive) * 1000, "o-", label="naive O(N^2)", color="crimson")
    ax.axhline(1000 / USABLE_CYCLES_PER_SEC, color="gray", linestyle="--", alpha=0.6,
               label=f"{USABLE_CYCLES_PER_SEC:.0f} cycles/s threshold")
    ax.set_xlabel("population size"); ax.set_ylabel("ms per cycle")
    ax.set_title("Per-cycle cost vs population")
    ax.set_yscale("log")
    ax.legend(); ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.plot(sizes_hashed, [1.0 / t for t in hashed], "o-", label="spatial hash", color="seagreen")
    ax.plot(sizes_naive, [1.0 / t for t in naive], "o-", label="naive O(N^2)", color="crimson")
    ax.axhline(USABLE_CYCLES_PER_SEC, color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("population size"); ax.set_ylabel("cycles / second")
    ax.set_title("Throughput vs population")
    ax.set_yscale("log")
    ax.legend(); ax.grid(alpha=0.3, which="both")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    import os
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "scaling_test.png")
    plt.savefig(out_path, dpi=130)
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    sizes_hashed, hashed, sizes_naive, naive = run()
    plot(sizes_hashed, hashed, sizes_naive, naive)
