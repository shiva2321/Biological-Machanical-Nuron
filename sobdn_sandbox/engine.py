"""
SOBDN Sandbox: Self-Organizing Bio-Digital Neurons.

A standalone research sandbox exploring the "brain grows its own hardware"
idea from a design conversation with Gemini (mitosis/apoptosis/chemotaxis-
driven structural growth, reward-modulated plasticity, an embodied agent).
Deliberately kept separate from the tested neuron.py/circuit.py framework
at the repo root -- this is a "see where it leads" experiment, not
production code.

This is a rewrite, not a port. The original sketch had real problems that
would have made it fail silently instead of failing informatively:

  1. Neurons were identified by Python list index. Any apoptosis or birth
     reordered the list, so surviving synapses silently pointed at the
     wrong neuron. Fixed: neurons live in a dict keyed by a stable id
     that is never reused or reordered.

  2. In the final consolidated version, only the 4 hand-coded sensor
     neurons ever called their own update step -- every interneuron and
     motor neuron, including every neuron produced by mitosis, never
     fired and never accumulated an eligibility trace. The reward-
     modulation line was multiplied by that trace, so learning was a
     no-op for the entire population even though the code "ran".
     Fixed: every neuron integrates input and can spike every cycle.

  3. Eligibility was tracked per-neuron (one trace shared by every
     outgoing synapse), which throws away exactly the information
     three-factor learning rules need -- which synapse was responsible.
     Fixed: eligibility is per-synapse, incremented on pre/post
     coincidence (Hebbian), and gated by a locally-sampled neuromodulator
     (reward-modulated Hebbian plasticity, e.g. Fremaux & Gerstner 2016).

  4. Growth/neighbor search was O(N^2) per cycle despite already having a
     voxel grid that could bucket neighbors cheaply. Fixed: a spatial
     hash rebuilt each cycle turns this into O(neighbors).

  5. Population had no real carrying capacity, just an ad hoc
     "if population > 200: pain" hack. Fixed: a finite, slowly-
     regenerating nutrient field is the only source of metabolic income;
     crowding a voxel means literally splitting a smaller resource, which
     is what makes a carrying capacity real instead of asserted.

  6. "Chemotaxis" in the original was a distance threshold, not gradient
     following. Fixed: growth targets are sampled with probability
     weighted by local reward-chemical density over distance, so it is
     actually gradient-biased.

  7. Sensing was hard-coded ("if distance-to-food < 10: neurons[1].fired
     = True"), bypassing the entire "structure emerges from space" story
     for the one part that mattered most (getting information in).
     Fixed: sensor/motor neurons are physically anchored to the agent's
     body in the *same* coordinate system the brain grows in, and they
     sense an analytic scent field via their actual position.

The "Memory Protein" from the original design is kept, honestly labeled:
it is spike-frequency adaptation (a second, slower leaky variable gating
threshold), which is what it actually is mechanically, whatever
biological name is attached to it.
"""

from __future__ import annotations

import itertools
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# World / physics constants (tuned empirically -- see FINDINGS.md)
# ---------------------------------------------------------------------------
GRID = 40
BUCKET_SIZE = 2.0          # spatial hash bucket edge length, in voxels

NUTRIENT_REGEN = 0.02       # ambient nutrient trickle per voxel per cycle
NUTRIENT_CAP = 1.0          # max nutrient a voxel can hold
NUTRIENT_UPTAKE_RATE = 0.05  # max a single neuron can draw from a voxel/cycle
NUTRIENT_TO_ENERGY = 4.0

UPKEEP_COST = 0.05           # baseline metabolic cost per neuron per cycle
FIRE_COST = 0.30             # extra cost of spiking
GROWTH_COST = 1.0
REWARD_ENERGY_GAIN = 5.0
PAIN_ENERGY_COST = 8.0

MP_ALPHA = 1.0               # memory-protein (adaptation) increment on spike
ELIG_DECAY = 0.7             # per-cycle decay of Hebbian eligibility trace
LR_PLUS = 0.4
LR_MINUS = 0.6
W_MIN, W_MAX = 0.0, 1.0

EVAPORATION = 0.9            # chem field decay per cycle (post-diffusion)
SCENT_TAU = 16.0             # length scale of the analytic food/hazard scent
SENSOR_GAIN = 3.0
HAZARD_SENSOR_FACTOR = 0.35  # hazard scent counts for less than food scent,
                              # so the two don't cancel to ~0 at typical range

THRUST_GAIN = 0.35
TURN_GAIN = 0.18

MAX_POP = 3000                # hard safety valve, distinct from the real
                               # nutrient-based carrying capacity -- if this
                               # is ever hit it is itself a finding.

FOOD_RADIUS = 2.0
HAZARD_RADIUS = 2.0
FOOD_REWARD = 3.0
APPROACH_REWARD = 0.05
HAZARD_PAIN = 2.0


def bucket_key(pos: np.ndarray) -> tuple[int, int, int]:
    return (int(pos[0] // BUCKET_SIZE), int(pos[1] // BUCKET_SIZE), int(pos[2] // BUCKET_SIZE))


# ---------------------------------------------------------------------------
# Genome
# ---------------------------------------------------------------------------
def random_genome(rng: random.Random) -> dict:
    return {
        "v_thresh": rng.uniform(0.6, 1.2),
        "leak": rng.uniform(0.5, 0.85),
        "mp_beta": rng.uniform(-0.3, 0.05),   # mostly facilitating, a few habituating
        "mp_gamma": rng.uniform(0.8, 0.97),
        "mitosis_thresh": rng.uniform(12.0, 22.0),
        "growth_radius": rng.uniform(3.0, 6.0),
        "max_synapses": rng.choice([4, 6, 8]),
        "spontaneity": rng.uniform(0.0, 0.03),  # baseline random-firing rate --
                                                  # "motor babbling" / spontaneous
                                                  # activity, without which no
                                                  # relay neuron ever gets excited
                                                  # enough to pass a signal on
                                                  # (see FINDINGS.md).
        "mutation_rate": 0.05,
    }


def mutate_genome(genome: dict, rng: random.Random) -> dict:
    child = dict(genome)
    for key in ("v_thresh", "leak", "mp_beta", "mp_gamma", "mitosis_thresh", "growth_radius"):
        base = genome[key]
        child[key] = base + rng.gauss(0.0, genome["mutation_rate"] * max(abs(base), 0.1))
    child["v_thresh"] = float(np.clip(child["v_thresh"], 0.2, 3.0))
    child["leak"] = float(np.clip(child["leak"], 0.3, 0.97))
    child["mp_gamma"] = float(np.clip(child["mp_gamma"], 0.5, 0.99))
    child["mp_beta"] = float(np.clip(child["mp_beta"], -0.6, 0.3))
    child["growth_radius"] = float(np.clip(child["growth_radius"], 1.5, 10.0))
    child["mitosis_thresh"] = float(np.clip(child["mitosis_thresh"], 6.0, 60.0))
    child["spontaneity"] = float(np.clip(
        genome.get("spontaneity", 0.01) + rng.gauss(0.0, 0.005), 0.0, 0.08))
    if rng.random() < 0.1:
        child["max_synapses"] = rng.choice([4, 6, 8, 10])
    return child


# ---------------------------------------------------------------------------
# Neuron
# ---------------------------------------------------------------------------
@dataclass
class Neuron:
    id: int
    pos: np.ndarray
    kind: str            # 'sensor' | 'motor' | 'inter'
    genome: dict
    energy: float = 5.0
    v: float = 0.0
    mp: float = 0.0
    rate: float = 0.0    # EMA of firing, used as the motor "thrust" readout
    fired: bool = False
    age: int = 0
    synapses: dict = field(default_factory=dict)   # target_id -> weight
    elig: dict = field(default_factory=dict)        # target_id -> eligibility trace

    def update(self, input_current: float, rng: random.Random) -> None:
        g = self.genome
        self.v = self.v * g["leak"] + input_current
        dynamic_thresh = g["v_thresh"] + self.mp * g["mp_beta"]
        spontaneous = rng.random() < g.get("spontaneity", 0.0)
        if self.v > dynamic_thresh or spontaneous:
            self.fired = True
            self.v = 0.0
            self.mp += MP_ALPHA
            self.energy -= FIRE_COST
        else:
            self.fired = False
        self.mp *= g["mp_gamma"]
        self.rate = self.rate * 0.85 + 0.15 * (1.0 if self.fired else 0.0)
        self.age += 1


# ---------------------------------------------------------------------------
# Embodied agent
# ---------------------------------------------------------------------------
class Agent:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)
        self.heading = 0.0
        self.vel = np.zeros(3)

    def anchor(self, forward: float, lateral: float, vertical: float = 0.0) -> np.ndarray:
        fwd = np.array([np.cos(self.heading), np.sin(self.heading), 0.0])
        right = np.array([-np.sin(self.heading), np.cos(self.heading), 0.0])
        return self.pos + fwd * forward + right * lateral + np.array([0.0, 0.0, vertical])

    def apply_thrust(self, left: float, right: float, grid: int) -> None:
        forward = (left + right) * THRUST_GAIN
        turn = (left - right) * TURN_GAIN
        self.heading += turn
        fwd_vec = np.array([np.cos(self.heading), np.sin(self.heading), 0.0])
        self.vel = fwd_vec * forward
        self.pos = np.clip(self.pos + self.vel, 0, grid - 1)


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------
class World:
    def __init__(self, grid: int = GRID, n_seed_interneurons: int = 40, seed: Optional[int] = None,
                 use_spatial_hash: bool = True, freeze_population: bool = False, max_pop: int = MAX_POP):
        self.grid = grid
        self.max_pop = max_pop
        self.rng = random.Random(seed)
        np_seed = None if seed is None else seed
        self._np_rng = np.random.default_rng(np_seed)

        self.chem = np.zeros((grid, grid, grid, 2))   # [..., 0]=reward  [..., 1]=pain
        self.nutrient = np.full((grid, grid, grid), NUTRIENT_CAP * 0.5)

        self._id_counter = itertools.count()
        self.neurons: dict[int, Neuron] = {}
        self.cycle = 0
        self.use_spatial_hash = use_spatial_hash
        # When True, mitosis/apoptosis are skipped so population size stays
        # exactly fixed -- used by scaling_test.py to benchmark per-cycle cost
        # at a controlled population size without organic growth confounding
        # the measurement.
        self.freeze_population = freeze_population
        self._hash: dict[tuple, list[int]] = defaultdict(list)

        center = np.array([grid / 2, grid / 2, grid / 2])
        self.agent = Agent(center)
        self.food_pos = self._random_point()
        self.hazard_pos = self._random_point()
        self.prev_dist = float(np.linalg.norm(self.agent.pos - self.food_pos))
        self.food_eaten = 0
        self.hazard_hits = 0
        self.max_pop_hit = False

        # --- anchored interface neurons (protected from apoptosis) ---
        self.sensor_offsets = {
            "left": (2.5, -1.5, 0.0),
            "center": (3.0, 0.0, 0.0),
            "right": (2.5, 1.5, 0.0),
        }
        self.sensor_ids: list[int] = []
        for _off in self.sensor_offsets.values():
            n = self._spawn(self.agent.pos.copy(), "sensor", random_genome(self.rng))
            self.sensor_ids.append(n.id)

        self.motor_offsets = {"left": (0.0, -0.8, 0.0), "right": (0.0, 0.8, 0.0)}
        self.motor_ids: dict[str, int] = {}
        for side, _off in self.motor_offsets.items():
            n = self._spawn(self.agent.pos.copy(), "motor", random_genome(self.rng))
            self.motor_ids[side] = n.id

        # --- seed interneuron cloud around the agent ---
        for _ in range(n_seed_interneurons):
            pos = np.clip(center + self._np_rng.normal(0, 2.0, 3), 0, grid - 1)
            self._spawn(pos, "inter", random_genome(self.rng))

    # -- bookkeeping -------------------------------------------------
    def _spawn(self, pos, kind, genome, energy: float = 5.0) -> Neuron:
        nid = next(self._id_counter)
        n = Neuron(id=nid, pos=np.array(pos, dtype=float), kind=kind, genome=genome, energy=energy)
        self.neurons[nid] = n
        return n

    def _random_point(self) -> np.ndarray:
        return self._np_rng.uniform(4, self.grid - 4, 3)

    def _voxel(self, pos: np.ndarray) -> tuple[int, int, int]:
        c = np.clip(pos.astype(int), 0, self.grid - 1)
        return int(c[0]), int(c[1]), int(c[2])

    # -- chemical / nutrient field ------------------------------------
    def _sample_chem(self, pos):
        x, y, z = self._voxel(pos)
        return self.chem[x, y, z, 0], self.chem[x, y, z, 1]

    def _deposit_chem(self, pos, reward, pain):
        x, y, z = self._voxel(pos)
        self.chem[x, y, z, 0] += reward
        self.chem[x, y, z, 1] += pain

    def _consume_nutrient(self, pos) -> float:
        x, y, z = self._voxel(pos)
        avail = self.nutrient[x, y, z]
        take = min(avail, NUTRIENT_UPTAKE_RATE)
        self.nutrient[x, y, z] -= take
        return take

    def _scent(self, pos: np.ndarray, source: np.ndarray) -> float:
        d = float(np.linalg.norm(pos - source))
        return float(np.exp(-d / SCENT_TAU))

    # -- spatial hash ---------------------------------------------------
    def _rebuild_spatial_hash(self):
        self._hash = defaultdict(list)
        for nid, n in self.neurons.items():
            self._hash[bucket_key(n.pos)].append(nid)

    def _nearby_neurons(self, pos: np.ndarray, radius: float) -> list[Neuron]:
        if not self.use_spatial_hash:
            return [n for n in self.neurons.values() if np.linalg.norm(n.pos - pos) <= radius]
        bx, by, bz = bucket_key(pos)
        r = int(np.ceil(radius / BUCKET_SIZE)) + 1
        out = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    for nid in self._hash.get((bx + dx, by + dy, bz + dz), ()):
                        n = self.neurons.get(nid)
                        if n is not None and np.linalg.norm(n.pos - pos) <= radius:
                            out.append(n)
        return out

    # -- connectivity introspection ---------------------------------
    def sensorimotor_path(self, weight_thresh: float = 0.05) -> Optional[int]:
        """BFS from sensors to motors over synapses with weight >= threshold.
        Returns path length in hops if a bridge exists, else None."""
        motor_set = set(self.motor_ids.values())
        visited = {sid: 0 for sid in self.sensor_ids}
        q = deque(self.sensor_ids)
        while q:
            cur = q.popleft()
            if cur in motor_set:
                return visited[cur]
            n = self.neurons.get(cur)
            if n is None:
                continue
            for tid, w in n.synapses.items():
                if w >= weight_thresh and tid not in visited and tid in self.neurons:
                    visited[tid] = visited[cur] + 1
                    q.append(tid)
        return None

    # -- main loop ----------------------------------------------------
    def step(self) -> dict:
        t0 = time.perf_counter()

        # 1. anchor sensors/motors to the agent's current body pose
        for name, nid in zip(self.sensor_offsets, self.sensor_ids):
            self.neurons[nid].pos = np.clip(self.agent.anchor(*self.sensor_offsets[name]), 0, self.grid - 1)
        for side, nid in self.motor_ids.items():
            self.neurons[nid].pos = np.clip(self.agent.anchor(*self.motor_offsets[side]), 0, self.grid - 1)

        # 2. spike propagation: push last cycle's fired neurons' current onto targets
        input_buf: dict[int, float] = defaultdict(float)
        for n in self.neurons.values():
            if n.fired:
                for tid, w in n.synapses.items():
                    if tid in self.neurons:
                        input_buf[tid] += w

        # sensors additionally get an external "scent" current from their body position
        food_scent_gain = SENSOR_GAIN
        hazard_scent_gain = -SENSOR_GAIN * HAZARD_SENSOR_FACTOR
        for nid in self.sensor_ids:
            n = self.neurons[nid]
            ext = (food_scent_gain * self._scent(n.pos, self.food_pos)
                   + hazard_scent_gain * self._scent(n.pos, self.hazard_pos))
            input_buf[nid] += ext

        for n in self.neurons.values():
            n.update(input_buf.get(n.id, 0.0), self.rng)

        # 3. Hebbian coincidence -> eligibility trace (per synapse, not per neuron)
        for n in self.neurons.values():
            if not n.synapses:
                continue
            for tid in n.synapses:
                target = self.neurons.get(tid)
                coincidence = 1.0 if (n.fired and target is not None and target.fired) else 0.0
                n.elig[tid] = n.elig.get(tid, 0.0) * ELIG_DECAY + coincidence

        # 4. motor integration -> move the body
        left_rate = self.neurons[self.motor_ids["left"]].rate
        right_rate = self.neurons[self.motor_ids["right"]].rate
        self.agent.apply_thrust(left_rate, right_rate, self.grid)

        # 5. reward / pain from the world, deposited locally at the agent's new position
        dist = float(np.linalg.norm(self.agent.pos - self.food_pos))
        reward = APPROACH_REWARD if dist < self.prev_dist else 0.0
        if dist < FOOD_RADIUS:
            reward += FOOD_REWARD
            self.food_eaten += 1
            self.food_pos = self._random_point()
            dist = float(np.linalg.norm(self.agent.pos - self.food_pos))
        self.prev_dist = dist

        pain = 0.0
        if float(np.linalg.norm(self.agent.pos - self.hazard_pos)) < HAZARD_RADIUS:
            pain += HAZARD_PAIN
            self.hazard_hits += 1
            self.hazard_pos = self._random_point()

        self._deposit_chem(self.agent.pos, reward, pain)

        # 6. reward-modulated weight update, sampled locally at each synapse's target
        for n in self.neurons.values():
            for tid in list(n.synapses.keys()):
                target = self.neurons.get(tid)
                if target is None:
                    del n.synapses[tid]
                    n.elig.pop(tid, None)
                    continue
                local_r, local_p = self._sample_chem(target.pos)
                trace = n.elig.get(tid, 0.0)
                dw = LR_PLUS * trace * local_r - LR_MINUS * trace * local_p
                if dw != 0.0:
                    n.synapses[tid] = float(np.clip(n.synapses[tid] + dw, W_MIN, W_MAX))

        # 7. metabolism: nutrient income, ambient chem exposure, upkeep
        for n in self.neurons.values():
            nutrient = self._consume_nutrient(n.pos)
            local_r, local_p = self._sample_chem(n.pos)
            n.energy += nutrient * NUTRIENT_TO_ENERGY - UPKEEP_COST
            n.energy += local_r * REWARD_ENERGY_GAIN - local_p * PAIN_ENERGY_COST

        # 8. growth (axon extension) applies to every neuron -- sensors and motors
        # must be able to originate synapses or they can never become part of any
        # signal path; only mitosis (population growth) is restricted to interneurons,
        # since sensors/motors are fixed structural anchors on the body, not a
        # population that should be multiplying.
        self._rebuild_spatial_hash()
        newborns = []
        for n in list(self.neurons.values()):
            if n.kind == "inter" and not self.freeze_population:
                if n.energy > n.genome["mitosis_thresh"] and len(self.neurons) + len(newborns) < self.max_pop:
                    newborns.append(self._mitosis(n))
                elif len(self.neurons) + len(newborns) >= self.max_pop:
                    self.max_pop_hit = True
            if len(n.synapses) < n.genome["max_synapses"] and n.energy > GROWTH_COST and self.rng.random() < 0.15:
                self._attempt_growth(n)
        for child in newborns:
            self.neurons[child.id] = child

        # 9. apoptosis (anchors are exempt -- the interface never disappears)
        if not self.freeze_population:
            dead = [nid for nid, n in self.neurons.items() if n.kind == "inter" and n.energy <= 0]
            for nid in dead:
                del self.neurons[nid]

        # 10. diffuse/evaporate chemical field, regenerate nutrient
        c = self.chem
        blurred = (c + np.roll(c, 1, 0) + np.roll(c, -1, 0) + np.roll(c, 1, 1)
                   + np.roll(c, -1, 1) + np.roll(c, 1, 2) + np.roll(c, -1, 2)) / 7.0
        self.chem = blurred * EVAPORATION
        self.nutrient = np.clip(self.nutrient + NUTRIENT_REGEN, 0, NUTRIENT_CAP)

        self.cycle += 1
        dt = time.perf_counter() - t0
        return {
            "cycle": self.cycle,
            "population": len(self.neurons),
            "n_interneurons": sum(1 for n in self.neurons.values() if n.kind == "inter"),
            "mean_energy": float(np.mean([n.energy for n in self.neurons.values()])) if self.neurons else 0.0,
            "n_edges": sum(len(n.synapses) for n in self.neurons.values()),
            "dist_to_food": dist,
            "food_eaten": self.food_eaten,
            "hazard_hits": self.hazard_hits,
            "step_time_s": dt,
            "max_pop_hit": self.max_pop_hit,
        }

    def _mitosis(self, parent: Neuron) -> Neuron:
        parent.energy /= 2.0
        child_genome = mutate_genome(parent.genome, self.rng)
        child_pos = np.clip(parent.pos + self._np_rng.normal(0, 0.75, 3), 0, self.grid - 1)
        return Neuron(id=next(self._id_counter), pos=child_pos, kind="inter",
                      genome=child_genome, energy=parent.energy)

    def _attempt_growth(self, n: Neuron) -> None:
        radius = n.genome["growth_radius"]
        candidates = [c for c in self._nearby_neurons(n.pos, radius) if c.id != n.id and c.id not in n.synapses]
        if not candidates:
            return
        scores = []
        for c in candidates:
            r, p = self._sample_chem(c.pos)
            d = float(np.linalg.norm(c.pos - n.pos)) + 1e-6
            scores.append(max(r - p, 0.01) / d)
        scores_arr = np.array(scores)
        probs = scores_arr / scores_arr.sum()
        choice = self._np_rng.choice(len(candidates), p=probs)
        target = candidates[choice]
        n.synapses[target.id] = self.rng.uniform(0.05, 0.2)
        n.elig[target.id] = 0.0
        n.energy -= GROWTH_COST

    def summary(self) -> dict:
        by_kind: dict[str, int] = defaultdict(int)
        for n in self.neurons.values():
            by_kind[n.kind] += 1
        return dict(by_kind)
