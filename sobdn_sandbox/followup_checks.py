"""
SOBDN Sandbox -- Second Follow-up Round

Three more things checked rather than asserted, all reusing the same
20,000-cycle, population-150, seed=7 harness as spontaneity_ablation.py:

1. Zero-reward control: rerun the fixed-spontaneity(0.02) condition with
   disable_reward=True. If the direct sensor->motor edge weight (0.205 in
   the real run) is genuinely reward-driven, it must never exceed 0.2 in
   this control -- 0.2 is the hard max of the initial random draw
   (uniform(0.05, 0.2)) and the ONLY other code path that changes a
   synapse's weight is the reward-modulated update, which is
   mathematically a no-op when reward and pain are both pinned to zero.

2. mp_beta drift: does the "Memory Protein" adaptation gene -- which is
   heritable and mutable exactly like spontaneity was -- also get
   selected away by the same individual-energy-cost logic? Reruns the
   evolvable-spontaneity condition (reproducing the original run
   bit-for-bit) with mp_beta/mp_gamma now tracked over time.

3. Continuous potential-based reward shaping (Ng, Harada & Russell 1999),
   Phi(s) = -distance-to-food: replaces the fixed binary "any improvement
   gets +0.05" bonus with a reward proportional to how much distance was
   actually closed each cycle (and negative when moving away). Tested on
   top of fixed_spontaneity=0.02 so only the shaping variable changes.
"""

import learning_wall_experiment as lwe

FIXED_SPONTANEITY = 0.02


def main():
    print("#" * 78)
    print("CHECK 1/3: ZERO-REWARD CONTROL")
    print("#" * 78)
    _, hist_d, summ_d = lwe.run(seed=7, fixed_spontaneity=FIXED_SPONTANEITY, disable_reward=True,
                                 label="D: fixed-spontaneity, reward DISABLED")
    lwe.plot(hist_d, filename="followup_D_zero_reward.png",
             suptitle="Zero-Reward Control -- spontaneity fixed, no reward/pain ever deposited")
    max_direct_w = max(hist_d["mean_direct_weight"]) if hist_d["mean_direct_weight"] else 0.0
    print(f"\n>>> Max direct sensor->motor weight ever observed with reward disabled: {max_direct_w:.4f}")
    print(f">>> (Real run B, reward enabled, reached 0.205 -- above the 0.2 max possible "
          f"from initial draw alone)")
    if max_direct_w > 0.2:
        print(">>> UNEXPECTED: weight exceeded 0.2 even with reward disabled -- would mean "
              "some other code path changes weights. Needs investigation.")
    else:
        print(">>> CONFIRMS: without reward, weights never exceed what growth alone can draw -- "
              "the 0.205 in the real run required reward-modulated plasticity.")

    print("\n\n" + "#" * 78)
    print("CHECK 2/3: mp_beta DRIFT (evolvable spontaneity, reproduces original run A)")
    print("#" * 78)
    _, hist_a, summ_a = lwe.run(seed=7, fixed_spontaneity=None, label="A2: evolvable-spontaneity (+ mp tracking)")
    lwe.plot(hist_a, filename="followup_A2_mp_drift.png",
             suptitle="mp_beta Drift -- does intrinsic-plasticity direction get selected away too?")
    beta_first, beta_last = hist_a["mean_mp_beta"][0], hist_a["mean_mp_beta"][-1]
    gamma_first, gamma_last = hist_a["mean_mp_gamma"][0], hist_a["mean_mp_gamma"][-1]
    print(f"\n>>> mean mp_beta:  first={beta_first:+.4f} -> last={beta_last:+.4f} "
          f"(more negative = more facilitating/easier to keep firing)")
    print(f">>> mean mp_gamma: first={gamma_first:.4f} -> last={gamma_last:.4f} (higher = adaptation lasts longer)")
    if beta_last > beta_first + 0.01:
        print(">>> mp_beta drifted LESS negative (toward habituating) -- same erosion pattern as spontaneity.")
    elif beta_last < beta_first - 0.01:
        print(">>> mp_beta drifted MORE negative (toward facilitating) -- opposite of the spontaneity pattern.")
    else:
        print(">>> mp_beta stayed roughly flat -- no strong selection pressure detected either way.")

    print("\n\n" + "#" * 78)
    print("CHECK 3/3: CONTINUOUS POTENTIAL-BASED REWARD SHAPING")
    print("#" * 78)
    _, hist_c, summ_c = lwe.run(seed=7, fixed_spontaneity=FIXED_SPONTANEITY, continuous_shaping=True,
                                 label="C: fixed-spontaneity, continuous shaping")
    lwe.plot(hist_c, filename="followup_C_continuous_shaping.png",
             suptitle="Continuous Shaping -- reward proportional to distance closed, not a fixed bonus")

    delta_c = summ_c["second_half_dist"] - summ_c["first_half_dist"]
    print(f"\n>>> dist-to-food: first half={summ_c['first_half_dist']:.2f} "
          f"second half={summ_c['second_half_dist']:.2f} (change {delta_c:+.2f})")
    print(f">>> food eaten: {summ_c['first_half_eaten'] + summ_c['second_half_eaten']}")
    print(f">>> Compare to fixed-spontaneity baseline (binary shaping, from spontaneity_ablation.py): "
          f"first=27.87 second=37.43 (change +9.55), food eaten=0")

    print("\n\n" + "#" * 78)
    print("SUMMARY")
    print("#" * 78)
    print(f"{'condition':40s} {'1st half':>10s} {'2nd half':>10s} {'change':>10s} {'eaten':>7s}")
    print(f"{'B: fixed-spontaneity (baseline)':40s} {27.87:10.2f} {37.43:10.2f} {'+9.55':>10s} {0:7d}")
    print(f"{'D: fixed-spontaneity, no reward':40s} {summ_d['first_half_dist']:10.2f} "
          f"{summ_d['second_half_dist']:10.2f} "
          f"{summ_d['second_half_dist']-summ_d['first_half_dist']:+10.2f} "
          f"{summ_d['first_half_eaten']+summ_d['second_half_eaten']:7d}")
    print(f"{'C: fixed-spontaneity, continuous shaping':40s} {summ_c['first_half_dist']:10.2f} "
          f"{summ_c['second_half_dist']:10.2f} {delta_c:+10.2f} "
          f"{summ_c['first_half_eaten']+summ_c['second_half_eaten']:7d}")


if __name__ == "__main__":
    main()
