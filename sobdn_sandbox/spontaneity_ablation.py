"""
SOBDN Sandbox -- Spontaneity Ablation

Direct test of the tragedy-of-the-commons finding in FINDINGS.md: if
individual-level energy selection erodes population-level exploration
because spontaneity is a heritable, individually-costed gene, then
removing that selection pressure (pin spontaneity to a fixed constant,
same for every neuron, immune to mutation/selection) should let the
mechanism actually learn -- if learning was being smothered by its own
selection dynamics rather than blocked by something more fundamental.

Runs the identical 20,000-cycle capped learning test twice, same seed:
  A) baseline: spontaneity heritable and individually selected
  B) ablated: spontaneity fixed at a constant for every neuron, always,
     immune to mutation

If (B) improves distance-to-food where (A) didn't: the mechanism
basically works and was fighting its own selection pressure (fixable).
If (B) still doesn't learn: the problem is deeper than the noise floor --
most likely the missing live sensor->motor bridge -- and no amount of
noise tuning fixes it.
"""

import learning_wall_experiment as lwe

FIXED_VALUE = 0.02  # comfortably above the ~0.011 the evolvable run converged to


def main():
    print("RUN A: evolvable spontaneity (replicates the original learning_wall run)\n")
    _, hist_a, summ_a = lwe.run(seed=7, fixed_spontaneity=None, label="A: evolvable-spontaneity")
    lwe.plot(hist_a, filename="ablation_A_evolvable.png",
             suptitle="Spontaneity Ablation -- A: Evolvable (selected against)")

    print("\n\nRUN B: spontaneity fixed at", FIXED_VALUE, "for every neuron, immune to selection\n")
    _, hist_b, summ_b = lwe.run(seed=7, fixed_spontaneity=FIXED_VALUE, label="B: fixed-spontaneity")
    lwe.plot(hist_b, filename="ablation_B_fixed.png",
             suptitle=f"Spontaneity Ablation -- B: Fixed at {FIXED_VALUE} (not selectable)")

    delta_a = summ_a["second_half_dist"] - summ_a["first_half_dist"]
    delta_b = summ_b["second_half_dist"] - summ_b["first_half_dist"]
    eaten_a = summ_a["first_half_eaten"] + summ_a["second_half_eaten"]
    eaten_b = summ_b["first_half_eaten"] + summ_b["second_half_eaten"]

    print("\n" + "=" * 78)
    print("ABLATION VERDICT")
    print("=" * 78)
    print(f"{'metric':38s} {'A: evolvable':>16s} {'B: fixed':>16s}")
    print(f"{'dist-to-food, first half':38s} {summ_a['first_half_dist']:16.2f} {summ_b['first_half_dist']:16.2f}")
    print(f"{'dist-to-food, second half':38s} {summ_a['second_half_dist']:16.2f} {summ_b['second_half_dist']:16.2f}")
    print(f"{'change (negative = improved)':38s} {delta_a:+16.2f} {delta_b:+16.2f}")
    print(f"{'total food eaten':38s} {eaten_a:16d} {eaten_b:16d}")
    print(f"{'% logged cycles with ANY path':38s} {summ_a['frac_path']*100:15.1f}% {summ_b['frac_path']*100:15.1f}%")
    print(f"{'% logged cycles with LIVE path':38s} {summ_a['frac_live']*100:15.1f}% {summ_b['frac_live']*100:15.1f}%")

    print()
    if delta_b < delta_a - 1.0 or eaten_b > eaten_a:
        print("VERDICT: fixing spontaneity meaningfully improved the trend --")
        print("the mechanism was fighting its own selection pressure. Noise-tuning is")
        print("a real, cheap fix; the growth/credit-assignment design is more viable")
        print("than the base run alone suggested.")
    else:
        print("VERDICT: fixing spontaneity did NOT meaningfully change the outcome --")
        print("the blocker is deeper than the noise floor. Most likely culprit per the")
        print("path-liveness data above: still no LIVE sensor->motor bridge forming,")
        print("so no amount of exploration noise has anything to reinforce.")


if __name__ == "__main__":
    main()
