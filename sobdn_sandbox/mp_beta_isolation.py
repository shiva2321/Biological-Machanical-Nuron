"""Run E -- the clean, unconfounded mp_beta isolation test.

Pinning spontaneity alone (spontaneity_ablation.py, followup_checks.py)
doesn't isolate mp_beta: v_thresh is still free to absorb the same
"reduce firing cost" selection pressure documented in FINDINGS.md point 7
(spontaneity fixed -> v_thresh climbs instead, replicated three-for-three
regardless of reward regime). This pins BOTH fixed_spontaneity and
fixed_v_thresh, leaving mp_beta/mp_gamma as the only evolvable
excitability-related genes -- the actual unconfounded test of whether
mp_beta erodes under selection the way spontaneity did.
"""

import learning_wall_experiment as lwe

world, history, summary = lwe.run(
    seed=7, fixed_spontaneity=0.02, fixed_v_thresh=0.88,
    label="E: spontaneity+v_thresh fixed, mp_beta isolated",
)
lwe.plot(history, filename="followup2_E_mp_beta_isolated.png",
         suptitle="mp_beta Isolated -- spontaneity AND v_thresh both pinned, only mp_beta/mp_gamma free")

print(f"\nmp_beta first/last: {history['mean_mp_beta'][0]} {history['mean_mp_beta'][-1]}")
print(f"mp_gamma first/last: {history['mean_mp_gamma'][0]} {history['mean_mp_gamma'][-1]}")
