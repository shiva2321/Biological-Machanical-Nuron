"""One-off: regenerate A2/C/D charts with the corrected self-verifying
plot() (now includes the mp_beta/mp_gamma panel). Deterministic reruns of
the same seed/params used in followup_checks.py -- same numbers, updated
chart only."""

import learning_wall_experiment as lwe

_, hist_a2, _ = lwe.run(seed=7, fixed_spontaneity=None, label="A2: evolvable-spontaneity (+ mp tracking)")
lwe.plot(hist_a2, filename="followup_A2_mp_drift.png",
         suptitle="mp_beta Drift -- does intrinsic-plasticity direction get selected away too?")

_, hist_d, _ = lwe.run(seed=7, fixed_spontaneity=0.02, disable_reward=True,
                        label="D: fixed-spontaneity, reward DISABLED")
lwe.plot(hist_d, filename="followup_D_zero_reward.png",
         suptitle="Zero-Reward Control -- spontaneity fixed, no reward/pain ever deposited")

_, hist_c, _ = lwe.run(seed=7, fixed_spontaneity=0.02, continuous_shaping=True,
                        label="C: fixed-spontaneity, continuous shaping")
lwe.plot(hist_c, filename="followup_C_continuous_shaping.png",
         suptitle="Continuous Shaping -- reward proportional to distance closed, not a fixed bonus")

print("Regenerated A2, D, C plots with mp_beta/mp_gamma panel.")
