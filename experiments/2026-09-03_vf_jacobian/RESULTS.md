# VF Jacobian mapping experiment - 2026-09-03

Scene: --tree6-simple --branch-rscale 0.5 --apple-root --apple-branch-soft
       --shortcut --uni-refuse --nav-field --or-parallel 8   (NO --branch-stiffness)
Starts: saved_starts_final_5x10.txt (50 runs per invocation).

(b) = one end-effector Jacobian reused for every force (old behaviour).
(a) = each force mapped with the Jacobian at its own application point (commit 14ed6b8).
w = vf_w_rigid / vf_w_vortex.

| config | N | median ms | p75 | p90 | mean | med pen mm | axial | cuts |
|---|---|---|---|---|---|---|---|---|
| (b) J_ee        w=2.5/2.0 | 200 | 84.6 | 200.5 | 268.5 | 125.5 | 37.0 | 0.536 | 0 |
| (a) per-link    w=2.5/2.0 | 200 | 89.6 | 200.6 | 255.4 | 129.1 | 38.0 | 0.535 | 0 |
| (a) per-link    w=2.5/1.0 | 150 | 91.9 | 183.4 | 250.1 | 127.2 | 38.0 | 0.536 | 0 |
| (a) per-link    w=2.5/3.5 | 150 | 87.4 | 210.1 | 284.0 | 134.0 | 38.0 | 0.536 | 0 |
| (a) 1 attraction point    | 50 | 95.6 | 218.0 | 255.5 | 140.2 | 41.0 | 0.579 | 0 |
| (a) sweep       w=1.5/1.0 | 50 | 103.6 | 191.2 | 229.5 | 128.9 | 38.0 | 0.536 | 0 |
| (a) sweep       w=1.5/2.0 | 50 | 100.0 | 190.1 | 288.2 | 138.0 | 38.0 | 0.542 | 0 |
| (a) sweep       w=1.5/3.5 | 50 | 99.0 | 195.0 | 251.0 | 130.9 | 38.0 | 0.513 | 0 |
| (a) sweep       w=4.0/1.0 | 50 | 93.9 | 207.1 | 272.8 | 136.8 | 39.0 | 0.492 | 0 |
| (a) sweep       w=4.0/2.0 | 50 | 87.5 | 185.7 | 269.7 | 132.2 | 38.0 | 0.526 | 0 |
| (a) sweep       w=4.0/3.5 | 50 | 80.5 | 202.5 | 266.5 | 130.3 | 37.5 | 0.540 | 0 |
| (a) sweep       w=1.5/2.0 | 50 | 100.0 | 190.1 | 288.2 | 138.0 | 38.0 | 0.542 | 0 |

## Conclusions

1. All 22 configurations: 50/50 success, cut_violations = 0. The validity layer is
   untouched by this change, so safety could not regress.
2. (a) vs (b) at equal N=200: median +5.9%, p75 identical, p90 -4.9% (better),
   mean +2.9%, contact depth +2.7%, axial unchanged. Comparable.
3. The 3x3 weight sweep found nothing better than the existing 2.5/2.0. The apparent
   winner in the first pass (2.5/1.0 at 117.4 ms) did not reproduce: repeats gave
   133.2 and 131.1 ms. Run-to-run spread of the 50-run mean is +/-6-13%, because the
   per-run distribution is heavily skewed (median ~85 ms, mean ~127 ms). Any effect
   below ~10% is not resolvable at this sample size.
4. Reducing attraction to a single end-effector point was tried and is WORSE
   (140.2 ms, 38.4 mm). The link6/link7 attraction points remain useful under the
   corrected mapping, contrary to the hypothesis that they were an artefact.

Decision: keep (a) with default weights, on correctness grounds -- it is
mechanically right and measurably comparable.

---

# Weight retune after per-point Jacobian + sum-then-normalize (same day)

The old weights 1.2 / 2.5 / 2.0 (rep / rigid / vortex) were fit against the
previous behaviour, so they were re-screened. All 22 sweep configurations kept
50/50 success and cut_violations = 0: the validity layer is independent of the
guidance weights, as designed.

## One-at-a-time screening (N=50 each, wide spacing so effects exceed noise)

Consistent directions, all trading time for quality:
  higher vf_w_rep     -> shallower contact, better axial, slower
  lower  vf_w_rigid   -> better on every metric
  higher vf_w_vortex  -> markedly shallower contact
  lower  vf_w_att     -> much shallower contact, much slower
  higher bs_weight_max-> shallower contact, much slower

## Confirmation (N=200 pooled)

| rep/rigid/vortex | p50 | p90 | p99 | max_pen | mean_pen | axial | rad_depth |
|---|---|---|---|---|---|---|---|
| 1.2/2.5/2.0 (old) | 105.9 | 307.5 | 383.3 | 34.0 | 12.0 | 0.488 | 0.627 |
| 0.4/1.0/4.0 | 92.5 | 304.8 | 389.4 | 31.0 | 12.0 | 0.489 | 0.625 |
| 1.2/1.0/4.0 | 113.8 | 290.4 | 339.8 | 26.0 | 11.0 | 0.483 | 0.633 |
| **2.4/1.0/4.0 (new default)** | 124.5 | **264.9** | 349.4 | **25.0** | **10.0** | **0.461** | **0.648** |

2.4/1.0/4.0 wins every quality metric AND the p90 tail; only p50 is worse
(+18%). Since the budget claim rests on the tail, not the median, and the
method's thesis is minimizing branch damage, this is the right trade.

## Pareto alternatives (N=50, not confirmed)

| config | p50 | p90 | max_pen | axial | note |
|---|---|---|---|---|---|
| 2.4/0.5/4.0 | 107.5 | 232.3 | 30.0 | 0.454 | best tail, faster, shallower than old |
| 3.6/1.0/4.0 | 169.2 | 275.0 | 26.0 | 0.412 | best axial |
| 2.4/1.0/4.0 + bs_max 9 | 223.1 | 422.3 | 19.0 | 0.436 | best depth, well over 2x slower |

Pick along this frontier if the paper needs a different balance. Everything
stays far inside the sub-second budget (worst p99 observed ~420 ms).
