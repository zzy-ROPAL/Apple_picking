#!/usr/bin/env python3
"""Five-way comparison of contact attribution variants.

Crosses two independent choices:

  FIT STRUCTURE   old = fit each branch ALONE against the measurement and take the
                        lowest residual (what attribute() does today)
                  new = solve ALL branches simultaneously in one least-squares system

  MEASUREMENT     tau           the torque itself
                  dtau/dt       its slope over a 0.5 s window (240 Hz, 120 samples)

  plus the stiffness variant, which is the only one that divides by the per-branch
  push rate v and therefore ranks by k rather than by k*x or k*v.

M1 old fit, tau                  -> ranks by FORCE
M2 new fit, tau                  -> ranks by FORCE
M3 new fit, 3 mm retreat         -> ranks by STIFFNESS   (k = df/dpen)
M4 old fit, dtau/dt over 0.5 s   -> ranks by FORCE CHANGE
M5 new fit, dtau/dt over 0.5 s   -> ranks by FORCE CHANGE

Rigid branch is always the SHALLOWEST, 5x stiffer -- the adversarial case.
"""
import argparse, collections, math, random, statistics, sys
import stiffness_probe as sp
from stiffness_probe import (contacts_at, branches_of, tau_of, dot, solve_lstsq,
                             fit_slopes, find_cases, est_probe, est_rate)
from contact_monitor import ContactAttributor, PandaKin
import os


def agg(by, b):
    n = [0.0] * 7
    for c in by[b]:
        for j in range(7):
            n[j] += c["h"][j]
    return n


def pick_old(order, by, meas):
    """OLD structure: each branch fitted alone, lowest residual wins."""
    best = None
    for b in order:
        n = agg(by, b)
        nn = dot(n, n)
        if nn < 1e-18:
            continue
        f = dot(meas, n) / nn
        res = math.sqrt(sum((meas[j] - f * n[j]) ** 2 for j in range(7)))
        if best is None or res < best[1]:
            best = (b, res)
    return best[0] if best else None


def pick_new(order, by, meas):
    """NEW structure: all branches solved together, largest coefficient wins."""
    cols = [agg(by, b) for b in order]
    sol, _ = solve_lstsq(cols, meas)
    if sol is None:
        return None
    return order[max(range(len(order)), key=lambda i: sol[i])]


def run_case(kin, att, case, ratio, probe_mm, noise, rng, win_s, win_n):
    cs = contacts_at(kin, att, case["q"])
    order, by = branches_of(cs)
    if not order:
        return None
    k0 = 300.0
    rigid = order[-1]                       # shallowest = rigid (adversarial)
    ks = {b: k0 for b in order}
    ks[rigid] = k0 * ratio

    tau = tau_of(cs, ks)
    tau_m = [v + rng.gauss(0, noise) for v in tau] if noise > 0 else list(tau)

    # probe / window direction: along the retreat
    d = [a - b for a, b in zip(case["q_prev"], case["q"])]
    nd = math.sqrt(dot(d, d))
    if nd < 1e-9:
        return None
    u = [x / nd for x in d]
    rate0 = abs(dot(u, cs[0]["h"]))
    if rate0 < 1e-6:
        return None
    dq = [x * (probe_mm * 1e-3) / rate0 for x in u]

    # M3: discrete 3 mm retreat, stiffness
    q2 = [a + b for a, b in zip(case["q"], dq)]
    cs2 = contacts_at(kin, att, q2)
    ks2 = collections.defaultdict(lambda: k0); ks2.update(ks)
    t2 = tau_of(cs2, ks2)
    t2_m = [v + rng.gauss(0, noise) for v in t2] if noise > 0 else list(t2)
    dtau = [a - b for a, b in zip(t2_m, tau_m)]
    sol, _ = est_probe(order, by, dtau, dq)
    m3 = order[max(range(len(order)), key=lambda i: sol[i])] if sol else None

    # window for M4/M5/M6: the final approach INTO this pose.
    #
    # BUG FIX 2026-10-26: the travel must not exceed the SHALLOWEST contact's
    # penetration, or that branch is simply not touching at the start of the window,
    # its torque contribution switches on partway through, and a straight-line fit
    # underestimates its slope. The rigid branch is the shallowest by construction,
    # so the bug hit exactly the branch being searched for. Cap the travel at 40% of
    # the shallowest penetration so every contact persists across the whole window.
    pen_min = min(c["pen"] for c in cs)
    win_mm = min(probe_mm, 0.4 * pen_min * 1e3)
    dqw = [x * (win_mm * 1e-3) / rate0 for x in u]
    qdot = [x / win_s for x in dqw]
    ts, taus = [], []
    for i in range(win_n):
        f = (i + 1) / win_n
        qi = [a - b * (1.0 - f) for a, b in zip(case["q"], dqw)]
        ci = contacts_at(kin, att, qi)
        ksi = collections.defaultdict(lambda: k0); ksi.update(ks)
        tv = tau_of(ci, ksi)
        if noise > 0:
            tv = [x + rng.gauss(0, noise) for x in tv]
        ts.append(f * win_s); taus.append(tv)
    slope = fit_slopes(ts, taus)

    # M6: same window, but DIVIDED by the per-branch push rate -> stiffness
    m6 = None
    if slope:
        sol6, _ = est_rate(order, by, slope, qdot)
        if sol6:
            m6 = order[max(range(len(order)), key=lambda i: sol6[i])]

    return {
        "rigid": rigid, "nb": len(order),
        "M1": pick_old(order, by, tau_m) == rigid,
        "M2": pick_new(order, by, tau_m) == rigid,
        "M3": m3 == rigid,
        "M4": (pick_old(order, by, slope) == rigid) if slope else False,
        "M5": (pick_new(order, by, slope) == rigid) if slope else False,
        "M6": m6 == rigid,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--cases", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--ratio", type=float, default=5.0)
    ap.add_argument("--probe-mm", type=float, default=3.0)
    ap.add_argument("--win-s", type=float, default=0.5)
    ap.add_argument("--win-n", type=int, default=120)
    a = ap.parse_args()

    kin = PandaKin()
    att = ContactAttributor(os.path.join(a.results, "scene_layout.json"), kin)
    LBL = {"M1": "1. OLD fit , tau            -> force",
           "M2": "2. NEW fit , tau            -> force",
           "M3": f"3. NEW fit , {a.probe_mm:.0f}mm retreat   -> STIFFNESS",
           "M4": f"4. OLD fit , dtau/dt {a.win_s}s   -> force change",
           "M5": f"5. NEW fit , dtau/dt {a.win_s}s   -> force change",
           "M6": f"6. NEW fit , dtau/dt {a.win_s}s   -> STIFFNESS"}
    for noise in (0.0, 0.5):
        print(f"\n{'='*74}\nnoise {noise} N*m   (rigid = shallowest branch, {a.ratio:.0f}x stiffer)\n{'='*74}")
        print(f"{'method':38s} {'1 branch':>10s} {'2 branches':>11s} {'3 branches':>11s}")
        tot = {m: {} for m in LBL}
        for nb in (1, 2, 3):
            cases = find_cases(kin, att, a.results, nb, a.cases, a.stride)
            acc = collections.defaultdict(list)
            for c in cases:
                for s in range(a.seeds):
                    r = run_case(kin, att, c, a.ratio, a.probe_mm, noise,
                                 random.Random(s), a.win_s, a.win_n)
                    if r is None:
                        continue
                    for m in LBL:
                        acc[m].append(r[m])
            for m in LBL:
                v = acc[m]
                tot[m][nb] = (sum(1 for x in v if x), len(v))
        for m in ("M1", "M2", "M3", "M4", "M5", "M6"):
            cells = []
            for nb in (1, 2, 3):
                ok, n = tot[m].get(nb, (0, 0))
                cells.append(f"{100*ok/n:5.1f}% ({ok}/{n})" if n else "   --")
            print(f"{LBL[m]:38s} {cells[0]:>10s} {cells[1]:>11s} {cells[2]:>11s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
