#!/usr/bin/env python3
"""Accuracy sweeps for the torque-based contact attribution in contact_monitor.py.

This file is COMMITTED on purpose. The numbers it produces are quoted in CLAUDE.md,
and the previous versions were one-off scratch scripts that were lost when the
scratchpad was cleared -- leaving a 10x units ambiguity (mm vs cm) in the headline
registration-tolerance figure that could not be resolved from the surviving output.

Two conventions worth stating up front, because both were implicit before:

UNITS. Every length is printed with its unit spelled out in the row label. The
perturbation sweeps are driven in METRES internally and labelled in cm.

SIGN. The forward model here uses tau = f * J^T(-r_hat), the SAME convention as
ContactAttributor.attribute(). Both are believed wrong relative to real hardware
(see "sign convention" in CLAUDE.md: r_hat points axis->probe, the branch pushes
the arm along +r_hat, so tau_ext = J^T(+r_hat)*f). Keeping the two matched means
these sweeps measure ESTIMATOR quality and are blind to the sign bug -- which is
exactly what we want for comparing against the historical numbers, and exactly
what must NOT be relied on when real torque data arrives.

CONTACT COUNT. The original self_test synthesised tau from a SINGLE contact. The
canopy routinely has several at once (the gate permits 15-75 mm of intentional
penetration), so every sweep here reports the single-contact case and the
multi-contact case side by side.
"""

import argparse
import glob
import json
import math
import os
import random
import statistics
import sys

import contact_monitor as cm
from contact_monitor import PEN_PROBES, LINK_JOINT_COUNT, ContactAttributor, PandaKin


# ======================================================================
# Ground-truth contact extraction
# ======================================================================

def contacts_at(kin, att, q, min_pen=0.0):
    """Every real (probe, soft cylinder) penetration at configuration q.

    Returns a list of dicts sorted deepest-first. 'rhat' points from the cylinder
    axis to the probe centre, matching the gate's and the attributor's convention.
    """
    out = []
    pos = kin.probe_positions(q)
    for pi, (pname, _off, prad) in enumerate(PEN_PROBES):
        for ci, cyl in enumerate(att.cyls):
            if not cyl["soft"]:
                continue
            seg = [cyl["p1"][d] - cyl["p0"][d] for d in range(3)]
            sq = sum(v * v for v in seg)
            if sq < 1e-12:
                continue
            u = max(0.0, min(1.0, sum((pos[pi][d] - cyl["p0"][d]) * seg[d]
                                      for d in range(3)) / sq))
            ax = cm._lerp(cyl["p0"], cyl["p1"], u)
            rad = [pos[pi][d] - ax[d] for d in range(3)]
            rn = math.sqrt(sum(v * v for v in rad))
            if rn < 1e-6:
                continue
            pen = (cyl["r"] + prad) - rn
            if pen > min_pen:
                out.append({"pen": pen, "probe": pi, "cyl": ci, "u": u,
                            "rhat": [v / rn for v in rad], "link": pname})
    out.sort(key=lambda c: -c["pen"])
    return out


def stall_poses(kin, att, path_csv):
    """The pose on this path with the deepest contact, plus all contacts there.

    One case per path: the deepest penetration anywhere is where a stall would
    actually occur (mid-path waypoints usually touch nothing).
    """
    with open(path_csv) as f:
        wps = [[float(x) for x in ln.split(",")] for ln in f if ln.strip()]
    best = None
    for q in wps:
        cs = contacts_at(kin, att, q)
        if cs and (best is None or cs[0]["pen"] > best[1][0]["pen"]):
            best = (q, cs)
    return best


# ======================================================================
# Forward model
# ======================================================================

def synth_tau(kin, q, contacts, forces, noise_nm=0.0, rng=None,
              rhat_override=None):
    """tau = sum_i f_i * J_i^T(-rhat_i), with joints distal to each contact zeroed.

    rhat_override: optional list parallel to `contacts`, used by the perturbation
    sweeps to generate truth from a DISPLACED branch while attribution still runs
    against the nominal model.
    """
    J = kin.probe_jacobians(q)
    tau = [0.0] * 7
    for idx, (c, f) in enumerate(zip(contacts, forces)):
        rhat = rhat_override[idx] if rhat_override else c["rhat"]
        k = LINK_JOINT_COUNT[c["link"]]
        for j in range(7):
            if j >= k:
                continue            # distal joints feel nothing from this contact
            tau[j] += -f * sum(J[c["probe"]][d][j] * rhat[d] for d in range(3))
    if noise_nm > 0.0:
        tau = [v + rng.gauss(0.0, noise_nm) for v in tau]
    return tau


def displaced_rhat(kin, att, q, c, offset_vec):
    """rhat as it would be if this cylinder's axis were displaced by offset_vec.

    This is what both the deflection and the registration sweep perturb: the TRUE
    branch sits somewhere else, the arm touches it there, but attribution still
    only knows the nominal axis.
    """
    cyl = att.cyls[c["cyl"]]
    p0 = [cyl["p0"][d] + offset_vec[d] for d in range(3)]
    p1 = [cyl["p1"][d] + offset_vec[d] for d in range(3)]
    pos = kin.probe_positions(q)[c["probe"]]
    seg = [p1[d] - p0[d] for d in range(3)]
    sq = sum(v * v for v in seg)
    u = max(0.0, min(1.0, sum((pos[d] - p0[d]) * seg[d] for d in range(3)) / sq))
    ax = cm._lerp(p0, p1, u)
    rad = [pos[d] - ax[d] for d in range(3)]
    rn = math.sqrt(sum(v * v for v in rad))
    if rn < 1e-6:
        return None
    return [v / rn for v in rad]


# ======================================================================
# Scoring
# ======================================================================

def score(att, kin, q, tau, target, F_target):
    """Did attribution name the target contact's cylinder? Plus error metrics."""
    got = att.attribute([(0.0, q, q, tau, 0.0)])
    if got is None:
        return {"ok": False, "none": True, "du_cm": None, "ferr": None,
                "res": None}
    cyl = att.cyls[target["cyl"]]
    seg_len = math.dist(cyl["p0"], cyl["p1"])
    return {
        "ok": got.cyl_idx == target["cyl"],
        "none": False,
        "du_cm": abs(got.u - target["u"]) * seg_len * 100.0,
        "ferr": abs(got.f_est - F_target),
        "res": got.residual,
    }


def summarize(rows):
    n = len(rows)
    ok = sum(1 for r in rows if r["ok"])
    nones = sum(1 for r in rows if r["none"])
    good = [r for r in rows if r["ok"]]
    du = statistics.median([r["du_cm"] for r in good]) if good else float("nan")
    fe = statistics.median([r["ferr"] for r in good]) if good else float("nan")
    rs = statistics.median([r["res"] for r in rows if r["res"] is not None]) \
        if any(r["res"] is not None for r in rows) else float("nan")
    return (f"branch correct {ok:3d}/{n:<3d} ({100.0*ok/max(1,n):5.1f}%)  "
            f"no-answer {nones:2d}  along-branch err median {du:5.1f} cm  "
            f"force err median {fe:5.2f} N  residual median {rs:6.3f} N*m")


# ======================================================================
# Cases
# ======================================================================

def build_cases(kin, att, results_dir, limit):
    cases = []
    for p in sorted(glob.glob(os.path.join(results_dir, "grasp_path_*.csv"))):
        sp = stall_poses(kin, att, p)
        if sp is None:
            continue
        q, cs = sp
        cases.append({"path": os.path.basename(p), "q": q, "contacts": cs})
        if len(cases) >= limit:
            break
    return cases


def forces_proportional(contacts, f_max=12.0):
    """f_i proportional to penetration -- a linear-spring branch model.

    The deepest contact carries the most load, so this is the FAVOURABLE
    multi-contact case ("the blocker dominates").
    """
    pmax = max(c["pen"] for c in contacts)
    return [f_max * c["pen"] / pmax for c in contacts]


# ======================================================================
# Sweeps
# ======================================================================

def sweep_noise(att, kin, cases, seeds, levels, multi):
    print(f"\n{'='*78}\nNOISE  ({'MULTI-contact' if multi else 'SINGLE contact'}) "
          f"-- {len(cases)} cases x {len(seeds)} seeds\n{'='*78}")
    for nm in levels:
        rows = []
        for case in cases:
            cs = case["contacts"] if multi else case["contacts"][:1]
            fs = forces_proportional(cs)
            target, F_t = cs[0], fs[0]
            for s in seeds:
                rng = random.Random(s)
                tau = synth_tau(kin, case["q"], cs, fs, nm, rng)
                rows.append(score(att, kin, case["q"], tau, target, F_t))
        print(f"  noise {nm:4.1f} N*m : {summarize(rows)}")


def sweep_offset(att, kin, cases, seeds, offsets_cm, multi, label, perturb_all):
    """DEFLECTION only: perturb the TRUE contacted branch, attribute against nominal.

    NOTE on a design trap found 2026-10-20: with a SINGLE contact, perturb_all has
    no effect at all (there are no other contacts to perturb), so a "registration"
    sweep written this way is bit-identical to the deflection sweep. It is also the
    wrong model of registration: displacing the TRUTH leaves every competing
    candidate at its nominal position relative to the probe, whereas a real
    registration error displaces the whole MODEL, moving the competitors too --
    and the competitors are what a wrong answer gets attributed to. Use
    sweep_registration() for that; this function is deflection only.
    """
    print(f"\n{'='*78}\n{label}  ({'MULTI-contact' if multi else 'SINGLE contact'}) "
          f"-- {len(cases)} cases x {len(seeds)} seeds\n{'='*78}")
    for off_cm in offsets_cm:
        rows = []
        for case in cases:
            cs = case["contacts"] if multi else case["contacts"][:1]
            fs = forces_proportional(cs)
            target, F_t = cs[0], fs[0]
            for s in seeds:
                rng = random.Random(s)
                # one random direction, magnitude off_cm
                v = [rng.gauss(0, 1) for _ in range(3)]
                n = math.sqrt(sum(x * x for x in v)) or 1.0
                vec = [x / n * off_cm * 0.01 for x in v]
                over = []
                bad = False
                for idx, c in enumerate(cs):
                    use = vec if (perturb_all or idx == 0) else [0.0, 0.0, 0.0]
                    rh = displaced_rhat(kin, att, case["q"], c, use)
                    if rh is None:
                        bad = True
                        break
                    over.append(rh)
                if bad:
                    continue
                tau = synth_tau(kin, case["q"], cs, fs, 0.0, rng,
                                rhat_override=over)
                rows.append(score(att, kin, case["q"], tau, target, F_t))
        print(f"  offset {off_cm:5.1f} cm : {summarize(rows)}")


def sweep_registration(att, kin, cases, seeds, offsets_cm, multi):
    """Registration error done properly: displace the whole MODEL, not the truth.

    The arm touches the real tree where it really is (truth from the nominal
    cylinders, which is what the planned path was generated against). Attribution
    then runs against a model whose every cylinder is offset by one rigid vector --
    so the competing candidates move too, which is the part that actually decides
    whether a wrong branch wins.
    """
    print(f"\n{'='*78}\nREGISTRATION, model-offset  "
          f"({'MULTI-contact' if multi else 'SINGLE contact'}) "
          f"-- {len(cases)} cases x {len(seeds)} seeds\n{'='*78}")
    nominal = [dict(c) for c in att.cyls]
    for off_cm in offsets_cm:
        rows = []
        for case in cases:
            cs = case["contacts"] if multi else case["contacts"][:1]
            fs = forces_proportional(cs)
            target, F_t = cs[0], fs[0]
            # truth comes from the NOMINAL model, once
            tau0 = synth_tau(kin, case["q"], cs, fs, 0.0, None)
            for s in seeds:
                rng = random.Random(s)
                v = [rng.gauss(0, 1) for _ in range(3)]
                n = math.sqrt(sum(x * x for x in v)) or 1.0
                vec = [x / n * off_cm * 0.01 for x in v]
                # shift the whole model, attribute, restore
                for idx, c in enumerate(att.cyls):
                    c["p0"] = [nominal[idx]["p0"][d] + vec[d] for d in range(3)]
                    c["p1"] = [nominal[idx]["p1"][d] + vec[d] for d in range(3)]
                try:
                    rows.append(score(att, kin, case["q"], tau0, target, F_t))
                finally:
                    for idx, c in enumerate(att.cyls):
                        c["p0"] = list(nominal[idx]["p0"])
                        c["p1"] = list(nominal[idx]["p1"])
        print(f"  model offset {off_cm:5.1f} cm : {summarize(rows)}")


def report_multiplicity(att, kin, cases):
    print(f"\n{'='*78}\nCONTACT AND CANDIDATE MULTIPLICITY AT THE STALL POSE\n{'='*78}")
    ncs, nbr, nprobe = [], [], []
    for case in cases:
        cs = case["contacts"]
        ncs.append(len(cs))
        nbr.append(len(set(c["cyl"] for c in cs)))
        nprobe.append(len(set(c["probe"] for c in cs)))
    print(f"  cases                              : {len(cases)}")
    print(f"  real (probe,cylinder) contacts      : "
          f"median {statistics.median(ncs):.0f}  range {min(ncs)}-{max(ncs)}")
    print(f"  distinct BRANCHES in contact        : "
          f"median {statistics.median(nbr):.0f}  range {min(nbr)}-{max(nbr)}")
    print(f"  distinct PROBES in contact          : "
          f"median {statistics.median(nprobe):.0f}  range {min(nprobe)}-{max(nprobe)}")
    single = sum(1 for n in nbr if n == 1)
    print(f"  cases with exactly ONE branch       : {single}/{len(cases)}  "
          f"({100.0*single/len(cases):.0f}%)   <- the regime the old sweeps tested")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="a benchmark results dir (scene_layout.json + grasp_path_*.csv)")
    ap.add_argument("--cases", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--only", default="", help="comma list: mult,noise,deflect,regist")
    args = ap.parse_args()

    scene = os.path.join(args.results, "scene_layout.json")
    kin = PandaKin()
    att = ContactAttributor(scene, kin)
    print(f"scene: {scene}")
    print(f"  cylinders {len(att.cyls)}  soft {sum(1 for c in att.cyls if c['soft'])}")

    cases = build_cases(kin, att, args.results, args.cases)
    if not cases:
        print("no cases with a real contact")
        return 1
    seeds = list(range(args.seeds))
    want = set(x for x in args.only.split(",") if x) or \
        {"mult", "noise", "deflect", "regist"}

    if "mult" in want:
        report_multiplicity(att, kin, cases)
    if "noise" in want:
        sweep_noise(att, kin, cases, seeds, [0.0, 0.2, 0.5, 1.0, 2.0], multi=False)
        sweep_noise(att, kin, cases, seeds, [0.0, 0.2, 0.5, 1.0, 2.0], multi=True)
    if "deflect" in want:
        sweep_offset(att, kin, cases, seeds, [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
                     multi=False, label="BRANCH DEFLECTION (target branch only)",
                     perturb_all=False)
    if "regist" in want:
        offs = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        sweep_registration(att, kin, cases, seeds, offs, multi=False)
        sweep_registration(att, kin, cases, seeds, offs, multi=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
