#!/usr/bin/env python3
"""Score the attribution estimators against PhysX ground truth, offline.

Reads what isaac_observe.py logged (steps.csv, contacts.csv, meta.json) and answers
the five questions that the whole attribution design currently rests on assumptions
for. Runs in the pybullet venv, NOT in Isaac -- which is why contact_monitor.py can
stay Isaac-free.

STATUS: WRITTEN, NEVER RUN against real logs (Isaac is not installed on the dev box).
The arithmetic is exercised by --demo, which feeds it the synthetic forward model so
the plumbing is at least known to execute.

The five questions:

  (1) CONTACT MODEL.  Is f = k*pen anything like reality?
      PhysX gives the true normal force (Jn/dt). The planner's model says it should be
      k*pen with pen from FK + the cylinder geometry. Regress one on the other: if the
      relationship is linear, the penalty-spring model holds and k is its slope. If it
      is not, est_static's whole premise is wrong, because it divides by pen.

  (2) FRICTION.  mu = Jt/Jn per contact point.
      This is the number the residual's reliability hinges on and it has never been
      measured -- 15-20 deg of direction error was a literature estimate for wood on
      metal, nothing more. mu also decides whether the probe estimator can use one
      bisector probe or needs one per normal.

  (3) SIGN.  Which convention does the logged torque use?
      Build J^T(+r_hat)*f from PhysX's own force and normal, and correlate against the
      measured torque. Positive correlation means tau_ext_sign = +1. This replaces the
      guesswork that produced the documented sign bug.

  (4) FALSE POSITIVES.  How often does StallDetector fire on a NOMINAL scene?
      It should be never. Anything else and the trigger needs retuning before the
      repair loop can be trusted.

  (5) WHICH ESTIMATOR.  Score force-ranking vs static-stiffness (and the probe, where
      a usable displacement exists) against "which branch did PhysX actually push
      hardest / least compliantly".

Usage:
  python score_observe.py --obs obs_run01 --scene <results>/scene_layout.json
  python score_observe.py --demo --scene <results>/scene_layout.json \
         --path <results>/grasp_path_01.csv
"""

import argparse
import collections
import csv
import json
import math
import os
import random
import sys

import contact_monitor as cm
from contact_monitor import (ContactAttributor, PandaKin, PEN_PROBES,
                             LINK_JOINT_COUNT, StallDetector)

# Which gate probe a PhysX robot prim corresponds to. The prim path contains the link
# name, so this maps link -> the probe indices living on it.
LINK_PROBES = collections.defaultdict(list)
for _i, (_ln, _off, _r) in enumerate(PEN_PROBES):
    LINK_PROBES[_ln].append(_i)


def load_steps(d):
    rows = []
    with open(os.path.join(d, "steps.csv")) as f:
        for r in csv.DictReader(f):
            rows.append({
                "t": float(r["t"]),
                "q": [float(r[f"q{i}"]) for i in range(7)],
                "qref": [float(r[f"qref{i}"]) for i in range(7)],
                "tau": [float(r[f"tau{i}"]) for i in range(7)],
                "arc": float(r["arc"]),
            })
    return rows


def load_contacts(d):
    rows = []
    with open(os.path.join(d, "contacts.csv")) as f:
        for r in csv.DictReader(f):
            rows.append({
                "t": float(r["t"]), "robot": r["robot_prim"],
                "other": r["other_prim"], "cls": r["cls"],
                "p": [float(r["px"]), float(r["py"]), float(r["pz"])],
                "n": [float(r["nx"]), float(r["ny"]), float(r["nz"])],
                "Jn": float(r["Jn"]), "Jt": float(r["Jt"]),
                "sep": float(r["sep"]),
            })
    return rows


def q1_contact_model(kin, att, steps, contacts, dt, out):
    """Regress PhysX normal force against the modelled penetration."""
    pairs = []
    by_t = collections.defaultdict(list)
    for c in contacts:
        if c["cls"] == "branch":
            by_t[round(c["t"] / dt)].append(c)
    for s in steps:
        key = round(s["t"] / dt)
        if key not in by_t:
            continue
        fn = sum(c["Jn"] for c in by_t[key]) / dt          # impulse -> force
        if fn <= 1e-6:
            continue
        cs = _contacts_model(kin, att, s["q"])
        if not cs:
            continue
        pen = max(c["pen"] for c in cs)
        pairs.append((pen, fn))
    out["q1_n"] = len(pairs)
    if len(pairs) < 10:
        out["q1"] = "insufficient data"
        return
    # least squares through the origin: f = k*pen
    sxy = sum(p * f for p, f in pairs)
    sxx = sum(p * p for p, f in pairs)
    k = sxy / sxx if sxx > 0 else float("nan")
    ybar = sum(f for _, f in pairs) / len(pairs)
    ss_res = sum((f - k * p) ** 2 for p, f in pairs)
    ss_tot = sum((f - ybar) ** 2 for _, f in pairs)
    out["q1_k_N_per_m"] = k
    out["q1_r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    out["q1"] = ("penalty-spring model plausible" if out["q1_r2"] > 0.5
                 else "penalty-spring model NOT supported -- est_static premise fails")


def q2_friction(contacts, out):
    mus = [c["Jt"] / c["Jn"] for c in contacts
           if c["cls"] == "branch" and c["Jn"] > 1e-6]
    out["q2_n"] = len(mus)
    if not mus:
        out["q2"] = "no branch contacts"
        return
    mus.sort()
    med = mus[len(mus) // 2]
    out["q2_mu_median"] = med
    out["q2_mu_p90"] = mus[min(len(mus) - 1, int(0.9 * len(mus)))]
    out["q2_dir_err_deg"] = math.degrees(math.atan(med))
    out["q2"] = (f"mu={med:.2f} -> {out['q2_dir_err_deg']:.0f} deg of direction error; "
                 + ("one bisector probe is fine" if med < 0.1 else
                    "probe along each normal separately (friction reverses on sliding)"))


def q3_sign(kin, att, steps, contacts, dt, out):
    """Correlate measured tau against J^T(+r_hat)*f built from PhysX truth."""
    num = den_a = den_b = 0.0
    n = 0
    by_t = collections.defaultdict(list)
    for c in contacts:
        if c["cls"] == "branch":
            by_t[round(c["t"] / dt)].append(c)
    for s in steps:
        key = round(s["t"] / dt)
        if key not in by_t or sum(c["Jn"] for c in by_t[key]) / dt < 1.0:
            continue
        J = kin.probe_jacobians(s["q"])
        pred = [0.0] * 7
        for c in by_t[key]:
            link = next((ln for ln in LINK_PROBES if ln in c["other"] or
                         ln in c["robot"]), None)
            if link is None:
                continue
            pi = LINK_PROBES[link][0]
            f = c["Jn"] / dt
            kj = LINK_JOINT_COUNT[link]
            for j in range(kj):
                pred[j] += f * sum(J[pi][d][j] * c["n"][d] for d in range(3))
        for a, b in zip(s["tau"], pred):
            num += a * b
            den_a += a * a
            den_b += b * b
        n += 1
    out["q3_n"] = n
    if n < 5 or den_a <= 0 or den_b <= 0:
        out["q3"] = "insufficient data"
        return
    corr = num / math.sqrt(den_a * den_b)
    out["q3_corr"] = corr
    out["q3_tau_ext_sign"] = 1.0 if corr > 0 else -1.0
    out["q3"] = (f"corr={corr:+.3f} -> use tau_ext_sign = "
                 f"{int(out['q3_tau_ext_sign']):+d}"
                 + ("" if abs(corr) > 0.3 else "  (WEAK -- do not trust)"))


def q4_false_positives(steps, effort_cap, out):
    det = StallDetector(effort_cap, observe_only=True)
    fires = 0
    for s in steps:
        if det.push(s["t"], s["q"], s["qref"], s["tau"], s["arc"]):
            fires += 1
        det.fired = False          # count every firing, not just the first
    out["q4_fires"] = fires
    out["q4_steps"] = len(steps)
    out["q4"] = ("no false positives on the nominal scene" if fires == 0
                 else f"FIRED {fires}x on a nominal replay -- retune before closing "
                      f"the loop")


def _contacts_model(kin, att, q):
    """The planner-model view: (probe, soft cylinder) penetrations and fingerprints."""
    out = []
    pos = kin.probe_positions(q)
    J = kin.probe_jacobians(q)
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
            if pen <= 0:
                continue
            rhat = [v / rn for v in rad]
            kj = LINK_JOINT_COUNT[pname]
            h = [(sum(J[pi][d][j] * rhat[d] for d in range(3)) if j < kj else 0.0)
                 for j in range(7)]
            out.append({"pen": pen, "probe": pi, "cyl": ci, "u": u, "h": h,
                        "rhat": rhat, "link": pname})
    return out


def q5_estimators(kin, att, steps, contacts, dt, out):
    """Score force-ranking vs static-stiffness at the deepest-contact step."""
    best = None
    for s in steps:
        cs = _contacts_model(kin, att, s["q"])
        if not cs:
            continue
        d = max(c["pen"] for c in cs)
        if best is None or d > best[0]:
            best = (d, s, cs)
    if best is None:
        out["q5"] = "no modelled contact anywhere in the replay"
        return
    _d, s, cs = best
    res = att.attribute_stiffness([(s["t"], s["q"], s["qref"], s["tau"], s["arc"])])
    by = collections.defaultdict(list)
    for c in cs:
        by[c["cyl"]].append(c)
    force_pick = max(by, key=lambda b: max(c["pen"] for c in by[b]))
    out["q5_n_branches"] = len(by)
    out["q5_force_pick"] = force_pick
    out["q5_stiffness_rank"] = [(r.cyl_idx, round(r.k_est)) for r in res]
    out["q5"] = ("force-ranking and stiffness-ranking AGREE"
                 if res and res[0].cyl_idx == force_pick
                 else "force-ranking and stiffness-ranking DISAGREE "
                      "(the case the redesign exists for)")


def demo(scene, path):
    """Exercise the arithmetic on the synthetic forward model, no Isaac needed."""
    import stiffness_probe as sp
    kin = PandaKin()
    att = ContactAttributor(scene, kin)
    with open(path) as f:
        wps = [[float(x) for x in ln.split(",")] for ln in f if ln.strip()]
    best = None
    for q in wps:
        cs = sp.contacts_at(kin, att, q)
        if cs and (best is None or cs[0]["pen"] > best[0]):
            best = (cs[0]["pen"], q, cs)
    if best is None:
        print("no contact on this path")
        return 1
    _p, q, cs = best
    order, by = sp.branches_of(cs)
    k0, rigid = 300.0, order[-1]
    ks = {b: k0 for b in order}
    ks[rigid] = k0 * 5
    tau = [0.0] * 7
    for c in cs:
        f = ks[c["cyl"]] * c["pen"]
        for j in range(7):
            tau[j] += f * c["h"][j]
    rng = random.Random(0)
    tau = [v + rng.gauss(0, 0.3) for v in tau]
    steps = [{"t": 0.0, "q": q, "qref": q, "tau": tau, "arc": 0.0}]
    out = {}
    q5_estimators(kin, att, steps, [], 1.0 / 240.0, out)
    q4_false_positives(steps, [87] * 4 + [12] * 3, out)
    print("DEMO (synthetic truth; rigid branch is the SHALLOWEST, cyl "
          f"{rigid}, k={ks[rigid]:.0f})")
    for k_, v in out.items():
        print(f"  {k_:22s} {v}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", help="isaac_observe.py output directory")
    ap.add_argument("--scene", required=True, help="planner scene_layout.json")
    ap.add_argument("--path", help="a planned path CSV (--demo only)")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--effort-cap", default="87,87,87,87,12,12,12")
    args = ap.parse_args()

    if args.demo:
        if not args.path:
            sys.exit("--demo needs --path")
        return demo(args.scene, args.path)
    if not args.obs:
        sys.exit("need --obs or --demo")

    meta = json.load(open(os.path.join(args.obs, "meta.json")))
    dt = meta["dt"]
    steps = load_steps(args.obs)
    contacts = load_contacts(args.obs)
    caps = [float(x) for x in args.effort_cap.split(",")]

    kin = PandaKin()
    att = ContactAttributor(args.scene, kin)
    out = {"obs": args.obs, "dt": dt, "n_steps": len(steps),
           "n_contacts": len(contacts), "trunk_hit": meta.get("trunk_hit")}
    q1_contact_model(kin, att, steps, contacts, dt, out)
    q2_friction(contacts, out)
    q3_sign(kin, att, steps, contacts, dt, out)
    q4_false_positives(steps, caps, out)
    q5_estimators(kin, att, steps, contacts, dt, out)

    print(json.dumps(out, indent=2))
    with open(os.path.join(args.obs, "score.json"), "w") as f:
        json.dump(out, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
