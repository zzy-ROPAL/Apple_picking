#!/usr/bin/env python3
"""
contact_monitor.py — stall detection and contact attribution for sim-to-real repair.

NO Isaac imports. Everything that makes a decision lives here so it can be unit
tested in seconds (see --self-test), with the Isaac side reduced to feeding it
signals. FK/Jacobians come from pybullet using the same panda.urdf and the same
z = 0.4 base offset the planner uses, so the geometry matches the planner exactly.

Two objects:

  StallDetector      consumes (t, q, q_ref, tau_ext, arc_progress) at control rate
                     and fires when effort is near the cap AND progress has stopped,
                     sustained for hold_s. Force alone is NOT a trigger: planned
                     paths press 20-50 mm into soft branches by design, so a force
                     threshold fires on every reach.

  ContactAttributor  given the stall-window samples, answers "which branch, and
                     roughly where along it". Two stages:
                       isolation  — highest joint index with significant residual;
                                    a contact on link k loads only joints 1..k
                       hypothesis — each candidate (point, direction) predicts a
                                    torque fingerprint g = J^T(-r_hat); the measured
                                    tau must be a scalar multiple of it

Accuracy needed is low: the inserted obstacle ends up ~9 cm across regardless
(2 cm capsule + 6-7 cm gate probe radius), and a 10 cm hard region was measured to
behave identically to a 4 cm one. So "right branch, roughly where" is the target,
not a precise contact point.
"""

import argparse
import json
import math
import os
import sys
from collections import deque

BASE_Z = 0.40
URDF_DEFAULT = ("/home/zhongyanzhang/zhongyan_zhang/Apple_picking/.venv/lib/"
                "python3.12/site-packages/pybullet_data/franka_panda/panda.urdf")

# The planner's penetration-gate probes (sdf_cost_evaluator.h enablePenetrationGate).
# Single source of truth: attribution must reason about the same points the gate
# enforces, or the obstacle we insert will not correspond to what stopped the arm.
PEN_PROBES = [
    ("panda_link3", (0.0,  0.00, 0.15), 0.07),
    ("panda_link4", (0.0,  0.00, 0.05), 0.07),
    ("panda_link5", (0.0,  0.00, 0.15), 0.06),
    ("panda_link5", (0.0,  0.00, 0.30), 0.06),
    ("panda_link6", (0.0,  0.00, 0.00), 0.06),
    ("panda_link7", (0.0,  0.00, 0.05), 0.06),
    ("panda_hand",  (0.0,  0.00, 0.08), 0.06),
    ("panda_hand",  (0.0,  0.04, 0.05), 0.06),
    ("panda_hand",  (0.0, -0.04, 0.05), 0.06),
]

# Which arm joint each probe's link can be loaded by. A contact on link k produces
# torque only in joints 1..k, so this is also the isolation lookup.
LINK_JOINT_COUNT = {
    "panda_link3": 3, "panda_link4": 4, "panda_link5": 5,
    "panda_link6": 6, "panda_link7": 7, "panda_hand": 7,
}


# ======================================================================
# Stall detection
# ======================================================================

class StallDetector:
    """Fires when the arm is pushing at its limit and no longer making progress.

    The discriminator is NOT force. A soft branch yields: force rises and the arm
    keeps moving. An unyielding one stops it. So the trigger is the conjunction,
    held long enough to reject acceleration transients and waypoint dwells.
    """

    def __init__(self, effort_cap, effort_frac=0.80, progress_eps=0.002,
                 track_err_min=0.02, hold_s=1.0, window_s=0.5, observe_only=True):
        # effort_cap: per-joint |tau| limit of the drives (N*m). "Near the cap" is
        # the meaningful signal, because a force-limited controller saturates there
        # rather than exceeding any threshold.
        self.effort_cap = list(effort_cap)
        self.effort_frac = effort_frac
        self.progress_eps = progress_eps      # m/s of arc length
        self.track_err_min = track_err_min    # rad, ||q_ref - q||
        self.hold_s = hold_s
        self.window_s = window_s
        self.observe_only = observe_only

        self.buf = deque()                    # (t, q, q_ref, tau_ext, arc)
        self._cond_since = None
        self.fired = False
        self.fire_t = None
        self.events = []                      # (t, reason) - "would have fired"

    def push(self, t, q, q_ref, tau_ext, arc):
        self.buf.append((t, list(q), list(q_ref), list(tau_ext), float(arc)))
        while self.buf and t - self.buf[0][0] > max(self.hold_s, self.window_s) * 2.0:
            self.buf.popleft()

        sat = self._saturated(tau_ext)
        prog = self._progress(t)
        terr = math.sqrt(sum((a - b) ** 2 for a, b in zip(q_ref, q)))

        cond = sat and (prog is not None and prog < self.progress_eps) \
               and terr > self.track_err_min
        if not cond:
            self._cond_since = None
            return False

        if self._cond_since is None:
            self._cond_since = t
            return False
        if t - self._cond_since >= self.hold_s and not self.fired:
            self.fired = True
            self.fire_t = t
            self.events.append((t, f"sat&stalled progress={prog:.5f} track={terr:.4f}"))
            return True
        return False

    def _saturated(self, tau_ext):
        return any(abs(v) >= self.effort_frac * c
                   for v, c in zip(tau_ext, self.effort_cap))

    def _progress(self, t):
        """Arc-length rate over the trailing window. None until the window fills."""
        t0 = t - self.window_s
        old = None
        for rec in self.buf:
            if rec[0] >= t0:
                old = rec
                break
        if old is None or t - old[0] < self.window_s * 0.5:
            return None
        dt = t - old[0]
        return abs(self.buf[-1][4] - old[4]) / dt if dt > 1e-6 else None

    def stall_window(self):
        """Samples spanning the sustained condition — the input to attribution.

        Stacking the whole window matters: the arm drifts slightly while pushing,
        so each sample projects the same force onto a slightly different set of
        joint directions. That turns an underdetermined single-sample problem on a
        proximal link into an overdetermined one.
        """
        if self._cond_since is None:
            return []
        return [r for r in self.buf if r[0] >= self._cond_since]


# ======================================================================
# Kinematics (pybullet, matching the planner)
# ======================================================================

class PandaKin:
    def __init__(self, urdf=None, gui=False):
        import pybullet as p
        import pybullet_data
        self.p = p
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        self.rid = p.loadURDF(urdf or os.environ.get("PANDA_URDF", URDF_DEFAULT),
                              [0, 0, BASE_Z], useFixedBase=True,
                              physicsClientId=self.cid)
        self.arm = [j for j in range(p.getNumJoints(self.rid, physicsClientId=self.cid))
                    if p.getJointInfo(self.rid, j, physicsClientId=self.cid)[2]
                    == p.JOINT_REVOLUTE][:7]
        # link name -> link index
        self.link_idx = {}
        for j in range(p.getNumJoints(self.rid, physicsClientId=self.cid)):
            info = p.getJointInfo(self.rid, j, physicsClientId=self.cid)
            self.link_idx[info[12].decode()] = j

    def set_q(self, q):
        for j, v in zip(self.arm, q):
            self.p.resetJointState(self.rid, j, float(v), physicsClientId=self.cid)

    def probe_positions(self, q):
        """World position of every gate probe at configuration q."""
        self.set_q(q)
        out = []
        for name, off, r in PEN_PROBES:
            li = self.link_idx[name]
            st = self.p.getLinkState(self.rid, li, computeForwardKinematics=True,
                                     physicsClientId=self.cid)
            pos, orn = st[4], st[5]            # URDF link frame, not COM
            wp, _ = self.p.multiplyTransforms(pos, orn, list(off), [0, 0, 0, 1])
            out.append(tuple(wp))
        return out

    def probe_jacobians(self, q, eps=1e-5):
        """Translational Jacobian (3x7) of every probe, by central differences.

        Done numerically rather than via calculateJacobian to sidestep pybullet's
        link-frame vs inertial-frame ambiguity for `localPosition`: this is exactly
        the derivative of the same FK used everywhere else in this file, and the
        cost (14 FK evaluations per attribution) is irrelevant at stall time.
        """
        n = len(PEN_PROBES)
        J = [[[0.0] * 7 for _ in range(3)] for _ in range(n)]
        for k in range(7):
            qp = list(q); qp[k] += eps
            qm = list(q); qm[k] -= eps
            pp = self.probe_positions(qp)
            pm = self.probe_positions(qm)
            for i in range(n):
                for d in range(3):
                    J[i][d][k] = (pp[i][d] - pm[i][d]) / (2.0 * eps)
        return J


# ======================================================================
# Contact attribution
# ======================================================================

class ContactHypothesis:
    def __init__(self, cyl_idx, u, probe_index, link, f_est, residual, point, normal):
        self.cyl_idx = cyl_idx
        self.u = u
        self.probe_index = probe_index
        self.link = link
        self.f_est = f_est
        self.residual = residual
        self.point = point
        self.normal = normal

    def __repr__(self):
        return (f"<cyl {self.cyl_idx} u={self.u:.2f} link={self.link} "
                f"f={self.f_est:.2f}N res={self.residual:.3f}>")


class BranchStiffness:
    """One branch's estimated stiffness, from attribute_stiffness()."""

    def __init__(self, cyl_idx, k_est, pen_max, u_at_max, probe_index, link,
                 point, normal, n_contacts):
        self.window_ok = True     # set by attribute_rate; see WINDOW VALIDITY there
        self.branch_id = None     # physical branch; cyl_idx is its deepest segment
        self.cyl_idx = cyl_idx
        self.k_est = k_est              # N/m
        self.pen_max = pen_max          # m, deepest contact on this branch
        self.u = u_at_max
        self.probe_index = probe_index
        self.link = link
        self.point = point              # on the branch surface, for capsule placement
        self.normal = normal            # r_hat at the deepest contact
        self.n_contacts = n_contacts    # (probe, cylinder) pairs on this branch

    def __repr__(self):
        return (f"<cyl {self.cyl_idx} k={self.k_est:.0f}N/m "
                f"pen={self.pen_max*1000:.1f}mm u={self.u:.2f} "
                f"link={self.link} n={self.n_contacts}>")


class ContactAttributor:
    def __init__(self, scene_json, kin, noise_tau=0.5, envelope=0.12, u_samples=9,
                 tau_ext_sign=+1.0):
        self.kin = kin
        self.noise_tau = noise_tau      # N*m below which a joint reads "unloaded"
        self.envelope = envelope        # m, candidate search radius around a probe
        self.u_samples = u_samples      # candidate points sampled along each cylinder
        # Sign of the supplied tau_ext relative to the physical convention
        #     tau_ext = sum_contacts J^T (+r_hat) * f ,   f > 0
        # r_hat points from the branch axis to the probe centre, and the branch pushes
        # the arm AWAY from itself, i.e. along +r_hat. So +1 is physically correct.
        #
        # Pass -1 when the supplied signal uses the opposite convention. This differs
        # by data source and MUST be pinned empirically per source before use:
        #   Franka  tau_ext_hat_filtered      }  opposite of each other -- determine
        #   Isaac   measured efforts - dynamics}  once against PhysX ground truth.
        #
        # 2026-10-20: this used to be hard-coded backwards (the fingerprint was
        # J^T(-r_hat) while the rejection test demanded f > 0), and self_test was blind
        # to it because it synthesised tau with the same flip. Both are now +r_hat.
        self.tau_ext_sign = float(tau_ext_sign)
        with open(scene_json) as f:
            d = json.load(f)
        self.apple = d["apple"]
        # Physical-branch grouping. A branch is split into several cylinder SEGMENTS
        # whose t0/t1 are arc-length fractions along the whole branch, chained
        # endpoint-to-endpoint, so a new branch begins wherever t0 resets to 0.
        #
        # The unknown must be one stiffness per BRANCH, not per segment. Two reasons,
        # one physical and one numerical: stiffness is a property of the branch, not of
        # our arbitrary segmentation; and adjacent segments have nearly collinear
        # fingerprints, so solving them separately splits the signal between them
        # arbitrarily. Measured: with per-segment unknowns and a 2 cm candidate margin,
        # the exact segment was named only 60% of the time -- but the correct BRANCH
        # was named 100% of the time, the other 40% landing on a neighbouring segment
        # of it. Grouping removes that split entirely.
        self.cyls = []
        self.branch_of = {}
        bid = -1
        for i, c in enumerate(d["cylinders"]):
            sub = c.get("subclass") or ("soft" if c["class"] == "soft" else "trunk")
            if sub == "soft" and abs(float(c.get("t0", 0.0))) < 1e-6:
                bid += 1
            self.cyls.append({
                "p0": c["p0"], "p1": c["p1"], "r": float(c["radius"]),
                "soft": sub == "soft", "t0": float(c.get("t0", 0.0)),
                "t1": float(c.get("t1", 1.0)),
            })
            self.branch_of[i] = bid if sub == "soft" else -1

    # ---- stage 1: which link ----
    @staticmethod
    def isolate(tau_ext, noise):
        """Highest joint index carrying load; contact is at or proximal to it."""
        k = 0
        for i, v in enumerate(tau_ext):
            if abs(v) > noise:
                k = i + 1
        return k

    # ---- stage 2: which cylinder, where along it ----
    def attribute(self, samples):
        """samples: list of (t, q, q_ref, tau_ext, arc) spanning the stall window.

        Stacks every sample into one least-squares problem. A candidate predicts a
        fingerprint g per sample; the measurement must be the SAME scalar multiple
        of it in all of them, which is a far stronger constraint than any single
        sample provides.
        """
        if not samples:
            return None
        # Isolation is reported as a diagnostic but deliberately NOT used to filter
        # candidates. A contact on link 7 can produce near-zero torque on joint 7
        # when the force line passes close to that joint's axis, so "highest loaded
        # joint" is a lower bound on the link, not an exclusion rule. Fitting all 7
        # components instead turns those small readings into evidence: a candidate
        # that predicts large distal torque is rejected by the very zeros that would
        # otherwise have excluded the right answer.
        k = 7
        if max(max(abs(v) for v in s[3]) for s in samples) < self.noise_tau:
            return None

        # Precompute per-sample probe positions and Jacobians once.
        prep = []
        for (_t, q, _qr, tau, _a) in samples:
            prep.append((self.kin.probe_positions(q),
                         self.kin.probe_jacobians(q),
                         list(tau)))

        best = None
        for pi, (pname, _poff, prad) in enumerate(PEN_PROBES):
            for ci, cyl in enumerate(self.cyls):
                if not cyl["soft"]:
                    continue
                for u in self._u_candidates(prep[0][0][pi], cyl):
                    g_all, tau_all = [], []
                    ok = True
                    for (pos, J, tau) in prep:
                        pp = pos[pi]
                        ax = _lerp(cyl["p0"], cyl["p1"], u)
                        rad = [pp[d] - ax[d] for d in range(3)]
                        rn = math.sqrt(sum(v * v for v in rad))
                        if rn < 1e-6 or rn - cyl["r"] > self.envelope:
                            ok = False
                            break
                        rhat = [v / rn for v in rad]
                        # g_j = (dp_j/dq)·r_hat: the torque joint j feels from a 1 N
                        # push along the OUTWARD normal, which is the direction the
                        # branch pushes the arm. tau_ext_sign maps the caller's
                        # convention onto that one (see __init__).
                        g = [self.tau_ext_sign
                             * sum(J[pi][d][j] * rhat[d] for d in range(3))
                             for j in range(k)]
                        g_all.extend(g)
                        tau_all.extend(tau[:k])
                    if not ok:
                        continue
                    gg = sum(v * v for v in g_all)
                    if gg < 1e-12:
                        continue
                    f = sum(a * b for a, b in zip(tau_all, g_all)) / gg
                    if f <= 0.0:
                        continue              # pushing outward: unphysical
                    res = math.sqrt(sum((a - f * b) ** 2
                                        for a, b in zip(tau_all, g_all)))
                    res /= math.sqrt(len(tau_all))        # per-joint-sample RMS
                    if best is None or res < best.residual:
                        pos0 = prep[0][0][pi]
                        ax = _lerp(cyl["p0"], cyl["p1"], u)
                        rad = [pos0[d] - ax[d] for d in range(3)]
                        rn = math.sqrt(sum(v * v for v in rad))
                        rhat = [v / rn for v in rad]
                        best = ContactHypothesis(
                            ci, u, pi, pname, f, res,
                            [ax[d] + rhat[d] * cyl["r"] for d in range(3)],
                            rhat)
        return best

    def _contacts_at(self, q, min_pen=0.0):
        """Every real (probe, soft cylinder) penetration at q, with its fingerprint.

        h = tau_ext_sign * J^T r_hat, zeroed on joints distal to the contact.
        """
        pos = self.kin.probe_positions(q)
        J = self.kin.probe_jacobians(q)
        out = []
        for pi, (pname, _off, prad) in enumerate(PEN_PROBES):
            for ci, cyl in enumerate(self.cyls):
                if not cyl["soft"]:
                    continue
                seg = [cyl["p1"][d] - cyl["p0"][d] for d in range(3)]
                sq = sum(v * v for v in seg)
                if sq < 1e-12:
                    continue
                u = max(0.0, min(1.0, sum((pos[pi][d] - cyl["p0"][d]) * seg[d]
                                          for d in range(3)) / sq))
                ax = _lerp(cyl["p0"], cyl["p1"], u)
                rad = [pos[pi][d] - ax[d] for d in range(3)]
                rn = math.sqrt(sum(v * v for v in rad))
                if rn < 1e-6:
                    continue
                pen = (cyl["r"] + prad) - rn
                if pen <= min_pen:
                    continue
                rhat = [v / rn for v in rad]
                kj = LINK_JOINT_COUNT[pname]
                h = [(self.tau_ext_sign
                      * sum(J[pi][d][j] * rhat[d] for d in range(3))
                      if j < kj else 0.0) for j in range(7)]
                out.append({"pen": pen, "probe": pi, "cyl": ci, "u": u,
                            "rhat": rhat, "link": pname, "h": h,
                            "surf": [ax[d] + rhat[d] * cyl["r"] for d in range(3)]})
        out.sort(key=lambda c: -c["pen"])
        return out

    # ---- stage 2'': PRIMARY -- stiffness from the RATE of torque change ----
    def attribute_rate(self, samples, margin=0.02):
        """Rank branches by stiffness, estimated from d(tau)/dt. THE PRIMARY ESTIMATOR.

        Measured against five alternatives on the same cases (6 scenes x 5 seeds, the
        rigid branch always the SHALLOWEST at 5x stiffness -- the adversarial case):

            metric / fit                       1 br    2 br    3 br   (0.5 N*m noise)
            old fit, tau        -> force       100%    100%      3%
            new fit, tau        -> force       100%    100%     23%
            new fit, 3mm retreat-> stiffness   100%     80%     70%
            old fit, dtau/dt    -> force rate  100%     33%     40%
            new fit, dtau/dt    -> force rate  100%     17%     60%
            THIS (dtau/dt       -> stiffness)  100%    100%     70%

        Two things make it work, and both are necessary:

        1. DIVIDE BY THE PUSH RATE. df/dt = k*v, where v = -qdot.h is how fast THIS
           branch's penetration is growing. Ranking by df/dt alone ranks by k*v, and v
           varies ~8x between branches whose fingerprints are 83 deg apart -- enough to
           bury a 5x stiffness difference. Dividing isolates k. Skipping the division
           is what drops the two rows above to 17-40%.

        2. FIT A SLOPE, DON'T DIFFERENCE TWO POINTS. Over ~120 samples the slope's
           standard error falls as 1/sqrt(N), and a constant bias (e.g. a gravity-model
           offset) is absorbed by the intercept instead of passing straight through.
           That is why this beats the discrete 3 mm retreat at 2 branches, 100% vs 80%.

        NO PENETRATION ANYWHERE IN THE ESTIMATE. `pen` is a modelling construct that
        cannot be measured in Isaac or on hardware, so the only model input here is
        r_hat -- a DIRECTION, which is forgiving (20 deg of error costs cos 20 = 6%)
        where a magnitude is brutal (a 10 mm radius error on a 2 mm deflection is
        520%). Contrast attribute_stiffness, which divides by pen and is therefore
        only as good as the tree model.

        It is also FREE: the window is the final approach INTO the stall pose, motion
        that already happened. No probe to command, no extra tree contact.

        THE CATCH: at a stall v -> 0 by definition, so the signal only exists while the
        arm is still moving. The window must be taken BEFORE the arm stops.

        WINDOW VALIDITY: every contact must persist across the whole window. If the
        shallowest branch is not yet touching at the window start, its torque switches
        on partway through and a straight-line fit underestimates its slope -- and the
        rigid branch is the shallowest by construction, so the bug targets exactly the
        branch being searched for. Branches that fail this check are returned with
        window_ok = False and should not be trusted.

        Returns [] if there is no contact, too few samples, or the solve is degenerate.
        """
        if len(samples) < 8:
            return []
        ts = [s[0] for s in samples]
        t0, t1 = ts[0], ts[-1]
        if t1 - t0 < 1e-6:
            return []

        q_end = samples[-1][1]
        # CANDIDATE GATE: proximity, NOT penetration.
        #
        # `pen` is a modelling fiction -- real branches bend, nothing is penetrated --
        # and it is not measurable in Isaac or on hardware. It appears nowhere in the
        # estimate below. The one place it used to matter was here: requiring pen > 0
        # meant a branch the model places slightly clear of the probe was never even
        # considered, which is precisely how a wrong model loses the rigid branch.
        #
        # So admit any pair within `margin` of touching. The asymmetry is the point:
        # over-inclusion is nearly free (a branch that is not really in contact
        # produces no force change, so the solve returns k ~ 0 and it sorts last),
        # while under-inclusion is fatal (no estimator can name a branch it never
        # enumerated). margin should cover the model's expected position error --
        # 2 cm is where attribution was measured to still be reliable.
        contacts = self._contacts_at(q_end, -abs(margin))
        if not contacts:
            return []

        # per-joint slope of tau, and of q (-> qdot), by least squares over the window
        n = len(samples)
        tbar = sum(ts) / n
        stt = sum((t - tbar) ** 2 for t in ts)
        if stt < 1e-18:
            return []

        def slope_of(get):
            out = []
            for j in range(7):
                ys = [get(s)[j] for s in samples]
                ybar = sum(ys) / n
                out.append(sum((t - tbar) * (y - ybar)
                               for t, y in zip(ts, ys)) / stt)
            return out

        dtau = slope_of(lambda s: s[3])
        qdot = slope_of(lambda s: s[1])

        # which contacts survived the whole window (see WINDOW VALIDITY above)
        start = {(c["probe"], c["cyl"]) for c in self._contacts_at(samples[0][1],
                                                                  -abs(margin))}

        by = {}
        for c in contacts:
            by.setdefault(self.branch_of.get(c["cyl"], c["cyl"]), []).append(c)
        order = sorted(by, key=lambda b: -max(c["pen"] for c in by[b]))

        cols = []
        for b in order:
            m = [0.0] * 7
            for c in by[b]:
                rate = -sum(a * h for a, h in zip(qdot, c["h"]))   # d(pen)/dt
                for j in range(7):
                    m[j] += rate * c["h"][j]
            cols.append(m)
        kvals = _solve_normal(cols, dtau)
        if kvals is None:
            return []

        out = []
        for i, b in enumerate(order):
            deepest = max(by[b], key=lambda c: c["pen"])
            r = BranchStiffness(deepest["cyl"], kvals[i], deepest["pen"],
                                deepest["u"], deepest["probe"], deepest["link"],
                                deepest["surf"], deepest["rhat"], len(by[b]))
            r.branch_id = b          # physical branch; cyl_idx is its deepest segment
            r.window_ok = all((c["probe"], c["cyl"]) in start for c in by[b])
            out.append(r)
        out.sort(key=lambda r: -r.k_est)
        return out

    # ---- stage 2': which branch is RIGID (the question that actually matters) ----
    def attribute_stiffness(self, samples, min_pen=0.0):
        """Rank the branches in contact by estimated stiffness, stiffest first.

        Why this exists, and why attribute() above is the wrong objective:

        attribute() leaves the force magnitude free and picks the candidate whose
        fingerprint best explains tau -- which ranks by FORCE. But the branch that
        stopped the arm is the one that is STIFF, and the gate deliberately permits
        15-75 mm of penetration into soft branches, so a soft branch pressed 50 mm
        routinely carries more force than a rigid branch pressed 8 mm. Force ranking
        then names a yielding branch. Measured on 3-branch cases with the rigid branch
        shallowest: force ranking 0/30, this estimator 28/30.

        Force ranking fails exactly when   pen_max/pen_min > k_rigid/k_soft.

        The fix needs no new sensing. The penetration is already known (FK + the
        cylinder geometry), so constrain f = k*pen instead of leaving f free:

            tau = sum_b k_b * A_b ,      A_b = sum_{p in b} pen_p * h_p

        with h_p = tau_ext_sign * J_p^T r_hat_p, zeroed on joints distal to contact p.
        That is 7 equations in (number of branches) unknowns -- overdetermined, so the
        residual still works as a consistency check.

        CAVEAT. k is only as good as pen, and pen comes from the model. Measured
        degradation under a rigid whole-model offset: 1 cm is survivable, 3 cm is not.
        The probe-based alternative in stiffness_probe.py never uses pen and is the
        better choice when the model's geometry is suspect -- but it costs arm motion
        and needs multi-sample averaging to beat this one. Unresolved; see CLAUDE.md.

        Returns [] if nothing is in contact or the solve is degenerate.
        """
        if not samples:
            return []
        # Geometry is taken from the FIRST sample: during a stall the arm has moved
        # < 2 mm (that is the trigger condition), so pen and h are effectively frozen.
        _t, q, _qr, _tau, _a = samples[0]
        contacts = self._contacts_at(q, min_pen)
        if not contacts:
            return []
        _UNUSED_SCAN = '''
            for ci, cyl in enumerate(self.cyls):
                if not cyl["soft"]:
                    continue
                seg = [cyl["p1"][d] - cyl["p0"][d] for d in range(3)]
                sq = sum(v * v for v in seg)
                if sq < 1e-12:
                    continue
                u = sum((pos[pi][d] - cyl["p0"][d]) * seg[d] for d in range(3)) / sq
                u = max(0.0, min(1.0, u))
                ax = _lerp(cyl["p0"], cyl["p1"], u)
                rad = [pos[pi][d] - ax[d] for d in range(3)]
                rn = math.sqrt(sum(v * v for v in rad))
                if rn < 1e-6:
                    continue
                pen = (cyl["r"] + prad) - rn
                if pen <= min_pen:
                    continue
                rhat = [v / rn for v in rad]
                kj = LINK_JOINT_COUNT[pname]
                h = [(self.tau_ext_sign
                      * sum(J[pi][d][j] * rhat[d] for d in range(3))
                      if j < kj else 0.0) for j in range(7)]
                contacts.append({"pen": pen, "probe": pi, "cyl": ci, "u": u,
                                 "rhat": rhat, "link": pname, "h": h,
                                 "surf": [ax[d] + rhat[d] * cyl["r"]
                                          for d in range(3)]})'''

        # group by branch, deepest branch first
        by = {}
        for c in contacts:
            by.setdefault(self.branch_of.get(c["cyl"], c["cyl"]), []).append(c)
        order = sorted(by, key=lambda b: -max(c["pen"] for c in by[b]))

        # average tau over the window: at a stall the configuration is frozen, so
        # every sample is a measurement of the same quantity and noise averages down
        # as 1/sqrt(n).
        n_s = len(samples)
        tau_avg = [sum(s[3][j] for s in samples) / n_s for j in range(7)]

        cols = []
        for b in order:
            A = [0.0] * 7
            for c in by[b]:
                for j in range(7):
                    A[j] += c["pen"] * c["h"][j]
            cols.append(A)
        kvals = _solve_normal(cols, tau_avg)
        if kvals is None:
            return []

        out = []
        for i, b in enumerate(order):
            deepest = max(by[b], key=lambda c: c["pen"])
            r = BranchStiffness(
                deepest["cyl"], kvals[i], deepest["pen"], deepest["u"],
                deepest["probe"], deepest["link"], deepest["surf"],
                deepest["rhat"], len(by[b]))
            r.branch_id = b
            out.append(r)
        out.sort(key=lambda r: -r.k_est)
        return out

    def _u_candidates(self, probe_pos, cyl):
        """Sample along the cylinder, centred on the closest point.

        Scanning rather than taking only the nearest point is what produces a
        residual profile along the branch; the width of that valley is the
        localization uncertainty, and therefore the length of capsule to insert.
        """
        seg = [cyl["p1"][d] - cyl["p0"][d] for d in range(3)]
        sq = sum(v * v for v in seg)
        if sq < 1e-12:
            return [0.0]
        u0 = sum((probe_pos[d] - cyl["p0"][d]) * seg[d] for d in range(3)) / sq
        u0 = max(0.0, min(1.0, u0))
        half = 0.15                           # +-15% of the segment
        n = self.u_samples
        return [max(0.0, min(1.0, u0 - half + 2 * half * i / (n - 1)))
                for i in range(n)] if n > 1 else [u0]

    def residual_profile(self, samples, cyl_idx, probe_index, n=21):
        """Residual vs u along one cylinder — the uncertainty the fit actually has."""
        prep = [(self.kin.probe_positions(q), self.kin.probe_jacobians(q), list(tau))
                for (_t, q, _qr, tau, _a) in samples]
        k = 7
        cyl = self.cyls[cyl_idx]
        out = []
        for i in range(n):
            u = i / (n - 1)
            g_all, tau_all = [], []
            bad = False
            for (pos, J, tau) in prep:
                pp = pos[probe_index]
                ax = _lerp(cyl["p0"], cyl["p1"], u)
                rad = [pp[d] - ax[d] for d in range(3)]
                rn = math.sqrt(sum(v * v for v in rad))
                if rn < 1e-6:
                    bad = True
                    break
                rhat = [v / rn for v in rad]
                g_all.extend([-sum(J[probe_index][d][j] * rhat[d] for d in range(3))
                              for j in range(k)])
                tau_all.extend(tau[:k])
            if bad:
                continue
            gg = sum(v * v for v in g_all)
            if gg < 1e-12:
                continue
            f = sum(a * b for a, b in zip(tau_all, g_all)) / gg
            res = math.sqrt(sum((a - f * b) ** 2 for a, b in zip(tau_all, g_all)))
            out.append((u, res / math.sqrt(len(tau_all)), f))
        return out


def _solve_normal(cols, rhs):
    """Least squares for rhs = sum_i x_i*cols[i] via normal equations. None if singular.

    Hand-rolled because contact_monitor.py deliberately has no numpy dependency (it
    must import in the Isaac python, which may not have one), and the system is at
    most 4x4.
    """
    n = len(cols)
    if n == 0:
        return None
    G = [[sum(a * b for a, b in zip(cols[i], cols[j])) for j in range(n)]
         for i in range(n)]
    v = [sum(a * b for a, b in zip(cols[i], rhs)) for i in range(n)]
    A = [G[i][:] + [v[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(A[r][c]))
        if abs(A[p][c]) < 1e-18:
            return None
        A[c], A[p] = A[p], A[c]
        pv = A[c][c]
        A[c] = [x / pv for x in A[c]]
        for r in range(n):
            if r != c and A[r][c]:
                f = A[r][c]
                A[r] = [xr - f * xc for xr, xc in zip(A[r], A[c])]
    return [A[i][n] for i in range(n)]


def _lerp(a, b, t):
    return [a[d] + (b[d] - a[d]) * t for d in range(3)]


# ======================================================================
# Self test — synthesise a known contact, check it is recovered
# ======================================================================

def self_test(scene_json, path_csv, seed=0, noise_nm=0.0, verbose=True):
    import random
    random.seed(seed)

    kin = PandaKin()
    att = ContactAttributor(scene_json, kin)

    # Scan the whole path for the deepest REAL penetration: that is the pose where
    # a stall would actually occur, and the mid-path waypoint usually touches
    # nothing at all.
    with open(path_csv) as f:
        wps = [[float(x) for x in ln.split(",")] for ln in f if ln.strip()]

    truth = None
    for q_i in wps:
        pos = kin.probe_positions(q_i)
        for pi, (pname, _o, prad) in enumerate(PEN_PROBES):
            for ci, cyl in enumerate(att.cyls):
                if not cyl["soft"]:
                    continue
                seg = [cyl["p1"][d] - cyl["p0"][d] for d in range(3)]
                sq = sum(v * v for v in seg)
                if sq < 1e-12:
                    continue
                u = max(0.0, min(1.0, sum((pos[pi][d] - cyl["p0"][d]) * seg[d]
                                          for d in range(3)) / sq))
                ax = _lerp(cyl["p0"], cyl["p1"], u)
                rad = [pos[pi][d] - ax[d] for d in range(3)]
                rn = math.sqrt(sum(v * v for v in rad))
                pen = (cyl["r"] + prad) - rn
                if rn > 1e-6 and (truth is None or pen > truth[0]):
                    truth = (pen, pi, ci, u, [v / rn for v in rad], pname, q_i)
    if truth is None or truth[0] <= 0.0:
        print("no actual contact anywhere on this path")
        return False
    pen, pi_t, ci_t, u_t, rhat_t, link_t, q = truth

    # Forward model: a 12 N push along the OUTWARD normal, which is the direction
    # the branch pushes the arm:  tau_ext = J^T(+r_hat) * f.
    # 2026-10-20: this used to carry a leading minus, matching the estimator's old
    # (wrong) fingerprint -- so the self-test passed while both were backwards and
    # could never have revealed it. Both are now the physical convention.
    F_MAG = 12.0
    J = kin.probe_jacobians(q)
    tau = [F_MAG * sum(J[pi_t][d][j] * rhat_t[d] for d in range(3))
           for j in range(7)]
    k_true = LINK_JOINT_COUNT[link_t]
    for j in range(k_true, 7):
        tau[j] = 0.0                          # joints distal to the contact feel nothing
    if noise_nm > 0:
        tau = [v + random.gauss(0.0, noise_nm) for v in tau]

    samples = [(0.0, q, q, tau, 0.0)]
    got = att.attribute(samples)

    if verbose:
        print(f"  truth : cyl {ci_t} u={u_t:.3f} link={link_t} "
              f"pen={pen*1000:.1f}mm f={F_MAG:.1f}N")
        print(f"  got   : {got}")
        if got:
            print(f"  isolation: joint {att.isolate(tau, att.noise_tau)} "
                  f"(true link loads joints 1..{k_true})")
            du = abs(got.u - u_t)
            seg_len = math.dist(att.cyls[ci_t]["p0"], att.cyls[ci_t]["p1"])
            print(f"  cylinder {'OK' if got.cyl_idx == ci_t else 'WRONG'}, "
                  f"du={du:.3f} ({du*seg_len*100:.1f} cm along the branch), "
                  f"force err {abs(got.f_est - F_MAG):.2f} N")
    return got is not None and got.cyl_idx == ci_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--path", required=True, help="a planned path CSV, for poses")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--noise", type=float, default=0.0, help="N*m of torque noise")
    args = ap.parse_args()
    if args.self_test:
        ok = self_test(args.scene, args.path, noise_nm=args.noise)
        print("SELF TEST:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
