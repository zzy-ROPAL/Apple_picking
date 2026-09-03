// measure_trajopt.cpp — measure path metrics before and after TrajOpt, offline
//
// Background: in the benchmark, the result of SdfTrajOptimizer::optimize() is only
// timed, never stored or measured (benchmark_grasp_rrt.cpp:1552). This tool reads
// already-saved raw RRT paths (grasp_path_XX.csv), re-runs TrajOpt offline in the
// same sparse-long scene, and compares, on the SAME path, before vs after:
//   - E4 smoothness: total turning, in-envelope turning, reversals (>90 deg turns)
//   - E1 anisotropy: axial_contact_fraction (whole arm) and _ee (hand+link6)
// The paired comparison is immune to the "no random seed" issue — the path is a fixed input.
//
// Usage:
//   ./measure_trajopt <urdf> <result_dir_1> [result_dir_2 ...]
// Output: one summary line (means) per directory, plus a per-path CSV (written into the directory
// trajopt_metrics.csv).

#include <chrono>
#include <cmath>
#include <cstdio>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "panda_pinocchio.h"
#include "sdf_cost_evaluator.h"
#include "sdf_traj_optimizer.h"
#include "types.h"

using namespace grasp_rrt;
using grasp_rrt::Config;
using Eigen::Vector3f;

static constexpr float kEnvelope = 0.05f;  // = GraspRrtConfig::winner_metric_envelope
static constexpr int   kK        = 3;      // = the benchmark default --winner-metric-k

struct PathMetrics {
    float total_turn = 0, env_turn = 0;
    int   reversals = 0, env_segs = 0;
    float axial = -1.0f, axial_ee = -1.0f;  // -1 = no contact
    int   n_wp = 0;
};

static PathMetrics measure(const std::vector<Config>& path,
                           const std::shared_ptr<PandaPinocchioRobot>& robot,
                           const std::shared_ptr<SdfCostEvaluator>& sdf) {
    PathMetrics m;
    m.n_wp = static_cast<int>(path.size());
    if (path.size() < 2) return m;

    // E1: anisotropy (formula identical to the benchmark's Group E)
    double ax = 0, tot = 0, ax_ee = 0, tot_ee = 0;
    for (size_t i = 0; i + 1 < path.size(); ++i) {
        Config v = path[i + 1] - path[i];
        auto c  = sdf->measureWaypointAnisotropy(path[i], v, false);
        auto ce = sdf->measureWaypointAnisotropy(path[i], v, true);
        ax += c.first;  tot += c.second;
        ax_ee += ce.first; tot_ee += ce.second;
    }
    if (tot    > 1e-9) m.axial    = static_cast<float>(ax / tot);
    if (tot_ee > 1e-9) m.axial_ee = static_cast<float>(ax_ee / tot_ee);

    // E4: EE path smoothness (same formula as the benchmark)
    std::vector<Vector3f> ee(path.size());
    for (size_t i = 0; i < path.size(); ++i) {
        robot->setJointValues(path[i]);
        ee[i] = robot->getGCPGlobalPose().block<3, 1>(0, 3);
    }
    double total_turn = 0, env_turn = 0;
    for (size_t i = 1; i + 1 < ee.size(); ++i) {
        Vector3f s0 = ee[i] - ee[i - 1], s1 = ee[i + 1] - ee[i];
        float n0 = s0.norm(), n1 = s1.norm();
        if (n0 < 1e-6f || n1 < 1e-6f) continue;
        float c = std::max(-1.0f, std::min(1.0f, s0.dot(s1) / (n0 * n1)));
        float ang = std::acos(c) * 180.0f / static_cast<float>(M_PI);
        total_turn += ang;
        if (c < 0.0f) ++m.reversals;
        if (sdf->maxWinnerBranchId(path[i - 1], path[i], kEnvelope, kK) >= 0) {
            env_turn += ang;
            ++m.env_segs;
        }
    }
    m.total_turn = static_cast<float>(total_turn);
    m.env_turn   = static_cast<float>(env_turn);
    return m;
}

static std::vector<Config> loadPath(const std::string& file) {
    std::vector<Config> path;
    std::ifstream f(file);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::vector<float> vals;
        std::string tok;
        while (std::getline(ss, tok, ',')) vals.push_back(std::stof(tok));
        if (vals.size() < 7) continue;
        Config q(7);
        for (int i = 0; i < 7; ++i) q[i] = vals[i];
        path.push_back(q);
    }
    return path;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <panda.urdf> <result_dir> [...]\n";
        return 1;
    }
    std::string urdf = argv[1];

    // ---- Scene: identical to the benchmark's --sparse-long ----
    auto robot = std::make_shared<PandaPinocchioRobot>(urdf, "panda_hand");
    auto sdf   = std::make_shared<SdfCostEvaluator>(robot, 0.01f);

    Vector3f trunk_c(0.7f, 0.0f, 0.75f);
    float trunk_h = 1.5f, trunk_r = 0.08f;
    sdf->addRigidCylinder(trunk_c - Vector3f(0, 0, trunk_h / 2),
                          trunk_c + Vector3f(0, 0, trunk_h / 2), trunk_r);
    struct Br { float h, angle, length, radius; };
    std::vector<Br> branches = {
        {0.60f, 3.1415f, 0.60f, 0.03f}, {0.60f, 2.40f, 0.55f, 0.03f}, {0.60f, 3.88f, 0.55f, 0.03f},
        {0.80f, 2.55f, 0.60f, 0.03f},   {0.80f, 3.73f, 0.60f, 0.03f},
        {0.80f, 1.90f, 0.55f, 0.03f},   {0.80f, 4.38f, 0.55f, 0.03f},
        {1.00f, 3.1415f, 0.60f, 0.03f}, {1.00f, 2.40f, 0.55f, 0.03f}, {1.00f, 3.88f, 0.55f, 0.03f}};
    for (const auto& br : branches) {
        float ca = std::cos(br.angle), sa = std::sin(br.angle);
        Vector3f p0(trunk_c.x() + trunk_r * ca, trunk_c.y() + trunk_r * sa, br.h);
        Vector3f p1 = p0 + Vector3f(ca, sa, 0) * br.length;
        sdf->addBranchCylinder(p0, p1, br.radius);
    }
    Vector3f overhead(0.45f, 0.0f, 0.91f);
    sdf->addRigidCylinder(overhead, overhead, 0.08f);

    struct LS { std::string name; float radius; std::vector<Vector3f> offsets; };
    std::vector<LS> robot_structure = {
        {"panda_link0", 0.12f, {Vector3f(0, 0, 0.05f)}},
        {"panda_link1", 0.10f, {Vector3f(0, 0, 0.1f), Vector3f(0, 0, 0.2f), Vector3f(0, 0, 0.3f)}},
        {"panda_link2", 0.08f, {Vector3f(0, 0, 0)}},
        {"panda_link3", 0.07f, {Vector3f(0, 0, 0.05f), Vector3f(0, 0, 0.15f), Vector3f(0, 0, 0.25f)}},
        {"panda_link4", 0.07f, {Vector3f(0, 0, 0.05f)}},
        {"panda_link5", 0.06f, {Vector3f(0, 0, 0.1f), Vector3f(0, 0, 0.2f), Vector3f(0, 0, 0.3f)}},
        {"panda_link6", 0.06f, {Vector3f(0, 0, 0)}},
        {"panda_link7", 0.06f, {Vector3f(0, 0, 0), Vector3f(0, 0, 0.05f)}},
        {"panda_hand",  0.06f, {Vector3f(0, 0, 0), Vector3f(0, 0, 0.08f),
                                Vector3f(0, 0.04f, 0.05f), Vector3f(0, -0.04f, 0.05f)}}};
    for (const auto& ls : robot_structure)
        for (const auto& off : ls.offsets)
            sdf->addRobotSphere(ls.name, ls.radius, off);
    sdf->buildSdf(0.40f);

    for (int d = 2; d < argc; ++d) {
        std::string dir = argv[d];
        std::vector<std::string> files;
        if (DIR* dp = opendir(dir.c_str())) {
            while (dirent* e = readdir(dp)) {
                std::string n = e->d_name;
                if (n.rfind("grasp_path_", 0) == 0 && n.size() > 4 &&
                    n.substr(n.size() - 4) == ".csv")
                    files.push_back(dir + "/" + n);
            }
            closedir(dp);
        }
        std::sort(files.begin(), files.end());
        if (files.empty()) {
            std::cerr << "[skip] no grasp_path_*.csv in " << dir << "\n";
            continue;
        }

        std::ofstream out(dir + "/trajopt_metrics.csv");
        out << "file,n_wp_raw,n_wp_opt,"
            << "env_turn_raw,env_turn_opt,total_turn_raw,total_turn_opt,"
            << "reversals_raw,reversals_opt,axial_raw,axial_opt,"
            << "axial_ee_raw,axial_ee_opt,trajopt_ms\n";

        // Summary accumulators (the axial mean excludes -1 no-contact paths, matching the benchmark)
        double s_et_r = 0, s_et_o = 0, s_tt_r = 0, s_tt_o = 0, s_ms = 0;
        double s_rv_r = 0, s_rv_o = 0;
        double s_ax_r = 0, s_ax_o = 0, s_ae_r = 0, s_ae_o = 0;
        int n = 0, n_ax = 0, n_ae = 0;

        for (const auto& f : files) {
            auto raw = loadPath(f);
            if (raw.size() < 3) continue;
            auto t0 = std::chrono::high_resolution_clock::now();
            auto opt = SdfTrajOptimizer::optimize(raw, robot, sdf);
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

            PathMetrics mr = measure(raw, robot, sdf);
            PathMetrics mo = measure(opt, robot, sdf);

            out << f.substr(f.find_last_of('/') + 1) << "," << mr.n_wp << "," << mo.n_wp << ","
                << mr.env_turn << "," << mo.env_turn << ","
                << mr.total_turn << "," << mo.total_turn << ","
                << mr.reversals << "," << mo.reversals << ","
                << mr.axial << "," << mo.axial << ","
                << mr.axial_ee << "," << mo.axial_ee << "," << ms << "\n";

            s_et_r += mr.env_turn;  s_et_o += mo.env_turn;
            s_tt_r += mr.total_turn; s_tt_o += mo.total_turn;
            s_rv_r += mr.reversals;  s_rv_o += mo.reversals;
            s_ms += ms;
            ++n;
            if (mr.axial >= 0 && mo.axial >= 0) {
                s_ax_r += mr.axial; s_ax_o += mo.axial; ++n_ax;
            }
            if (mr.axial_ee >= 0 && mo.axial_ee >= 0) {
                s_ae_r += mr.axial_ee; s_ae_o += mo.axial_ee; ++n_ae;
            }
        }
        if (n == 0) continue;
        std::printf(
            "%-50s n=%3d | env_turn %7.0f -> %7.0f | total %7.0f -> %7.0f | "
            "rev %5.1f -> %5.1f | axial %.3f -> %.3f (n=%d) | "
            "axial_ee %.3f -> %.3f (n=%d) | trajopt %.0f ms\n",
            dir.c_str(), n, s_et_r / n, s_et_o / n, s_tt_r / n, s_tt_o / n,
            s_rv_r / n, s_rv_o / n, s_ax_r / std::max(1, n_ax), s_ax_o / std::max(1, n_ax), n_ax,
            s_ae_r / std::max(1, n_ae), s_ae_o / std::max(1, n_ae), n_ae, s_ms / n);
    }
    return 0;
}
