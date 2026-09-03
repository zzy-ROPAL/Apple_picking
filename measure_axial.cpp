// measure_axial.cpp — recompute axial metrics offline on already-saved paths
//
// Purpose: axial_ee now follows the planner's target sphere set (erSphereActive).
// This tool recomputes the pair of metrics on historical grasp_path_XX.csv files
// without re-running the benchmark (avoiding unseeded noise). Scene = --sparse-long
//
// Usage:
//   ./measure_axial <urdf> <dir1> <links1> [<dir2> <links2> ...]
// links looks like "panda_link7:0,panda_link5:2"; "-" means default (hand+link6).

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
#include "types.h"

using namespace grasp_rrt;
using Eigen::Vector3f;

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

static std::vector<std::pair<std::string,int>> parseLinks(const std::string& s) {
    std::vector<std::pair<std::string,int>> sel;
    if (s == "-") return sel;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        auto pos = tok.find(':');
        if (pos == std::string::npos) continue;
        sel.push_back({tok.substr(0, pos), std::atoi(tok.substr(pos+1).c_str())});
    }
    return sel;
}

int main(int argc, char** argv) {
    if (argc < 4 || (argc - 2) % 2 != 0) {
        std::cerr << "usage: " << argv[0] << " <urdf> <dir> <links|-> [...]\n";
        return 1;
    }

    auto robot = std::make_shared<PandaPinocchioRobot>(argv[1], "panda_hand");
    auto sdf   = std::make_shared<SdfCostEvaluator>(robot, 0.01f);

    // ---- Scene: identical to benchmark --sparse-long ----
    Vector3f trunk_c(0.7f, 0.0f, 0.75f);
    sdf->addRigidCylinder(trunk_c - Vector3f(0,0,0.75f),
                          trunk_c + Vector3f(0,0,0.75f), 0.08f);
    struct Br { float h, angle, length, radius; };
    std::vector<Br> branches = {
        {0.60f, 3.1415f, 0.60f, 0.03f}, {0.60f, 2.40f, 0.55f, 0.03f}, {0.60f, 3.88f, 0.55f, 0.03f},
        {0.80f, 2.55f, 0.60f, 0.03f},   {0.80f, 3.73f, 0.60f, 0.03f},
        {0.80f, 1.90f, 0.55f, 0.03f},   {0.80f, 4.38f, 0.55f, 0.03f},
        {1.00f, 3.1415f, 0.60f, 0.03f}, {1.00f, 2.40f, 0.55f, 0.03f}, {1.00f, 3.88f, 0.55f, 0.03f}};
    for (const auto& br : branches) {
        float ca = std::cos(br.angle), sa = std::sin(br.angle);
        Vector3f p0(trunk_c.x() + 0.08f*ca, trunk_c.y() + 0.08f*sa, br.h);
        sdf->addBranchCylinder(p0, p0 + Vector3f(ca, sa, 0)*br.length, br.radius);
    }
    Vector3f overhead(0.45f, 0.0f, 0.91f);
    sdf->addRigidCylinder(overhead, overhead, 0.08f);

    struct LS { std::string name; float radius; std::vector<Vector3f> offsets; };
    std::vector<LS> robot_structure = {
        {"panda_link0", 0.12f, {Vector3f(0,0,0.05f)}},
        {"panda_link1", 0.10f, {Vector3f(0,0,0.1f), Vector3f(0,0,0.2f), Vector3f(0,0,0.3f)}},
        {"panda_link2", 0.08f, {Vector3f(0,0,0)}},
        {"panda_link3", 0.07f, {Vector3f(0,0,0.05f), Vector3f(0,0,0.15f), Vector3f(0,0,0.25f)}},
        {"panda_link4", 0.07f, {Vector3f(0,0,0.05f)}},
        {"panda_link5", 0.06f, {Vector3f(0,0,0.1f), Vector3f(0,0,0.2f), Vector3f(0,0,0.3f)}},
        {"panda_link6", 0.06f, {Vector3f(0,0,0)}},
        {"panda_link7", 0.06f, {Vector3f(0,0,0), Vector3f(0,0,0.05f)}},
        {"panda_hand",  0.06f, {Vector3f(0,0,0), Vector3f(0,0,0.08f),
                                Vector3f(0,0.04f,0.05f), Vector3f(0,-0.04f,0.05f)}}};
    for (const auto& ls : robot_structure)
        for (const auto& off : ls.offsets)
            sdf->addRobotSphere(ls.name, ls.radius, off);
    sdf->buildSdf(0.40f);

    for (int a = 2; a + 1 < argc; a += 2) {
        std::string dir = argv[a];
        sdf->setEdgeRefuseSphereSet(parseLinks(argv[a+1]));

        std::vector<std::string> files;
        if (DIR* dp = opendir(dir.c_str())) {
            while (dirent* e = readdir(dp)) {
                std::string n = e->d_name;
                if (n.rfind("grasp_path_", 0) == 0 &&
                    n.size() > 4 && n.substr(n.size()-4) == ".csv")
                    files.push_back(dir + "/" + n);
            }
            closedir(dp);
        }
        std::sort(files.begin(), files.end());

        double s_all = 0, s_tgt = 0;
        int n_all = 0, n_tgt = 0;
        for (const auto& f : files) {
            auto path = loadPath(f);
            if (path.size() < 2) continue;
            double ax=0, tot=0, axt=0, tott=0;
            for (size_t i = 0; i + 1 < path.size(); ++i) {
                Config v = path[i+1] - path[i];
                auto c  = sdf->measureWaypointAnisotropy(path[i], v, false);
                auto ct = sdf->measureWaypointAnisotropy(path[i], v, true);
                ax += c.first;  tot += c.second;
                axt += ct.first; tott += ct.second;
            }
            if (tot  > 1e-9) { s_all += ax/tot;   ++n_all; }
            if (tott > 1e-9) { s_tgt += axt/tott; ++n_tgt; }
        }
        std::printf("%-42s targets=%-38s axial_all %.3f (n=%d)  axial_target %.3f (n=%d)\n",
                    dir.c_str(), argv[a+1],
                    n_all ? s_all/n_all : -1.0, n_all,
                    n_tgt ? s_tgt/n_tgt : -1.0, n_tgt);
    }
    return 0;
}
