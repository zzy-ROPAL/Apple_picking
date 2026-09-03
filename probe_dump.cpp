// probe_dump.cpp — print the world coordinates of the 9 penetration-gate probes
// Purpose: cross-validate frame conventions against Python (pybullet FK).
// Usage:   ./probe_dump <urdf> "q1,q2,...,q7"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "panda_pinocchio.h"

using namespace grasp_rrt;
using Eigen::Vector3f;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: probe_dump <urdf> \"q1,...,q7\"\n";
        return 1;
    }
    auto robot = std::make_shared<PandaPinocchioRobot>(argv[1]);
    Config q(7);
    {
        std::stringstream ss(argv[2]);
        std::string tok;
        int i = 0;
        while (std::getline(ss, tok, ',') && i < 7) q[i++] = std::stof(tok);
        if (i != 7) { std::cerr << "need 7 joint values\n"; return 1; }
    }
    robot->setJointValues(q);

    struct Probe { const char* frame; Vector3f off; float r; };
    const Probe probes[9] = {
        {"panda_link3", Vector3f(0,0,0.15f), 0.07f},
        {"panda_link4", Vector3f(0,0,0.05f), 0.07f},
        {"panda_link5", Vector3f(0,0,0.15f), 0.06f},
        {"panda_link5", Vector3f(0,0,0.30f), 0.06f},
        {"panda_link6", Vector3f(0,0,0),     0.06f},
        {"panda_link7", Vector3f(0,0,0.05f), 0.06f},
        {"panda_hand",  Vector3f(0,0,0.08f), 0.06f},
        {"panda_hand",  Vector3f(0,0.04f,0.05f),  0.06f},
        {"panda_hand",  Vector3f(0,-0.04f,0.05f), 0.06f},
    };
    for (const auto& pr : probes) {
        Matrix4f pose = robot->getFramePose(pr.frame);
        Vector3f p = pose.block<3,1>(0,3) + pose.block<3,3>(0,0) * pr.off;
        std::printf("%s %.6f %.6f %.6f %.6f %.6f %.6f r=%.3f\n",
                    pr.frame, p.x(), p.y(), p.z(),
                    pose(0,3), pose(1,3), pose(2,3), pr.r);
    }
    return 0;
}
