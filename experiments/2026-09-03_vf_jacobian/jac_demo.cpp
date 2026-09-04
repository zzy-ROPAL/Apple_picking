// Demonstrate the difference between mapping (a) per-point Jacobian and
// (b) reusing the end-effector Jacobian, on the real Panda model.
#include "panda_pinocchio.h"
#include <iostream>
#include <iomanip>
using namespace grasp_rrt;

static void printRow(const char* name, const Eigen::VectorXf& v) {
    std::cout << std::setw(26) << name << " [";
    for (int i = 0; i < v.size(); ++i)
        std::cout << std::setw(8) << std::fixed << std::setprecision(3) << v[i];
    std::cout << " ]\n";
}

int main(int argc, char** argv) {
    const char* urdf = (argc > 1) ? argv[1] : std::getenv("PANDA_URDF");
    auto robot = std::make_shared<PandaPinocchioRobot>(urdf, "panda_hand");

    Config q(7);
    q << 0.0f, -0.4f, 0.0f, -2.0f, 0.0f, 1.6f, 0.8f;   // a typical mid-reach pose
    robot->setJointValues(q);
    robot->prepareJacobians();

    int fid_l3 = robot->getFrameId("panda_link3");
    Eigen::Vector3f off(0, 0, 0.15f);                  // the point used by trunk_points[]

    MatrixXf J_l3 = robot->computeSphereJacobianDirect(fid_l3, off);   // (a) 3x7 at link3
    MatrixXf J_ee = robot->computeJacobian().topRows(3);               // (b) 3x7 at the gripper

    std::cout << "Translational Jacobian columns = effect of each joint on the point\n\n";
    std::cout << "(a) J at link3 (+0.15) -- column norms, i.e. how much each joint moves link3:\n";
    Eigen::VectorXf n_l3(7), n_ee(7);
    for (int j = 0; j < 7; ++j) { n_l3[j] = J_l3.col(j).norm(); n_ee[j] = J_ee.col(j).norm(); }
    printRow("|col| joints 1..7:", n_l3);
    std::cout << "\n(b) J at the gripper -- how much each joint moves the GRIPPER:\n";
    printRow("|col| joints 1..7:", n_ee);

    // A branch pushes on link3: repulsion points along +x with magnitude 1.
    Eigen::Vector3f F(1.0f, 0.0f, 0.0f);
    Config tau_a = J_l3.transpose() * F;    // correct
    Config tau_b = J_ee.transpose() * F;    // what the code currently does

    std::cout << "\nForce F = (+1, 0, 0) APPLIED AT LINK3. Resulting joint command:\n";
    printRow("(a) J_link3^T F  correct:", tau_a);
    printRow("(b) J_ee^T F     current:", tau_b);

    Config dir_a = tau_a.normalized(), dir_b = tau_b.normalized();
    std::cout << "\ncos angle between the two joint directions = "
              << std::setprecision(3) << dir_a.dot(dir_b) << "\n";

    // Does each command actually move link3 away from the branch?
    float eff_a = (J_l3 * tau_a.normalized()).dot(F);
    float eff_b = (J_l3 * tau_b.normalized()).dot(F);
    std::cout << "\nActual escape speed of LINK3 along F, per unit joint motion:\n"
              << "   using (a): " << eff_a << "\n"
              << "   using (b): " << eff_b << "\n";
    return 0;
}
