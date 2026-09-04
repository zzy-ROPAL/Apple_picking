#include "panda_pinocchio.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>
using namespace grasp_rrt;
int main(int argc,char**argv){
    auto robot=std::make_shared<PandaPinocchioRobot>(std::getenv("PANDA_URDF"),"panda_hand");
    const char* names[]={"panda_link2","panda_link3","panda_link4","panda_link5","panda_link6","panda_link7","panda_hand"};
    Eigen::Vector3f offs[]={{0,0,0},{0,0,0.15f},{0,0,0.05f},{0,0,0.15f},{0,0,0},{0,0,0.05f},{0,0,0.08f}};
    std::mt19937 rng(42); std::uniform_real_distribution<float> U(-2.0f,2.0f), Ud(-1.f,1.f);
    for(int k=0;k<7;k++){
        int fid=robot->getFrameId(names[k]);
        std::vector<float> eff;
        for(int t=0;t<400;t++){
            Config q(7); for(int i=0;i<7;i++) q[i]=U(rng);
            robot->setJointValues(q); robot->prepareJacobians();
            MatrixXf Jp=robot->computeSphereJacobianDirect(fid,offs[k]);
            MatrixXf Je=robot->computeJacobian().topRows(3);
            Eigen::Vector3f F(Ud(rng),Ud(rng),Ud(rng)); if(F.norm()<1e-6f) continue; F.normalize();
            Config ta=Jp.transpose()*F, tb=Je.transpose()*F;
            if(ta.norm()<1e-9f||tb.norm()<1e-9f) continue;
            float ea=(Jp*ta.normalized()).dot(F), eb=(Jp*tb.normalized()).dot(F);
            if(ea<1e-9f) continue;
            eff.push_back(eb/ea);            // 1.0 = as good as correct; <0 = pushes INTO the branch
        }
        if(eff.size()<20){ std::cout<<std::setw(12)<<names[k]<<"  (only "<<eff.size()<<" valid samples, fid="<<fid<<")\n"; continue; }
        std::sort(eff.begin(),eff.end());
        auto q=[&](double p){return eff[(size_t)(p*(eff.size()-1))];};
        int neg=std::count_if(eff.begin(),eff.end(),[](float v){return v<0.f;});
        std::cout<<std::setw(12)<<names[k]<<"  median "<<std::fixed<<std::setprecision(2)<<q(0.5)
                 <<"   p10 "<<q(0.1)<<"   p01 "<<q(0.01)<<"   min "<<eff.front()
                 <<"   wrong-direction "<<std::setprecision(1)<<(100.0*neg/eff.size())<<"%\n";
    }
}
