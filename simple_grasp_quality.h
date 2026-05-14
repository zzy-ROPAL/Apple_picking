#pragma once

#include "robot_interface.h"
#include <cmath>
#include <iostream>

namespace grasp_rrt {

// ============================================================================
// SimpleGraspQuality — 简化的抓取质量评估
//
// 使用几何启发式代替完整的力封闭 (Force Closure) 分析:
//   1. 接触点数量 >= 最小值
//   2. 接触法线对称性 (两侧夹紧)
//   3. 接触力分布均匀性
//
// 这是一个近似实现, 足以验证 GraspRRT 流程。
// 如需真实力封闭分析, 可用 GraspIt! 或自行实现凸包算法。
// ============================================================================

class SimpleGraspQuality : public GraspQualityInterface {
public:
    SimpleGraspQuality() : score_(0.0f), valid_(false), min_contacts_(2) {}

    /**
     * @param min_contacts 最少接触点数
     */
    explicit SimpleGraspQuality(int min_contacts)
        : score_(0.0f), valid_(false), min_contacts_(min_contacts) {}

    void setContactPoints(const std::vector<ContactInfo>& contacts) override {
        contacts_ = contacts;
    }

    void calculateGraspQuality() override {
        score_ = 0.0f;
        valid_ = false;

        int n = static_cast<int>(contacts_.size());
        if (n < min_contacts_) return;

        // ---- 评分组件 ----

        // 1. 接触点数量评分 (越多越好, 上限 6 个)
        float count_score = std::min(1.0f, static_cast<float>(n) / 6.0f);

        // 2. 法线对称性评分
        //    好的抓取: 有来自不同方向的法线 (对握)
        float symmetry_score = computeNormalSymmetry();

        // 3. 接触分布评分 (接触点不要都集中在一个点)
        float spread_score = computeContactSpread();

        // 4. 力的合理性 (所有接触力 > 0)
        float force_score = 1.0f;
        for (const auto& c : contacts_) {
            if (c.force <= 0.0f) {
                force_score *= 0.5f;
            }
        }

        // 综合评分 (加权平均)
        score_ = 0.3f * count_score +
                 0.4f * symmetry_score +
                 0.2f * spread_score +
                 0.1f * force_score;

        // 力封闭判定: 对称性 > 0.3 且有足够接触点
        valid_ = (symmetry_score > 0.3f && n >= min_contacts_);
    }

    float getGraspQuality() const override {
        return score_;
    }

    bool isValid() const override {
        return valid_;
    }

private:
    std::vector<ContactInfo> contacts_;
    float score_;
    bool valid_;
    int min_contacts_;

    /**
     * 计算法线对称性: 好的 antipodal grasp 应有近似相反的法线对
     */
    float computeNormalSymmetry() const {
        if (contacts_.size() < 2) return 0.0f;

        float best_antisym = 0.0f;
        for (size_t i = 0; i < contacts_.size(); ++i) {
            for (size_t j = i + 1; j < contacts_.size(); ++j) {
                // 两个法线的反对称程度: -n1·n2 越大越好 (理想值 = 1)
                float dot = -contacts_[i].normal.dot(contacts_[j].normal);
                if (dot > best_antisym) {
                    best_antisym = dot;
                }
            }
        }
        return std::max(0.0f, best_antisym);
    }

    /**
     * 计算接触点的空间分布
     */
    float computeContactSpread() const {
        if (contacts_.size() < 2) return 0.0f;

        // 计算接触点的最大跨度
        float max_dist = 0.0f;
        for (size_t i = 0; i < contacts_.size(); ++i) {
            for (size_t j = i + 1; j < contacts_.size(); ++j) {
                float d = (contacts_[i].position - contacts_[j].position).norm();
                if (d > max_dist) max_dist = d;
            }
        }

        // 归一化 (假设 5cm 以上的跨度是理想的)
        return std::min(1.0f, max_dist / 0.05f);
    }
};

} // namespace grasp_rrt
