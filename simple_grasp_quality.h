#pragma once

#include "robot_interface.h"
#include <cmath>
#include <iostream>

namespace grasp_rrt {

// ============================================================================
// SimpleGraspQuality — simplified grasp-quality evaluation
//
// Uses geometric heuristics in place of a full force-closure analysis:
//   1. Number of contact points >= minimum
//   2. Contact-normal symmetry (clamping from both sides)
//   3. Uniformity of the contact-force distribution
//
// This is an approximation, sufficient to validate the GraspRRT pipeline.
// For real force-closure analysis, use GraspIt! or implement a convex-hull method.
// ============================================================================

class SimpleGraspQuality : public GraspQualityInterface {
public:
    SimpleGraspQuality() : score_(0.0f), valid_(false), min_contacts_(2) {}

    /**
     * @param min_contacts minimum number of contact points
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

        // ---- Score components ----

        // 1. Contact-count score (more is better, capped at 6)
        float count_score = std::min(1.0f, static_cast<float>(n) / 6.0f);

        // 2. Normal-symmetry score
        //    A good grasp: normals coming from opposing directions (antipodal)
        float symmetry_score = computeNormalSymmetry();

        // 3. Contact-spread score (contacts should not all cluster at one point)
        float spread_score = computeContactSpread();

        // 4. Force plausibility (all contact forces > 0)
        float force_score = 1.0f;
        for (const auto& c : contacts_) {
            if (c.force <= 0.0f) {
                force_score *= 0.5f;
            }
        }

        // Combined score (weighted average)
        score_ = 0.3f * count_score +
                 0.4f * symmetry_score +
                 0.2f * spread_score +
                 0.1f * force_score;

        // Force-closure verdict: symmetry > 0.3 and enough contact points
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
     * Compute normal symmetry: a good antipodal grasp has near-opposite normal pairs
     */
    float computeNormalSymmetry() const {
        if (contacts_.size() < 2) return 0.0f;

        float best_antisym = 0.0f;
        for (size_t i = 0; i < contacts_.size(); ++i) {
            for (size_t j = i + 1; j < contacts_.size(); ++j) {
                // Antisymmetry of the two normals: larger -n1*n2 is better (ideal = 1)
                float dot = -contacts_[i].normal.dot(contacts_[j].normal);
                if (dot > best_antisym) {
                    best_antisym = dot;
                }
            }
        }
        return std::max(0.0f, best_antisym);
    }

    /**
     * Compute the spatial spread of the contact points
     */
    float computeContactSpread() const {
        if (contacts_.size() < 2) return 0.0f;

        // Compute the maximum span across contact points
        float max_dist = 0.0f;
        for (size_t i = 0; i < contacts_.size(); ++i) {
            for (size_t j = i + 1; j < contacts_.size(); ++j) {
                float d = (contacts_[i].position - contacts_[j].position).norm();
                if (d > max_dist) max_dist = d;
            }
        }

        // Normalize (a span of 5 cm or more is assumed ideal)
        return std::min(1.0f, max_dist / 0.05f);
    }
};

} // namespace grasp_rrt
