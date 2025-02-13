#pragma once

#include "optimizationProblem.hpp"

namespace op
{

struct SecondOrderConeProgram : public GenericOptimizationProblem
{
    std::vector<internal::EqualityConstraint> equalityConstraints;
    std::vector<internal::PositiveConstraint> positiveConstraints;
    std::vector<internal::SecondOrderConeConstraint> secondOrderConeConstraints;
    internal::AffineSum costFunction;

    void addConstraint(std::vector<internal::EqualityConstraint> constraints);
    void addConstraint(std::vector<internal::PositiveConstraint> constraints);
    void addConstraint(std::vector<internal::SecondOrderConeConstraint> constraints);

    void addMinimizationTerm(const Affine &affine);

    void cleanUp();

    bool isFeasible() const;

    friend std::ostream &operator<<(std::ostream &os, const SecondOrderConeProgram &socp);
};

} // namespace op