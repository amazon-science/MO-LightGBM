#include "secondOrderConeProgram.hpp"

#include <iostream>
#include <cmath>

namespace op
{

void SecondOrderConeProgram::addConstraint(std::vector<internal::EqualityConstraint> constraints)
{
    equalityConstraints.insert(equalityConstraints.end(), constraints.begin(), constraints.end());
}

void SecondOrderConeProgram::addConstraint(std::vector<internal::PositiveConstraint> constraints)
{
    positiveConstraints.insert(positiveConstraints.end(), constraints.begin(), constraints.end());
}

void SecondOrderConeProgram::addConstraint(std::vector<internal::SecondOrderConeConstraint> constraints)
{
    secondOrderConeConstraints.insert(secondOrderConeConstraints.end(), constraints.begin(), constraints.end());
}

void SecondOrderConeProgram::addMinimizationTerm(const Affine &affine)
{
    assert(affine.is_scalar());
    costFunction += affine.coeff(0);
}

std::ostream &operator<<(std::ostream &os, const SecondOrderConeProgram &socp)
{
    os << "Second order cone problem with " << socp.solution_vector.size() << " variables.\n";

    os << "Number of equality constraints:          " << socp.equalityConstraints.size() << "\n";
    os << "Number of positive constraints:          " << socp.positiveConstraints.size() << "\n";
    os << "Number of second order cone constraints: " << socp.secondOrderConeConstraints.size() << "\n";
    os << "\n";

    os << "Minimize:"
       << "\n";
    os << socp.costFunction << "\n";

    if (not socp.equalityConstraints.empty())
    {
        os << "\n"
           << "Subject to equality constraints:"
           << "\n";
        for (const auto &equalityConstraint : socp.equalityConstraints)
        {
            os << equalityConstraint << "\n";
        }
    }

    if (not socp.positiveConstraints.empty())
    {
        os << "\n"
           << "Subject to linear inequalities:"
           << "\n";
        for (const auto &positiveConstraint : socp.positiveConstraints)
        {
            os << positiveConstraint << "\n";
        }
    }

    if (not socp.secondOrderConeConstraints.empty())
    {
        os << "\n"
           << "Subject to cone constraints:"
           << "\n";
        for (const auto &secondOrderConeConstraint : socp.secondOrderConeConstraints)
        {
            os << secondOrderConeConstraint << "\n";
        }
    }
    return os;
}

template <typename T>
bool check_constraint(double tol, double val, const T &constraint)
{
    if (val > tol)
    {
        std::cout << "Infeasible solution, constraint value: " + std::to_string(val) << "\n"
                  << constraint << "\n";

        return false;
    }
    return true;
}

bool SecondOrderConeProgram::isFeasible() const
{
    const double tol = 0.01;
    bool feasible = true;
    auto check = [&](const auto &constraint) { return check_constraint(tol,
                                                                       constraint.evaluate(solution_vector),
                                                                       constraint); };
    auto check_abs = [&](const auto &constraint) { return check_constraint(tol,
                                                                           std::fabs(constraint.evaluate(solution_vector)),
                                                                           constraint); };

    feasible &= std::all_of(secondOrderConeConstraints.begin(),
                            secondOrderConeConstraints.end(),
                            check);
    feasible &= std::all_of(positiveConstraints.begin(),
                            positiveConstraints.end(),
                            check);
    feasible &= std::all_of(equalityConstraints.begin(),
                            equalityConstraints.end(),
                            check_abs);

    return feasible;
}

void SecondOrderConeProgram::cleanUp()
{
    // std::cout << "Cleaning up problem...\n";

    size_t variables_removed = 0;

    variables_removed += costFunction.clean();

    for (auto &equalityConstraint : equalityConstraints)
    {
        variables_removed += equalityConstraint.affine.clean();
    }
    for (auto &positiveConstraint : positiveConstraints)
    {
        variables_removed += positiveConstraint.affine.clean();
    }
    for (auto &secondOrderConeConstraint : secondOrderConeConstraints)
    {
        { // Affine
            variables_removed += secondOrderConeConstraint.affine.clean();
        }

        { // Norm2
            variables_removed += secondOrderConeConstraint.affine.clean();
            for (auto &affineSum : secondOrderConeConstraint.norm2.arguments)
            {
                variables_removed += affineSum.clean();
            }
        }
    }

    // std::cout << "Removed " << variables_removed << " term(s) from constraints.\n";

    size_t constraints_removed = 0;
    {
        auto erase_from = std::remove_if(equalityConstraints.begin(),
                                         equalityConstraints.end(),
                                         [](const internal::EqualityConstraint &constraint) { return constraint.affine.is_constant(); });
        const auto erase_to = equalityConstraints.end();
        const size_t erased_elements = std::distance(erase_from, erase_to);
        equalityConstraints.erase(erase_from, erase_to);

        constraints_removed += erased_elements;
    }
    {
        auto erase_from = std::remove_if(positiveConstraints.begin(),
                                         positiveConstraints.end(),
                                         [](const internal::PositiveConstraint &constraint) { return constraint.affine.is_constant(); });
        const auto erase_to = positiveConstraints.end();
        const size_t erased_elements = std::distance(erase_from, erase_to);
        positiveConstraints.erase(erase_from, erase_to);

        constraints_removed += erased_elements;
    }
    {
        auto erase_from = std::remove_if(secondOrderConeConstraints.begin(),
                                         secondOrderConeConstraints.end(),
                                         [](const internal::SecondOrderConeConstraint &constraint) {
                                             bool all_terms_constant = constraint.affine.is_constant();
                                             for (const internal::AffineSum &affineSum : constraint.norm2.arguments)
                                             {
                                                 all_terms_constant &= affineSum.is_constant();
                                             }
                                             return all_terms_constant;
                                         });
        const auto erase_to = secondOrderConeConstraints.end();
        const size_t erased_elements = std::distance(erase_from, erase_to);
        secondOrderConeConstraints.erase(erase_from, erase_to);

        constraints_removed += erased_elements;
    }

    // std::cout << "Removed " << constraints_removed << " constraint(s).\n";
}

} // namespace op