#pragma once

#include "expression.hpp"

#include <vector>
#include <ostream>

namespace op
{

namespace internal
{

// represents a constraint like
//     p_1*x_1 + p_2*x_2 + ... + b == 0
struct EqualityConstraint
{
    explicit EqualityConstraint(const internal::AffineSum &affine);
    internal::AffineSum affine;
    friend std::ostream &operator<<(std::ostream &os, const EqualityConstraint &constraint);
    double evaluate(const std::vector<double> &soln_values) const;
};

// represents a constraint like
//     p_1*x_1 + p_2*x_2 + ... + b >= 0
struct PositiveConstraint
{
    explicit PositiveConstraint(const internal::AffineSum &affine);
    internal::AffineSum affine;
    friend std::ostream &operator<<(std::ostream &os, const PositiveConstraint &constraint);
    double evaluate(const std::vector<double> &soln_values) const;
};

// represents a constraint like
//      norm2([p_1*x_1 + p_2*x_2 + ... + b_1,   p_3*x_3 + p_4*x_4 + ... + b_2 ])
//        <= p_5*x_5 + p_6*x_6 + ... + b_3
struct SecondOrderConeConstraint
{
    SecondOrderConeConstraint(const internal::Norm2Term &norm2, const internal::AffineSum &affine);
    internal::Norm2Term norm2;
    internal::AffineSum affine;
    friend std::ostream &operator<<(std::ostream &os, const SecondOrderConeConstraint &constraint);
    double evaluate(const std::vector<double> &soln_values) const;
};

} // namespace internal

std::vector<internal::EqualityConstraint> operator==(const Affine &affine, const double zero);
std::vector<internal::EqualityConstraint> operator==(const Affine &lhs, const Affine &rhs);

std::vector<internal::PositiveConstraint> operator>=(const Affine &affine, const double zero);
std::vector<internal::PositiveConstraint> operator<=(const double zero, const Affine &affine);
std::vector<internal::PositiveConstraint> operator>=(const Affine &lhs, const Affine &rhs);
std::vector<internal::PositiveConstraint> operator<=(const Affine &lhs, const Affine &rhs);

std::vector<internal::SecondOrderConeConstraint> operator<=(const SOCLhs &socLhs, const Affine &affine);

} // namespace op