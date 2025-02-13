#include "constraint.hpp"

#include <sstream>
#include <cassert>

namespace op
{

namespace internal
{

EqualityConstraint::EqualityConstraint(const internal::AffineSum &affine)
    : affine(affine) {}

EqualityConstraint operator==(const internal::AffineSum &affine, const double zero)
{
    assert(zero == 0.0);
    (void)zero;

    return EqualityConstraint(affine);
}

std::ostream &operator<<(std::ostream &os, const EqualityConstraint &constraint)
{
    os << constraint.affine << " == 0";
    return os;
}

double EqualityConstraint::evaluate(const std::vector<double> &soln_values) const
{
    return -affine.evaluate(soln_values);
}

PositiveConstraint::PositiveConstraint(const internal::AffineSum &affine)
    : affine(affine) {}

std::ostream &operator<<(std::ostream &os, const PositiveConstraint &constraint)
{
    os << constraint.affine << " >= 0";
    return os;
}

double PositiveConstraint::evaluate(const std::vector<double> &soln_values) const
{
    return -affine.evaluate(soln_values);
}

PositiveConstraint operator>=(const internal::AffineSum &affine, const double zero)
{
    assert(zero == 0.0);
    (void)zero;

    return PositiveConstraint(affine);
}

PositiveConstraint operator<=(const double zero, const internal::AffineSum &affine)
{
    return affine >= zero;
}

SecondOrderConeConstraint::SecondOrderConeConstraint(const internal::Norm2Term &norm2,
                                                     const internal::AffineSum &affine)
    : norm2(norm2), affine(affine) {}

std::ostream &operator<<(std::ostream &os, const SecondOrderConeConstraint &constraint)
{
    os << constraint.norm2 << " <= " << constraint.affine;
    return os;
}

double SecondOrderConeConstraint::evaluate(const std::vector<double> &soln_values) const
{
    return (norm2.evaluate(soln_values) - affine.evaluate(soln_values));
}

} // namespace internal

std::vector<internal::EqualityConstraint> operator==(const Affine &affine, const double zero)
{
    assert(zero == 0.0);
    (void)zero;

    std::vector<internal::EqualityConstraint> constraints;
    constraints.reserve(affine.size());
    for (auto [row, col] : affine.all_indices())
    {
        constraints.push_back(affine.coeff(row, col) == 0.0);
    }
    return constraints;
}
std::vector<internal::EqualityConstraint> operator==(const Affine &lhs, const Affine &rhs)
{
    assert(lhs.shape() == rhs.shape() or lhs.is_scalar() or rhs.is_scalar());

    std::vector<internal::EqualityConstraint> constraints;
    constraints.reserve(std::max(lhs.size(), rhs.size()));
    for (size_t row = 0; row < std::max(lhs.rows(), rhs.rows()); row++)
    {
        for (size_t col = 0; col < std::max(lhs.cols(), rhs.cols()); col++)
        {
            if (not lhs.is_scalar() and not rhs.is_scalar())
            {
                constraints.push_back(rhs.coeff(row, col) + -lhs.coeff(row, col) == 0.);
            }
            else if (lhs.is_scalar())
            {
                constraints.push_back(rhs.coeff(row, col) + -lhs.coeff(0) == 0.);
            }
            else if (rhs.is_scalar())
            {
                constraints.push_back(lhs.coeff(row, col) + -rhs.coeff(0) == 0.);
            }
        }
    }
    return constraints;
}

std::vector<internal::PositiveConstraint> operator>=(const Affine &affine, const double zero)
{
    assert(zero == 0.0);
    (void)zero;

    std::vector<internal::PositiveConstraint> constraints;
    for (auto [row, col] : affine.all_indices())
    {
        constraints.push_back(affine.coeff(row, col) >= 0.0);
    }
    return constraints;
}

std::vector<internal::PositiveConstraint> operator<=(const double zero, const Affine &affine)
{
    return affine >= zero;
}

std::vector<internal::PositiveConstraint> operator<=(const double &zero, const Affine &affine)
{
    return affine >= zero;
}

std::vector<internal::PositiveConstraint> operator>=(const Affine &lhs, const Affine &rhs)
{
    assert(lhs.shape() == rhs.shape() or lhs.is_scalar() or rhs.is_scalar());

    std::vector<internal::PositiveConstraint> constraints;
    constraints.reserve(std::max(lhs.size(), rhs.size()));
    for (size_t row = 0; row < std::max(lhs.rows(), rhs.rows()); row++)
    {
        for (size_t col = 0; col < std::max(lhs.cols(), rhs.cols()); col++)
        {
            if (not lhs.is_scalar() and not rhs.is_scalar())
            {
                constraints.push_back(-rhs.coeff(row, col) + lhs.coeff(row, col) >= 0.);
            }
            else if (lhs.is_scalar())
            {
                constraints.push_back(lhs.coeff(0) + -rhs.coeff(row, col) >= 0.);
            }
            else if (rhs.is_scalar())
            {
                constraints.push_back(lhs.coeff(row, col) + -rhs.coeff(0) >= 0.);
            }
        }
    }
    return constraints;
}

std::vector<internal::PositiveConstraint> operator<=(const Affine &lhs, const Affine &rhs)
{
    return rhs >= lhs;
}

std::vector<internal::SecondOrderConeConstraint> operator<=(const SOCLhs &socLhs, const Affine &affine)
{
    assert(socLhs.shape() == affine.shape() or affine.is_scalar()); //or socLhs.is_scalar()); // Maybe TODO
    std::vector<internal::SecondOrderConeConstraint> constraints;

    for (auto [row, col] : socLhs.all_indices())
    {
        if (affine.is_scalar())
        {
            internal::AffineSum rhs = affine.coeff(0);
            rhs += -socLhs.coeff(0).second;
            constraints.emplace_back(socLhs.coeff(row, col).first, rhs);
        }
        else
        {
            internal::AffineSum rhs = affine.coeff(row, col);
            rhs += -socLhs.coeff(row, col).second;

            constraints.emplace_back(socLhs.coeff(row, col).first, rhs);
        }
    }
    return constraints;
}

} // namespace op