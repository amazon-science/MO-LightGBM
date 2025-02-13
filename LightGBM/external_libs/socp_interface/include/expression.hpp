#pragma once

#include "parameter.hpp"
#include "variable.hpp"

#include <optional>

namespace op
{

namespace internal
{

// A linear term (p_1*x_1) or constant term (p_1)
struct AffineTerm
{
    AffineTerm();
    explicit AffineTerm(const ParameterSource &parameter);
    explicit AffineTerm(const VariableSource &variable);
    AffineTerm(const ParameterSource &parameter,
               const VariableSource &variable);

    operator AffineSum() const;

    ParameterSource parameter;
    std::optional<VariableSource> variable; // a missing Variable represents a constant 1.0
    friend std::ostream &operator<<(std::ostream &os, const AffineTerm &term);
    double evaluate(const std::vector<double> &soln_values) const;
    AffineTerm operator*(const ParameterSource &parameter) const;
    AffineTerm &operator*=(const ParameterSource &parameter);
};
AffineTerm operator*(const ParameterSource &parameter, const VariableSource &variable);

// A term like (p_1*x_1 + p_2*x_2 + ... + b)
struct AffineSum
{
    AffineSum() = default;
    explicit AffineSum(const ParameterSource &parameter);
    explicit AffineSum(const VariableSource &variable);
    explicit AffineSum(const AffineTerm &term);

    std::vector<AffineTerm> terms;
    friend std::ostream &operator<<(std::ostream &os, const AffineSum &expression);
    double evaluate(const std::vector<double> &soln_values) const;
    AffineSum &operator+=(const AffineSum &other);
    AffineSum operator+(const AffineSum &other) const;
    AffineSum operator*(const ParameterSource &parameter) const;
    AffineSum operator-() const;
    size_t clean();
    bool is_constant() const;
};

} // namespace internal

// A scalar/vector/matrix of affine expressions
class Affine : public DynamicMatrix<internal::AffineSum, Affine>
{
public:
    using DynamicMatrix<internal::AffineSum, Affine>::DynamicMatrix;
    explicit Affine(const Parameter &parameter);
    explicit Affine(const Variable &variable);
    explicit Affine(const internal::AffineSum &expression);
    friend std::ostream &operator<<(std::ostream &os, const Affine &expression);
    Affine &operator+=(const Affine &other);
    Affine operator-() const;
};
Affine operator+(const Affine &lhs, const Affine &rhs);
Affine operator*(const Parameter &parameter, const Variable &variable);
Affine operator*(const Variable &variable, const Parameter &parameter);
Affine operator*(const Parameter &parameter, const Affine &affine);

namespace internal
{

// A term like norm2([p_1*x_1 + p_2*x_2 + ... + b_1,   p_3*x_3 + p_4*x_4 + ... + b_2, ... ])
struct Norm2Term
{
    Norm2Term() = default;
    explicit Norm2Term(const Affine &affine);
    std::vector<internal::AffineSum> arguments;
    friend std::ostream &operator<<(std::ostream &os, const Norm2Term &norm2);
    double evaluate(const std::vector<double> &soln_values) const;
};

} // namespace internal

Affine sum(const Affine &affine);
Affine sum(const Affine &affine, size_t axis);

class SOCLhs : public DynamicMatrix<std::pair<internal::Norm2Term, internal::AffineSum>, SOCLhs>
{
public:
    SOCLhs operator+(const Affine &affine) const;
    SOCLhs &operator+=(const Affine &affine);
};

SOCLhs norm2(const Affine &affine);
SOCLhs norm2(const Affine &affine, size_t axis);

} // namespace op