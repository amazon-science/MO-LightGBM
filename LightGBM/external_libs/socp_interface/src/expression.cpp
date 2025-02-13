#include "expression.hpp"

#include <sstream>
#include <cassert>
#include <numeric>
#include <cmath>

namespace op
{

namespace internal
{

AffineTerm::AffineTerm()
    : parameter(0.) {}

AffineTerm::AffineTerm(const ParameterSource &parameter)
    : parameter(parameter) {}

AffineTerm::AffineTerm(const VariableSource &variable)
    : parameter(1.0), variable(variable) {}

AffineTerm::AffineTerm(const ParameterSource &parameter, const VariableSource &variable)
    : parameter(parameter), variable(variable) {}

std::ostream &operator<<(std::ostream &os, const AffineTerm &term)
{
    os << term.parameter.get_value();
    if (term.variable)
        os << "*" << term.variable.value();
    return os;
}

double AffineTerm::evaluate(const std::vector<double> &soln_values) const
{
    double p = parameter.get_value();
    if (variable)
    {
        return p * soln_values[variable.value().getProblemIndex()];
    }
    else
    {
        return p;
    }
}

AffineTerm &AffineTerm::operator*=(const internal::ParameterSource &parameter)
{
    this->parameter = this->parameter * parameter;
    return *this;
}

AffineTerm AffineTerm::operator*(const internal::ParameterSource &parameter) const
{
    AffineTerm result = *this;
    result.parameter = parameter * result.parameter;
    return result;
}

AffineTerm operator*(const internal::ParameterSource &parameter, const VariableSource &variable)
{
    return AffineTerm(parameter, variable);
}

ParameterSource::operator AffineTerm() const
{
    return AffineTerm(*this);
}

VariableSource::operator AffineTerm() const
{
    return AffineTerm(*this);
}

AffineSum::AffineSum(const ParameterSource &parameter) : terms{parameter} {}

AffineSum::AffineSum(const VariableSource &variable) : terms{variable} {}

AffineSum::AffineSum(const AffineTerm &term) : terms{term} {}

std::ostream &operator<<(std::ostream &os, const AffineSum &expression)
{
    for (size_t i = 0; i < expression.terms.size(); i++)
    {
        if (i > 0)
            os << " + ";
        os << expression.terms[i];
    }
    return os;
}

double AffineSum::evaluate(const std::vector<double> &soln_values) const
{
    double result = 0.;
    for (const auto &term : terms)
    {
        result += term.evaluate(soln_values);
    }
    return result;
}

AffineSum AffineSum::operator+(const AffineSum &other) const
{
    AffineSum result = *this;
    result.terms.insert(result.terms.end(), other.terms.begin(), other.terms.end());
    return result;
}

AffineSum AffineSum::operator*(const ParameterSource &parameter) const
{
    AffineSum result;
    result.terms = terms;
    for (AffineTerm &term : result.terms)
    {
        term *= parameter;
    }
    return result;
}

AffineSum AffineSum::operator-() const
{
    return *this * ParameterSource(-1.0);
}

size_t AffineSum::clean()
{
    // erase variables that are multiplied by zero
    auto erase_from = std::remove_if(terms.begin(),
                                     terms.end(),
                                     [](const AffineTerm &term) {
                                         return term.parameter.is_zero();
                                     });
    const auto erase_to = terms.end();
    const size_t erased_elements = std::distance(erase_from, erase_to);
    terms.erase(erase_from, erase_to);

    return erased_elements;
}

bool AffineSum::is_constant() const
{
    return not std::any_of(terms.begin(), terms.end(),
                           [](const AffineTerm &term) {
                               return term.variable.has_value();
                           });
}

AffineSum &AffineSum::operator+=(const AffineSum &other)
{
    terms.insert(terms.end(), other.terms.begin(), other.terms.end());
    return *this;
}

ParameterSource::operator AffineSum() const
{
    return AffineSum(*this);
}

VariableSource::operator AffineSum() const
{
    return AffineSum(*this);
}

AffineTerm::operator AffineSum() const
{
    return AffineSum(*this);
}

} // namespace internal

Affine::Affine(const Parameter &parameter)
{
    resize(parameter.rows(), parameter.cols());
    for (auto [row, col] : all_indices())
    {
        coeffRef(row, col) = internal::AffineSum(parameter.coeff(row, col));
    }
}

Affine::Affine(const Variable &variable)
{
    resize(variable.rows(), variable.cols());
    for (auto [row, col] : variable.all_indices())
    {
        coeffRef(row, col) = internal::AffineSum(variable.coeff(row, col));
    }
}

Affine::Affine(const internal::AffineSum &expression)
{
    resize(1, 1);
    coeffRef(0) = expression;
}

namespace internal
{

std::ostream &operator<<(std::ostream &os, const internal::Norm2Term &norm2)
{
    os << "norm2([";
    for (size_t i = 0; i < norm2.arguments.size(); i++)
    {
        if (i > 0)
            os << ", ";
        os << norm2.arguments[i];
    }
    os << "])";
    return os;
}

} // namespace internal

Affine operator+(const Affine &lhs, const Affine &rhs)
{
    assert(lhs.shape() == rhs.shape());
    Affine result = lhs;
    for (auto [row, col] : lhs.all_indices())
    {
        result.coeffRef(row, col) += rhs.coeff(row, col);
    }
    return result;
}

Affine operator*(const Parameter &parameter, const Variable &variable)
{
    if (parameter.is_scalar() and variable.is_scalar())
    {
        return Affine(parameter.coeff(0) * variable.coeff(0));
    }
    else if (!parameter.is_scalar() and !variable.is_scalar())
    {
        assert(parameter.cols() == variable.rows());

        Affine result(parameter.rows(), variable.cols());
        for (auto [row, col] : result.all_indices())
        {
            internal::AffineSum expression;
            for (size_t inner = 0; inner < parameter.cols(); inner++)
            {
                expression.terms.emplace_back(parameter.coeff(row, inner),
                                              variable.coeff(inner, col));
            }
            result.coeffRef(row, col) = expression;
        }
        return result;
    }
    else if (parameter.is_scalar())
    {
        Affine result(variable.rows(), variable.cols());
        for (auto [row, col] : result.all_indices())
        {
            result.coeffRef(row, col) = internal::AffineTerm(parameter.coeff(0),
                                                             variable.coeff(row, col));
        }
        return result;
    }
    else // if (variable.is_scalar())
    {
        Affine result(parameter.rows(), parameter.cols());
        for (auto [row, col] : result.all_indices())
        {
            result.coeffRef(row, col) = internal::AffineTerm(parameter.coeff(row, col),
                                                             variable.coeff(0));
        }
        return result;
    }
}

Affine operator*(const Variable &variable, const Parameter &parameter)
{
    if (variable.is_scalar() and parameter.is_scalar())
    {
        return Affine(parameter.coeff(0) * variable.coeff(0));
    }
    else if (!variable.is_scalar() and !parameter.is_scalar())
    {
        assert(variable.cols() == parameter.rows());

        Affine result(variable.rows(), parameter.cols());
        for (auto [row, col] : result.all_indices())
        {
            internal::AffineSum expression;
            for (size_t inner = 0; inner < variable.cols(); inner++)
            {
                expression.terms.emplace_back(parameter.coeff(inner, col),
                                              variable.coeff(row, inner));
            }
            result.coeffRef(row, col) = expression;
        }
        return result;
    }
    else if (variable.is_scalar())
    {
        Affine result(variable.rows(), parameter.cols());
        for (auto [row, col] : result.all_indices())
        {
            result.coeffRef(row, col) = internal::AffineTerm(parameter.coeff(row, col),
                                                             variable.coeff(0));
        }
        return result;
    }
    else // if (parameter.is_scalar())
    {
        Affine result(parameter.rows(), parameter.cols());
        for (auto [row, col] : result.all_indices())
        {
            result.coeffRef(row, col) = internal::AffineTerm(parameter.coeff(0), variable.coeff(row, col));
        }
        return result;
    }
}

Affine operator*(const Parameter &parameter, const Affine &affine)
{
    if (not parameter.is_scalar())
    {
        throw std::runtime_error("This operation is not implemented for parameter matrices.");
    }
    Affine result(affine.rows(), affine.cols());
    for (auto [row, col] : affine.all_indices())
    {
        result.coeffRef(row, col) = affine.coeff(row, col) * parameter.coeff(0);
    }
    return result;
}

std::ostream &operator<<(std::ostream &os, const Affine &expression)
{
    for (size_t row = 0; row < expression.rows(); row++)
    {
        os << "[";
        for (size_t col = 0; col < expression.cols(); col++)
        {
            os << expression.coeff(row, col);
            if (col < expression.cols() - 1)
            {
                os << ", ";
            }
        }
        os << "]\n";
    }
    return os;
}

Affine &Affine::operator+=(const Affine &other)
{
    for (auto [row, col] : all_indices())
    {
        coeffRef(row, col) += other.coeff(row, col);
    }
    return *this;
}

Affine Affine::operator-() const
{
    return Parameter(-1.0) * *this;
}

Parameter::operator Affine() const
{
    return Affine(*this);
}

Variable::operator Affine() const
{
    return Affine(*this);
}

namespace internal
{

Norm2Term::Norm2Term(const Affine &affine)
{
    assert(affine.rows() == 1 or affine.cols() == 1);

    if (affine.rows() == 1)
    {
        arguments = affine.rowElements(0);
    }
    else
    {
        arguments = affine.colElements(0);
    }
}

double Norm2Term::evaluate(const std::vector<double> &soln_values) const
{
    auto sum_squares = [&soln_values](double sum, const auto &arg) {
        double val = arg.evaluate(soln_values);
        return sum + val * val;
    };
    return std::sqrt(std::accumulate(arguments.begin(), arguments.end(), 0., sum_squares));
}

} // namespace internal

SOCLhs norm2(const Affine &affine)
{
    assert(affine.rows() == 1 or affine.cols() == 1);

    SOCLhs socLhs;
    for (auto [row, col] : affine.all_indices())
    {
        socLhs.coeffRef(0, 0).first.arguments.push_back(affine.coeff(row, col));
    }
    return socLhs;
}

SOCLhs norm2(const Affine &affine, size_t axis)
{
    assert(axis == 0 or axis == 1);

    SOCLhs socLhs;
    if (axis == 0)
    {
        socLhs.resize(1, affine.cols());
        for (auto [row, col] : affine.all_indices())
        {
            socLhs.coeffRef(0, col).first.arguments.push_back(affine.coeff(row, col));
        }
    }
    else if (axis == 1)
    {
        socLhs.resize(affine.rows(), 1);
        for (auto [row, col] : affine.all_indices())
        {
            socLhs.coeffRef(row, 0).first.arguments.push_back(affine.coeff(row, col));
        }
    }
    return socLhs;
}

Affine Parameter::cwiseProduct(const Affine &affine) const
{
    assert(affine.shape() == shape());
    op::Affine product(rows(), cols());
    for (auto [row, col] : affine.all_indices())
    {
        product.coeffRef(row, col) = affine.coeff(row, col) * coeff(row, col);
    }
    return product;
}

Affine sum(const Affine &affine)
{
    op::Affine sum;
    for (auto [row, col] : affine.all_indices())
    {
        sum += affine(row, col);
    }
    return sum;
}

Affine sum(const Affine &affine, size_t axis)
{
    assert(axis == 0 or axis == 1);
    op::Affine sum;
    if (axis == 0)
    {
        sum.resize(1, affine.cols());
        for (size_t col = 0; col < affine.cols(); col++)
        {
            op::internal::AffineSum row_sum;
            for (size_t row = 0; row < affine.rows(); row++)
            {
                row_sum += affine.coeff(row, col);
            }
            sum.coeffRef(0, col) = row_sum;
        }
    }
    else if (axis == 1)
    {
        sum.resize(affine.rows(), 1);
        for (size_t row = 0; row < affine.rows(); row++)
        {
            op::internal::AffineSum col_sum;
            for (size_t col = 0; col < affine.cols(); col++)
            {
                col_sum += affine.coeff(row, col);
            }
            sum.coeffRef(row, 0) = col_sum;
        }
    }
    return sum;
}

Affine Variable::operator-() const
{
    return Parameter(-1.) * *this;
}

SOCLhs SOCLhs::operator+(const Affine &affine) const
{
    assert(affine.shape() == shape());

    SOCLhs result = *this;

    result += affine;

    return result;
}

SOCLhs &SOCLhs::operator+=(const Affine &affine)
{
    assert(affine.shape() == shape());
    for (auto [row, col] : all_indices())
    {
        coeffRef(row, col).second += affine.coeff(row, col);
    }
    return *this;
}

} // namespace op