#include "parameter.hpp"

#include <sstream>
#include <cassert>

namespace op
{

namespace internal
{

ParameterSource::ParameterSource(const double const_value)
    : source(const_value) {}

ParameterSource::ParameterSource(double *value_ptr)
    : source(value_ptr) {}

ParameterSource::ParameterSource(const std::function<double()> &callback)
    : source(callback) {}

double ParameterSource::get_value() const
{
    switch (source.index())
    {
    case 0:
        return std::get<0>(source);
    case 1:
        return *std::get<1>(source);
    default:
        return std::get<2>(source)();
    }
}

bool ParameterSource::is_constant() const
{
    return source.index() == 0;
}

bool ParameterSource::is_pointer() const
{
    return source.index() == 1;
}

bool ParameterSource::is_callback() const
{
    return source.index() == 2;
}

bool ParameterSource::is_zero() const
{
    return is_constant() and std::abs(get_value()) < 1e-10;
}

bool ParameterSource::is_one() const
{
    return is_constant() and std::abs(get_value() - 1.) < 1e-10;
}

ParameterSource ParameterSource::operator+(const ParameterSource &other) const
{
    if (other.is_zero())
    {
        return *this;
    }
    if (is_constant() and other.is_constant())
    {
        return ParameterSource(get_value() + other.get_value());
    }

    return ParameterSource([p1 = *this,
                            p2 = other]() {
        return p1.get_value() + p2.get_value();
    });
}

ParameterSource ParameterSource::operator-(const ParameterSource &other) const
{
    if (other.is_zero())
    {
        return *this;
    }
    if (is_constant() and other.is_constant())
    {
        return ParameterSource(get_value() - other.get_value());
    }

    return ParameterSource([p1 = *this,
                            p2 = other]() {
        return p1.get_value() - p2.get_value();
    });
}

ParameterSource ParameterSource::operator*(const ParameterSource &other) const
{
    if (is_zero() or other.is_zero())
    {
        return ParameterSource(0.);
    }
    if (is_constant() and other.is_constant())
    {
        return ParameterSource(get_value() * other.get_value());
    }

    return ParameterSource([p1 = *this,
                            p2 = other]() {
        return p1.get_value() * p2.get_value();
    });
}

ParameterSource ParameterSource::operator/(const ParameterSource &other) const
{
    assert(not other.is_zero());

    if (other.is_one())
    {
        return *this;
    }
    if (is_constant() and other.is_constant())
    {
        return ParameterSource(get_value() / other.get_value());
    }

    return ParameterSource([p1 = *this,
                            p2 = other]() {
        return p1.get_value() / p2.get_value();
    });
}

} // namespace internal

Parameter::Parameter(const double const_value)
{
    resize(1, 1);
    coeffRef(0) = internal::ParameterSource(const_value);
}

Parameter::Parameter(double *value_ptr)
{
    resize(1, 1);
    coeffRef(0) = internal::ParameterSource(value_ptr);
}

Parameter::Parameter(const std::function<double()> &callback)
{
    resize(1, 1);
    coeffRef(0) = internal::ParameterSource(callback);
}

double Parameter::get_value(const size_t row, const size_t col) const
{
    return coeff(row, col).get_value();
}

Eigen::MatrixXd Parameter::get_values() const
{
    Eigen::MatrixXd result_matrix(rows(), cols());

    for (auto [row, col] : all_indices())
    {
        result_matrix.coeffRef(row, col) = get_value(row, col);
    }
    return result_matrix;
}

Parameter Parameter::operator+(const Parameter &other) const
{
    assert(shape() == other.shape());

    Parameter parameter(rows(), cols());
    for (auto [row, col] : all_indices())
    {
        parameter.coeffRef(row, col) = coeff(row, col) + other.coeff(row, col);
    }
    return parameter;
}

Parameter Parameter::operator-() const
{
    return Parameter(-1.0) * *this;
}

Parameter Parameter::operator-(const Parameter &other) const
{
    assert(shape() == other.shape());

    Parameter parameter(rows(), cols());
    for (auto [row, col] : all_indices())
    {
        parameter.coeffRef(row, col) = coeff(row, col) - other.coeff(row, col);
    }
    return parameter;
}

Parameter scaleMatrix(const internal::ParameterSource &scalar, const Parameter &matrix)
{
    Parameter parameter(matrix.rows(), matrix.cols());
    for (auto [row, col] : matrix.all_indices())
    {
        parameter.coeffRef(row, col) = scalar * matrix.coeff(row, col);
    }
    return parameter;
}

Parameter multiplyMatrices(const Parameter &matrix1, const Parameter &matrix2)
{
    assert(matrix1.cols() == matrix2.rows());

    Parameter parameter(matrix1.rows(), matrix2.cols());
    for (size_t row = 0; row < matrix1.rows(); row++)
    {
        for (size_t col = 0; col < matrix2.cols(); col++)
        {
            std::vector<internal::ParameterSource> row_vector(matrix1.cols());
            std::vector<internal::ParameterSource> column_vector(matrix2.rows());
            for (size_t inner = 0; inner < matrix1.cols(); inner++)
            {
                row_vector.at(inner) = matrix1.coeff(row, inner);
                column_vector.at(inner) = matrix2.coeff(inner, col);
            }
            auto sum_op = [rv = row_vector, cv = column_vector] {
                double element = 0;
                for (size_t inner = 0; inner < rv.size(); inner++)
                {
                    element += rv[inner].get_value() * cv[inner].get_value();
                }
                return element;
            };
            parameter.coeffRef(row, col) = internal::ParameterSource(sum_op);
        }
    }
    return parameter;
}

Parameter Parameter::operator*(const Parameter &other) const
{
    if (is_scalar() and other.is_scalar())
    {
        return Parameter({{coeff(0) * other.coeff(0)}});
    }
    if (!is_scalar() and !other.is_scalar())
    {
        return multiplyMatrices(*this, other);
    }
    if (is_scalar())
    {
        return scaleMatrix(coeff(0), other);
    }
    else // if (other.is_scalar())
    {
        return scaleMatrix(other.coeff(0), *this);
    }
}

Parameter Parameter::operator/(const Parameter &other) const
{
    assert(other.is_scalar());

    Parameter parameter(rows(), cols());
    for (auto [row, col] : all_indices())
    {
        parameter.coeffRef(row, col) = coeff(row, col) / other.coeff(0);
    }
    return parameter;
}

} // namespace op