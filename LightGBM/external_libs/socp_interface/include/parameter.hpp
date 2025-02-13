#pragma once

#include <functional>
#include <string>
#include <utility>
#include <variant>
#include <memory>

#include "dynamicMatrix.hpp"

#include <Eigen/Dense>

namespace op
{

class Affine;

namespace internal
{

struct AffineTerm;
struct AffineSum;

class ParameterSource
{
public:
    ParameterSource() = default;
    explicit ParameterSource(const double const_value);
    explicit ParameterSource(double *value_ptr);
    explicit ParameterSource(const std::function<double()> &callback);
    double get_value() const;
    bool is_constant() const;
    bool is_pointer() const;
    bool is_callback() const;
    bool is_zero() const;
    bool is_one() const;

    ParameterSource operator+(const ParameterSource &other) const;
    ParameterSource operator-(const ParameterSource &other) const;
    ParameterSource operator*(const ParameterSource &other) const;
    ParameterSource operator/(const ParameterSource &other) const;
    operator AffineTerm() const;
    operator AffineSum() const;

private:
    using source_variant_t = std::variant<double,
                                          const double *,
                                          std::function<double()>>;
    source_variant_t source;
};

} // namespace internal

class Parameter : public DynamicMatrix<internal::ParameterSource, Parameter>
{
public:
    using DynamicMatrix<internal::ParameterSource, Parameter>::DynamicMatrix;
    explicit Parameter(const double const_value);
    explicit Parameter(double *value_ptr);
    explicit Parameter(const std::function<double()> &callback);

    template <typename Derived>
    explicit Parameter(const Eigen::DenseBase<Derived> &matrix);
    template <typename Derived>
    explicit Parameter(Eigen::DenseBase<Derived> *matrix);

    Parameter operator+(const Parameter &other) const;
    Parameter operator-() const;
    Parameter operator-(const Parameter &other) const;
    Parameter operator*(const Parameter &other) const;
    Parameter operator/(const Parameter &other) const;
    Affine cwiseProduct(const Affine &affine) const;
    double get_value(const size_t row = 0,
                     const size_t col = 0) const;
    Eigen::MatrixXd get_values() const;

    operator Affine() const;
};

template <typename Derived>
Parameter::Parameter(const Eigen::DenseBase<Derived> &matrix)
{
    resize(matrix.rows(), matrix.cols());
    for (size_t row = 0; row < size_t(matrix.rows()); row++)
    {
        for (size_t col = 0; col < size_t(matrix.cols()); col++)
        {
            coeffRef(row, col) = internal::ParameterSource(matrix(row, col));
        }
    }
}

template <typename Derived>
Parameter::Parameter(Eigen::DenseBase<Derived> *matrix)
{
    resize(matrix->rows(), matrix->cols());
    for (size_t row = 0; row < size_t(matrix->rows()); row++)
    {
        for (size_t col = 0; col < size_t(matrix->cols()); col++)
        {
            coeffRef(row, col) = internal::ParameterSource(&matrix->coeffRef(row, col));
        }
    }
}

} // namespace op