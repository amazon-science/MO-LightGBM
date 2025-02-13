#include "parameter.hpp"

#include <iostream>
#include <cassert>

#include <Eigen/Dense>

int main()
{
    // Scalar
    op::Parameter scalar_param_1(1.0);
    op::Parameter scalar_param_2(2.0);
    op::Parameter scalar_param_3(3.0);
    op::Parameter scalar_param_4(4.0);
    double scalar = 3.;
    op::Parameter scalar_param_ptr(&scalar);

    op::Parameter result_scalar = scalar_param_ptr + scalar_param_2 + scalar_param_ptr + scalar_param_2;
    assert(result_scalar.get_value() == 10.);

    result_scalar = scalar_param_ptr + scalar_param_2 - scalar_param_ptr - scalar_param_2;
    assert(result_scalar.get_value() == 0.);

    result_scalar = scalar_param_ptr * scalar_param_2 - scalar_param_ptr * scalar_param_2;
    assert(result_scalar.get_value() == 0.);

    result_scalar = scalar_param_ptr * scalar_param_2 * scalar_param_ptr * scalar_param_2;
    assert(result_scalar.get_value() == 36.);

    scalar = 1.;
    result_scalar = scalar_param_2 + scalar_param_ptr;
    assert(result_scalar.get_value() == 3.);
    result_scalar = scalar_param_2 - scalar_param_ptr;
    assert(result_scalar.get_value() == 1.);
    result_scalar = scalar_param_2 * scalar_param_ptr;
    assert(result_scalar.get_value() == 2.);
    result_scalar = scalar_param_2 / scalar_param_ptr;
    assert(result_scalar.get_value() == 2.);

    // Matrix
    // multiply 2x2/2x2
    Eigen::Matrix2d dyn_matrix;
    dyn_matrix << 1., 2., 3., 4.;
    op::Parameter matrix_parameter(&dyn_matrix);
    op::Parameter result_matrix = matrix_parameter * matrix_parameter;

    Eigen::Matrix2d reference_matrix;

    reference_matrix << 7., 10., 15., 22.;
    assert(result_matrix.get_values() == reference_matrix);

    dyn_matrix(1, 0) = 1.;
    reference_matrix << 3., 10., 5., 18.;
    assert(result_matrix.get_values() == reference_matrix);

    dyn_matrix(1, 0) = 3.;

    Eigen::Vector2d dyn_vector(1., 2.);

    // multiply 2x2/2x1
    op::Parameter vector_parameter(&dyn_vector);
    result_matrix = matrix_parameter * vector_parameter;
    assert(result_matrix.get_values() == Eigen::Vector2d(5., 11));

    // multiply 1x2/2x2
    result_matrix = vector_parameter.transpose() * matrix_parameter;
    assert(result_matrix.get_values() == Eigen::Vector2d(7., 10.).transpose());

    // multiply 1x1/2x2
    result_matrix = scalar_param_2 * matrix_parameter;
    reference_matrix << 2., 4., 6., 8.;
    assert(result_matrix.get_values() == reference_matrix);

    // multiply 2x2/1x1
    result_matrix = matrix_parameter * scalar_param_2;
    assert(result_matrix.get_values() == reference_matrix);

    Eigen::Matrix3d m1, m2;
    m1.setRandom();
    m2.setRandom();
    op::Parameter eigen1(&m1);
    op::Parameter eigen2(m2);

    scalar = 3.;
    Eigen::MatrixXd m = scalar * m1 * m2;
    op::Parameter result = scalar_param_ptr * eigen1 * eigen2;
    assert((result.get_values() - m).cwiseAbs().sum() < 1e-10);

    scalar = 10.;
    m = scalar * m1 * m2;
    assert((result.get_values() - m).cwiseAbs().sum() < 1e-10);

    Eigen::MatrixXd m3x2(3, 2);
    Eigen::MatrixXd m2x5(2, 5);
    m3x2.setRandom();
    m2x5.setRandom();

    m = m3x2 * m2x5;
    result = op::Parameter(m3x2) * op::Parameter(m2x5);

    assert((result.get_values() - m).cwiseAbs().sum() < 1e-10);

    std::cout << "All tests were successful."
              << "\n";
}