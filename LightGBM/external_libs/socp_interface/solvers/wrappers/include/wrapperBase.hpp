#pragma once

#include "secondOrderConeProgram.hpp"

namespace op
{

class WrapperBase
{
protected:
    SecondOrderConeProgram &socp;

    int n_variables;
    int n_constraint_rows;
    int n_equalities;
    int n_positive_constraints;
    int n_cone_constraints;
    std::vector<int> cone_constraint_dimensions;
    int n_exponential_cones;
    std::vector<internal::ParameterSource> G_data_CCS;
    std::vector<int> G_columns_CCS;
    std::vector<int> G_rows_CCS;
    std::vector<internal::ParameterSource> A_data_CCS;
    std::vector<int> A_columns_CCS;
    std::vector<int> A_rows_CCS;
    std::vector<internal::ParameterSource> c;
    std::vector<internal::ParameterSource> h;
    std::vector<internal::ParameterSource> b;

public:
    explicit WrapperBase(SecondOrderConeProgram &_socp);
    virtual bool solveProblem(bool verbose = false) = 0;
    virtual std::string getResultString() const = 0;
    virtual void initialize() = 0;
};

} // namespace op