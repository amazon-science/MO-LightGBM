#pragma once

#include "wrapperBase.hpp"

#include <any>

namespace op
{

class EcosWrapper : public WrapperBase
{
    using WrapperBase::WrapperBase;

    long last_exit_flag = -99;

    std::any work;

    size_t step = 0;

    std::vector<long> cone_constraint_dimensions_l;
    std::vector<long> A_columns_CCS_l;
    std::vector<long> G_columns_CCS_l;
    std::vector<long> A_rows_CCS_l;
    std::vector<long> G_rows_CCS_l;

    std::vector<double> c_values1;
    std::vector<double> h_values1;
    std::vector<double> b_values1;
    std::vector<double> G_data_CCS_values1;
    std::vector<double> A_data_CCS_values1;

    std::vector<double> c_values2;
    std::vector<double> h_values2;
    std::vector<double> b_values2;
    std::vector<double> G_data_CCS_values2;
    std::vector<double> A_data_CCS_values2;

public:
    void initialize() override;
    bool solveProblem(bool verbose = false) override;
    std::string getResultString() const override;
};

} // namespace op