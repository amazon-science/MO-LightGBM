#pragma once

#include "wrapperBase.hpp"

#include "EiCOS/include/eicos.hpp"

namespace op
{

class EicosWrapper : public WrapperBase
{
    using WrapperBase::WrapperBase;

    EiCOS::exitcode last_exit_flag;

    std::unique_ptr<EiCOS::Solver> solver;

    std::vector<double> c_values;
    std::vector<double> h_values;
    std::vector<double> b_values;
    std::vector<double> G_data_CCS_values;
    std::vector<double> A_data_CCS_values;

public:
    void initialize() override;
    bool solveProblem(bool verbose = false) override;
    std::string getResultString() const override;
};

} // namespace op