#include "eicosWrapper.hpp"

namespace op
{

void EicosWrapper::initialize()
{
    G_data_CCS_values.resize(G_data_CCS.size());
    A_data_CCS_values.resize(A_data_CCS.size());
    c_values.resize(n_variables);
    h_values.resize(n_constraint_rows);
    b_values.resize(n_equalities);

    solver = std::make_unique<EiCOS::Solver>(n_variables,
                                             n_constraint_rows,
                                             n_equalities,
                                             n_positive_constraints,
                                             n_cone_constraints,
                                             cone_constraint_dimensions.data(),
                                             G_data_CCS_values.data(),
                                             G_columns_CCS.data(),
                                             G_rows_CCS.data(),
                                             A_data_CCS_values.data(),
                                             A_columns_CCS.data(),
                                             A_rows_CCS.data(),
                                             c_values.data(),
                                             h_values.data(),
                                             b_values.data());
}

void get_parameter_values(const std::vector<internal::ParameterSource> &params,
                          double factor,
                          std::vector<double> &values)
{
    assert(values.size() == params.size());
    std::transform(params.begin(), params.end(), values.begin(),
                   [factor](const auto &param) { return param.get_value() * factor; });
}

bool EicosWrapper::solveProblem(bool verbose)
{
    assert(solver != nullptr && "You must first call initialize()!");

    get_parameter_values(c, 1.0, c_values);
    get_parameter_values(h, 1.0, h_values);
    get_parameter_values(b, 1.0, b_values);
    get_parameter_values(G_data_CCS, -1.0, G_data_CCS_values);
    get_parameter_values(A_data_CCS, -1.0, A_data_CCS_values);
    // The signs for A and G must be flipped because they are negative in the EiCOS interface

    solver->updateData(G_data_CCS_values.data(),
                       A_data_CCS_values.data(),
                       c_values.data(),
                       h_values.data(),
                       b_values.data());

    EiCOS::exitcode exitflag = solver->solve(verbose);

    // copy solution
    for (int i = 0; i < solver->solution().size(); i++)
    {
        socp.solution_vector[i] = solver->solution()(i);
    }

    last_exit_flag = exitflag;

    return exitflag != EiCOS::exitcode::fatal;
}

std::string EicosWrapper::getResultString() const
{
    switch (last_exit_flag)
    {
    case EiCOS::exitcode::optimal:
        return "Optimal solution found.";
    case EiCOS::exitcode::primal_infeasible:
        return "Certificate of primal infeasibility found.";
    case EiCOS::exitcode::dual_infeasible:
        return "Certificate of dual infeasibility found.";

    case EiCOS::exitcode::close_to_optimal:
        return "Optimal solution found subject to reduced tolerances.";
    case EiCOS::exitcode::close_to_primal_infeasible:
        return "Certificate of primal infeasibility found subject to reduced tolerances.";
    case EiCOS::exitcode::close_to_dual_infeasible:
        return "Certificate of dual infeasibility found subject to reduced tolerances.";

    case EiCOS::exitcode::maxit:
        return "Maximum number of iterations reached.";

    case EiCOS::exitcode::numerics:
        return "Numerical problems (unreliable search direction).";
    case EiCOS::exitcode::outcone:
        return "Numerical problems (slacks or multipliers outside cone)";

    default: // EiCOS::exitcode::fatal
        return "Unknown problem in solver.";
    }
}

} // namespace op
