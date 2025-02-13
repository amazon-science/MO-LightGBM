#include "ecosWrapper.hpp"

#define DCTRLC = 1
#define DLONG
#define LDL_LONG
#include "ecos/include/ecos.h"

static_assert(std::is_same_v<idxint, SuiteSparse_long> and std::is_same_v<idxint, long>,
              "Definitions of idxint are not consistent."
              "Make sure ECOS is compiled with USE_LONG = 1.");

namespace op
{

void ecos_get_parameter_values(const std::vector<internal::ParameterSource> &params,
                               double factor,
                               std::vector<double> &values)
{
    assert(values.size() == params.size());
    std::transform(params.begin(), params.end(), values.begin(),
                   [factor](const auto &param) { return param.get_value() * factor; });
}

void EcosWrapper::initialize()
{
    G_data_CCS_values1.resize(G_data_CCS.size());
    A_data_CCS_values1.resize(A_data_CCS.size());
    c_values1.resize(n_variables);
    h_values1.resize(n_constraint_rows);
    b_values1.resize(n_equalities);

    G_data_CCS_values2.resize(G_data_CCS.size());
    A_data_CCS_values2.resize(A_data_CCS.size());
    c_values2.resize(n_variables);
    h_values2.resize(n_constraint_rows);
    b_values2.resize(n_equalities);

    cone_constraint_dimensions_l = std::vector<long>(cone_constraint_dimensions.begin(), cone_constraint_dimensions.end());
    A_rows_CCS_l = std::vector<long>(A_rows_CCS.begin(), A_rows_CCS.end());
    A_columns_CCS_l = std::vector<long>(A_columns_CCS.begin(), A_columns_CCS.end());
    G_rows_CCS_l = std::vector<long>(G_rows_CCS.begin(), G_rows_CCS.end());
    G_columns_CCS_l = std::vector<long>(G_columns_CCS.begin(), G_columns_CCS.end());

    work = ECOS_setup(
        n_variables,
        n_constraint_rows,
        n_equalities,
        n_positive_constraints,
        n_cone_constraints,
        cone_constraint_dimensions_l.data(),
        n_exponential_cones,
        G_data_CCS_values1.data(),
        G_columns_CCS_l.data(),
        G_rows_CCS_l.data(),
        A_data_CCS_values1.data(),
        A_columns_CCS_l.data(),
        A_rows_CCS_l.data(),
        c_values1.data(),
        h_values1.data(),
        b_values1.data());

    if (std::any_cast<pwork*>(work) == nullptr)
    {
        throw std::runtime_error("Could not set up problem.");
    }
}

bool EcosWrapper::solveProblem(bool verbose)
{
    auto ecos_work = std::any_cast<pwork*>(work);

    assert(ecos_work != nullptr && "You must first call initialize()!");

    long exitflag;
    ecos_work->stgs->verbose = verbose;

    // awkward switching between memory locations
    std::vector<double> *c_values;
    std::vector<double> *h_values;
    std::vector<double> *b_values;
    std::vector<double> *G_data_CCS_values;
    std::vector<double> *A_data_CCS_values;
    if (step % 2 == 0)
    {
        c_values = &c_values1;
        h_values = &h_values1;
        b_values = &b_values1;
        G_data_CCS_values = &G_data_CCS_values1;
        A_data_CCS_values = &A_data_CCS_values1;
    }
    else
    {
        c_values = &c_values2;
        h_values = &h_values2;
        b_values = &b_values2;
        G_data_CCS_values = &G_data_CCS_values2;
        A_data_CCS_values = &A_data_CCS_values2;
    }
    step++;

    ecos_get_parameter_values(c, 1.0, *c_values);
    ecos_get_parameter_values(h, 1.0, *h_values);
    ecos_get_parameter_values(b, 1.0, *b_values);
    ecos_get_parameter_values(G_data_CCS, -1.0, *G_data_CCS_values);
    ecos_get_parameter_values(A_data_CCS, -1.0, *A_data_CCS_values);
    // The signs for A and G must be flipped because they are negative in the ECOS interface

    ECOS_updateData(ecos_work,
                    G_data_CCS_values->data(),
                    A_data_CCS_values->data(),
                    c_values->data(),
                    h_values->data(),
                    b_values->data());

    exitflag = ECOS_solve(ecos_work);

    // copy solution
    std::copy(ecos_work->x, std::next(ecos_work->x, n_variables),
              socp.solution_vector.begin());

    if (exitflag == ECOS_SIGINT)
    {
        std::exit(130);
    }

    last_exit_flag = exitflag;

    return exitflag != ECOS_FATAL;
}

std::string EcosWrapper::getResultString() const
{
    switch (last_exit_flag)
    {
    case -99:
        return "Problem not solved yet.";
    case ECOS_OPTIMAL:
        return "Optimal solution found.";
    case ECOS_PINF:
        return "Certificate of primal infeasibility found.";
    case ECOS_DINF:
        return "Certificate of dual infeasibility found.";

    case ECOS_OPTIMAL + ECOS_INACC_OFFSET:
        return "Optimal solution found subject to reduced tolerances.";
    case ECOS_PINF + ECOS_INACC_OFFSET:
        return "Certificate of primal infeasibility found subject to reduced tolerances.";
    case ECOS_DINF + ECOS_INACC_OFFSET:
        return "Certificate of dual infeasibility found subject to reduced tolerances.";

    case ECOS_MAXIT:
        return "Maximum number of iterations reached.";

    case ECOS_NUMERICS:
        return "Numerical problems (unreliable search direction).";
    case ECOS_OUTCONE:
        return "Numerical problems (slacks or multipliers outside cone)";

    case ECOS_SIGINT:
        return "Interrupted by signal or CTRL-C.";
    default: // ECOS_FATAL:
        return "Unknown problem in solver.";
    }
}

} // namespace op
