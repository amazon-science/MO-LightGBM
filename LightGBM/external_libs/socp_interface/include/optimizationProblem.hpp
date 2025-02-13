#pragma once

#include "constraint.hpp"

#include <map>
#include <vector>

namespace op
{

class GenericOptimizationProblem
{
public:
    Variable createVariable(const std::string &name,
                            size_t rows = 1, size_t cols = 1);

    Variable getVariable(const std::string &name) const;

    size_t getNumVariables() const;

    void readSolution(const std::string &name,
                      double &solution) const;

    template <typename Derived>
    void readSolution(const std::string &name,
                      Eigen::PlainObjectBase<Derived> &solution) const;

    std::vector<double> solution_vector;

protected:
    std::map<std::string, Variable> variables;
};

template <typename Derived>
void GenericOptimizationProblem::readSolution(const std::string &name,
                                              Eigen::PlainObjectBase<Derived> &solution) const
{
    const Variable &variable = variables.at(name);
    solution.resize(variable.rows(), variable.cols());
    for (size_t row = 0; row < variable.rows(); row++)
    {
        for (size_t col = 0; col < variable.cols(); col++)
        {
            solution(row, col) = solution_vector[variable.coeff(row, col).getProblemIndex()];
        }
    }
}

} // namespace op