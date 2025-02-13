#include "variable.hpp"

#include <sstream>

namespace op
{

namespace internal
{

VariableSource::VariableSource(const std::string &name, size_t problem_index,
                               size_t row, size_t col)
    : name(name), problem_index(problem_index), index(row, col) {}

size_t VariableSource::getProblemIndex() const
{
    return problem_index;
}

std::ostream &operator<<(std::ostream &os, const VariableSource &variable)
{
    os << variable.name
       << "[" << variable.index.first << ", " << variable.index.second << "]"
       << "@(" << variable.problem_index << ")";
    return os;
}

} // namespace internal

Variable::Variable(const std::string &name, size_t start_index,
                   size_t rows, size_t cols)
    : name(name)
{
    resize(rows, cols);

    size_t index = start_index;
    for (auto [row, col] : all_indices())
    {
        coeffRef(row, col) = internal::VariableSource(name, index, row, col);
        index++;
    }
}

std::ostream &operator<<(std::ostream &os, const Variable &variable)
{
    os << variable.name
       << "(" << variable.rows() << "x" << variable.cols() << ")";
    return os;
}

} // namespace op