#pragma once

#include <utility>
#include <ostream>
#include <vector>
#include <string>

#include "dynamicMatrix.hpp"

namespace op
{

struct Affine;

namespace internal
{

struct AffineTerm;
struct AffineSum;

class VariableSource
{
public:
    VariableSource() = default;
    VariableSource(const std::string &name, size_t problem_index,
                   size_t row, size_t col);
    size_t getProblemIndex() const;
    friend std::ostream &operator<<(std::ostream &os,
                                    const VariableSource &variable);
    operator AffineTerm() const;
    operator AffineSum() const;
    std::string name;

private:
    size_t problem_index;
    std::pair<size_t, size_t> index;
};

} // namespace internal

// A scalar/vector/matrix optimization variable
class Variable : public DynamicMatrix<internal::VariableSource, Variable>
{
public:
    using DynamicMatrix<internal::VariableSource, Variable>::DynamicMatrix;
    Variable(const std::string &name, size_t start_index,
             size_t rows = 1, size_t cols = 1);
    friend std::ostream &operator<<(std::ostream &os,
                                    const Variable &variable);
    operator Affine() const;

    Affine operator-() const;

private:
    std::string name;
    // indices of the value in the solution vector x
};

} // namespace op