#pragma once

#include <vector>
#include <cassert>
#include <memory>

namespace op
{

template <typename T>
T vstack(const std::initializer_list<T> elements)
{
    if (elements.size() == 1)
    {
        return *elements.begin();
    }

    T stacked = *elements.begin();
    for (auto e_ptr = std::next(elements.begin()); e_ptr != elements.end(); e_ptr++)
    {
        assert(stacked.cols() == e_ptr->cols());

        const size_t start_row = stacked.rows();
        stacked.resize(start_row + e_ptr->rows(), stacked.cols());

        for (auto [row, col] : e_ptr->all_indices())
        {
            stacked.coeffRef(start_row + row, col) = e_ptr->coeff(row, col);
        }
    }
    return stacked;
}

template <typename T>
T hstack(const std::initializer_list<T> elements)
{
    if (elements.size() == 1)
    {
        return *elements.begin();
    }

    T stacked = *elements.begin();
    for (auto e_ptr = std::next(elements.begin()); e_ptr != elements.end(); e_ptr++)
    {
        assert(stacked.rows() == e_ptr->rows());

        const size_t start_col = stacked.cols();
        stacked.resize(stacked.rows(), start_col + e_ptr->cols());

        for (auto [row, col] : e_ptr->all_indices())
        {
            stacked.coeffRef(row, start_col + col) = e_ptr->coeff(row, col);
        }
    }
    return stacked;
}

template <typename T, class Derived>
class DynamicMatrix
{

public:
    DynamicMatrix();
    explicit DynamicMatrix(size_t rows, size_t cols = 1);
    explicit DynamicMatrix(const std::vector<std::vector<T>> &matrix);

    bool empty() const;
    size_t rows() const;
    size_t cols() const;
    size_t size() const;
    bool is_scalar() const;

    std::pair<size_t, size_t> shape() const;

    std::vector<T> rowElements(size_t index) const;
    std::vector<T> colElements(size_t index) const;

    auto row(size_t index) const;
    auto col(size_t index) const;
    auto block(size_t start_row, size_t start_col,
               size_t n_rows, size_t n_cols) const;
    auto topLeftCorner(size_t n_rows, size_t n_cols) const;
    auto bottomLeftCorner(size_t n_rows, size_t n_cols) const;
    auto topRightCorner(size_t n_rows, size_t n_cols) const;
    auto bottomRightCorner(size_t n_rows, size_t n_cols) const;
    auto topRows(size_t n_rows) const;
    auto bottomRows(size_t n_rows) const;
    auto leftCols(size_t n_cols) const;
    auto rightCols(size_t n_cols) const;
    auto transpose() const;
    auto head(size_t n) const;
    auto tail(size_t n) const;
    auto segment(size_t i, size_t n) const;
    std::vector<std::pair<size_t, size_t>> all_indices() const;

    T &coeffRef(size_t row, size_t col = 0);

    auto operator()(size_t row, size_t col = 0) const;
    const T &coeff(size_t row, size_t col = 0) const;

    void resize(size_t rows, size_t cols);

private:
    std::vector<std::vector<T>> data_matrix;
};

template <typename T, class Derived>
DynamicMatrix<T, Derived>::DynamicMatrix()
{
    resize(1, 1);
}

template <typename T, class Derived>
DynamicMatrix<T, Derived>::DynamicMatrix(size_t rows, size_t cols)
{
    resize(rows, cols);
}

template <typename T, class Derived>
DynamicMatrix<T, Derived>::DynamicMatrix(const std::vector<std::vector<T>> &matrix)
    : data_matrix(matrix) {}

template <typename T, class Derived>
bool DynamicMatrix<T, Derived>::empty() const
{
    return rows() == 0;
}

template <typename T, class Derived>
size_t DynamicMatrix<T, Derived>::rows() const
{
    return data_matrix.size();
}

template <typename T, class Derived>
size_t DynamicMatrix<T, Derived>::cols() const
{
    if (rows() > 0)
    {
        return data_matrix.front().size();
    }
    else
    {
        return 0;
    }
}

template <typename T, class Derived>
size_t DynamicMatrix<T, Derived>::size() const
{
    return rows() * cols();
}

template <typename T, class Derived>
bool DynamicMatrix<T, Derived>::is_scalar() const
{
    return rows() == 1 and cols() == 1;
}

template <typename T, class Derived>
std::vector<T> DynamicMatrix<T, Derived>::rowElements(size_t index) const
{
    assert(index < rows());
    return data_matrix.at(index);
}

template <typename T, class Derived>
std::vector<T> DynamicMatrix<T, Derived>::colElements(size_t index) const
{
    assert(index < cols());
    std::vector<T> column;
    for (const auto &row : data_matrix)
    {
        column.push_back(row.at(index));
    }
    return column;
}

template <typename T, class Derived>
std::pair<size_t, size_t> DynamicMatrix<T, Derived>::shape() const
{
    return {rows(), cols()};
}

template <typename T, class Derived>
auto DynamicMatrix<T, Derived>::operator()(size_t row, size_t col) const
{
    assert(row < rows() and col < cols());
    Derived result(1, 1);
    result.coeffRef(0, 0) = coeff(row, col);
    return result;
}

template <typename T, class Derived>
auto DynamicMatrix<T, Derived>::row(size_t index) const
{
    assert(index < rows());

    Derived result(1, cols());

    for (size_t col = 0; col < cols(); col++)
    {
        result.coeffRef(0, col) = coeff(index, col);
    }

    return result;
}

template <typename T, class Derived>
auto DynamicMatrix<T, Derived>::col(size_t index) const
{
    assert(index < cols());

    Derived result(rows(), 1);

    for (size_t row = 0; row < rows(); row++)
    {
        result.coeffRef(row, 0) = coeff(row, index);
    }

    return result;
}

template <typename T, class Derived>
auto DynamicMatrix<T, Derived>::block(size_t start_row, size_t start_col,
                                      size_t n_rows, size_t n_cols) const
{
    assert((start_row + n_rows - 1) < rows() and
           (start_col + n_cols - 1) < cols());

    Derived result(n_rows, n_cols);

    for (auto [row, col] : result.all_indices())
    {
        result.coeffRef(row, col) = coeff(start_row + row, start_col + col);
    }
    return result;
}

template <typename T, class Derived>
auto DynamicMatrix<T, Derived>::topLeftCorner(size_t n_rows, size_t n_cols) const
{
    return block(0, 0, n_rows, n_cols);
}

template <typename T, class Derived>
auto DynamicMatrix<T, Derived>::bottomLeftCorner(size_t n_rows, size_t n_cols) const
{
    return block(rows() - n_rows - 1, 0, n_rows, n_cols);
}

template <typename T, class Derived>
auto DynamicMatrix<T, Derived>::topRightCorner(size_t n_rows, size_t n_cols) const
{
    return block(0, cols() - n_cols - 1, n_rows, n_cols);
}

template <typename T, class Derived>
auto DynamicMatrix<T, Derived>::bottomRightCorner(size_t n_rows, size_t n_cols) const
{
    return block(rows() - n_rows - 1, cols() - n_cols - 1, n_rows, n_cols);
}

template <typename T, class Derived>
auto DynamicMatrix<T, Derived>::topRows(size_t n_rows) const
{
    return block(0, 0, n_rows, cols());
}

template <typename T, class Derived>
auto DynamicMatrix<T, Derived>::bottomRows(size_t n_rows) const
{
    return block(rows() - n_rows - 1, 0, n_rows, cols());
}

template <typename T, class Derived>
auto DynamicMatrix<T, Derived>::leftCols(size_t n_cols) const
{
    return block(0, 0, rows(), n_cols);
}

template <typename T, class Derived>
auto DynamicMatrix<T, Derived>::rightCols(size_t n_cols) const
{
    return block(0, cols() - n_cols - 1, rows(), n_cols);
}

template <typename T, class Derived>
auto DynamicMatrix<T, Derived>::head(size_t n) const
{
    assert(cols() == 1);
    return block(0, 0, n, 1);
}

template <typename T, class Derived>
auto DynamicMatrix<T, Derived>::tail(size_t n) const
{
    assert(cols() == 1);
    return block(rows() - n - 1, 0, n, 1);
}

template <typename T, class Derived>
auto DynamicMatrix<T, Derived>::segment(size_t i, size_t n) const
{
    assert(cols() == 1);
    return block(i, 0, n, 1);
}

template <typename T, class Derived>
auto DynamicMatrix<T, Derived>::transpose() const
{
    Derived result(cols(), rows());

    for (auto [row, col] : all_indices())
    {
        result.coeffRef(col, row) = coeff(row, col);
    }

    return result;
}

template <typename T, class Derived>
const T &DynamicMatrix<T, Derived>::coeff(size_t row, size_t col) const
{
    assert(row < rows() and col < cols());
    return data_matrix[row][col];
}

template <typename T, class Derived>
T &DynamicMatrix<T, Derived>::coeffRef(size_t row, size_t col)
{
    assert(row < rows() and col < cols());
    return data_matrix[row][col];
}

template <typename T, class Derived>
void DynamicMatrix<T, Derived>::resize(size_t rows, size_t cols)
{
    data_matrix.resize(rows);
    for (auto &row : data_matrix)
    {
        row.resize(cols);
    }
}

template <typename T, class Derived>
std::vector<std::pair<size_t, size_t>> DynamicMatrix<T, Derived>::all_indices() const
{
    std::vector<std::pair<size_t, size_t>> indices;

    for (size_t row = 0; row < rows(); row++)
    {
        for (size_t col = 0; col < cols(); col++)
        {
            indices.emplace_back(row, col);
        }
    }

    return indices;
}

} // namespace op