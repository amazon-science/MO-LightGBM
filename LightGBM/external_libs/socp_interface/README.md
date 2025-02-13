# socp_interface

A C++ interface to formulate and solve Second Order Cone Problems.

## Dependencies

* C++17
* Eigen

## Supported Solvers
* [EiCOS](https://github.com/embersarc/eicos)
* [ECOS](https://github.com/embotech/ecos)

## Installation
``` 
git clone --recurse-submodules https://github.com/EmbersArc/socp_interface.git
cd socp_interface  
mkdir build  
cd build  
cmake ..  
make  
```

## Concepts

### Parameters
Parameters are symbolic representations of constants. Parameters can be scalars, vectors or matrices. They can be constant or point to a dynamic value. This makes it possible to change the value of a constant in a problem without reconstructing the entire problem. Parameters support basic arithmetic functions `+` `-` `*` `/`.
```cpp
// scalar parameter
const double scalar = 5.;
op::Parameter scalar_par(scalar);

// vector parameter
const auto vector = Eigen::Vector3d::Ones();
op::Parameter vector_par(vector);

// matrix parameter
const auto matrix = Eigen::Matrix3d::Identity();
op::Parameter matrix_par(matrix);

// all parameters may also be pointers to values so that their values can be changed dynamically
auto mutable_matrix = Eigen::Matrix3d::Identity();
op::Parameter matrix_ptr_par(&matrix);
```

### Variables
Problem variables can again be scalars, vectors or matrices and have to be added directly to the problem.
```cpp
op::SecondOrderConeProgram socp;

// scalar variable
op::Variable scalar_var = socp.createVariable("x_scalar");

// vector variable
op::Variable vector_var = socp.createVariable("x_vector", 3);

// matrix variable
op::Variable matrix_var = socp.createVariable("x_matrix", 3, 3);

// the problem can always return existing variables if you lose track of them
op::Variable matrix_var_copy = socp.getVariable("x_matrix");
```

### Expressions

#### Affine
Affine expressions include Parameters, Variables and sums or products of the two.

```cpp
op::Affine affine_vector = op::Parameter(Eigen::Matrix3d::Random()) * vector_var;
```

#### SOCLhs
This is the left hand side of a second order cone constraint and gets created by calling `op::norm(Affine affine, int axis)`. Apart from the 2-norm, it can also contain an `Affine` expression of the same dimension.

```cpp
op::SOCLhs soc_lhs = op::norm(vector_var);
soc_lhs += scalar_par;
soc_lhs += scalar_var;
```

### Constraints
A second order cone program generally supports three kinds of constraints. Those are equality, linear and second order cone constraints.
Constraints are added to the SOCP with the `addConstraint` method e.g. `socp.addConstraint(soc_lhs <= affine_vector)`.

#### Equality Constraints
Either side can be a scalar while the other is a matrix. If both have the same shape, the equality is applied coefficient-wise.
```
<Affine> == 0.
<Affine> == <Affine>
```
#### Linear Constraints
Either side can be a scalar while the other is a matrix. If both have the same shape, the inequality is applied coefficient-wise.
```
<Affine> >= 0.
0. <= <Affine>
<Affine> >= <Affine>
<Affine> <= <Affine>
```
#### Second Order Cone Constraints
As opposed to the other constraint types, the left hand side can not be a scalar when the right hand side has a higher dimension.
```
<SOCLhs> <= <Affine>
```

### Cost Function
A cost term can be added to the SOCP with the `addMinimizationTerm` method e.g. `socp.addMinimizationTerm(op::sum(affine_vector))`. The `Affine` term has to be a scalar.

### Solving the Problem
First, create a solver instance with `op::Solver solver(socp)` and call `solver.solveProblem()` to solve the problem. If `true` is passed to the function, the solver output will be shown. This method returns `true` if it was successful and a solution is available. The solution for a variable `x` can be retrieved by calling `socp.readSolution("x", x_sol)` where `x_sol` is the solution variable of type `double` for scalars and `Eigen::Matrix` for higher dimensional variables.

### Matrix Access
All matrix expressions can be accessed like Eigen matrices i.e. `operator()` for coefficient-wise access and [Eigen Block Operations](https://eigen.tuxfamily.org/dox/group__TutorialBlockOperations.html) that return matrices.

### Other Functions
Matrices can be stacked horizontally and vertically using the `op::hstack({...})` and `op::vstack({...})` functions. For convenience, `op::sum(Affine affine, int axis)` can sum up elements of `Affine` matrices.

## Example

```cpp
#include "socpInterface.hpp"

#include <array>
#include <iostream>
#include <chrono>

#include <Eigen/Dense>

// This example solves a simple random second order cone problem
// based on https://www.cvxpy.org/examples/basic/socp.html

int main()
{
    // Set up problem data.

    // number of second order cone constraints
    const size_t m = 3;
    // number of variables
    const size_t n = 10;
    // dimension of equality constraints
    const size_t p = 5;
    // dimension of second order cone constraints
    const size_t n_i = 5;

    std::array<Eigen::Matrix<double, n_i, n>, m> A;
    std::array<Eigen::Matrix<double, n_i, 1>, m> b;
    std::array<Eigen::Matrix<double, n, 1>, m> c;
    std::array<double, m> d;

    Eigen::Matrix<double, n, 1> x0;
    x0.setRandom();
    Eigen::Matrix<double, n, 1> f;
    f.setRandom();

    for (size_t i = 0; i < m; i++)
    {
        A[i].setRandom();
        b[i].setRandom();
        c[i].setRandom();
        d[i] = (A[i] * x0).norm() - c[i].dot(x0);
    }

    Eigen::Matrix<double, p, n> F;
    F.setRandom();
    Eigen::Matrix<double, p, 1> g = F * x0;

    // Formulate SOCP.
    auto t0 = std::chrono::high_resolution_clock::now();

    // Create the SOCP instance.
    op::SecondOrderConeProgram socp;

    // Add variables. Those can be scalars, vectors or matrices.
    op::Variable x = socp.createVariable("x", n);

    // Add constraints.
    for (size_t i = 0; i < m; i++)
    {
        socp.addConstraint(op::norm2(op::Parameter(A[i]) * x + op::Parameter(b[i])) <=
                           op::Parameter(c[i]).transpose() * x + op::Parameter(d[i]));
    }
    socp.addConstraint(op::Parameter(F) * x == op::Parameter(g));

    // Here we use a pointer to a parameter. This allows changing it dynamically.
    socp.addMinimizationTerm(op::Parameter(&f).transpose() * x);

    // Print the problem for inspection.
    std::cout << socp << "\n\n";

    // Create and initialize the solver instance.
    op::Solver solver(socp);
    solver.initialize();

    auto t = std::chrono::high_resolution_clock::now();
    auto t_setup = std::chrono::duration_cast<std::chrono::microseconds>(t - t0).count();
    std::cout << "\nSetup duration: " << t_setup << "μs.\n\n";

    // Solve the problem and show solver output.
    t0 = std::chrono::high_resolution_clock::now();
    solver.solveProblem(true);

    // Check if the solver has produced a valid solution.
    assert(socp.isFeasible());

    t = std::chrono::high_resolution_clock::now();
    auto t_solve = std::chrono::duration_cast<std::chrono::microseconds>(t - t0).count();
    std::cout << "\nSolver duration: " << t_solve << "μs.\n\n";

    // Get Solution.
    Eigen::Matrix<double, n, 1> x_sol;
    socp.readSolution("x", x_sol);
    std::cout << "Solver message: " << solver.getResultString() << "\n";

    // Print the first solution.
    std::cout << "First solution:\n"
              << x_sol << "\n\n";

    // Change the problem parameters and solve again.
    f.setRandom();
    solver.solveProblem(false);
    socp.readSolution("x", x_sol);

    // Print the new solution.
    std::cout << "Solution after changing the cost function:\n"
              << x_sol << "\n\n";
}
```

For more examples, see the [SCpp](https://github.com/EmbersArc/SCpp) optimal guidance library.

