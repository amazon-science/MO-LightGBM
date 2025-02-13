#ifndef LIGHTGBM_BASE_OPTIMIZER_HPP_
#define LIGHTGBM_BASE_OPTIMIZER_HPP_

#include <iostream>

// #include <ecos.h>
#include <amatrix.h>
#include <Eigen/Sparse>

namespace LightGBM {
/*!
* \brief The interface of Generic Cone Programming Solver
*/

class BaseOptimizer {
 public:
  ~BaseOptimizer() {
//   ECOS_cleanup(sol, 0);
  }

  int max_trials;

  int n_variables;
  int n_constraint_rows;
  int n_equalities;
  int n_positive_constraints;
  int n_cone_constraints;
  std::vector<int> cone_constraint_dimensions;
  int n_exponential_cones;
  std::vector<double> G_data_CCS;
  std::vector<int> G_columns_CCS;
  std::vector<int> G_rows_CCS;
  std::vector<double> A_data_CCS;
  std::vector<int> A_columns_CCS;
  std::vector<int> A_rows_CCS;
  std::vector<double> c;
  std::vector<double> h;
  std::vector<double> b;

  virtual bool solveProblem(bool verbose = false) = 0;
  virtual std::string getResultString() const = 0;

  void initialize(int n, int m, int f, int l, int qsize, int* q) {
    this->n_variables = m ;
    this->n_constraint_rows =  n ;
    this->n_equalities = f ;
    this->n_positive_constraints = l ;
    this->n_cone_constraints = qsize ;
    this->cone_constraint_dimensions = q ;
    this->n_exponential_cones = 0 ;
    this->A_data_CCS = NULL ;
    this->A_columns_CCS = NULL ;
    this->A_rows_CCS = NULL ;
    this->b = NULL ;

    max_trials = 5;

  }

};


}
#endif   // LIGHTGBM_BASE_OPTIMIZER_HPP_
