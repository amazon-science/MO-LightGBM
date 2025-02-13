#ifndef LIGHTGBM_ECOS_OPTIMIZER_V2_HPP_
#define LIGHTGBM_ECOS_OPTIMIZER_V2_HPP_

#include <iostream>

#include <ecos.h>
#include <amatrix.h>
#include <Eigen/Sparse>

#include "base_optimizer.hpp"

namespace LightGBM {
/*!
* \brief The interface of Generic Cone Programming Solver
*/

class ECOSOptimizerV2 : public BaseOptimizer {
 public:
  ~ECOSOptimizerV2() {
  ECOS_cleanup(sol, 0);
  }


  bool solveProblem(Eigen::MatrixXd &A, Eigen::VectorXd &h, Eigen::VectorXd &c) {
    Eigen::SparseMatrix<double> Asp = A.sparseView();
    Asp.makeCompressed();

    Gpr = (pfloat *) Asp.valuePtr() ;
    Gjc = (idxint *) Asp.outerIndexPtr() ;
    Gir = (idxint *) Asp.innerIndexPtr() ;

    // dual mywork->z ;
    // cidx (sum q[i])+l+3*nexc(e) should be m
    // std::cout<< "q[0] = " << q[0] << "; l = " << l << "; e = " << e << "; m =" << m << "\n" ;
    sol = ECOS_setup(
              n_variables // num vars
              , n_constraint_rows // num constraints (row)
              , n_equalities// num equalities
              , n_positive_constraints// num positive constraints
              , n_cone_constraints // num cone constraints
              , cone_constraint_dimensions // cone constraint dim
              , n_exponential_cones // num exponential cones
              , Gpr // G CCS values
              , Gjc // G CCS columns
              , Gir // G CCS rows
              , A_data_CCS // A CCS values
              , A_columns_CCS // A CCS columns
              , A_rows_CCS // A CCS rows
              , c.data() // c values
              , h.data()// h values
              , b // b values
    ) ;

/*
    if (ecos_work == nullptr)
    {
        throw std::runtime_error("Could not set up problem.");
    }
*/
    bool solved = false;
    for (int i=0; i < max_trials; i++){
      idxint exitflag ;
      exitflag = ECOS_solve( sol ) ;
      std::cout << "[ECOS] exitcode = " << exitflag << "\n";
      if (exitflag < 0) {
        std::cout << "[ECOS] exitcode = " << exitflag << "\n";
      }
      if (exitflag >= 0) {
        solved = true;
        break;
      }
    }
    if (solved == false) {
      std::cout<< "***: ecos couldn't find solution of the following problem in " << max_trials << " trials" << std::endl;
      std::cout<< "A=\n" << A << "\n h = "<< h.transpose() <<"\n c = " << c.transpose() << std::endl;
    }
    return solved;
  }

  /*
  pwork* sol ;
  int max_trials;

  idxint n, m, p, l ;
  idxint ncones ;
  idxint* q ;
  idxint e ;
  pfloat* Gpr ;
  idxint* Gjc ;
  idxint* Gir ;
  pfloat* Apr ;
  idxint* Ajc ;
  idxint* Air ;
  pfloat* b ;
*/

};


}
#endif   // LIGHTGBM_ECOS_OPTIMIZER_V2_HPP_
