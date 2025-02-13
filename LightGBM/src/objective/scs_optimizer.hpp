#ifndef LIGHTGBM_SCS_OPTIMIZER_HPP_
#define LIGHTGBM_SCS_OPTIMIZER_HPP_

#include <iostream>

#include <scs.h>
#include <amatrix.h>
//#include <scs_matrix.h>
//#include <csparse.h>
#include <Eigen/Sparse>

namespace LightGBM {
/*!
* \brief The interface of Generic Cone Programming Solver
*/

class SCSOptimizer {
 public:
  ~SCSOptimizer() {
    SCS(free_sol)(sol);
//    scs_free(sol); // 3.0
    scs_free(d->A);
    scs_free(k);
    scs_free(d->stgs);
    scs_free(d);
  }

  /*
   * f: number of eq constraint
   * l: length of LP cone
   */
  void Init(int m, int n, int f, int l, int qsize, int* q) {
    d = (ScsData *)scs_calloc(1, sizeof(ScsData));
    init_m = m ;
    d->m = (scs_int) m;
    d->n = (scs_int) n;
    d->A = (ScsMatrix *)scs_calloc(1, sizeof(ScsMatrix));
    d->A->m = (scs_int) m;
    d->A->n = (scs_int) n;
    d->stgs = (ScsSettings *)scs_calloc(1, sizeof(ScsSettings));
    SCS(set_default_settings)(d);
    d->stgs->eps = 1e-4;
    d->stgs->verbose = 0;
//    d->stgs->eps = 1e-4;

    k = (ScsCone *)scs_calloc(1, sizeof(ScsCone));
    k->f = (scs_int) f;
    k->l = (scs_int) l;
    k->q = (scs_int *) q;
    //k->q = q;
    k->qsize = (scs_int) qsize;
    
    max_trials = 5;
    sol = (ScsSolution *)scs_calloc(1, sizeof(ScsSolution));

    //SCS(set_default_settings)(d);
  }

  void SetAb(Eigen::MatrixXd &AA, Eigen::VectorXd &bb){
//      Eigen::SparseMatrix<double> AAsp = AA.sparseView();
//    Asp.makeCompressed();//
//      std::cout << "size of A = " << AA.rows() << " x " << AA.cols() << std::endl ;
//      std::cout<< "size A=\n" << AA.size() <<  std::endl;
      AAmat = AA ;
      bbvec = bb ;

//      std::cout<< "A=\n" << AA << "\n b = "<< bb << std::endl;
//      std::cout<< "size A=\n" << Amat.size() <<  std::endl;

  }

  bool Solve(Eigen::MatrixXd &A, Eigen::VectorXd &b, Eigen::VectorXd &c) {

  Eigen::MatrixXd A_all ;
  Eigen::VectorXd b_all ;
  if (AAmat.size() > 0) {
      scs_int m2 = AAmat.rows();
      d->m = init_m + m2 ;
      d->A->m = init_m + m2 ;
      k->f = m2 ;

      A_all.setZero(d->A->m, d->A->n) ;
      A_all << AAmat, A ;

      b_all.setZero(d->A->m) ;
      b_all << bbvec, b ;

//      std::cout << "A_all = " << A_all << "; b_all = " << b_all << std::endl ;
  }
  else{
      A_all.setZero(d->A->m, d->A->n) ;
      A_all = A ;
      b_all.setZero(d->A->m) ;
      b_all = b ;
//      std::cout << "A_all = " << A_all << "; b_all = " << b_all << std::endl ;

  }

    Eigen::SparseMatrix<double> Asp = A_all.sparseView();
    Asp.makeCompressed();
    d->A->x = Asp.valuePtr();
    d->A->i = (scs_int *) Asp.innerIndexPtr();
    d->A->p = (scs_int *) Asp.outerIndexPtr();

    d->b = b_all.data() ;
    d->c = c.data();
    d->stgs->acceleration_lookback = 10;
    
    bool solved = false;
    for (int i=0; i < max_trials; i++){
      ScsInfo info;
      scs_int exitflag;
      exitflag = scs(d, k, sol, &info);
      if (exitflag != 1) {
        std::cout << "[SCS] status: " << info.status << "; exitcode = " << info.status_val << "; Solving again with acceleration_lookback = 0\n";
        d->stgs->acceleration_lookback = 0;
        exitflag = scs(d, k, sol, &info);
      }
      if (exitflag == 1) {
        solved = true;
        break;
      }
    }
    if (solved == true) {
        Eigen::Map<const Eigen::VectorXd> _primal_var(sol->x, d->n ) ;
        Eigen::Map<const Eigen::VectorXd> _dual_var(sol->y, d->m ) ;
        primal_var = _primal_var ;
        dual_var = _dual_var ;
    }
    if (solved == false) {
      std::cout<< "***: scs couldn't find solution of the following problem in " << max_trials << " trials" << std::endl;
      std::cout<< "A=\n" << A << "\n b = "<< b.transpose() <<"\n c = " << c.transpose() << std::endl;   
    }
    return solved;
  }

  ScsCone* k;
  ScsData* d;
  ScsSolution* sol;
  int max_trials;

  scs_int init_m ;

  Eigen::VectorXd primal_var ;
  Eigen::VectorXd dual_var ;

  Eigen::MatrixXd AAmat ;
  Eigen::VectorXd bbvec ;
    // 3.0
//  ScsSettings stgs ;
};


}
#endif   // LIGHTGBM_SCS_OPTIMIZER_HPP_
