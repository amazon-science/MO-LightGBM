#ifndef LIGHTGBM_ECOS_OPTIMIZER_HPP_
#define LIGHTGBM_ECOS_OPTIMIZER_HPP_

#include <iostream>

#include <ecos.h>
#include <amatrix.h>
#include <Eigen/Sparse>

namespace LightGBM {
/*!
* \brief The interface of Generic Cone Programming Solver
*/



class ECOSOptimizer {
 public:
  ~ECOSOptimizer() {
  ECOS_cleanup(sol, 0);
  }

  void Init(int m, int n, int f, int l, int qsize, int* q) {
    // assming no equality constraint and no exponential cones

    this->m = (idxint) m ;
    this->n = (idxint) n ;
    this->p = (idxint) f ;
    this->l = (idxint) l ;
    this->ncones = qsize ;
    this->q = (idxint *) q ;
    this->e = 0 ;
    this->Apr = NULL ;
    this->Ajc = NULL ;
    this->Air = NULL ;
    this->b = NULL ;
    max_trials = 5;
  }


  void SetAb(Eigen::MatrixXd &A, Eigen::VectorXd &bb){
      Eigen::SparseMatrix<double> Asp = A.sparseView();
//      Asp.makeCompressed();
//      std::cout << "size of A = " << Asp.rows() << " x " << Asp.cols() << std::endl ;
//      std::cout<< "size A=\n" << Amat.size() <<  std::endl;
      Amat.setZero(1, n) ;
      bvec.setZero(1) ;
      Amat = A ;
      bvec = bb ;

//      std::cout<< "A=\n" << A << "\n b = "<< bb << std::endl;
//      std::cout<< "size A=\n" << Amat.size() <<  std::endl;


  }

  bool Solve(Eigen::MatrixXd &G, Eigen::VectorXd &h, Eigen::VectorXd &c) {
    Eigen::SparseMatrix<double> Gsp = G.sparseView();
//    Gsp.makeCompressed();

//    std::cout<< "G=\n" << G<< "\n h = "<< h.transpose() << "\n c = " << c << std::endl;
//    std::cout<< "A=\n" << Amat<< "\n b = "<< bvec.transpose() << std::endl;

    Gpr = (pfloat *) Gsp.valuePtr() ;
    Gjc = (idxint *) Gsp.outerIndexPtr() ;
    Gir = (idxint *) Gsp.innerIndexPtr() ;

    Eigen::SparseMatrix<double> Asp ;
    if (Amat.size() > 0) {
        Asp = Amat.sparseView();
        p = Asp.rows();
        Apr = (pfloat *) Asp.valuePtr();
        Ajc = (idxint *) Asp.outerIndexPtr();
        Air = (idxint *) Asp.innerIndexPtr();
        b = (pfloat *) bvec.data();
    }
      // dual mywork->z ;
    // cidx (sum q[i])+l+3*nexc(e) should be n
    // std::cout<< "q[0] = " << q[0] << "; l = " << l << "; e = " << e << "; m =" << m << "\n" ;
    sol = ECOS_setup(
              n // num vars
              , m // num constraints (row of matrix G)
              , p // num equalities (row of matrix A)
              , l // num positive constraints (sub matrix of G -- linear)
              , ncones // num cone constraints
              , q // cone constraint dim
              , e // num exponential cones (zero)
              , Gpr // G CCS values
              , Gjc // G CCS columns
              , Gir // G CCS rows
              , Apr // A CCS values
              , Ajc // A CCS columns
              , Air // A CCS rows
              , (pfloat *) c.data() // c values (objective)
              , (pfloat *) h.data()// h values (upper bound Gx <= h)
              , b // b values (Ax = b)
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
      //if (exitflag < 0) {
      //  std::cout << "[ECOS] exitcode = " << exitflag << "\n";
      //}
      if (exitflag >= 0) {
        solved = true;
        break;
      }
    }

    if (solved == true) {
        Eigen::Map<const Eigen::VectorXd> _primal_var(sol->x, n ) ;
        Eigen::Map<const Eigen::VectorXd> _dual_var(sol->z, m ) ;
        primal_var = _primal_var ;
        dual_var = _dual_var ;
    }

    if (solved == false) {
      std::cout<< "***: ecos couldn't find solution of the following problem in " << max_trials << " trials" << std::endl;
      std::cout<< "G=\n" << G << "\n h = "<< h.transpose() <<"\n c = " << c.transpose() << std::endl;
    }
    return solved;
  }

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

  Eigen::MatrixXd Amat ;
  Eigen::VectorXd bvec ;
  Eigen::VectorXd primal_var ;
  Eigen::VectorXd dual_var ;


};


}
#endif   // LIGHTGBM_ECOS_OPTIMIZER_HPP_
