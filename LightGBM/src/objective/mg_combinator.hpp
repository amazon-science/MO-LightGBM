#ifndef LIGHTGBM_MG_COMBINATOR_HPP_
#define LIGHTGBM_MG_COMBINATOR_HPP_

#include <iostream>

#include <cstdio>
#include <cstring>
#include <cmath>

#include <vector>
#include <algorithm>
#include <limits>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "scs_optimizer.hpp"
#include "ecos_optimizer.hpp"

#include <fstream>

namespace LightGBM {

/* for opeining a file containing reference vectors for PMTL
    template<typename M>
    M load_csv (const std::string & path) {
        std::ifstream indata;
        indata.open(path);
        std::string line;
        std::vector<double> values;
        uint rows = 0;
        while (std::getline(indata, line)) {
            std::stringstream lineStream(line);
            std::string cell;
            while (std::getline(lineStream, cell, '\t')) {
                values.push_back(std::stod(cell));
            }
            ++rows;
        }
        return Map<const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, Eigen::RowMajor>>(values.data(), rows, values.size()/rows);
    }
*/
    /*!
* \brief The interface of Multi-Gradient Combination
*/
class MGCombinator {
  /*
  Any derived object of MGcombinator finds combination coefficients for num_mo_ gradients.
  */
 public:
  virtual ~MGCombinator() {}

  static MGCombinator* CreateMGCombinator(const std::string& type, const Config& config);

  virtual void Init(const Metadata& metadata, data_size_t num_data) {
    num_mo_ = metadata.num_mo();
    num_data_ = num_data;
    // todo: check for negative preferences
    preferences_ = Eigen::Map<const Eigen::VectorXf>(metadata.mo_preferences(), num_mo_).cast <double> ();
    // todo: need to define custom type for VectorX(f|d)
//    std::cout << "preferences = " << preferences_ << std::endl;
    std::stringstream tmp_buf;
    tmp_buf <<"Preferences: [" << preferences_.transpose() << "]";
    Log::Info(tmp_buf.str().c_str());
  }

  virtual bool GetCoefficients(const std::vector<score_t>*,
                               const double*, score_t*) {return false;}

  void GetCombination(const std::vector<score_t>* multi_vecs,
          const score_t* coefficients, score_t* out_vec) {
    for (auto i=0; i < num_data_; i++) {
      out_vec[i] = 0;
      for (auto mo_idx=0; mo_idx < num_mo_; mo_idx++) {
        out_vec[i] += multi_vecs[mo_idx][i] * coefficients[mo_idx];
      }
    }
  }

  void RefreshGG(const std::vector<score_t>* multi_gradients, Eigen::MatrixXd& GG_) {
    GG_.array() = 0.0;
    std::cout << "start refresh GG" << std::endl ;
    for (auto i = 0; i < num_mo_; i++) {
      for (auto j = i; j < num_mo_; j++) {
        for (auto k = 0; k < num_data_; k++) {
          GG_(i,j) += static_cast<double>(multi_gradients[i][k]) * static_cast<double>(multi_gradients[j][k]);
        }
        if (j > i) {
          GG_(j,i) = GG_(i,j); }
      }
    }
      std::cout << "finish refresh GG" << std::endl ;
  }

 protected:
  data_size_t num_mo_;
  data_size_t num_data_;
  Eigen::VectorXd preferences_;
};

class LinearScalarization : public MGCombinator {
  /*
  The combination coefficients of gradients are the preference values given as user input
  */
 public:
  explicit LinearScalarization(const Config&) {}

  ~LinearScalarization() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    MGCombinator::Init(metadata, num_data);
  }

  bool GetCoefficients(const std::vector<score_t>*, const double*,
                       score_t* coefficients) override {
    Eigen::Map<Eigen::VectorX<score_t>>(coefficients, num_mo_) = preferences_.cast <score_t> ();
    return true;
  }
};

class ChebychevScalarization : public MGCombinator {
 /*
 The combination coefficients are zero for all gradients except for the one that has maximum relative cost.
 The num_mo_ relative costs are preference[i] * cost[i] for i=0 to i < num_mo_
 */
 public:
  explicit ChebychevScalarization(const Config& config) {
    if(config.mg_combination == "chebychev_scalarization_decay"){
        cs_type = "alpha_decay" ;
    }
    else{
        cs_type = "no_alpha_dacay" ;
    }
    mo_ema_multiplier = config.mo_ema_multiplier ;

  }

  ~ChebychevScalarization() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    MGCombinator::Init(metadata, num_data);
    num_queries = metadata.num_queries();
    alpha.setZero(num_mo_);
    prev_alpha.setZero(num_mo_);
    ref_point = Eigen::Map<const Eigen::VectorXf>(metadata.mo_wc_mgda_refpt(), num_mo_ ).cast <double> ();
    std::stringstream tmp_buf;
    tmp_buf << "ref point: [" << ref_point.transpose() << "]" << std::endl ;
    Log::Info(tmp_buf.str().c_str());
  }

  bool GetCoefficients(const std::vector<score_t>*, const double* mo_costs,
                       score_t* coefficients) override {
    Eigen::Map<const Eigen::VectorXd> costs(mo_costs, num_mo_);
    Eigen::VectorXd::Index k_star;
    Eigen::VectorXd losses = costs / num_queries ;
    Eigen::VectorXd relative_losses = losses - ref_point ;
    double max_relcost = ( relative_losses.array() * preferences_.array()).maxCoeff(&k_star);
    log_details.str("\n");
    log_details << "losses = "<< losses.transpose() << std::endl;
    log_details << "losses - ref_point = "<< relative_losses.transpose() << std::endl;
    log_details << "max relative cost = " << max_relcost << " at k = " << k_star;
    Log::Debug(log_details.str().c_str());
    alpha.array() = 0;
    alpha(k_star) = 1.0;
    if (n_iter > 0 && cs_type == "alpha_decay") {
//        alpha = (alpha * 0.1 + prev_alpha * 0.9)  ;
        alpha = (alpha * mo_ema_multiplier + prev_alpha * (1 - mo_ema_multiplier) )  ;
        prev_alpha = alpha ;
    }
    Eigen::Map<Eigen::VectorX<score_t>>(coefficients, num_mo_) = alpha.cast <score_t> ();
    n_iter += 1;
    return true;
  }

 private:
  Eigen::VectorXd alpha, ref_point;
  Eigen::VectorXd prev_alpha;
  std::stringstream log_details;
  int num_queries ;
  int n_iter = 0;
  std::string cs_type ;
    double mo_ema_multiplier;
};

class EPOSearch : public MGCombinator {
  /*
  The algorithm implemented here is a combination of methods from these papers:
  ICML: http://proceedings.mlr.press/v119/mahapatra20a.html
  ArXiv: https://arxiv.org/abs/2108.00597

  The combination coefficients are decided by solving a quadratic programming problem.
  */
 public:
  explicit EPOSearch(const Config& config) {
      if(config.mg_combination == "epo_search_decay"){
          epo_type = "alpha_decay" ;
      }
      else{
          epo_type = "no_alpha_dacay" ;
      }
      mo_ema_multiplier = config.mo_ema_multiplier ;

  }

  ~EPOSearch() {}



  void Init(const Metadata& metadata, data_size_t num_data) override {
    MGCombinator::Init(metadata, num_data);
    preferences_ = (preferences_.array() <= 0).select(1e-6, preferences_);
    pref_inv_ = 1.0 / preferences_.array();
    pref_inv_ /= pref_inv_.norm();
    GG.setZero(num_mo_, num_mo_);
    prev_alpha.setZero(num_mo_);

    restricted_des = Eigen::MatrixXd::Identity(num_mo_,num_mo_) - pref_inv_ * pref_inv_.transpose();

    setSOCPData_opt_bal();
    setSOCPData_opt_des();

    /*
    int m, n, f;
    // Set common parameters for both balance and descent mode
    n = 1 + num_mo_;
    q.setConstant(1, n);
    c.setZero(n); c(n-1) = 1.0;

    // Set optimizer for balance mode
    //    m = 1 + num_mo_ + (1 + num_mo_);   // = 2*num_mo_+2 : eq constraint(1) + ineq constraint + cone constraint
    m = num_mo_ + (1 + num_mo_);   // = 2*num_mo_+1 : ineq constraint + cone constraint
    f = 1 ; // one ineq constraint
    opt_bal.Init(m, n, f, num_mo_, 1, q.data());
    A_bal.setZero(m, n);
//      A_bal.topLeftCorner(1, num_mo_).array() = 1.0; // old
    A_bal.block(0,0,num_mo_,num_mo_) = -Eigen::MatrixXd::Identity(num_mo_,num_mo_); // new
    //A_bal.block(1,0,num_mo_,num_mo_) = -Eigen::MatrixXd::Identity(num_mo_,num_mo_); //old
    A_bal(num_mo_,num_mo_) = -1.0;
    b_bal.setZero(m);

    A_eq_bal.setZero(f, n) ;
    A_eq_bal.topLeftCorner(f, num_mo_).array() = 1.0 ;
    b_eq_bal.setZero(f) ;
    b_eq_bal.head(f).array() = 1;

    opt_bal.SetAb(A_eq_bal, b_eq_bal) ;

    // b_bal(0) = 1.0;
    // Set optimizer for descent mode
    f = num_mo_ ;
    m = num_mo_ + (1 + num_mo_); // = 3*num_mo_ + 1
    opt_des.Init(m, n, f, num_mo_, 1, q.data());
    A_des.setZero(m, n);
    //A_des.topLeftCorner(1, num_mo_).array() = 1.0;  // Next num_mo_ - 1 rows are for constraints of restricted descent
    A_des.block(0,0,num_mo_,num_mo_) = -Eigen::MatrixXd::Identity(num_mo_,num_mo_);
    A_des(num_mo_, num_mo_) = -1.0;
    b_des.setZero(m);
    //b_des(0) = 2.0;

    A_eq_des.setZero(f, n) ;
    A_eq_des.topLeftCorner(1, num_mo_).array() = 1.0 ;
    b_eq_des.setZero(f) ;
    b_eq_des(0) = 2.0 ;

    opt_des.SetAb(A_eq_des, b_eq_des) ;
*/
  }

  void setSOCPData_opt_bal(){
      int m, n, f;
      // Set common parameters for both balance and descent mode
      n = 1 + num_mo_;
      q.setConstant(1, n);
      c.setZero(n); c(n-1) = 1.0;

      // Set optimizer for balance mode
      //    m = 1 + num_mo_ + (1 + num_mo_);   // = 2*num_mo_+2 : eq constraint(1) + ineq constraint + cone constraint
      m = num_mo_ + (1 + num_mo_);   // = 2*num_mo_+1 : ineq constraint + cone constraint
      f = 1 ; // one ineq constraint
      opt_bal.Init(m, n, f, num_mo_, 1, q.data());
      A_bal.setZero(m, n);
//      A_bal.topLeftCorner(1, num_mo_).array() = 1.0; // old
      A_bal.block(0,0,num_mo_,num_mo_) = -Eigen::MatrixXd::Identity(num_mo_,num_mo_); // new
      //A_bal.block(1,0,num_mo_,num_mo_) = -Eigen::MatrixXd::Identity(num_mo_,num_mo_); //old
      A_bal(num_mo_,num_mo_) = -1.0;
      b_bal.setZero(m);
      A_eq_bal.setZero(f, n) ;
      A_eq_bal.topLeftCorner(f, num_mo_).array() = 1.0 ;
      b_eq_bal.setZero(f) ;
      b_eq_bal.head(f).array() = 1;
      opt_bal.SetAb(A_eq_bal, b_eq_bal) ;
  }

  void setSOCPData_opt_des(){
      int m, n, f;
      // Set common parameters for both balance and descent mode
      n = 1 + num_mo_;
      q.setConstant(1, n);
      c.setZero(n); c(n-1) = 1.0;

      f = num_mo_ ;
      m = num_mo_ + (1 + num_mo_); // = 3*num_mo_ + 1
      opt_des.Init(m, n, f, num_mo_, 1, q.data());
      A_des.setZero(m, n);
      //A_des.topLeftCorner(1, num_mo_).array() = 1.0;  // Next num_mo_ - 1 rows are for constraints of restricted descent
      A_des.block(0,0,num_mo_,num_mo_) = -Eigen::MatrixXd::Identity(num_mo_,num_mo_);
      A_des(num_mo_, num_mo_) = -1.0;
      b_des.setZero(m);
      //b_des(0) = 2.0;
      A_eq_des.setZero(f, n) ;
      A_eq_des.topLeftCorner(1, num_mo_).array() = 1.0 ;
      b_eq_des.setZero(f) ;
//      b_eq_des(0) = 2.0 ;
      b_eq_des(0) = 1.0 ;
      opt_des.SetAb(A_eq_des, b_eq_des) ;
  }

  bool GetCoefficients(const std::vector<score_t>* multi_gradients,
                       const double* mo_costs, score_t* coefficients) override {
    RefreshGG(multi_gradients, GG);
    Eigen::VectorXd alpha, anchor(num_mo_);
    Eigen::Map<const Eigen::VectorXd> costs(mo_costs, num_mo_);
    log_details.str("\n");
    double mu = CauchySchwarzMu(costs);   // LagrangeMu(costs)
    bool solved;
    if (mu >= mu_thresh_) {
      log_details<<"[EPO] balancing... mu="<<mu<<std::endl;
      setSOCPData_opt_bal() ;
      GetLagrangeAnchor(costs, anchor);   // GetCauchySchwarzAnchor(costs, anchor);
      b_bal.tail(num_mo_) = -anchor;
      A_bal.bottomLeftCorner(num_mo_, num_mo_) = -GG;
//      std::cout<<"A_bal\n" << A_bal << "\nb_bal=" << b_bal.transpose() << "\nc_bal="<<c.transpose()<<"\n";
//      std::cout << "A_eq_bal\n" << A_eq_bal << "\nbb_bal=" << b_eq_bal.transpose() << "\nc_bal=" << c.transpose() << "\n";
      opt_bal.SetAb(A_eq_bal, b_eq_bal) ;
      solved = opt_bal.Solve(A_bal, b_bal, c);
      if (solved){
        //alpha = Eigen::Map<Eigen::VectorXd>(opt_bal.sol->x, num_mo_);
        //std::cout<<"original alpha="<<alpha.transpose()<<"\n";
        alpha = opt_bal.primal_var.head(num_mo_) ;
//        std::cout<<"new alpha="<<alpha.transpose()<<"\n";
      } else {
        Eigen::VectorXd::Index k_star;
        (costs.array() * preferences_.array()).maxCoeff(&k_star);
        log_details << "max relative cost at k = " << k_star << std::endl;
        alpha.setZero(num_mo_);
        alpha(k_star) = 1.0;
      }
      // std::cout<<"alpha="<<alpha.transpose()<<"\n";
    } else {
      log_details<<"[EPO] descending... mu="<<mu<<std::endl;
      setSOCPData_opt_des();
      anchor = pref_inv_ * costs.norm();
      b_des.tail(num_mo_) = -anchor;
      //A_des.block(1,0,num_mo_-1,num_mo_) = (restricted_des * GG).topLeftCorner(num_mo_-1,num_mo_); // cause rank(restricted_des*GG) < num_mo_
      A_eq_des.block(1, 0, num_mo_ - 1, num_mo_) = (restricted_des * GG).topLeftCorner(num_mo_ - 1, num_mo_); // cause rank(restricted_des*GG) < num_mo_

//      std::cout<<"A_des\n" << A_des << "\nb_des=" << b_des.transpose() << "\nc_des="<<c.transpose()<<"\n";
//      std::cout << "A_eq_des\n" << A_eq_des << "\nb_eq_des=" << b_des.transpose() << "\nc_des=" << c.transpose() << "\n";
      A_des.bottomLeftCorner(num_mo_, num_mo_) = -GG;
      opt_des.SetAb(A_eq_des, b_eq_des) ;
      solved = opt_des.Solve(A_des, b_des, c);
      if (solved) {
        //alpha = Eigen::Map<Eigen::VectorXd>(opt_des.sol->x, num_mo_);
        alpha = opt_des.primal_var.head(num_mo_) ;
      } else {
        alpha = preferences_;
      }
      //std::cout<<"alpha="<<alpha.transpose()<<"\n";
    }

    alpha = (alpha.array() < 0).select(0, alpha);

    if (n_iter > 0 && epo_type == "alpha_decay") {
//        alpha = (alpha * 0.1 + prev_alpha * 0.9)  ;
        alpha = (alpha * mo_ema_multiplier + prev_alpha * (1 - mo_ema_multiplier) )  ;
        prev_alpha = alpha ;
    }

    //std::cout<< "GG:\n" << GG <<std::endl;
    //std::cout<< "cost: " << costs.transpose() <<"; anchor: " << anchor.transpose() << std::endl;
    log_details<< "obj: ||GG*alpha - anchor|| = " << (GG*alpha - anchor).norm() << "; sum : " << alpha.sum();
    Log::Debug(log_details.str().c_str());
    Eigen::Map<Eigen::VectorX<score_t>>(coefficients, num_mo_) = alpha.cast <score_t> ();
    n_iter += 1;
    return solved;
  }


  inline void GetLagrangeAnchor(const Eigen::VectorXd& costs, Eigen::VectorXd& anchor) {
    anchor = costs - pref_inv_ * costs.dot(pref_inv_);
    if (pref_inv_.dot(anchor) > 0.0001) {
      Log::Fatal("[EPO] anchor direction is not orthogonal to preference inverse, dot_prod= %f", pref_inv_.dot(anchor));
    }
  }

  inline void GetCauchySchwarzAnchor(const Eigen::VectorXd& costs, Eigen::VectorXd& anchor) {
    double cost_norm = costs.norm();
    anchor = costs * costs.dot(pref_inv_) / cost_norm - pref_inv_ * cost_norm;
    if (costs.dot(anchor) > 0.0001) {
      Log::Fatal("[EPO] anchor direction is not orthogonal to cost, dot_prod= %f", costs.dot(anchor)); }
  }

  inline double LagrangeMu(const Eigen::VectorXd& costs) {
    Eigen::VectorXd n_costs = costs / costs.norm();
    return (n_costs * pref_inv_.transpose() - pref_inv_ * n_costs.transpose()).squaredNorm() / 2.0;
    // return std::pow(mu, 0.5);
  }

  inline double CauchySchwarzMu(const Eigen::VectorXd& costs) {
    double cos_angle = costs.dot(pref_inv_) / costs.norm();
    return 1.0 - std::pow(cos_angle, 2);
    // return std::pow(1.0 - std::pow(cos_angle, 2), 0.5);
  }

 private:

  std::string epo_type ;
  Eigen::VectorXd prev_alpha;
  int n_iter = 0;
  // For EPO Search algorithm
  Eigen::VectorXd pref_inv_;
  Eigen::MatrixXd GG, restricted_des;
  double mu_thresh_ = 0.001;

  // For second order cone programming
  ECOSOptimizer opt_bal, opt_des;
  Eigen::MatrixXd A_bal, A_des;
  Eigen::VectorXd b_bal, b_des, c;
  Eigen::VectorXi q;

  std::stringstream log_details;

  Eigen::MatrixXd A_eq_bal, A_eq_des ;
  Eigen::VectorXd b_eq_bal, b_eq_des ;

    double mo_ema_multiplier;
};

class WCMGDA: public MGCombinator {
/*
  The combination coefficients are decided by solving a second order cone programming problem.
*/
 public:
  explicit WCMGDA(const Config& config) {
      min_u = config.mo_mgda_min_u;
      max_u = fmax(min_u, max_u);
      mo_ema_multiplier = config.mo_ema_multiplier ;
      use_fix_u = config.mo_mgda_use_fix_u ;

      if(config.mg_combination == "wc_mgda_decay"){
          mo_wc_mgda_type = "alpha_decay" ;
      }
      else {
          mo_wc_mgda_type = "no_alpha_decay" ;
      }

      log_details << "mg_combination = " << config.mg_combination << "\n" ;
      log_details << "mo_mgda_min_u = " << config.mo_mgda_min_u << "\n" ;
      log_details << "use_fix_u = " << use_fix_u << "\n" ;

      Log::Info(log_details.str().c_str());
  }

  ~WCMGDA() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    MGCombinator::Init(metadata, num_data);
    num_queries = metadata.num_queries();
    //std::cout << "metadata.mo_wc_mgda_lb() = " << metadata.mo_wc_mgda_lb() << std::endl ;
    lb = Eigen::Map<const Eigen::VectorXf>(metadata.mo_wc_mgda_lb(), num_mo_ ).cast <double> ();
    GG.setZero(num_mo_, num_mo_);
    K.setZero(num_mo_, num_mo_);
    n_iter = 0;
    //u_value = min_u ;
    //u_val = u_value ;
    //g_value = max_g ;

    ref_point = Eigen::Map<const Eigen::VectorXf>(metadata.mo_wc_mgda_refpt(), num_mo_ ).cast <double> ();
    std::stringstream tmp_buf;
    tmp_buf << "ref point: [" << ref_point.transpose() << "]" << ", lb: [" << lb.transpose() << "]" ;
    Log::Info(tmp_buf.str().c_str());
    alpha.setZero(num_mo_);
    setSOCPData_opt(min_u);
    setSCOPData_opt_u();
    //setSCOPData_opt_alt();
    //setSCOPData_opt_alt_g();
    //use_alpha_decay = true ;
  }

  void setSOCPData_opt(double u_value){
      // Set common parameters for both balance and descent mode
      q.setConstant(1, 1 + num_mo_);
      // Set optimizer for un-tuned u
      // m: number of rows in G, n: number of variables, f: number of rows in A, l: number of linear constraints in G
      int m = num_mo_ + (1 + num_mo_) ;
      int n = 1 + num_mo_ ;
      int f = 1;
      int l = num_mo_ ;
      opt.Init(m, n, f, l , q.size(), q.data());
      G.setZero(m, n);
      G.topLeftCorner(num_mo_,num_mo_) = -Eigen::MatrixXd::Identity(num_mo_,num_mo_);
      G(num_mo_,num_mo_) = -1.0;
      h.setZero(m);
      h.head(num_mo_) = -lb ;
      c.setZero(n); c(n-1) = u_value * (init_K_norm / K_norm);
      A.setZero(1, n) ;
      A.topLeftCorner(1, num_mo_).array() = 1.0 ;
      b.setZero(1) ;
      b(0) = 1.0 ;
      opt.SetAb(A, b) ;
  }

  void setSCOPData_opt_u(){
      q.setConstant(1, 1 + num_mo_);
      int m = num_mo_ + 1 + num_mo_ + (1 + num_mo_);  // = 3*num_mo + 2
      int n = 1 + 1 + num_mo_ + num_mo_;  // = 2*num_mo_+2  /* (u, rho, z, d) */
      int f = 0;
      int l = 2 * num_mo_ + 1;
      opt_u.Init(m, n, f, l, 1, q.data());
      G_u.setZero(m, n);
      G_u.block(0, 1, num_mo_, 1) = -Eigen::VectorXd::Ones(num_mo_);
      G_u.block(0, 2, num_mo_, num_mo_) = Eigen::MatrixXd::Identity(num_mo_,num_mo_);
      G_u(num_mo_, 1) = 1.0;
      G_u.block(num_mo_, 2, 1, num_mo_) = -lb.transpose();
      G_u.block(num_mo_+1, 2, num_mo_, num_mo_) = -Eigen::MatrixXd::Identity(num_mo_,num_mo_);
      G_u(2*num_mo_+1, 0) = -1.0;
      G_u.bottomRightCorner(num_mo_, num_mo_) = -Eigen::MatrixXd::Identity(num_mo_,num_mo_);
      h_u.setZero(m);
      c_u.setZero(n);
      c_u(0) = 1.0;
  }

  bool GetCoefficients(const std::vector<score_t>* multi_gradients,
                       const double* mo_costs, score_t* coefficients) override {
    RefreshGG(multi_gradients, GG);
    Eigen::Map<const Eigen::VectorXd> costs(mo_costs, num_mo_) ;
    log_details.str("\n");
    log_details << "sum loss = " << costs.transpose() << "\n" ;
    Eigen::VectorXd losses = costs / num_queries ;
//    if (n_iter > 0 && mo_wc_mgda_type == "alpha_decay"){
//        losses = mo_ema_multiplier * losses + (1-mo_ema_multiplier) * prev_losses ;
//        GG = mo_ema_multiplier * GG + (1-mo_ema_multiplier) * prev_GG ;
//    }
//    else{
//        prev_losses = losses ;
//        prev_GG = GG ;
//    }
    bool scale_each_iteration = false ;
    if (n_iter == 0) {
      init_rc_norm = preferences_.cwiseProduct((losses)).norm();
//      init_rc_norm = preferences_.cwiseProduct((losses - ref_point)).norm();
      r_pref = preferences_ / (init_rc_norm) ;
//      inv_r_pref = 1 / r_pref.array() ;
      std::cout << "r_pref = " << r_pref << "; inv_r_pref = " << 1 / r_pref.array() << std::endl;
      //u_value = 1 + u_value ;
    }
    else if (scale_each_iteration == true){
//        init_rc_norm = preferences_.cwiseProduct((losses - ref_point)).norm();
        init_rc_norm = preferences_.cwiseProduct((losses)).norm();
        r_pref = preferences_ / (init_rc_norm) ;
    }
    // (element)coefficient-wise product
    wc = r_pref.array() * losses.array();
    log_details << "r_pref = " << r_pref.transpose() << "\n" ;
    log_details << "loss = " << losses.transpose() << "\n" ;
    log_details << "ref = " << ref_point.transpose() << "\n" ;

    wb = r_pref.array() * ref_point.array();
//    loss_div = wc-wb;
    log_details << "wc-wb = " << (wc - wb).transpose() << "; ||wc-wb|| = " << (wc-wb).norm() << "\n" ;

//    if (n_iter == 0){
//    double max_div = (loss_div.array() - loss_div.minCoeff()).maxCoeff();
//    wc = wc.array() / max_div ;
//    wb = wb.array() / max_div ;
//    log_details << "wc-wb (after rescaling) = " << (wc - wb).transpose() << "; ||wc-wb|| = " << (wc-wb).norm() << "\n" ;

//        if(max_div > u_value){
//            log_details << "max_div is too large:" << max_div << std::endl ;
//            u_value = (max_div + u_value * n_iter)/(1 + n_iter) ;
//            log_details << "max_div = " << max_div << "; use u_value = " << u_value << "\n" ;
//        }

//    }
//    Log::Debug(log_details.str().c_str());
//    G2 = GG;
    G2 = GG.array() * (r_pref * r_pref.transpose()).array();
//    G2 = GG.array() * (inv_r_pref * inv_r_pref.transpose()).array();
//    G2 /= init_K_norm;
    double tau = 1e-12 ;
    double g2trace = G2.trace() ;
    K = (  g2trace * ( (1-tau)/g2trace * G2 + (tau/num_mo_) * Eigen::MatrixXd::Identity(num_mo_,num_mo_) )
            ).pow(0.5);
    K_norm = K.lpNorm<2>();
    if (n_iter == 0) {
        init_K_norm = K_norm;
    }
    if(scale_each_iteration == true) {
        K = (1 / K_norm) * K ;
    }
    else {
        K = (1 / init_K_norm) * K;
    }
//    Eigen::FullPivLU<Eigen::MatrixXd> lu_decomp(K);
//    K.eigenvalues()
    log_details << "eigenvalues of K = " << K.eigenvalues().transpose() << "\n" ;
    Log::Debug(log_details.str().c_str());

    bool solved ;
    Eigen::VectorXd _alpha ;
    double _out_u_value ;

    std::tie(solved, _alpha, _out_u_value) = uOptimizeLoop(p0);

    alpha = _alpha ;
//    gu = _gu ;
//    if(solved == true){
//        u_value = _out_u_value ;
//        p0 = _p0 ;
//    }

    log_details<<"alpha = " << alpha.transpose() << "\n";
    alpha = (alpha.array() < 0).select(0, alpha);
    alpha = r_pref.cwiseProduct(alpha);
    alpha /= alpha.sum();
    if (n_iter > 0 && mo_wc_mgda_type == "alpha_decay"){
        alpha = (alpha * mo_ema_multiplier + prev_alpha * (1 - mo_ema_multiplier) )  ;
    }
    log_details<<"p0: = " << p0;
    Log::Debug(log_details.str().c_str());
    Eigen::Map<Eigen::VectorX<score_t>>(coefficients, num_mo_) = alpha.cast <score_t> ();
    n_iter += 1;
    prev_alpha = alpha ;
    return true;
  }

  // output: alpha, u, in_p0, gu
  std::tuple<bool, Eigen::VectorXd, double> uOptimizeLoop(double in_p0) {
      double _u, _gu, objval;
      double out_u_value = 0 ;
      Eigen::VectorXd alpha ;
      int status_flag ;
      bool solved = false ;
      if ((n_iter > 1) & (use_fix_u == false)) {
          std::tie(status_flag, _u, _gu) = getUTunedAlpha(in_p0);

          if( status_flag == 0 ){
              log_details << "Tuning u no solution " << std::endl ;
              // should p0 be relaxed?
              Log::Debug(log_details.str().c_str());
              std::tie( alpha, objval, _gu) = getAlpha(prev_u_value) ;
          }
          else if( status_flag == 1){
              std::tie( alpha, p0, _gu) = getAlpha(prev_u_value) ;
              log_details << "Tuning u succeed, but trivial solution (primal sub-optimal). use u = "
                          << prev_u_value << "; update p0 = " << p0 << "\n";
              Log::Debug(log_details.str().c_str());
          }
          else if(_u < min_u){
              log_details << "Tuning u succeed. min u satisfied. u = " << _u
                          << "; use min_u = " << min_u << "\n" ;
              Log::Debug(log_details.str().c_str());
              // do not update p0
              std::tie( alpha, objval, _gu) = getAlpha(min_u) ;
          }
          else {
              out_u_value = fmin(max_u, _u);
              log_details << "Tuning u succeed. u updated: " << out_u_value << "\n";
              // no need to update p0
              Log::Debug(log_details.str().c_str());
              std::tie(alpha, objval, _gu) = getAlpha(out_u_value);
              solved = true ;
          }
          Log::Debug(log_details.str().c_str());
      }
      else {
          std::tie(alpha, p0, _gu) = getAlpha(min_u);
          log_details << "Tuning u bypassed. p0 updated: " << p0 << "\n";
          solved = true ;
      }

      return std::make_tuple(solved, alpha, out_u_value);
  }

  double get_EMA_u(double in_u_val){
      double out_u_val = in_u_val ;
//      if(n_iter > 0 && mo_wc_mgda_type == "alpha_decay"){
//          out_u_val = (in_u_val * mo_ema_multiplier + prev_u_value * (1 - mo_ema_multiplier)) ;
//      }
      prev_u_value = out_u_val ;
      return out_u_val ;
  }

  std::tuple <Eigen::VectorXd,double,double> getAlpha(double u_val) {
    //std::cout<<"normal alpha"<<std::endl;
    log_details.str("\n");
//    double u_fac = fmax(1, init_K_norm / K_norm) ;
//    double u_fac = 1.0 ;
//    double _u_val = u_val * u_fac ;
//    log_details << "u_fac = " << u_fac << "; u_val = " << _u_val << "\n";

    double _u_val;
    _u_val = get_EMA_u(u_val) ;
    std::cout << "u_val (getAlpha) = " << _u_val << std::endl;
    log_details << "u_val (getAlpha) = " << _u_val << "\n";

    setSOCPData_opt(_u_val) ;
    Eigen::VectorXd _alpha ;
    _alpha.setZero(num_mo_) ;
    log_details << "in get alpha; wb = " << wb.transpose() << "; wc = " << wc.transpose() << "\n";
    c.head(num_mo_) = wb - wc;
    c(num_mo_) = _u_val;
    G.bottomLeftCorner(num_mo_, num_mo_) = -K;
    h.head(num_mo_) = -lb ;
    //std::cout<<"A=\n"<<A<<"\nb= "<<b.transpose()<<"\nc= "<<c.transpose()<<"\n";

      log_details << "c = " << c << "\n" ;
      log_details << "G = " << G << "\n" ;
      log_details << "h = " << h << "\n" ;
      log_details << "Amat = " << opt.Amat << "\n" ;
      log_details << "bvec = " << opt.bvec << "\n" ;

    opt.SetAb(A, b) ;
    opt.Solve(G, h, c);

    Eigen::VectorXd primal_var = opt.primal_var ;
    _alpha = primal_var.head(num_mo_);
    double gamma = primal_var(num_mo_);
    double gu = gamma * _u_val ;

    Eigen::VectorXd dual_var = opt.dual_var ;
    Eigen::VectorXd d = dual_var.tail(num_mo_);
    std::cout << "getalpha: wc-wb = " << (wc-wb).transpose()
    << "; Kd = " << (K*d).transpose()
    << "; alpha = " << _alpha.transpose()
    << std::endl ;

    return std::make_tuple(_alpha, -c.dot(primal_var), gu) ;
  }

  std::tuple<int, double, double> getUTunedAlpha(double p0) {
    //std::cout<<"u-tuned alpha"<<std::endl;
    setSCOPData_opt_u();
    Eigen::VectorXd alpha ;
    alpha.setZero(num_mo_) ;
    std::cout << "in get utuned alpha; wb = " << wb.transpose() << std::endl;
    h_u.head(num_mo_) = wb - wc;
    h_u(num_mo_) = p0;
    G_u.topRightCorner(num_mo_, num_mo_) = K;
    /*
      log_details << "c_u = " << c_u << std::endl ;
      log_details << "G_u = " << G_u << std::endl ;
      log_details << "h_u = " << h_u << std::endl ;
      log_details << "Amat = " << opt_u.Amat << std::endl ;
      log_details << "bvec = " << opt_u.bvec << std::endl ;
    */
    //std::cout<<"A_u=\n"<<A_u<<"\nb_u = "<<b_u.transpose()<<"\nc_u = "<<c_u.transpose()<<"\n";
    bool solved = opt_u.Solve(G_u, h_u, c_u);
//    Eigen::Map<const Eigen::VectorXd> primal_var(opt_u.sol->x, 2*num_mo_+2);
//    Eigen::Map<const Eigen::VectorXd> dual_var(opt_u.sol->y, 3*num_mo_+2);
    Eigen::VectorXd primal_var = opt_u.primal_var ;
    Eigen::VectorXd dual_var = opt_u.dual_var ;
    double u = primal_var(0);
    double rho = primal_var(1);
    Eigen::VectorXd z = primal_var.segment(2,num_mo_), d = primal_var.tail(num_mo_);
    //std::cout<< "z = "<< z.transpose() << "\nd = " << d.transpose() << "\n";
    //double _p0 = rho - lb.dot(z);
    //double pdiff = (p0 - _p0) / p0;
    alpha = dual_var.head(num_mo_);
    double delta = dual_var(num_mo_);
    log_details << "solved = " << solved << "; alpha / delta " << (alpha / delta).transpose() << "; sum = " << (alpha / delta).sum() << std::endl;
    alpha /= delta;
    log_details<< "u = " << u << " ; rho = " << rho << "; delta = " << delta << "; abs(delta) = " << fabs(delta) << "\n";
    // return ((fabs(delta) > 1e-2) && (alpha.sum() > 1e-2) && (pdiff < 1e-6)) ? true : false;

    double _gu = u / delta ;

    if( solved == false ){
        return std::make_tuple(0, u, _gu) ;
    }
    else if ( (fabs(delta) < 1e-6) || (u < 1e-6) ){
        return std::make_tuple(1, u, _gu) ;
    }
    else{
        return std::make_tuple(2, u, _gu) ;
    }
  }


 private:
  // For WC-MGDA algorithm
  Eigen::VectorXd lb, ref_point, alpha, wc, wb, r_pref, prev_alpha;
//  Eigen::VectorXd inv_r_pref ;
//  Eigen::VectorXd loss_div ;
//  bool use_alpha_decay ;
  //Eigen::VectorXd primal_var, dual_var ;
  Eigen::MatrixXd GG, G2, K;
  double K_norm;
  double p0, init_K_norm, init_rc_norm ;
  double prev_u_value ;
//  double u_val ;
//  double g_value ;
  // mo_wc_mgda_lb;
  double min_u = 0.1, max_u = 10 ;
//  double min_g = 0.01, max_g = 0.01 ;
  double gu ;
  int n_iter;
  int num_queries ;
  double mo_ema_multiplier ;

  std::string mo_wc_mgda_type ;
  bool use_fix_u = false ;

  // For Second Order Cone Programming
  ECOSOptimizer opt, opt_u ;
//  ECOSOptimizer opt_alt, opt_alt_g ;
  //SCSOptimizer opt, opt_u;
  // Gx <= h
  Eigen::MatrixXd G, G_u;
//  Eigen::MatrixXd G_alt, G_alt_g ;
  Eigen::VectorXd h, h_u, c, c_u ;
//  Eigen::VectorXd h_alt, h_alt_g, c_alt, c_alt_g ;
  Eigen::VectorXi q;
  // Ax = b
  Eigen::MatrixXd A;
  Eigen::VectorXd b;
  // Eigen::MatrixXd A_alt, A_alt_g ;
  // Eigen::VectorXd b_alt, b_alt_g ;
  std::stringstream log_details;
  Eigen::VectorXd prev_losses;
  Eigen::MatrixXd prev_GG;

};

class ECMGDA: public MGCombinator {
/*
  The combination coefficients are decided by solving a second order cone programming problem.
*/
 public:
  explicit ECMGDA(const Config& config) {
      min_u = config.mo_mgda_min_u;
      max_u = fmax(min_u, max_u);
      mo_ema_multiplier = config.mo_ema_multiplier ;
      use_fix_u = config.mo_mgda_use_fix_u ;

      if(config.mg_combination == "ec_mgda_decay"){
          mo_ec_mgda_type = "alpha_decay" ;
      }
      else {
          mo_ec_mgda_type = "no_alpha_decay" ;
      }
      log_details << "mg_combination = " << config.mg_combination << "\n" ;
      log_details << "mo_mgda_min_u = " << config.mo_mgda_min_u << "\n" ;
      log_details << "use_fix_u = " << use_fix_u << "\n" ;

      Log::Info(log_details.str().c_str());
  }

  ~ECMGDA() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    MGCombinator::Init(metadata, num_data);
    num_queries = metadata.num_queries();
    orig_ub = Eigen::Map<const Eigen::VectorXf>(metadata.mo_ub_sec_obj(), num_mo_ - 1).cast <double> ();
    ub.setOnes(num_mo_ -1) ;
    init_lnorm.setZero( num_mo_ ) ;
    w = Eigen::Map<const Eigen::VectorXf>(metadata.mo_ec_mgda_w(), num_mo_-1).cast <double> ();
    std::stringstream tmp_buf;
    tmp_buf << "upper bounds: [" << orig_ub.transpose() << "]" << "\nw = [" << w.transpose() << "]";
    Log::Info(tmp_buf.str().c_str());
    wp = 1.0 - w.mean();
    //wp = 1.0 ;
//    max_u = max_u / sqrt( num_mo_ );
//    min_u = min_u / sqrt( num_mo_ );
    //u_val = min_u ;

    GG.setZero(num_mo_, num_mo_);
    K.setZero(num_mo_, num_mo_);
    Kp = Eigen::VectorXd::Zero(num_mo_);
    Ks.setZero(num_mo_, num_mo_-1);
    lsec.setZero(num_mo_-1);
    prim_idx = num_mo_-1;
    sec_start = 0;

    n_iter = 0;
    alpha.setZero(num_mo_);

    // For modified formulation
    sK.setZero(num_mo_, num_mo_);
    sKp = Eigen::VectorXd::Zero(num_mo_);
    sKs.setZero(num_mo_, num_mo_-1);

    setSOCPData_opt();
    // Set optimizer for auto-tuned u
    setSOCPData_opt_u();

  }

  void setSOCPData_opt(){
      int l, m, n;
      // Set common parameters for both balance and descent mode
      q.setConstant(1, 1 + num_mo_);

      // Set optimizer for un-tuned u
      l = 2 * num_mo_-2 ;
      // m = (num_mo_-1) + (num_mo_-1) + (1 + num_mo_);   // = 3*num_mo_-1: ineq_1 + ineq_2 + cone_constraint
      m = 3 * num_mo_ -1 ;
      // num_mo_ = 2; 1 + 1 + 3 = 5
      n = num_mo_;
      opt.Init(m, n, 0, l, 1, q.data()) ;
      // opt.Init(m, n, 0, l, 1, q.data());

      A.setZero(m, n);
      A.topLeftCorner(num_mo_-1, num_mo_-1) = -Eigen::MatrixXd::Identity(num_mo_-1,num_mo_-1);
      A.block(num_mo_-1,0,num_mo_-1,num_mo_-1) = Eigen::MatrixXd::Identity(num_mo_-1,num_mo_-1);
      A(2*num_mo_-2,num_mo_-1) = -1.0;
      b.setZero(m); b.segment(num_mo_-1, num_mo_-1) = w;
      c.setZero(n);
  }

  void setSOCPData_opt_u(){
      int m, n;
      // Set common parameters for both balance and descent mode
      q.setConstant(1, 1 + num_mo_);
      m = (num_mo_ - 1) + 1 + (num_mo_-1) + (1 + num_mo_);  // = 3*num_mo: ineq_1 + ineq_2 + ineq_3 + cone_constraint
      /*  num_mo_-1: lsec <= orig_ub + z + K2.T @ d,
          1 : lprim + w.T @ z - K1.T @ d <= old_objv,
          num_mo_-1: z >= 0,
          1 + num_mo_: cvx.SOC(u/k, d),  */
      n = 1 + (num_mo_-1) + num_mo_;  // = 2*num_mo_  /* (u, z, d) */
      // opt_u.Init(m, n, 0, 2*num_mo_-1, 1, q.data());
      opt_u.Init(m, n, 0, 2 * num_mo_ - 1, 1, q.data());
      A_u.setZero(m, n);
      A_u.block(0, 1, num_mo_-1, num_mo_-1) = -Eigen::MatrixXd::Identity(num_mo_-1,num_mo_-1);
      A_u.block(num_mo_-1, 1, 1, num_mo_-1) = w.transpose();
      A_u.block(num_mo_, 1, num_mo_-1, num_mo_-1) = -Eigen::MatrixXd::Identity(num_mo_-1,num_mo_-1);
      A_u(2*num_mo_-1, 0) = -1.0;
      A_u.bottomRightCorner(num_mo_, num_mo_) = -Eigen::MatrixXd::Identity(num_mo_,num_mo_);
      b_u.setZero(m);
      c_u.setZero(n); c_u(0) = 1.0;
  }


  bool GetCoefficients(const std::vector<score_t>* multi_gradients,
                       const double* mo_costs, score_t* coefficients) override {
    log_details.str("\n");
    Eigen::Map<const Eigen::VectorXd> costs(mo_costs, num_mo_);
    Eigen::VectorXd losses = costs / num_queries;
    if (n_iter == 0){
//    if(true){
        init_lnorm(prim_idx) = losses(prim_idx) ;
        init_lnorm.segment(sec_start, num_mo_ -1) = (losses.segment(sec_start, num_mo_-1) - orig_ub).cwiseAbs() ;
//        init_lnorm = init_lnorm * sqrt( num_mo_ ) ;
        log_details << "orig ub = " << orig_ub.transpose() << "\n" ;
        ub = orig_ub.array() / init_lnorm.segment(sec_start, num_mo_ -1).array() ;
        log_details << "normalized ub = " << ub.transpose() << "\n" ;
//        init_lnorm.segment(sec_start, num_mo_ -1) = orig_ub ;
    }
    log_details << "losses = " << losses.transpose() << "\n" ;
    losses = losses.array() / init_lnorm.array() ;
    log_details << "normalized losses = " << losses.transpose() << "\n" ;

    lprim = losses(prim_idx) * wp ;
    lsec = losses.segment(sec_start, num_mo_-1);
    // std::cout << "lprim = " << lprim << "; lsec = " << lsec.transpose() << "; orig_ub = " << orig_ub.transpose() << "\n";
    RefreshGG(multi_gradients, GG);
    double tau = 1e-12 ;
    double g2trace = GG.trace() ;
//    K = (GG + 1e-12 * Eigen::MatrixXd::Identity(num_mo_,num_mo_)).pow(0.5);
    K = ( g2trace * ((1-tau)/g2trace) * GG + (tau/num_mo_) * Eigen::MatrixXd::Identity(num_mo_,num_mo_) ).pow(0.5);
    K_norm = K.lpNorm<2>();
    if (n_iter == 0){
        init_K_norm = K_norm;
    }
    K /= init_K_norm;
//    K /= K_norm ;
    Kp = K.col(prim_idx);
    Ks = K.block(0, sec_start, num_mo_, num_mo_-1);

    // For modified formulation
    sK = K;
    sK = sK.pow(0.5);
    sKp = sK.col(prim_idx);
    sKs = sK.block(0, sec_start, num_mo_, num_mo_-1);

    double _u, _gu ;
    int status_flag ;
    double objval ;

    //std::cout << "K=\n"<<K<<"\n";
    log_details << "Kp = " << Kp.transpose() << "\nKs = "<<Ks.transpose()<<"\n";
    if ( (n_iter > 1) & (use_fix_u == false)) {
      std::tie(status_flag, _u,_gu) = getUTunedAlpha(p0, gu);

      if( status_flag == 0 ){
          log_details << "Tuning u no solution " << "\n" ;
          std::tie( alpha, objval, _gu) = getAlpha(prev_u_value) ;
      }
      else if( status_flag == 1){
          log_details << "Tuning u succeed, but trivial solution (primal sub-optimal). use u = "
          << prev_u_value << "\n";
          std::tie( alpha, p0, _gu) = getAlpha(prev_u_value) ;
      }
      else if(_u < min_u){
          log_details << "Tuning u succeed. min u satisfied. u = " << _u
                      << "; use min_u = " << min_u << "\n";
          // do not update objval
          std::tie( alpha, objval, _gu) = getAlpha(min_u) ;
      }
      else{
              _u = fmin(_u, max_u);
              log_details << "Tuning u succeed. u = " << _u << "\n";
              std::tie(alpha, objval, _gu) = getAlpha(_u);
      }

    }
  else{
    std::tie(alpha,p0, _gu) = getAlpha(min_u);
    log_details << "Tuning u bypassed. p0 updated: " << p0 << "\n" ;
  }

    gu = _gu ;

    //assignAlpha2Coefs:
    log_details<<"alpha = " << alpha.transpose() << "\n";
    alpha = (alpha.array() < 0).select(0, alpha);  //  alpha[alpha < 0] = 0
    alpha /= alpha.sum();

    if (n_iter > 0 && mo_ec_mgda_type == "alpha_decay"){
        alpha = (alpha * mo_ema_multiplier + prev_alpha * (1 - mo_ema_multiplier) )  ;
    }

    log_details<<"p0: = " << p0 << "; gu = " << gu ;
    Log::Debug(log_details.str().c_str());
    Eigen::Map<Eigen::VectorX<score_t>>(coefficients, num_mo_) = alpha.cast <score_t> ();
    n_iter += 1;
    prev_alpha = alpha ;
    return true;
  }

    double get_EMA_u(double in_u_val){
        double out_u_val = in_u_val ;
        if(n_iter > 0 && mo_ec_mgda_type == "alpha_decay"){
            out_u_val = (in_u_val * mo_ema_multiplier + prev_u_value * (1 - mo_ema_multiplier)) ;
        }
        prev_u_value = out_u_val ;
        return out_u_val ;
    }

  std::tuple <Eigen::VectorXd,double,double> getAlpha(double u_val) {
    //std::cout<<"normal alpha"<<std::endl;
    setSOCPData_opt();
    Eigen::VectorXd _alpha ;
    _alpha.setZero(num_mo_) ;

    u_val = get_EMA_u(u_val) ;

    if(use_modified == true){
        A.bottomLeftCorner(num_mo_, num_mo_-1) = -sKs;
        b.tail(num_mo_) = wp * sKp;
    }
    else {
        A.bottomLeftCorner(num_mo_, num_mo_ - 1) = -Ks;
        b.tail(num_mo_) = wp * Kp;
    }
    c.head(num_mo_-1) = ub - lsec;
    c(num_mo_-1) = u_val;
    //std::cout<<"A=\n"<<A<<"\nb= "<<b.transpose()<<"\nc= "<<c.transpose()<<"\n";

/*      log_details << "c = " << c << "\n" ;
      log_details << "G = " << A << "\n" ;
      log_details << "h = " << b << "\n" ;
      log_details << "Amat = " << opt.Amat << "\n" ;
      log_details << "bvec = " << opt.bvec << "\n" ;
*/

    opt.Solve(A, b, c);
//    Eigen::Map<const Eigen::VectorXd> primal_var(opt.sol->x, num_mo_);
    Eigen::VectorXd primal_var = opt.primal_var ;
    double gamma = primal_var(num_mo_-1);
    double gu = gamma * u_val ;
    _alpha(prim_idx) = wp;
    _alpha.segment(sec_start, num_mo_-1) = primal_var.head(num_mo_-1);
    log_details << "In getAlpha:\nalpha = " << _alpha.transpose()
    << "; gamma = " << gamma << "; u_val = " << u_val
    << "; gu = " << gu
    << "\n";
    return std::make_tuple(_alpha, lprim -c.dot(primal_var), gu) ;
  }


  std::tuple <int,double,double> getUTunedAlpha(double p0, double gu) {
      //std::cout<<"u-tuned alpha"<<std::endl;
      setSOCPData_opt_u();

      Eigen::VectorXd _alpha ;
      _alpha.setZero(num_mo_) ;

      A_u.topRightCorner(num_mo_-1, num_mo_) = -Ks.transpose();
      A_u.block(num_mo_-1, num_mo_, 1, num_mo_) = -wp * Kp.transpose();
      if(use_modified == true){
          A_u.bottomRightCorner(num_mo_, num_mo_) = -sK.transpose();
      }
      b_u.head(num_mo_-1) = ub - lsec;
      b_u(num_mo_-1) = p0 - lprim;
      //std::cout<<"A_u=\n"<<A_u<<"\nb_u = "<<b_u.transpose()<<"\nc_u = "<<c_u.transpose()<<"\n";
/*
      log_details << "c_u = " << c_u << "\n" ;
      log_details << "G_u = " << A_u << "\n" ;
      log_details << "h_u = " << b_u << "\n" ;
      log_details << "Amat = " << opt_u.Amat << "\n" ;
      log_details << "bvec = " << opt_u.bvec << "\n" ;
*/
      bool solved = opt_u.Solve(A_u, b_u, c_u);

      Eigen::VectorXd primal_var = opt_u.primal_var ;
      Eigen::VectorXd dual_var = opt_u.dual_var ;
      double _u = primal_var(0);
      Eigen::VectorXd z = primal_var.segment(1,num_mo_-1), d = primal_var.tail(num_mo_);
      log_details<< "In getUTunedAlpha:\nz = "<< z.transpose() << "\nd = " << d.transpose() << "\n";
      // p0 = lprim + w.dot(z) - K1.dot(d);
      double delta = dual_var(num_mo_-1);
      _alpha(prim_idx) = wp;
      _alpha.segment(sec_start, num_mo_-1) = dual_var.head(num_mo_-1) / delta;
      log_details << "u = " << _u << "; g = " << 1.0 / delta
                  << "; gu (u tuned alpha) = " << _u / delta << "; input gu = " << gu
                  << std::endl;
      log_details << "alpha_u = " << _alpha.transpose() << "; delta = " << delta << "\n";

      double _gu =  _u/delta ;

      if( solved == false ){
          return std::make_tuple(0, _u, gu) ;
      }
      else if ( (fabs(delta) < 1e-6) || (_u < 1e-6) ){
          return std::make_tuple(1, _u, _gu) ;
      }
      else{
          return std::make_tuple(2, _u, _gu) ;
      };

//      if( (fabs(delta) < 1e-6) || (_u < 1e-3) ){
//          // there is room to improve primary objective
//          return std::make_tuple(0, u_val, _gu) ;
//      }
//      else if( fabs(_gu) < fabs(gu) ){
//          // successfully found optimal u / smaller gu
//          //        gu = _gu ;
//          return std::make_tuple(2, _u, _gu) ;
//      }
//      else{
//          // successfully found optimal u / gu increased
//            log_details << "gu is up; gu = " << gu
//                        << "; _gu = " << _gu <<  std::endl ;
//            // gu = _gu ;
//            return std::make_tuple(1, _u, _gu) ;
//      }
  }

  private:
  // For EC-MGDA algorithm
  Eigen::VectorXd orig_ub, w, alpha, r_pref, lsec, Kp;
  Eigen::VectorXd prev_alpha ;
  Eigen::MatrixXd GG, Ks, K;
  // For modified formulation
  Eigen::VectorXd sKp;
  Eigen::MatrixXd sKs, sK;
  Eigen::VectorXd ub ;
  bool use_modified = false ;

  double p0, lprim, init_K_norm, wp;
  double K_norm;
  double min_u = 0.1, max_u = 0.3 ;
  Eigen::VectorXd init_lnorm ;
  int n_iter, prim_idx, sec_start, num_queries;
  double gu = 1e6 ;

  std::string mo_ec_mgda_type ;
  double mo_ema_multiplier ;
        // For Second Order Cone Programming
  // SCSOptimizer opt, opt_u;
//  ECOSOptimizer opt, opt_u ;
  ECOSOptimizer opt, opt_u ;

  Eigen::MatrixXd A, A_u;
  Eigen::VectorXd b, b_u, c, c_u;
  Eigen::VectorXi q;
  // Opt g
  ECOSOptimizer opt_g;
  Eigen::MatrixXd A_g;
  Eigen::VectorXd b_g, c_g;

  std::stringstream log_details;
  bool use_fix_u;
  double prev_u_value;

};

class w_MGDA: public MGCombinator {
/*
The combination coefficients are decided by solving a second order cone programming problem.
*/
public:
    explicit w_MGDA(const Config& config) {
        log_details << "mg_combination = " << config.mg_combination << "\n" ;
        Log::Info(log_details.str().c_str());
    }

    ~w_MGDA() {}

    void Init(const Metadata& metadata, data_size_t num_data) override {
        MGCombinator::Init(metadata, num_data);
        num_queries = metadata.num_queries();
        GG.setZero(num_mo_, num_mo_);
        K.setZero(num_mo_, num_mo_);
        n_iter = 0;
        alpha.setZero(num_mo_);
        setSOCPData_opt();
    }

    void setSOCPData_opt(){
        q.setConstant(1, 1 + num_mo_);
        // m: number of rows in G, n: number of variables, f: number of rows in A, l: number of linear constraints in G
        int m = num_mo_ + (1 + num_mo_) ;
        int n = 1 + num_mo_ ;
        int f = 1;
        int l = num_mo_ ;
        opt.Init(m, n, f, l , q.size(), q.data());
        G.setZero(m, n);
        G.topLeftCorner(num_mo_,num_mo_) = -Eigen::MatrixXd::Identity(num_mo_,num_mo_);
        G(num_mo_,num_mo_) = -1.0;
        h.setZero(m);
        c.setZero(n); c(n-1) = 1.0 ;
        A.setZero(1, n) ;
        A.topLeftCorner(1, num_mo_).array() = 1.0 ;
        b.setZero(1) ;
        b(0) = 1.0 ;
        opt.SetAb(A, b) ;
    }

    bool GetCoefficients(const std::vector<score_t>* multi_gradients,
                         const double* mo_costs, score_t* coefficients) override {
        RefreshGG(multi_gradients, GG);
        Eigen::Map<const Eigen::VectorXd> costs(mo_costs, num_mo_) ;
        log_details.str("\n");
        log_details << "sum loss = " << costs.transpose() << std::endl ;
        Eigen::VectorXd losses = costs / num_queries ;
        if (n_iter == 0) {
            init_rc_norm = preferences_.cwiseProduct(losses).norm();
            r_pref = preferences_ / init_rc_norm ;
            inv_r_pref = 1 / r_pref.array() ;
            std::cout << "r_pref = " << r_pref << std::endl;
        }
        log_details << "r_pref = " << r_pref.transpose() << std::endl ;
        log_details << "loss = " << losses.transpose() << std::endl ;
        Log::Info(log_details.str().c_str());
//        G2 = GG.array() * (r_pref * r_pref.transpose()).array();
        G2 = GG.array() * (inv_r_pref * inv_r_pref.transpose()).array();
        if (n_iter == 0) {
            init_G2_trace = G2.trace();

        }
        G2 /= init_G2_trace;
        K = (G2 + 1e-6 * Eigen::MatrixXd::Identity(num_mo_,num_mo_)).pow(0.5);

        Eigen::VectorXd _alpha ;
        double objval ;
        std::tie(alpha, objval) = getAlpha();

        log_details<<"alpha = " << alpha.transpose() << "\n";
        alpha = (alpha.array() < 0).select(0, alpha);
        //alpha = r_pref.cwiseProduct(alpha);
        alpha /= alpha.sum();
        if (n_iter > 0){
            alpha = (alpha + prev_alpha) / 2.0 ;
        }
        Log::Debug(log_details.str().c_str());
        Eigen::Map<Eigen::VectorX<score_t>>(coefficients, num_mo_) = alpha.cast <score_t> ();
        n_iter += 1;
        prev_alpha = alpha ;
        return true;
    }

    std::tuple <Eigen::VectorXd,double> getAlpha() {
        //std::cout<<"normal alpha"<<std::endl;
        setSOCPData_opt() ;
        Eigen::VectorXd _alpha ;
        _alpha.setZero(num_mo_) ;
        log_details.str("\n");
        G.bottomLeftCorner(num_mo_, num_mo_) = -K;

//          log_details << "c = " << c << std::endl ;
//          log_details << "G = " << G << std::endl ;
//          log_details << "h = " << h << std::endl ;
//          log_details << "Amat = " << opt.Amat << std::endl ;
//          log_details << "bvec = " << opt.bvec << std::endl ;

        opt.SetAb(A, b) ;
        opt.Solve(G, h, c);
        Eigen::VectorXd primal_var = opt.primal_var ;
        _alpha = primal_var.head(num_mo_);
        return std::make_tuple(_alpha, -c.dot(primal_var)) ;
    }

    template<typename M>
    M load_csv (const std::string & path) {
        std::ifstream indata;
        indata.open(path);
        std::string line;
        std::vector<double> values;
        uint rows = 0;
        while (std::getline(indata, line)) {
            std::stringstream lineStream(line);
            std::string cell;
            while (std::getline(lineStream, cell, '\t')) {
                values.push_back(std::stod(cell));
            }
            ++rows;
        }
        return Eigen::Map<const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, Eigen::RowMajor>>(values.data(), rows, values.size()/rows);
    }


private:
    // For w-MGDA algorithm
    Eigen::VectorXd alpha, r_pref, prev_alpha, inv_r_pref;
    Eigen::MatrixXd GG, G2, K;
    double init_G2_trace, init_rc_norm ;
    int n_iter;
    int num_queries ;

    // For Second Order Cone Programming
    ECOSOptimizer opt;
    Eigen::MatrixXd G;
    Eigen::VectorXd h;
    Eigen::VectorXd c ;
    Eigen::VectorXi q;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::stringstream log_details;
};



class PMTL: public MGCombinator {
/*
The combination coefficients are decided by solving a second order cone programming problem.
*/
public:
    explicit PMTL(const Config& config) {
        log_details << "mg_combination = " << config.mg_combination << "\n" ;
        Log::Info(log_details.str().c_str());
        mo_pmtl_preferencefile_path = config.mo_pmtl_preferencefile_path ;
        mo_ema_multiplier = config.mo_ema_multiplier ;
    }

    ~PMTL() {}

    void Init(const Metadata& metadata, data_size_t num_data) override {
        MGCombinator::Init(metadata, num_data);
 //       std::string pf = config.mo_pmtl_preferencefile_path ;
        Eigen::MatrixXd matA = load_csv<Eigen::MatrixXf>(mo_pmtl_preferencefile_path).cast <double> ();
        //Eigen::MatrixXd matA = matAf.cast <double> ();
        // todo need to define a custom type for MatrixX(f|d)
//        matA.rowwise() = matA.rowwise() / matA.rowwise().sum();
        matA.rowwise().normalize();
//        std::cout << matA << std::endl;
//        std::cout << matA.cwiseAbs().rowwise().sum() << std::endl;
//        std::cout << preferences_ << std::endl;
        Eigen::VectorXd a = preferences_.normalized() ; //({0}, Eigen::all);
//        std::cout << a << std::endl;
        matA.rowwise() -= a.transpose();
        std::cout << matA << std::endl;
        matA = removeZeroRow(matA);
        std::cout << matA << std::endl;

        prefMatrix = matA;

        num_queries = metadata.num_queries();
        GG.setZero(num_mo_, num_mo_);
        K.setZero(num_mo_, num_mo_);
        n_iter = 0;
        alpha.setZero(num_mo_);

        C = 0.1;
//        setSOCPData_opt();
    }

    Eigen::MatrixXd removeZeroRow(Eigen::MatrixXd prefMatrix){
//        Eigen::Matrix<bool, 1, Eigen::Dynamic> non_zeros = matA.cast<bool>().rowwise().any();
//        std::cout << "matA:\n" << matA << "\nnon_zeros:\n" << non_zeros << "\n\n";
        // allocate result matrix:
//        Eigen::MatrixXd res(non_zeros.count(), matA.cols());
        // fill result matrix:
        Eigen::VectorXd rownorm = prefMatrix.rowwise().norm();
        rownorm = (rownorm.array() < 1e-10).select(0, rownorm);
//        rownorm.nonZeros();
//        std::cout << rownorm << std::endl;
//        std::cout << rownorm.isZero() << std::endl;
//        std::cout << rownorm.nonZeros() << std::endl;
//        std::cout << rownorm.count() << std::endl;

        Eigen::MatrixXd res(rownorm.count(), prefMatrix.cols());

//        res.setZero(prefMatrix.rows(), rownorm.nonZeros());
        Eigen::Index j = 0;
        for(Eigen::Index i = 0; i < prefMatrix.rows(); ++i)
        {
            if(rownorm(i) > 0)
                res.row(j++) = prefMatrix.row(i);
        }
//        std::cout << res << std::endl ;
        return res ;
    }

    Eigen::MatrixXd removeNegRow(Eigen::MatrixXd prefMatrix){
        std::cout << prefMatrix << std::endl;
//        Eigen::VectorXd pl = prefMatrix * lossVector ;
        Eigen::VectorXd ul_ = (ul.array() < 1e-10).select(0, ul);
        std::cout << ul_ << std::endl;
        Eigen::MatrixXd res(ul_.count(), prefMatrix.cols());

        Eigen::Index j = 0;
        for(Eigen::Index i = 0; i < prefMatrix.rows(); ++i)
        {
            if(ul_(i) > 1e-10)
                res.row(j++) = prefMatrix.row(i);
        }
        std::cout << res << std::endl;

        return res ;
    }


    void setSOCPData_opt_ini(int num_pref){
        q.setConstant(1, 1 + num_mo_);
        // m: number of rows in G, n: number of variables, f: number of rows in A, l: number of linear constraints in G
//        int num_pref = prefMatrix.rows();
        int m = num_pref + 1 + num_mo_ ;
        int n = 1 + num_pref ;
        int f = 1;
        int l = num_pref ;
        opt.Init(m, n, f, l , q.size(), q.data());
        G = Eigen::MatrixXd(m,n);
        G.setZero(m, n);
        G.topLeftCorner(l, l) = -Eigen::MatrixXd::Identity(l, l);
        G(l, l) = -1.0;
        h = Eigen::VectorXd(m);
        h.setZero(m);
        c = Eigen::VectorXd(n);
        c.setZero(n); c(n-1) = 1.0 ;
        A = Eigen::MatrixXd(1, n);
        A.setZero(1, n) ;
        A.topLeftCorner(1, l).array() = 1.0 ;
        b.setZero(1) ;
        b(0) = 1.0 ;
        opt.SetAb(A, b) ;
    }

    void setSOCPData_opt(int num_pref){
        q.setConstant(1, 1 + num_mo_);
        // m: number of rows in G, n: number of variables, f: number of rows in A, l: number of linear constraints in G
//        int num_pref = prefMatrix.rows();
        int m = num_mo_ + num_pref + 1 + num_mo_ ;
        int n = 1 + num_mo_ + num_pref ;
        int f = 1;
        int l = num_mo_ + num_pref ;
        opt.Init(m, n, f, l , q.size(), q.data());
        G = Eigen::MatrixXd(m,n);
        G.setZero(m, n);
        G.topLeftCorner(l, l) = -Eigen::MatrixXd::Identity(l, l);
        G(l, l) = -1.0;
        h = Eigen::VectorXd(m);
        h.setZero(m);
        c = Eigen::VectorXd(n);
        c.setZero(n); c(n-1) = 1.0 ;
        A = Eigen::MatrixXd(1, n);
        A.setZero(1, n) ;
        A.topLeftCorner(1, l).array() = 1.0 ;
        b.setZero(1) ;
        b(0) = 1.0 ;
        opt.SetAb(A, b) ;
    }

    void setSOCPData_opt_const(int num_pref){
        q.setConstant(1, 1 + num_mo_);
        // m: number of rows in G, n: number of variables, f: number of rows in A, l: number of linear constraints in G
//        int num_pref = prefMatrix.rows();
        int num_alpha_beta_ = num_mo_ + num_pref ;

        int m = num_alpha_beta_ + 1 + 1 + num_mo_ ;
        int n = num_alpha_beta_ + 1;
        int f = 1;
        int l = num_alpha_beta_ + 1;
        opt.Init(m, n, f, l , q.size(), q.data());

        G = Eigen::MatrixXd(m,n);
        G.setZero(m, n);
        G.topLeftCorner(num_alpha_beta_, num_alpha_beta_) = -Eigen::MatrixXd::Identity(num_alpha_beta_, num_alpha_beta_);
//        std::cout << G << std::endl;
        G.block(num_alpha_beta_, num_mo_, 1, num_pref).array() = 1.0;
//        std::cout << G << std::endl;
        G(l, n-1) = -1.0;
//        std::cout << G << std::endl;
        h = Eigen::VectorXd(m);
        h.setZero(m);
        h(num_alpha_beta_) = C;
        c = Eigen::VectorXd(n);
        c.setZero(n);
        c.segment(num_mo_, num_pref) = -ul ;
        c(n-1) = 1.0 ;
        A = Eigen::MatrixXd(1, n);
        A.setZero(1, n) ;
        A.topLeftCorner(1, num_mo_).array() = 1.0 ;
        b.setZero(1) ;
        b(0) = 1.0 ;
//        std::cout << "G = " << G << std::endl;
//        std::cout << "h = " << h << std::endl;
//        std::cout << "A = " << A << std::endl;
//        std::cout << "b = " << b << std::endl;
//        std::cout << "c = " << c << std::endl;
        opt.SetAb(A, b) ;
    }

    void setSOCPData_opt_mgda(int num_pref){
        q.setConstant(1, 1 + num_mo_);
        // m: number of rows in G, n: number of variables, f: number of rows in A, l: number of linear constraints in G

        int m = num_pref + 1 + num_mo_ ;
        int n = num_pref + 1;
        int f = 1;
        int l = num_pref;
        opt.Init(m, n, f, l , q.size(), q.data());

        G = Eigen::MatrixXd(m,n);
        G.setZero(m, n);
        G.topLeftCorner(num_pref, num_pref) = -Eigen::MatrixXd::Identity(num_pref, num_pref);
//        std::cout << G << std::endl;
        G(l, n-1) = -1.0;
//        std::cout << G << std::endl;
        h = Eigen::VectorXd(m);
        h.setZero(m);
        c = Eigen::VectorXd(n);
        c.setZero(n);
        c.head(num_pref) = -ul ;
        c(n-1) = C ;
        A = Eigen::MatrixXd(1, n);
        A.setZero(1, n) ;
        A.topLeftCorner(1, num_pref).array() = 1.0 ;
        b.setZero(1) ;
        b(0) = 1.0 ;
        std::cout << "G = " << G << std::endl;
        std::cout << "h = " << h.transpose() << std::endl;
        std::cout << "A = " << A << std::endl;
        std::cout << "b = " << b << std::endl;
        std::cout << "c = " << c.transpose() << std::endl;
        opt.SetAb(A, b) ;
    }


    bool GetCoefficients(const std::vector<score_t>* multi_gradients,
                         const double* mo_costs, score_t* coefficients) override {
        RefreshGG(multi_gradients, GG);
        Eigen::Map<const Eigen::VectorXd> costs(mo_costs, num_mo_) ;
        log_details.str("\n");
        log_details << "sum loss = " << costs.transpose() << "\n" ;
        Eigen::VectorXd losses = costs / num_queries ;
        log_details << "loss = " << losses.transpose() << "\n" ;

        ul = prefMatrix * losses;
        log_details << "ul = " << ul.transpose() << "\n" ;
        G2 = GG ;
        if (n_iter == 0) {
            init_G2_trace = G2.trace();
        }
        G2 /= init_G2_trace;
        K = (G2 + 1e-6 * Eigen::MatrixXd::Identity(num_mo_,num_mo_)).pow(0.5);

//        alpha = opt_orig_pmtl();
//        alpha = opt_modified_pmtl();
        alpha = opt_modified_mgda();

        log_details<<"alpha = " << alpha.transpose() << "\n";
        alpha /= alpha.lpNorm<1>();
        Log::Debug(log_details.str().c_str());
        if(n_iter > 0){
//            alpha = prev_alpha * 0.9 + alpha * 0.1;
            alpha = (alpha * mo_ema_multiplier + prev_alpha * (1 - mo_ema_multiplier) )  ;
        }
        Eigen::Map<Eigen::VectorX<score_t>>(coefficients, num_mo_) = alpha.cast <score_t> ();
        n_iter += 1;
        prev_alpha = alpha ;
        return true;
    }

    Eigen::VectorXd opt_orig_pmtl(){
        Eigen::VectorXd alpha ;
        Eigen::MatrixXd prefMatrix_ = removeNegRow(prefMatrix);
        std::cout << "prefmatrix = " << prefMatrix_ << std::endl;
        std::cout << "prefmatrix size = " << prefMatrix_.size() << std::endl;
        if(prefMatrix_.size() > 0) {
            KU = K * prefMatrix_.transpose();
        }
        else{
            KU = K;
        }
        double objval ;
        if (n_iter < 2 && prefMatrix_.size() > 0){
            bool solved;
            std::tie(solved,alpha, objval) = getAlpha_ini(prefMatrix_);
            if(solved == false || alpha.lpNorm<1>() < 1e-3){
                std::tie(alpha, objval) = getAlpha(prefMatrix_);
            }
        }
        else{
            std::tie(alpha, objval) = getAlpha(prefMatrix_);
        }

        return alpha;
    }

    Eigen::VectorXd opt_modified_pmtl(){
        Eigen::VectorXd alpha ;

        Eigen::MatrixXd prefMatrix_ = prefMatrix;

        double objval ;
        std::tie(alpha, objval) = getAlpha_const(prefMatrix_);

        return alpha;
    }

    Eigen::VectorXd opt_modified_mgda(){
        Eigen::VectorXd alpha ;
        Eigen::MatrixXd prefMatrix_ = prefMatrix;
        KU = K * prefMatrix_.transpose();

        double objval ;
        std::tie(alpha, objval) = getAlpha_mgda(prefMatrix_);

        return alpha;
    }



    std::tuple <Eigen::VectorXd,double> getAlpha(Eigen::MatrixXd prefMatrix_) {
        //std::cout<<"normal alpha"<<std::endl;
        setSOCPData_opt(KU.cols()) ;
        Eigen::VectorXd _alpha ;
        Eigen::VectorXd _beta ;
        _alpha.setZero(num_mo_) ;
        _beta.setZero(KU.rows()) ;
        log_details.str("\n");
        G.bottomLeftCorner(num_mo_, num_mo_) = -K;
        G.block(num_mo_ + KU.cols() +1, num_mo_, KU.rows(), KU.cols()) = -KU ;

         log_details << "c = " << c << "\n" ;
          log_details << "G = " << G << "\n" ;
          log_details << "h = " << h << "\n" ;
          log_details << "Amat = " << opt.Amat << "\n" ;
          log_details << "bvec = " << opt.bvec << "\n" ;

        opt.SetAb(A, b) ;
        opt.Solve(G, h, c);
        Eigen::VectorXd primal_var = opt.primal_var ;
        _alpha = primal_var.head(num_mo_);
        _beta = primal_var.tail(KU.cols());
        if(prefMatrix_.size() > 0) {
            _alpha = _alpha + prefMatrix_.transpose() * _beta;
        }
        return std::make_tuple(_alpha, c.dot(primal_var)) ;
    }

    std::tuple <Eigen::VectorXd,double> getAlpha_const(Eigen::MatrixXd prefMatrix_) {
        std::cout<<"in getalpha const: prefMatrix = "<< prefMatrix_ << std::endl;
        setSOCPData_opt_const(prefMatrix_.rows()) ;
        Eigen::VectorXd _alpha ;
        Eigen::VectorXd _beta ;
        _alpha.setZero(num_mo_) ;
        _beta.setZero(prefMatrix_.rows()) ;
//        log_details.str("\n");
        G.bottomLeftCorner(num_mo_, num_mo_) = -K;

        log_details << "c = " << c.transpose() << "\n" ;
        log_details << "G = " << G << "\n" ;
        log_details << "h = " << h.transpose() << "\n" ;
        log_details << "Amat = " << opt.Amat << "\n" ;
        log_details << "bvec = " << opt.bvec << "\n" ;

        opt.SetAb(A, b) ;
        opt.Solve(G, h, c);
        Eigen::VectorXd primal_var = opt.primal_var ;
        _alpha = primal_var.head(num_mo_);
        _beta = primal_var.tail(prefMatrix_.rows());
        Eigen::VectorXd ubeta = prefMatrix_.transpose() * _beta;
        log_details << "primal_var = " << primal_var.transpose() << "\n" ;
        log_details << "ubeta = " << ubeta.transpose() << "\n";
        return std::make_tuple(_alpha, -c.dot(primal_var)) ;
    }

    std::tuple <Eigen::VectorXd,double> getAlpha_mgda(Eigen::MatrixXd prefMatrix_) {
        std::cout<<"in getalpha const: prefMatrix = "<< prefMatrix_ << std::endl;
        setSOCPData_opt_mgda(prefMatrix_.rows()) ;
        Eigen::VectorXd _alpha ;
        _alpha.setZero(prefMatrix_.rows()) ;
        G.bottomLeftCorner(num_mo_, prefMatrix_.rows()) = -KU;

        log_details << "c = " << c.transpose() << "\n" ;
        log_details << "G = " << G << "\n" ;
        log_details << "h = " << h.transpose() << "\n" ;
        log_details << "Amat = " << opt.Amat << "\n" ;
        log_details << "bvec = " << opt.bvec << "\n" ;

        opt.SetAb(A, b) ;
        opt.Solve(G, h, c);
        Eigen::VectorXd primal_var = opt.primal_var ;
        _alpha = primal_var.head(prefMatrix_.rows());
        Eigen::VectorXd ualpha = prefMatrix_.transpose() * _alpha;
        log_details << "primal_var = " << primal_var.transpose() << "\n" ;
        log_details << "ualpha = " << ualpha.transpose() << "\n";
        return std::make_tuple(ualpha, -c.dot(primal_var)) ;
    }

    std::tuple <bool, Eigen::VectorXd,double> getAlpha_ini(Eigen::MatrixXd prefMatrix_) {
        //std::cout<<"normal alpha"<<std::endl;
        int num_pref = KU.cols();
        setSOCPData_opt_ini(num_pref) ;
        Eigen::VectorXd _beta ;
//        _alpha.setZero(num_mo_) ;
        _beta.setZero(KU.rows()) ;
        log_details.str("\n");
        G.bottomLeftCorner(KU.rows(), KU.cols()) = -KU;

        log_details << "getalpha_ini" << "\n" ;
        log_details << "c = " << c << "\n" ;
        log_details << "G = " << G << "\n" ;
        log_details << "h = " << h << "\n" ;
        log_details << "Amat = " << opt.Amat << "\n" ;
        log_details << "bvec = " << opt.bvec << "\n" ;

        opt.SetAb(A, b) ;
        bool solved = opt.Solve(G, h, c);
        Eigen::VectorXd primal_var = opt.primal_var ;
//        _alpha = primal_var.head(num_mo_);
        _beta = primal_var.head(num_pref);
        _beta = prefMatrix_.transpose() * _beta ;
        return std::make_tuple(solved, _beta, c.dot(primal_var)) ;
    }

    template<typename M>
    M load_csv (const std::string & path) {
        std::ifstream indata;
        indata.open(path);
        std::string line;
        std::vector<label_t> values; // todo later use custom type
        uint rows = 0;
        while (std::getline(indata, line)) {
            std::stringstream lineStream(line);
            std::string cell;
            while (std::getline(lineStream, cell, '\t')) {
                values.push_back(std::stod(cell));
            }
            ++rows;
        }
        return Eigen::Map<const Eigen::Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, Eigen::RowMajor>>(values.data(), rows, values.size()/rows);
    }


private:
    // For w-MGDA algorithm
    Eigen::VectorXd alpha, r_pref, prev_alpha, inv_r_pref;
    Eigen::MatrixXd GG, G2, K, KU;
    double init_G2_trace;
    //, init_rc_norm ;
    int n_iter;
    int num_queries ;

    // For Second Order Cone Programming
    ECOSOptimizer opt;
    Eigen::MatrixXd G;
    Eigen::VectorXd h;
    Eigen::VectorXd c ;
    Eigen::VectorXi q;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::stringstream log_details;
    std::string mo_pmtl_preferencefile_path;
    Eigen::MatrixXd prefMatrix;
    Eigen::VectorXd ul;
    double C;
    double mo_ema_multiplier;
    };


class EConstraint: public MGCombinator {
/*
  The combination coefficients (cc) are decided a simple proximal update rule, by using the cc of previous iteration.
*/
 public:
  explicit EConstraint(const Config&) {}

  ~EConstraint() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    MGCombinator::Init(metadata, num_data);
    num_queries = metadata.num_queries();
    orig_ub = Eigen::Map<const Eigen::VectorXf>(metadata.mo_ub_sec_obj(), num_mo_ - 1).cast <double> ();
    ub.setZero(num_mo_ -1) ;
    std::stringstream tmp_buf;
    tmp_buf << "upper bounds: [" << orig_ub.transpose() << "]";

    prim_idx = num_mo_-1;
    sec_start = 0;
    alpha.setZero(num_mo_);
    alpha(prim_idx) = 1.0;
//    mu.setConstant(num_mo_-1, 10);
//    std::cout << metadata.mo_ec_mu() << std::endl ;
    mu = Eigen::Map<const Eigen::VectorXf>(metadata.mo_ec_mu(), num_mo_ -1).cast <double> () ;
    tmp_buf << "mu = " << mu.transpose() << std::endl ;
    Log::Info(tmp_buf.str().c_str());
    n_iter = 0;

  }

  bool GetCoefficients(const std::vector<score_t>*,
                       const double* mo_costs, score_t* coefficients) override {
    log_details.str("\n");
    Eigen::Map<const Eigen::VectorXd> costs(mo_costs, num_mo_);
    Eigen::VectorXd losses = costs / num_queries;
    if( n_iter == 0 ){
        loss_norm = losses ;
//        loss_norm.segment(sec_start, num_mo_ -1) = (orig_ub - losses.segment(sec_start, num_mo_-1)).cwiseAbs() ;
//        ub = orig_ub.array() / loss_norm.segment(sec_start, num_mo_ -1).array() ;
//        loss_norm.segment(sec_start, num_mo_ -1) = loss_norm.segment(sec_start, num_mo_ -1).array() / orig_ub.array() ;
        loss_norm.segment(sec_start, num_mo_ -1) = orig_ub.array() ;
        ub.setOnes(num_mo_-1);
        log_details << "ub (after norm) = " << ub.transpose() << "\n" ;
    }
    log_details << "losses (before norm) = " << losses.transpose() << "\n" ;
    losses = losses.array() / loss_norm.array() ;
    log_details << "losses (after norm) = " << losses.transpose() << "\n" ;
    log_details << "losses - ub (after norm) = " << (losses.segment(sec_start, num_mo_-1) - ub).transpose() << std::endl ;
    Eigen::VectorXd alpha_update = mu.array() * (losses.segment(sec_start, num_mo_-1) - ub).array();

    // alpha update in old version
//    alpha.segment(sec_start, num_mo_-1)
//        = (alpha_update.array() < 0).select(0, alpha.segment(sec_start, num_mo_-1));
//    alpha_update = (alpha_update.array() < 0).select(0, alpha_update);  //  alpha_update[alpha_update < 0] = 0
//    log_details<<"alpha_update = " << alpha_update.transpose() << "\n";
//    alpha.segment(sec_start, num_mo_-1) += alpha_update;

// alpha update -- new
    log_details<<"alpha_update = " << alpha_update.transpose() << "\n";
    alpha.segment(sec_start, num_mo_-1) += alpha_update;
    alpha.segment(sec_start, num_mo_-1)
        = (alpha.segment(sec_start, num_mo_-1).array() < 0).select(0, alpha.segment(sec_start, num_mo_-1));

    //for (data_size_t i = 0; i < num_mo_-1; i++) {
    //  int idx = sec_start + i;
    //  if (mo_n0_ < 0) {
    //    // constraint satisfied, reset alpha
    //    if (losses[idx] <= orig_ub[idx]) {
    //      alpha(idx) = 0;
    //    } else {  // update alpha
    //      alpha(idx) += mu(idx) * (losses[idx] - orig_ub[idx]);
    //      if (alpha(idx) < 0) alpha(idx) = 0;
    //    }
    //  }
    //}
    alpha(prim_idx) = 1.0;
    log_details <<"alpha = " << alpha.transpose();
    Log::Debug(log_details.str().c_str());
    alpha /= alpha.sum();
    Eigen::Map<Eigen::VectorX<score_t>>(coefficients, num_mo_) = alpha.cast <score_t> ();
    n_iter += 1;
    return true;
  }
 private:
  // For EC-MGDA algorithm
  Eigen::VectorXd alpha, orig_ub, mu, ub;
  Eigen::VectorXd loss_norm ;
  int n_iter, prim_idx, num_queries, sec_start;

  std::stringstream log_details;
};

MGCombinator* MGCombinator::CreateMGCombinator(const std::string& type, const Config& config) {
  if (type == "linear_scalarization") {
    return new LinearScalarization(config);
  }
  if (type == "chebychev_scalarization" || type == "chebychev_scalarization_decay") {
    return new ChebychevScalarization(config);
  }
  if (type == "epo_search" || type == "epo_search_decay") {
    return new EPOSearch(config);
  }
  if (type == "wc_mgda" || type == "wc_mgda_decay" ) {
    return new WCMGDA(config);
  }
  if (type == "w_mgda") {
    return new w_MGDA(config);
  }
  if (type == "ec_mgda" || type == "ec_mgda_decay") {
    return new ECMGDA(config);
  }
  if (type == "pmtl") {
    return new PMTL(config);
  }

  if (type == "e_constraint") {
    return new EConstraint(config);
  }
  return nullptr;
}


}

#endif   // LightGBM_MG_COMBINATOR_HPP_
