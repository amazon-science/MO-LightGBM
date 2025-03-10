/*!
 * Copyright (c) 2020 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_OBJECTIVE_RANK_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_RANK_OBJECTIVE_HPP_

#include <LightGBM/metric.h>
#include <LightGBM/objective_function.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <random>

#include "mg_combinator.hpp"

namespace LightGBM {

/*!
 * \brief Objective function for Ranking
 */
class RankingObjective : public ObjectiveFunction {
 public:
  explicit RankingObjective(const Config& config)
      : seed_(config.objective_seed) {
    objective_type_ = config.objective;
    all_cost_.clear();
    multi_gradients_.clear();
    mo_combinator_name_ = config.mg_combination;
    if (mo_combinator_name_ != "stochastic_label_aggregation") {
      mg_combinator_.reset(MGCombinator::CreateMGCombinator(config.mg_combination, config));
    }
  }

  explicit RankingObjective(const std::vector<std::string>&) : seed_(0) {}

  ~RankingObjective() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    // get label
    label_ = metadata.label();
    // get weights
    //weights_ = metadata.weights();
    // get boundries
    query_boundaries_ = metadata.query_boundaries();
    if (query_boundaries_ == nullptr) {
      Log::Fatal("Ranking tasks require query information");
    }
    num_queries_ = metadata.num_queries();

    num_mo_ = metadata.num_mo();
    mo_preferences_ = metadata.mo_preferences();
    mo_data_ = metadata.mo_data();
    all_cost_.resize(num_mo_, 0.0);
    mg_coefficients_.resize(num_mo_, 1.0f);
    mo_weights_.resize(num_mo_);
    multi_gradients_.resize(num_mo_);
    multi_hessians_.resize(num_mo_);
    for (data_size_t mo_idx = 0; mo_idx < num_mo_; mo_idx++) {
      mo_weights_[mo_idx] = metadata.mo_weights(mo_idx);
      multi_gradients_[mo_idx].clear();
      multi_gradients_[mo_idx] = std::vector<score_t>(num_data_, 0.0f);
      multi_hessians_[mo_idx].clear();
      multi_hessians_[mo_idx] = std::vector<score_t>(num_data_, 0.0f);
    }
    if (mo_combinator_name_ != "stochastic_label_aggregation") {
      mg_combinator_->Init(metadata, num_data);
    } else {
      mo_label_distribution_ = std::discrete_distribution<int>(mo_preferences_, mo_preferences_ + num_mo_);
    }
  }

  void GetGradients(const double* score, score_t* gradients,
                    score_t* hessians) const override {
    if (mo_combinator_name_ != "stochastic_label_aggregation") {
      for (auto mo_idx = 0; mo_idx < num_mo_; mo_idx++) {
        all_cost_[mo_idx] = 0.0;
        std::vector<double> q_cost(num_queries_);
        #pragma omp parallel for schedule(guided)
        for (data_size_t i = 0; i < num_queries_; ++i) {
          q_cost[i] = GetGradientsForOneQuery(i, mo_idx, score, mo_weights_[mo_idx]);
          //if (weights_ != nullptr) {
          //  for (data_size_t j = 0; j < cnt; ++j) {
          //    gradients[start + j] =
          //        static_cast<score_t>(gradients[start + j] * weights_[start + j]);
          //    hessians[start + j] =
          //        static_cast<score_t>(hessians[start + j] * weights_[start + j]);
          //  }
          //}
        }
        // cost here is sum of costs over all queries
        all_cost_[mo_idx] = accumulate(q_cost.begin(), q_cost.end(), 0.0) ;
        // / static_cast<double>(num_queries_);
      }
      mg_combinator_->GetCoefficients(multi_gradients_.data(),
                                      all_cost_.data(), mg_coefficients_.data());
      std::stringstream tmp_buf;
      tmp_buf <<"multigrad-combination-coefficients: [";
      for (auto i = 0; i < num_mo_; i++) {
        tmp_buf << mg_coefficients_[i] <<",";
      }
      tmp_buf<<"]";
      Log::Info(tmp_buf.str().c_str());

      mg_combinator_->GetCombination(multi_gradients_.data(), mg_coefficients_.data(), gradients);
      mg_combinator_->GetCombination(multi_hessians_.data(), mg_coefficients_.data(), hessians);
    } else {
      gradients_ = gradients; hessians_ = hessians;
      std::vector<int> mo_idx_samples(num_queries_);
      #pragma omp parallel for schedule(guided)
      for (data_size_t i = 0; i < num_queries_; ++i) {
        mo_idx_samples[i] = mo_label_distribution_(generator_);
        GetGradientsForOneQuery(i, mo_idx_samples[i], score, mo_weights_[ mo_idx_samples[i] ]);
        //if (weights_ != nullptr) {
        //  for (data_size_t j = 0; j < cnt; ++j) {
        //    gradients[start + j] =
        //        static_cast<score_t>(gradients[start + j] * weights_[start + j]);
        //    hessians[start + j] =
        //        static_cast<score_t>(hessians[start + j] * weights_[start + j]);
        //  }
        //}
      }
      float least_probable = 1.0f/float(num_queries_);
      std::vector<float> sample_dist(num_mo_, 0.0f);
      for (data_size_t i = 0; i < num_queries_; ++i) {
        sample_dist[mo_idx_samples[i]] += least_probable;
      }
      std::stringstream tmp_buf;
      tmp_buf <<"multilable-sample-distribution: [";
      for (auto i = 0; i < num_mo_; i++) {
        tmp_buf << sample_dist[i] <<",";
      }
      tmp_buf<<"]";
      Log::Info(tmp_buf.str().c_str());
    }
  }

  virtual double GetGradientsForOneQuery(data_size_t query_id, data_size_t mo_idx,
                                         const double* all_score, const float* weight) const = 0;

  const char* GetName() const override = 0;

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName();
    return str_buf.str();
  }

  bool NeedAccuratePrediction() const override { return false; }

 protected:
  int seed_;
  std::string objective_type_;
  data_size_t num_queries_;
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const label_t* label_;
  /*! \brief Pointer of weights */
  const label_t* weights_;
  /*! \brief Query boundaries */
  const data_size_t* query_boundaries_;
  /*! \brief Number of multi-objective ranking */
  data_size_t num_mo_ = 1;
  /*! \brief Matrix for target labels, dimn: num_mo_ x num_data_ */
  const std::vector<label_t>* mo_data_;
  /*! \brief Cost of primary objective and multi-objectives, dim = num_mo_ + 1 */
  mutable std::vector<double> all_cost_;
  /*! \brief weight of multi-objectives, dim = num_mo_, note vector<const float*> is different from vector<float>*, which used in mo_data_ */
  std::vector<const float*> mo_weights_;
  /*! \brief container for multi-gradients */
  mutable std::vector<std::vector<score_t>> multi_gradients_;
  /*! \brief container for multi-gradients */
  mutable std::vector<std::vector<score_t>> multi_hessians_;
  /*! \brief Multi-Gradient Combinator */
  std::unique_ptr<MGCombinator> mg_combinator_;
  mutable std::vector<score_t> mg_coefficients_;
  std::string mo_combinator_name_;
  //const label_t* mo_preferences_;
  const label_t* mo_preferences_; //todo need to revisit to switch to custom type
  mutable score_t* gradients_;
  mutable score_t* hessians_;
  mutable std::default_random_engine generator_;
  mutable std::discrete_distribution<int> mo_label_distribution_;

  mutable std::vector<label_t> mo_data_label_max_;


};

/*!
 * \brief Objective function for LambdaRank with NDCG
 */
class LambdarankNDCG : public RankingObjective {
 public:
  explicit LambdarankNDCG(const Config& config)
      : RankingObjective(config),
        sigmoid_(config.sigmoid),
        norm_(config.lambdarank_norm),
        truncation_level_(config.lambdarank_truncation_level) {
    label_gain_ = config.label_gain;
    // initialize DCG calculator
    DCGCalculator::DefaultLabelGain(&label_gain_);
    use_quicksort_ndcg_ = config.use_quicksort_ndcg;
    DCGCalculator::Init(label_gain_, use_quicksort_ndcg_);
    sigmoid_table_.clear();
    inverse_max_dcgs_.clear();
    mo_inverse_max_dcgs_.clear();
    if (sigmoid_ <= 0.0) {
      Log::Fatal("Sigmoid param %f should be greater than zero", sigmoid_);
    }
  }

  explicit LambdarankNDCG(const std::vector<std::string>& strs)
      : RankingObjective(strs) {}

  ~LambdarankNDCG() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    RankingObjective::Init(metadata, num_data);
    DCGCalculator::CheckMetadata(metadata, num_queries_);
    //DCGCalculator::CheckLabel(label_, num_data_);
    init_inv_max_dcgs(inverse_max_dcgs_, label_);

    mo_inverse_max_dcgs_.resize(num_mo_);
    for (data_size_t mo_idx = 0; mo_idx < num_mo_; mo_idx++) {
      // Check if mo data is legal
      DCGCalculator::CheckLabel(mo_data_[mo_idx].data(), num_data_);
      // Init inverse_max_ndcg for each obj
      init_inv_max_dcgs(mo_inverse_max_dcgs_[mo_idx], mo_data_[mo_idx].data());
    }
    // construct Sigmoid table to speed up Sigmoid transform
    ConstructSigmoidTable();
  }

  inline void init_inv_max_dcgs(std::vector<double>& inv_mdcgs, const label_t* lbl) {
    inv_mdcgs.resize(num_queries_);
    #pragma omp parallel for schedule(static)
    for (data_size_t i = 0; i < num_queries_; ++i) {
      inv_mdcgs[i] = DCGCalculator::CalMaxDCGAtK(
          truncation_level_, lbl + query_boundaries_[i],
          query_boundaries_[i + 1] - query_boundaries_[i]);
      if (inv_mdcgs[i] > 0.0) {
        inv_mdcgs[i] = 1.0f / inv_mdcgs[i];
      }
    }
  }

  double GetGradientsForOneQuery(data_size_t query_id, data_size_t mo_idx,
                                        const double* all_score, const float* weight) const override {
    const data_size_t start = query_boundaries_[query_id];
    const data_size_t cnt = query_boundaries_[query_id + 1] - query_boundaries_[query_id];
    const double* score = all_score + start;
    const label_t* label = mo_data_[mo_idx].data() + start;
    score_t* lambdas;
    score_t* hessians;
    if (mo_combinator_name_ != "stochastic_label_aggregation") {
      lambdas = multi_gradients_[mo_idx].data() + start;
      hessians = multi_hessians_[mo_idx].data() + start;
    } else {
      lambdas = gradients_ + start;
      hessians = hessians_ + start;
    }

    // get max DCG on current query
    const double inverse_max_dcg = mo_inverse_max_dcgs_[mo_idx][query_id];

    // initialize with zero
    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] = 0.0f;
      hessians[i] = 0.0f;
    }
    // get sorted indices for scores
    std::vector<data_size_t> sorted_idx(cnt);
    for (data_size_t i = 0; i < cnt; ++i) {
      sorted_idx[i] = i;
    }
    std::stable_sort(
        sorted_idx.begin(), sorted_idx.end(),
        [score](data_size_t a, data_size_t b) { return score[a] > score[b]; });
    // get best and worst score
    const double best_score = score[sorted_idx[0]];
    data_size_t worst_idx = cnt - 1;
    if (worst_idx > 0 && score[sorted_idx[worst_idx]] == kMinScore) {
      worst_idx -= 1;
    }
    const double worst_score = score[sorted_idx[worst_idx]];
    double sum_lambdas = 0.0;
    std::vector<double> cost(cnt, 0.0);
    // start accmulate lambdas by pairs that contain at least one document above truncation level
    for (data_size_t i = 0; i < cnt - 1 && i < truncation_level_; ++i) {
      if (score[sorted_idx[i]] == kMinScore) { continue; }
      for (data_size_t j = i + 1; j < cnt; ++j) {
        if (score[sorted_idx[j]] == kMinScore) { continue; }
        // skip pairs with the same labels
        if (label[sorted_idx[i]] == label[sorted_idx[j]]) { continue; }
        data_size_t high_rank, low_rank;
        if (label[sorted_idx[i]] > label[sorted_idx[j]]) {
          high_rank = i;
          low_rank = j;
        } else {
          high_rank = j;
          low_rank = i;
        }
        const data_size_t high = sorted_idx[high_rank];
        const int high_label = static_cast<int>(label[high]);
        const double high_score = score[high];
        const double high_discount = DCGCalculator::GetDiscount(high_rank);
        const data_size_t low = sorted_idx[low_rank];
        const int low_label = static_cast<int>(label[low]);
        const double low_score = score[low];
        const double low_discount = DCGCalculator::GetDiscount(low_rank);

        const double delta_score = high_score - low_score;
        double delta_pair_NDCG = 1.0;
        if (objective_type_ == std::string("lambdarank")) {
          // get dcg gap
          double dcg_gap = 0.0;
          if (use_quicksort_ndcg_) {
            dcg_gap = high_label - low_label;
          } else {
            dcg_gap = label_gain_[high_label] - label_gain_[low_label];
          }
          // get discount of this pair
          const double paired_discount = fabs(high_discount - low_discount);
          // get delta NDCG
          delta_pair_NDCG = dcg_gap * paired_discount * inverse_max_dcg;
          // regular the delta_pair_NDCG by score distance
          if (norm_ && best_score != worst_score) {
            delta_pair_NDCG /= (0.01 + fabs(delta_score));
          }
        }
        // calculate lambda for this pair
        double p_lambda = GetSigmoid(delta_score);
        double p_hessian = p_lambda * (1.0 - p_lambda);
        // update
        p_lambda *= -sigmoid_ * delta_pair_NDCG;
        p_hessian *= sigmoid_ * sigmoid_ * delta_pair_NDCG;
        lambdas[low] -= static_cast<score_t>(p_lambda);
        hessians[low] += static_cast<score_t>(p_hessian);
        lambdas[high] += static_cast<score_t>(p_lambda);
        hessians[high] += static_cast<score_t>(p_hessian);
        // lambda is negative, so use minus to accumulate
        sum_lambdas -= 2.0 * p_lambda;
        cost[high] += delta_pair_NDCG * std::log(1.0 / GetSigmoid(-delta_score));
      }
    }
    if (norm_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
          lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor );
          hessians[i] = static_cast<score_t>(hessians[i] * norm_factor );
      }
    }
    if( weight != nullptr ){
        for (data_size_t i = 0; i < cnt; ++i) {
            lambdas[i] = static_cast<score_t>(lambdas[i] * weight[ start + i] );
            hessians[i] = static_cast<score_t>(hessians[i] * weight[ start + i] );
            cost[i] = cost[i] * weight[ start + i ] ;
        }
    }
    return accumulate(cost.begin(), cost.end(), 0.0);
  }

  inline double GetSigmoid(double score) const {
    if (score <= min_sigmoid_input_) {
      // too small, use lower bound
      return sigmoid_table_[0];
    } else if (score >= max_sigmoid_input_) {
      // too large, use upper bound
      return sigmoid_table_[_sigmoid_bins - 1];
    } else {
      return sigmoid_table_[static_cast<size_t>((score - min_sigmoid_input_) *
                                                sigmoid_table_idx_factor_)];
    }
  }

  void ConstructSigmoidTable() {
    // get boundary
    min_sigmoid_input_ = min_sigmoid_input_ / sigmoid_ / 2;
    max_sigmoid_input_ = -min_sigmoid_input_;
    sigmoid_table_.resize(_sigmoid_bins);
    // get score to bin factor
    sigmoid_table_idx_factor_ =
        _sigmoid_bins / (max_sigmoid_input_ - min_sigmoid_input_);
    // cache
    for (size_t i = 0; i < _sigmoid_bins; ++i) {
      const double score = i / sigmoid_table_idx_factor_ + min_sigmoid_input_;
      sigmoid_table_[i] = 1.0f / (1.0f + std::exp(score * sigmoid_));
    }
  }

  const char* GetName() const override { return "lambdarank"; }

  double* LabelGain() {return label_gain_.data();}

  void ConvertOutput(const double* input, double* output) const override {
    output[0] = GetSigmoid(input[0]);
  }

 private:
  /*! \brief Sigmoid param */
  double sigmoid_;
  /*! \brief Normalize the lambdas or not */
  bool norm_;
  /*! \brief Truncation position for max DCG */
  int truncation_level_;
  /*! \brief Cache inverse max DCG, speed up calculation */
  std::vector<double> inverse_max_dcgs_;
  /*! \brief Cache result for sigmoid transform to speed up */
  std::vector<double> sigmoid_table_;
  /*! \brief Gains for labels */
  std::vector<double> label_gain_;
  /*! \brief Number of bins in simoid table */
  size_t _sigmoid_bins = 1024 * 1024;
  /*! \brief Minimal input of sigmoid table */
  double min_sigmoid_input_ = -50;
  /*! \brief Maximal input of Sigmoid table */
  double max_sigmoid_input_ = 50;
  /*! \brief Factor that covert score to bin in Sigmoid table */
  double sigmoid_table_idx_factor_;
  /*! \brief Cache mo inverse max DCG, speed up calculation */
  std::vector<std::vector<double>> mo_inverse_max_dcgs_;
  /*! \brief Indicate current iteration */
  bool use_quicksort_ndcg_;
};

/*!
 * \brief Implementation of the learning-to-rank objective function, XE_NDCG
 * [arxiv.org/abs/1911.09798].
 */
class RankXENDCG : public RankingObjective {
 public:
  explicit RankXENDCG(const Config& config) : RankingObjective(config) {}

  explicit RankXENDCG(const std::vector<std::string>& strs)
      : RankingObjective(strs) {}

  ~RankXENDCG() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    RankingObjective::Init(metadata, num_data);
    for (data_size_t i = 0; i < num_queries_; ++i) {
      rands_.emplace_back(seed_ + i);
    }

    for (auto mo_idx = 0; mo_idx < num_mo_; mo_idx++){
        label_t _maxlabel = *std::max_element(std::begin(mo_data_[mo_idx]), std::end(mo_data_[mo_idx]) );
        mo_data_label_max_.push_back( _maxlabel );
        std::cout << "mo_idx = " << mo_idx << "; max label = " << mo_data_label_max_[mo_idx] << std::endl;
    }

  }

  double GetGradientsForOneQuery(data_size_t query_id, data_size_t mo_idx,
                                        const double* all_score, const float* weight) const override {
    const data_size_t start = query_boundaries_[query_id];
    const data_size_t cnt = query_boundaries_[query_id + 1] - query_boundaries_[query_id];
    const double* score = all_score + start;
    const label_t* label = mo_data_[mo_idx].data() + start;
    score_t* lambdas;
    score_t* hessians;
    if (mo_combinator_name_ != "stochastic_label_aggregation") {
        lambdas = multi_gradients_[mo_idx].data() + start;
        hessians = multi_hessians_[mo_idx].data() + start;
    } else {
        lambdas = gradients_ + start;
        hessians = hessians_ + start;
    }

    // Skip groups with too few items.
    if (cnt <= 1) {
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = 0.0f;
        hessians[i] = 0.0f;
      }
      return 0;
    }

    // Turn scores into a probability distribution using Softmax.
    std::vector<double> rho(cnt, 0.0);
    Common::Softmax(score, rho.data(), cnt);

    // An auxiliary buffer of parameters used to form the ground-truth
    // distribution and compute the loss.
    std::vector<double> params(cnt);

    double inv_denominator = 0;
    for (data_size_t i = 0; i < cnt; ++i) {
        double _rd = rands_[query_id].NextFloat() ;
//        std::cout << "label max = " << mo_data_label_max_[mo_idx]  <<
//                  "normalized label = " << label[i] / mo_data_label_max_[mo_idx] <<
//                  "rand = " << _rd << std::endl ;
//        params[i] = Phi(label[i] / mo_data_label_max_[mo_idx] * 10, _rd);
        params[i] = Phi(label[i], _rd);
        inv_denominator += params[i];
//        std::cout << "params = " << params[i] << std::endl;
    }
//    std::cout << "inv_demoninator = " << inv_denominator << std::endl ;
    // sum_labels will always be positive number
//    double max_eps_invd = std::max<double>(kEpsilon, inv_denominator) ;
//    std::cout << "max_eps_invd = " << max_eps_invd << std::endl ;
    inv_denominator = 1. / std::max<double>(kEpsilon, inv_denominator);
//      std::cout << "inv_demoninator = " << inv_denominator << std::endl ;
    // Get the costs
    std::vector<double> cost(cnt, 0.0);
    for (auto i=0; i < cnt; i++) {
      cost[i] -= params[i] * inv_denominator * std::log(rho[i]);
//      std::cout << "cost = " << cost[i] << std::endl ;
    }

    // Approximate gradients and inverse Hessian.
    // First order terms.
    double sum_l1 = 0.0;
    for (data_size_t i = 0; i < cnt; ++i) {
      double term = -params[i] * inv_denominator + rho[i];
      lambdas[i] = static_cast<score_t>(term);
      // Params will now store terms needed to compute second-order terms.
      params[i] = term / (1. - rho[i]);
      sum_l1 += params[i];
    }
    // Second order terms.
    double sum_l2 = 0.0;
    for (data_size_t i = 0; i < cnt; ++i) {
      double term = rho[i] * (sum_l1 - params[i]);
      lambdas[i] += static_cast<score_t>(term);
      // Params will now store terms needed to compute third-order terms.
      params[i] = term / (1. - rho[i]);
      sum_l2 += params[i];
    }
    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] += static_cast<score_t>(rho[i] * (sum_l2 - params[i]));
      hessians[i] = static_cast<score_t>(rho[i] * (1.0 - rho[i]));
    }
    if( weight != nullptr ){
        for (data_size_t i = 0; i < cnt; ++i) {
            lambdas[i] = static_cast<score_t>(lambdas[i] * weight[ start + i] );
            hessians[i] = static_cast<score_t>(hessians[i] * weight[ start + i] );
            cost[i] = cost[i] * weight[ start + i ] ;
        }
    }
    return accumulate(cost.begin(), cost.end(), 0.0);
  }

  double Phi(const label_t l, double g) const {
    //return Common::Pow(2, static_cast<int>(l)) - g;
    return (l + 1) - g;
  }

  const char* GetName() const override { return "rank_xendcg"; }

 private:
  mutable std::vector<Random> rands_;

};

//class ListNet : public RankingObjective {
// public:
//  explicit ListNet(const Config& config) : RankingObjective(config) {}
//
//  explicit ListNet(const std::vector<std::string>& strs)
//      : RankingObjective(strs) {}
//
//  ~ListNet() {}
//
//  void Init(const Metadata& metadata, data_size_t num_data) override {
//    RankingObjective::Init(metadata, num_data);
//  }
//
//  double GetGradientsForOneQuery(data_size_t query_id, data_size_t mo_idx,
//                                        const double* all_score) const override {
//    const data_size_t start = query_boundaries_[query_id];
//    const data_size_t cnt = query_boundaries_[query_id + 1] - query_boundaries_[query_id];
//    const double* score = all_score + start;
//    const label_t* label = mo_data_[mo_idx].data() + start;
//    score_t* lambdas = multi_gradients_[mo_idx].data() + start;
//    score_t* hessians = multi_hessians_[mo_idx].data() + start;
//    // Skip groups with too few items.
//    if (cnt <= 1) {
//      for (data_size_t i = 0; i < cnt; ++i) {
//        lambdas[i] = 0.0f;
//        hessians[i] = 0.0f;
//      }
//      return;
//    }
//
//    // Turn scores into a probability distribution using Softmax.
//    std::vector<double> rho(cnt, 0.0);
//    Common::Softmax(score, rho.data(), cnt);
//
//    std::vector<double> phi(cnt, 0.0);
//    Common::Softmax(label, phi.data(), cnt);
//    // An auxiliary buffer of parameters used to form the ground-truth
//    // distribution and compute the loss.
//    std::vector<double> params(cnt);
//
//    double inv_denominator = 0;
//    for (data_size_t i = 0; i < cnt; ++i) {
//      params[i] = Phi(label[i], rands_[query_id].NextFloat());
//      inv_denominator += params[i];
//    }
//    // sum_labels will always be positive number
//    inv_denominator = 1. / std::max<double>(kEpsilon, inv_denominator);
//
//    // Get the costs
//    std::vector<double> cost(cnt, 0.0);
//    for (auto i=0; i < cnt; i++) {
//      cost[i] += phi[i] * std::log(rho[i]);
//    }
//
//    // Approximate gradients and inverse Hessian.
//    // First order terms.
//    double sum_l1 = 0.0;
//    for (data_size_t i = 0; i < cnt; ++i) {
//      double term = -params[i] * inv_denominator + rho[i];
//      lambdas[i] = static_cast<score_t>(term);
//      // Params will now store terms needed to compute second-order terms.
//      params[i] = term / (1. - rho[i]);
//      sum_l1 += params[i];
//    }
//    // Second order terms.
//    double sum_l2 = 0.0;
//    for (data_size_t i = 0; i < cnt; ++i) {
//      double term = rho[i] * (sum_l1 - params[i]);
//      lambdas[i] += static_cast<score_t>(term);
//      // Params will now store terms needed to compute third-order terms.
//      params[i] = term / (1. - rho[i]);
//      sum_l2 += params[i];
//    }
//    for (data_size_t i = 0; i < cnt; ++i) {
//      lambdas[i] += static_cast<score_t>(rho[i] * (sum_l2 - params[i]));
//      hessians[i] = static_cast<score_t>(rho[i] * (1.0 - rho[i]));
//    }
//    return accumulate(cost.begin(), cost.end(), 0.0);
//  }
//
//  double Phi(const label_t l, double g) const {
//    return Common::Pow(2, static_cast<int>(l)) - g;
//  }
//
//  const char* GetName() const override { return "rank_xendcg"; }
//
// private:
//  mutable std::vector<Random> rands_;
//};


}  // namespace LightGBM
#endif  // LightGBM_OBJECTIVE_RANK_OBJECTIVE_HPP_
