/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_METRIC_RANK_METRIC_HPP_
#define LIGHTGBM_METRIC_RANK_METRIC_HPP_

#include <LightGBM/metric.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>
#include <LightGBM/utils/openmp_wrapper.h>

#include <string>
#include <sstream>
#include <vector>


namespace LightGBM {

class RankMetric: public Metric {
 public:
  explicit RankMetric(const Config& config)
          : seed_(config.objective_seed) {}

  explicit RankMetric(const std::vector<std::string>&) : seed_(0) {}

  ~RankMetric(){}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    std::string data_filename = metadata.data_filename();
    num_queries_ = metadata.num_queries();
    if (data_filenames_.count(data_filename) == 0) {
      DCGCalculator::CheckMetadata(metadata, num_queries_);
    }
    query_boundaries_ = metadata.query_boundaries();
    if (query_boundaries_ == nullptr) {
      Log::Fatal("The NDCG metric requires query information");
    }
    num_data_ = num_data;
    num_mo_ = metadata.num_mo();
    if (num_mo_ == 1) {
      label_ = metadata.label();      // get label
      if (data_filenames_.count(data_filename) == 0) {
        DCGCalculator::CheckLabel(label_, num_data_);
      }
      query_weights_ = metadata.query_weights();   // get query weights
      init_sum_query_weight(query_weights_, sum_query_weights_); // cache sum_query_weight
      weights_ = metadata.weights();
    } else {      // get MO info
      mo_data_ = metadata.mo_data();
      mo_query_weights_.resize(num_mo_);
      mo_sum_query_weights_.resize(num_mo_, 0.0f);
      mo_weights_.resize(num_mo_);
      for (auto mo_idx = 0; mo_idx < num_mo_; ++mo_idx) {
        if (data_filenames_.count(data_filename) == 0) {
          DCGCalculator::CheckLabel(mo_data_[mo_idx].data(), num_data_);
        }
        mo_query_weights_[mo_idx] = metadata.mo_query_weights(mo_idx);
        init_sum_query_weight(mo_query_weights_[mo_idx], mo_sum_query_weights_[mo_idx]);
        mo_weights_[mo_idx] = metadata.mo_weights(mo_idx) ;
      }
    }
    data_filenames_.insert(data_filename);
  }

  inline void init_sum_query_weight(const float* query_weight, double &sum_query_weight) {
    if (query_weight == nullptr) {
      sum_query_weight = static_cast<double>(num_queries_);
    } else {
      sum_query_weight = 0.0f;
      for (data_size_t i = 0; i < num_queries_; ++i) {
        sum_query_weight += query_weight[i];
      }
    }
  }

  const std::vector<std::string>& GetName() const override {
    return name_;
  }

  data_size_t num_mo() const override {
    return num_mo_;
  }
 protected:
  int seed_;
  static std::unordered_set<std::string> data_filenames_;
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const label_t* label_;
  /*! \brief Name of test set */
  std::vector<std::string> name_;
  /*! \brief Query boundaries information */
  const data_size_t* query_boundaries_;
  /*! \brief Number of queries */
  data_size_t num_queries_;
  /*! \brief Weights of queries */
  const label_t* query_weights_;
  /*! \brief Sum weights of queries */
  double sum_query_weights_;
  /*! \brief Multi-objective weights of queries */
  std::vector<const float*> mo_query_weights_;
  /*! \brief Multi-objective sum weights of queries */
  std::vector<double> mo_sum_query_weights_;
  /*! \brief Number of multi-objective ranking */
  data_size_t num_mo_;
  /*! \brief multi-objective ranking data */
  const std::vector<float>* mo_data_;

  /*! \brief Weights of documents */
  const label_t* weights_;
  /*! \brief Multi-objective weights of documents */
  std::vector<const float*> mo_weights_;


};

std::unordered_set<std::string> RankMetric::data_filenames_;

class NDCGMetric: public RankMetric {
 public:
  explicit NDCGMetric(const Config& config) : RankMetric(config) {
    // get eval position
    eval_at_ = config.eval_at;
    auto label_gain = config.label_gain;
    DCGCalculator::DefaultEvalAt(&eval_at_);
    DCGCalculator::DefaultLabelGain(&label_gain);
    // initialize DCG calculator
    // DCGCalculator::Init(label_gain);
    DCGCalculator::Init(label_gain, config.use_quicksort_ndcg);
    // whether ignore zero max_dcg
    ignore_zero_max_dcg_ = config.ignore_zero_max_dcg;
    mo_inverse_max_dcgs_.clear();
  }

  ~NDCGMetric() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    RankMetric::Init(metadata, num_data);
    for (auto k : eval_at_) {
      name_.emplace_back(std::string("ndcg@") + std::to_string(k));
    }
    // compute inverse max DCG for each of multi-objective
    if (num_mo_ > 1) {
      mo_inverse_max_dcgs_.resize(num_mo_);
      for (auto mo_idx = 0; mo_idx < num_mo_; ++mo_idx) {
        init_inv_max_dcgs(mo_inverse_max_dcgs_[mo_idx], mo_data_[mo_idx].data());
      }
    } else {
      init_inv_max_dcgs(inverse_max_dcgs_, label_);
    }
  }

  inline void init_inv_max_dcgs(std::vector<std::vector<double>> &inv_mdcgs, const float * lbl) {
    inv_mdcgs.resize(num_queries_);
    #pragma omp parallel for schedule(static)
    for (data_size_t i = 0; i < num_queries_; ++i) {
      inv_mdcgs[i].resize(eval_at_.size(), 0.0f);
      DCGCalculator::CalMaxDCG(eval_at_, lbl + query_boundaries_[i],
                               query_boundaries_[i + 1] - query_boundaries_[i],
                               &inv_mdcgs[i]);
      for (size_t j = 0; j < inv_mdcgs[i].size(); ++j) {
        if (inv_mdcgs[i][j] > 0.0f) {
          inv_mdcgs[i][j] = 1.0f / inv_mdcgs[i][j];
        } else {
          // marking negative for all negative querys.
          // if one meet this query, it's ndcg will be set as -1.
          inv_mdcgs[i][j] = -1.0f;
        }
      }
    }
  }

  double factor_to_bigger_better() const override {
    return 1.0f;
  }

  std::vector<double> Eval(const double* score, const ObjectiveFunction*) const override {
    return Eval_(score, nullptr, inverse_max_dcgs_, label_, query_weights_, sum_query_weights_);
  }

  std::vector<std::vector<double>> MOEval(const double* score, const ObjectiveFunction*) const override {
    std::vector<std::vector<double>> results(eval_at_.size());
    for (size_t i = 0; i < eval_at_.size(); i++) {
      results[i].resize(num_mo_);
    }
    for (data_size_t mo_idx = 0; mo_idx < num_mo_; mo_idx++){
//        std::cout << "mo_idx = " << mo_idx << std::endl ;
      auto res_mo_idx = Eval_(score, nullptr, mo_inverse_max_dcgs_[mo_idx], mo_data_[mo_idx].data(),
                              mo_query_weights_[mo_idx], mo_sum_query_weights_[mo_idx]);
      for (size_t i = 0; i < eval_at_.size(); i++){
        results[i][mo_idx] = res_mo_idx[i];
      }
    }
    return results;
  }

  inline std::vector<double> Eval_(const double* score, const ObjectiveFunction*,
                                 std::vector<std::vector<double>> inv_mdcgs, const float* lbl,
                                 const float* query_weight, double sum_query_weight) const {
    int num_threads = OMP_NUM_THREADS();
    // some buffers for multi-threading sum up
    std::vector<std::vector<double>> result_buffer_;
    for (int i = 0; i < num_threads; ++i) {
      result_buffer_.emplace_back(eval_at_.size(), 0.0f);
    }
    std::vector<double> tmp_dcg(eval_at_.size(), 0.0f);
    if (query_weight == nullptr) {
      #pragma omp parallel for schedule(static) firstprivate(tmp_dcg)
      for (data_size_t i = 0; i < num_queries_; ++i) {
        const int tid = omp_get_thread_num();
        // if all doc in this query are all negative, let its NDCG=1
        if (inv_mdcgs[i][0] <= 0.0f) {
          if (ignore_zero_max_dcg_) continue;
          for (size_t j = 0; j < eval_at_.size(); ++j) {
            result_buffer_[tid][j] += 1.0f;
          }
        } else {
          // calculate DCG
          DCGCalculator::CalDCG(eval_at_, lbl + query_boundaries_[i],
                                score + query_boundaries_[i],
                                query_boundaries_[i + 1] - query_boundaries_[i], &tmp_dcg);
          // calculate NDCG
          for (size_t j = 0; j < eval_at_.size(); ++j) {
            // if (i < 10) {
            //   std::cout<<"DCG(";
            //   for (auto k = query_boundaries_[i]; k < query_boundaries_[i] + 5; k++) {
            //     std::cout<< score[k] << ",";
            //   }
            //   std::cout<<"...) = "<< tmp_dcg[j] <<" * " << inv_mdcgs[i][j]<< "\n";
            // }
            result_buffer_[tid][j] += tmp_dcg[j] * inv_mdcgs[i][j];
          }
        }
      }
    } else {
      #pragma omp parallel for schedule(static) firstprivate(tmp_dcg)
      for (data_size_t i = 0; i < num_queries_; ++i) {
        //std::cout << "query_weight = " << query_weight[i] << std::endl ;
        const int tid = omp_get_thread_num();
        // if all doc in this query are all negative, let its NDCG=1
        if (inv_mdcgs[i][0] <= 0.0f) {
          if (ignore_zero_max_dcg_) continue;
          for (size_t j = 0; j < eval_at_.size(); ++j) {
            result_buffer_[tid][j] += 1.0f;
          }
        } else {
          // calculate DCG
          DCGCalculator::CalDCG(eval_at_, lbl + query_boundaries_[i],
                                score + query_boundaries_[i],
                                query_boundaries_[i + 1] - query_boundaries_[i], &tmp_dcg);
          // calculate NDCG
          for (size_t j = 0; j < eval_at_.size(); ++j) {
            result_buffer_[tid][j] += tmp_dcg[j] * inv_mdcgs[i][j] * query_weight[i];
          }
        }
      }
    }
    // Get final average NDCG
    //std::cout << "num_queries = " << num_queries_ << std::endl ;

    std::vector<double> result(eval_at_.size(), 0.0f);
    for (size_t j = 0; j < result.size(); ++j) {
      for (int i = 0; i < num_threads; ++i) {
        result[j] += result_buffer_[i][j];
      }
      result[j] /= sum_query_weight;
    }
    return result;
  }

 private:
  /*! \brief Evaluate position of NDCG */
  std::vector<data_size_t> eval_at_;
  /*! \brief Cache the inverse max dcg for all queries */
  std::vector<std::vector<double>> inverse_max_dcgs_;
  /*! \brief If true, ignore the query group whose max_dcg is 0
   *  So if a column is sparse, ndcg will not be very large */
  bool ignore_zero_max_dcg_;
  /*! \brief cached inverse DCD for mo */
  std::vector<std::vector<std::vector<double>>> mo_inverse_max_dcgs_;
};

class RankLossMetric: public RankMetric {
 public:
  RankLossMetric(const Config& config) : RankMetric(config) {
    objective_type_ = config.objective;
  }

  ~RankLossMetric() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
      RankMetric::Init(metadata, num_data);
      name_.emplace_back(std::string("loss"));
  }

  std::vector<double> Eval(const double* score, const ObjectiveFunction* obj
                           //, const float* weight
  ) const override {
      std::vector<double> q_cost(num_queries_);
      #pragma omp parallel for schedule(guided)
      for (data_size_t i = 0; i < num_queries_; i++) {
        q_cost[i] = GetCostForOneQuery(i, -1, score, obj
                                       //, weight
                                       );
      }
      double cost = accumulate(q_cost.begin(), q_cost.end(), 0.0) / static_cast<double>(num_queries_);
      return std::vector<double>{cost};
    }

  std::vector<std::vector<double>> MOEval(const double* score, const ObjectiveFunction* obj
                                          // , const float* weight
  ) const override {
    std::vector<double> all_costs(num_mo_);
    std::vector<double> q_cost(num_queries_);
    for (data_size_t mo_idx = 0; mo_idx < num_mo_; mo_idx++) {
      #pragma omp parallel for schedule(guided)
      for (data_size_t i = 0; i < num_queries_; ++i) {
        q_cost[i] = GetCostForOneQuery(i, mo_idx, score, obj
                                       //, weight
        );
      }
      all_costs[mo_idx] = accumulate(q_cost.begin(), q_cost.end(), 0.0) / static_cast<double>(num_queries_);
    }
    return std::vector<std::vector<double>>{all_costs};
  }

  virtual double GetCostForOneQuery(data_size_t query_id, int mo_idx, const double* all_score,
                                                    const ObjectiveFunction* obj
                                                    //, const float* weight
                                                    ) const = 0;

  double factor_to_bigger_better() const override {
    return -1.0f;
  }

  std::string objective_type_;
};

class LambdarankNDCGLossMetric: public RankLossMetric {
 public:
  explicit LambdarankNDCGLossMetric(const Config& config)
        : RankLossMetric(config),
          norm_(config.lambdarank_norm),
          truncation_level_(config.lambdarank_truncation_level) {
    use_quicksort_ndcg_ = config.use_quicksort_ndcg;
    label_gain_ = config.label_gain;
    // initialize DCG calculator
    DCGCalculator::DefaultLabelGain(&label_gain_);
    DCGCalculator::Init(label_gain_, use_quicksort_ndcg_);
    inverse_max_dcgs_.clear();
    mo_inverse_max_dcgs_.clear();
  }

  ~LambdarankNDCGLossMetric() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    RankLossMetric::Init(metadata, num_data);
    // compute inverse max DCG  for each of multi-objective
    if (num_mo_ > 1) {
      mo_inverse_max_dcgs_.resize(num_mo_);
      for (auto mo_idx = 0; mo_idx < num_mo_; ++mo_idx) {
        init_inv_max_dcgs(mo_inverse_max_dcgs_[mo_idx], mo_data_[mo_idx].data());
      }
    } else {
      init_inv_max_dcgs(inverse_max_dcgs_, label_);
    }
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

  double GetCostForOneQuery(data_size_t query_id, int mo_idx, const double* all_score,
                                                  const ObjectiveFunction* obj
                                                  //, const float* weight
                                                  ) const override {
    const data_size_t start = query_boundaries_[query_id];
    const data_size_t cnt = query_boundaries_[query_id + 1] - query_boundaries_[query_id];
    const double* score = all_score + start;
    const label_t* label = (mo_idx < 0) ?  label_ + start : mo_data_[mo_idx].data() + start;

    // get max DCG on current query
    const double inverse_max_dcg = (mo_idx < 0) ? inverse_max_dcgs_[query_id] : mo_inverse_max_dcgs_[mo_idx][query_id];

    // get sorted indices for scores
    std::vector<data_size_t> sorted_idx(cnt);
    for (data_size_t i = 0; i < cnt; ++i) {
      sorted_idx[i] = i;
    }
    std::stable_sort(sorted_idx.begin(), sorted_idx.end(),
                     [score](data_size_t a, data_size_t b) { return score[a] > score[b]; });
    // get best and worst score
    const double best_score = score[sorted_idx[0]];
    data_size_t worst_idx = cnt - 1;
    if (worst_idx > 0 && score[sorted_idx[worst_idx]] == kMinScore) {
      worst_idx -= 1;
    }
    const double worst_score = score[sorted_idx[worst_idx]];
    std::vector<double> cost(cnt, 0.0);
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
        double sigmoid = 1.0, neg_delta_score = -delta_score;
        obj->ConvertOutput(&neg_delta_score, &sigmoid);
        cost[high] += delta_pair_NDCG * std::log(1.0 / sigmoid);
      }
    }

    if( mo_weights_[mo_idx] != nullptr ){
        for (data_size_t i = 0; i < cnt; ++i) {
            cost[i] = cost[i] * mo_weights_[ mo_idx ][ start + i] ;
        }
    }

    return accumulate(cost.begin(), cost.end(), 0.0);
  }

 private:
  /*! \brief Gains for labels */
  std::vector<double> label_gain_;
  /*! \brief Normalize the lambdas or not */
  bool norm_;
  /*! \brief Truncation position for max DCG */
  int truncation_level_;
  /*! \brief Cache inverse max DCG, speed up calculation */
  std::vector<double> inverse_max_dcgs_;
  /*! \brief cached inverse DCD for mo */
  std::vector<std::vector<double>> mo_inverse_max_dcgs_;
  /*! \brief Indicate current iteration */
  bool use_quicksort_ndcg_;
};

class RankXeNDGCLossMetric: public RankLossMetric {
    public:
    explicit RankXeNDGCLossMetric(const Config& config)
        : RankLossMetric(config) {}

    ~RankXeNDGCLossMetric() {}
    void Init(const Metadata& metadata, data_size_t num_data) override {
        RankLossMetric::Init(metadata, num_data);
        for (data_size_t i = 0; i < num_queries_; ++i) {
            rands_.emplace_back(seed_ + i);
        }
    }
    double GetCostForOneQuery(data_size_t query_id, int mo_idx, const double* all_score,
                              const ObjectiveFunction* obj
                              // , const float* weight
                              ) const override {
        const data_size_t start = query_boundaries_[query_id];
        const data_size_t cnt = query_boundaries_[query_id + 1] - query_boundaries_[query_id];
        const double* score = all_score + start;
        const label_t* label = mo_data_[mo_idx].data() + start;

        // Turn scores into a probability distribution using Softmax.
        std::vector<double> rho(cnt, 0.0);
        Common::Softmax(score, rho.data(), cnt);

        // An auxiliary buffer of parameters used to form the ground-truth
        // distribution and compute the loss.
        std::vector<double> params(cnt);

        double inv_denominator = 0;
        for (data_size_t i = 0; i < cnt; ++i) {
                params[i] = Phi(label[i], rands_[query_id].NextFloat());
                inv_denominator += params[i];
        }
        // sum_labels will always be positive number
        inv_denominator = 1. / std::max<double>(kEpsilon, inv_denominator);
        // Get the costs
        std::vector<double> cost(cnt, 0.0);
        for (auto i=0; i < cnt; i++) {
            cost[i] -= params[i] * inv_denominator * std::log(rho[i]);
        }

        if( mo_weights_[mo_idx] != nullptr ){
            for (data_size_t i = 0; i < cnt; ++i) {
                cost[i] = cost[i] * mo_weights_[ mo_idx ][ start + i] ;
            }
        }
        return accumulate(cost.begin(), cost.end(), 0.0) ;
    }

    double Phi(const label_t l, double g) const {
        //return Common::Pow(2, static_cast<int>(l)) - g;
        return (l + 1) - g;
    }

    private:
        mutable std::vector<Random> rands_;

};



}  // namespace LightGBM

#endif   // LightGBM_METRIC_RANK_METRIC_HPP_
