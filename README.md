# Multi-Label Learning to Rank through Multi-objective Optimization with LightGBM

## Introduction
MO-LightGBM is a gradient boosting (GBT, GBDT, GBRT, GBM or MART) framework based on decision tree algorithms, used for multi-label learning to rank tasks. It is based on [LightGBM](https://github.com/microsoft/LightGBM).
## Installation
1. Install blas and lapack, 
    1. In mac:
       ```
       $brew install openblas
       $brew install lapack
       ```
    2. In Debian-based systems like Ubuntu:
       ```
       $sudo apt install libblas-dev liblapack-dev
       ```
    3. In Red Hat systems like CentOS and Fedora:
       ```
       $sudo yum install lapack-devel blas-devel
       ```

2. from the scs root, build the package, install cmake if needed
   
    ```
    cd LightGBM/external_libs/scs/
    rm -r build
    mkdir build
    cd build
    cmake ..
    make -j$(nproc)
    cd ../../../../
    ```
4. from the ecos root, build the package
   
    ```
    cd LightGBM/external_libs/ecos.2.0.8/
    rm -r build
    mkdir build
    cd build
    cmake ..
    make -j$(nproc)
    cd ../../../../
    ```
5. Now build LightGBM
   
   ```
   cd LightGBM
   rm -r build
   mkdir build
   cd build
   cmake ..
   make -j$(nproc)
   cd ../../
   ```
## How to Use
### Assumptions about Dataset
1. The dataset should be in csv or tsv format, not in the libsvm format. 
2. The csv/tsv file should have header that names each column


### General process for bi-objective experiments
1. Prepare the `<dataset_name>_config.yml`
2. Generate configure files for baseline experiment

   `python baselines.py <dataset_name>_config.yml`
3. Go the `baseline_results_<dataset_name>` folder and run the experiments

   ```
   cd baseline_results_<dataset_name>
   sh run_experiment.sh
   ```
4. Generate configure files for biobjective experiments for preference based methods

   `python biobjectives.py <dataset_name>_config.yml`
5. Go to the `biobjective_results_<dataset_name>` folder, and, either run all the experiments or run for individual combinator

   ```
   cd biobjective_results_<dataset_name>
   sh run_experiment.sh   # will run for the all combinators
   sh epo_search_run_experiment.sh     # will run only for epo_search  
   ```
6. Generate configure files for biobjective experiments for constraints based methods

   `python biobjectives_ec.py <dataset_name>_config.yml`
7. Go to the `biobjective_results_<dataset_name>` folder, and, either run all the experiments or run for individual combinator

   ```
   cd biobjective_results_<dataset_name>
   sh ec_run_experiment.sh   # will run for the all combinators
   sh epo_search_run_experiment.sh     # will 
8. Run plotting script
   1. plot bound and ray in one figure
       ```
      python plot_biobjectives.py <dataset_name>_config.yml
       ```
   2. plot only ray
       ```
        python plot_biobjectives_results.py <dataset_name>_config.yml
       ```
   3. plot only bound
       ```
       python plot_biobjectives_ec.py <dataset_name>_config.yml
       ```
9. Run plotting script for evolution (under construction using plot_biobjective_results.py)

   `python plot_biobjectives_evolution.py <dataset_name>_config.yml`

### General process for multi-objective experiments

### Sample Config for bi-objective Experiment
```
dataset:
  name: istella
  train_file: "datasets/istella-letor/full/train.tsv"
  valid_file: "datasets/istella-letor/full/test.tsv"
  query_column: "1" # query group id column
  main_label: "0" # main objective
  all_labels: ['11', '194', '203', '214', '0']    # candidate labels for objectives
  bilabels_idx: [[0, 1], [0, -1], [2, 3], [2, -1]] # combination of labels from all_labels
  ignore_columns: ['11', '194', '203', '214', '0']  # all_lables + some extra feature columns that needs to be ignored

lightgbm_parameters:
  objective_type: lambdarank   # current supported options: lambdarank and ranknet
  num_iterations: '1000'   # all numbers should be in quotes
  num_thread: '47'
  ndcg_eval_at: '5,30'

lightgbm_path: "../LightGBM/lightgbm"
sample_lightgbm_config: "sample_config.conf"
num_tradeoffs: 5 # number of preference verctors for objectives

mg_combinators:
  preference_based:
    - linear_scalarization
    - stochastic_label_aggregation
    - chebychev_scalarization
    - epo_search
    - wc_mgda
  constraint_based:
    - e_constraint
    - ec_mgda

plotting:
  label_legends:
    "11": "Relevance 1"
    "194": "Relevance 2"
    "203": "Relevance 3"
    "214": "Relevance 4"
    "0": "Relevance 5"

  to_track:
    loss: "Training-loss"     # options: Validation-loss
    ndcg: "Validation-ndcg@5" # options: Training-ndcg@k or Validation-ndcg@k, k can be any number in the ndcg_eval_at

  tracker_titles:
    loss: "Training Cost"
    ndcg: "Validation NDCG@5"

  snapshots_at: [10, 20, 40, 80, 160, 240, 480, 600, 800, 1000] # snapshots of intermidate results at speficif interation

  combinator_legends:
    linear_scalarization: "LS"
    stochastic_label_aggregation: "SLA"
    chebychev_scalarization: "CS"
    epo_search: "EPO-Search"
    wc_mgda: "WC-MGDA"
    e_constraint: '$\epsilon-$Constraint'
    ec_mgda: "EC-MGDA"

  combinator_markers:
    linear_scalarization: "s"
    stochastic_label_aggregation: "P"
    chebychev_scalarization: "^"
    epo_search: "*"
    wc_mgda: "X"
    e_constraint: '$\clubsuit$'
    ec_mgda: '$\spadesuit$'

  combinator_marker_sizes:
    linear_scalarization: 18
    stochastic_label_aggregation: 20
    chebychev_scalarization: 25
    epo_search: 40
    wc_mgda: 20
    e_constraint: 30
    ec_mgda: 25
```

### Sample Command
```
#dataset=mslr
#dataset=yahoo
dataset=istella
root=./moro/
folder=${root}/test_istella/
cd ${folder}
chmod 777 *
python baselines.py ${dataset}_config.yml
cd baseline_results_${dataset}
sh run_experiment.sh
cd ..

python biobjectives.py ${dataset}_config.yml
cd biobjective_results_${dataset}
sh run_experiment.sh 
cd ..

python biobjectives_ec.py ${dataset}_config.yml
cd biobjective_results_${dataset}
sh ec_run_experiment.sh 
cd ..

python plot_biobjectives.py ${dataset}_config.yml
```



## Citation 
If you use this work, please consider citing the papers:

[Multi-Label Learning to Rank through Multi-Objective
Optimization](https://dl.acm.org/doi/pdf/10.1145/3580305.3599870)

```
@article{chaoshengmo-lightgbm2025,
  title={MO-LightGBM: A Package for Multi-Label Learning to Rank in LightGBM},
  author={Dong, Chaosheng and Momma, Michinari},
  journal={arXiv preprint arXiv:2412.10418},
  year={2025}
}

@inproceedings{mahapatra2023multi,
  title={Multi-label learning to rank through multi-objective optimization},
  author={Mahapatra, Debabrata and Dong, Chaosheng and Chen, Yetian and Momma, Michinari},
  booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2023}
}

```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

