# Multi-Label Learning to Rank through Multi-objective Optimization with LightGBM

TODO: Fill this README out!

Be sure to:

* Change the title in this README
* Edit your repository description on GitHub

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

### Sample config
```

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

