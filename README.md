# Hybrid Models for Learning to Branch

Prateek Gupta, Maxime Gasse, Elias B. Khalil, M. Pawan Kumar, Andrea Lodi, Yoshua Bengio

This is the official implementation of our NeurIPS 2020 [paper](https://arxiv.org/abs/2006.15212)

## Installation

This work is built upon [learn2branch](https://github.com/ds4dm/learn2branch), which proposes Graph Neural Network for learning to branch.
We use it as a [git submodule](https://www.git-scm.com/book/en/v2/Git-Tools-Submodules).
Follow installation instructions of [learn2branch](https://github.com/ds4dm/learn2branch/blob/master/INSTALL.md) to install [SCIP](https://www.scipopt.org/) and PySCIPOpt.

**UPDATE** As pointed out in [PR#2](https://github.com/pg2455/Hybrid-learn2branch/issues/2), a function needs to be added in the [class Column](https://github.com/scipopt/PySCIPOpt/blob/v3.0.4/src/pyscipopt/scip.pyx#L314) of PySCIPOpt. Please add the following function there before installation of PySCIPOpt-
```cpython
def getIndex(self):
  return SCIPcolGetIndex(self.scip_col)
```

Following python dependencies were used to run the code in this repository
```
torch==1.4.0.dev20191031
scipy==1.5.2
numpy==1.18.1
networkx==2.4
Cython==0.29.13
PySCIPOpt==2.1.5
scikit-learn==0.20.2
```

To setup this repo, follow
```bash
git clone https://github.com/pg2455/Hybrid-learn2branch.git
cd Hybrid-learn2branch
git submodule update --init
```

## How to run it?
In the instructions below we assumed that a bash variable `PROBLEM` exists. For example,
```bash
PROBLEM=setcover
```
Below instructions assume access to `data/` folder in the repo. Please look at the argument flags in each of the script to use another folder.

### Generate Instances
```bash
# generate instances
python learn2branch/01_generate_instances.py $PROBLEM
```

### Generate dataset
```bash
# generate dataset
python 02_generate_dataset.py $PROBLEM
```
### Train models

```bash
# GNN
python 03_train_gcnn_torch.py $PROBLEM # PyTorch version of learn2branch GNN

# COMP
python learn2branch/03_train_competitor.py $PROBLEM -m extratrees --hybrid_data_structure
python learn2branch/03_train_competitor.py $PROBLEM -m svmrank --hybrid_data_structure
python learn2branch/03_train_competitor.py $PROBLEM -m lambdamart --hybrid_data_structure

# MLP
python 03_train_mlp.py $PROBLEM

# Hybrid models
python 03_train_hybrid.py $PROBLEM -m concat --no_e2e # (pre)
python 03_train_hybrid.py $PROBLEM -m concat --no_e2e --distilled # (pre + KD)

python 03_train_hybrid.py $PROBLEM -m film --no_e2e # (pre)
python 03_train_hybrid.py $PROBLEM -m film --no_e2e --distilled # (pre + KD)

## CONCAT
python 03_train_hybrid.py $PROBLEM -m concat # (e2e)
python 03_train_hybrid.py $PROBLEM -m concat --distilled # (e2e + KD)

## FILM
python 03_train_hybrid.py $PROBLEM -m film # (e2e)
python 03_train_hybrid.py $PROBLEM -m film --distilled # (e2e + KD)

## HybridSVM
python 03_train_hybrid.py $PROBLEM -m hybridsvm # (e2e)
python 03_train_hybrid.py $PROBLEM -m hybridsvm --distilled  # (e2e + KD)

## HybridSVM-FiLM
python 03_train_hybrid.py $PROBLEM -m hybridsvm-film # (e2e)
python 03_train_hybrid.py $PROBLEM -m hybridsvm-film --distilled  # (e2e + KD)

# Auxiliary task (AT)
python 03_train_hybrid.py $PROBLEM -m film --at ED --beta_at 0.001 # (e2e + AT)
python 03_train_hybrid.py $PROBLEM -m film --distilled --at ED --beta_at 0.001 # (e2e + KD + AT)

# l2 regularization
python 03_train_hybrid.py $PROBLEM -m film --at ED --beta_at 0.001 --l2 0.001
```

### Test model performance
```bash
# test models

python 04_test_gcnn_torch.py $PROBLEM # GNN
python 04_test_mlp.py $PROBLEM # MLP

# ml-comp (COMP is the one with best accuracy)
python learn2branch/04_test.py $PROBLEM --no_gnn --ml_comp_brancher svmrank_khalil --hybrid_data_structure
python learn2branch/04_test.py $PROBLEM --no_gnn --ml_comp_brancher lambdamark_khalil --hybrid_data_structure
python learn2branch/04_test.py $PROBLEM --no_gnn --ml_comp_brancher extratrees_gcnn_agg --hybrid_data_structure

# Hybrid models
python 04_test_hybrid.py $PROBLEM # tests all available hybrid models in trained_models/$PROBLEM
```

### Evaluate models
```bash
# evaluate models

python 05_evaluate_gcnn_torch.py $PROBLEM -g -1 # GNN-CPU
python 05_evaluate_gcnn_torch.py $PROBLEM -g 0 # GNN-GPU
python 05_evaluate_mlp.py $PROBLEM -g -1

# COMP
python learn2branch/05_evaluate.py $PROBLEM --ml_comp_brancher use_best_performing_ml_competitor_folder_name --time_limit 2700 --no_gnn --hybrid_data_structure -g -1


# FiLM
python 05_evaluate_hybrid.py $PROBLEM -g -1 --model_string use_best_performing_model_folder_name


# internal branchers
python learn2branch/05_evaluate.py $PROBLEM --internal_brancher pscost --time_limit 2700 --no_gnn -g -1 --hybrid_data_structure # PB
python learn2branch/05_evaluate.py $PROBLEM --internal_brancher relpscost --time_limit 2700 --no_gnn  -g -1 --hybrid_data_structure # RPB
python learn2branch/05_evaluate.py $PROBLEM --internal_brancher fullstrong --time_limit 2700 --no_gnn  -g -1 --hybrid_data_structure # FSB
```

Follow instructions [here](https://github.com/pg2455/Hybrid-learn2branch/blob/master/RESULTS.md) to reproduce the evaluation results (Table 4).

## Citation
Please cite our paper if you use this code in your work.
```
@inproceedings{conf/nips/Gupta20hybrid,
  title={Hybrid Models for Learning to Branch},
  author={Gupta, Prateek and Gasse, Maxime and Khalil, Elias B and Kumar, M Pawan and Lodi, Andrea and Bengio, Yoshua},
  booktitle={Advances in Neural Information Processing Systems 33},
  year={2020}
}
```

## Questions / Bugs
Please feel free to submit a Github issue if you have any questions or find any bugs. We do not guarantee any support, but will do our best if we can help.
