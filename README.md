# Hybrid-learn2branch


```bash
git clone repo
cd repo
git submodule update --init
```


```bash
PROBLEM=setcover

# generate instances
python learn2branch/01_generate_instances.py $PROBLEM

# generate dataset
python 02_generate_dataset.py $PROBLEM

# train models

# GCNN
python 03_train_gcnn_torch.py $PROBLEM # PyTorch version of learn2branch GCNN

# COMP
## Note: use symlink inside learn2branch/ to point to data/ in the current directory
python learn2branch/03_train_competitor.py $PROBLEM -m extratrees
python learn2branch/03_train_competitor.py $PROBLEM -m svmrank
python learn2branch/03_train_competitor.py $PROBLEM -m lambdamart

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

# AT

# l2 regularization


# test models
python 04_test_gcnn_torch.py $PROBLEM


# evaluate models

python 05_evaluate_gcnn_torch.py $PROBLEM

```
