##

There are several combinations of training protocols that one can use to train the hybrid models.
We report several empirical insights in the supplementary material.

Here, we focus on the procedure to obtain the models that are used for evaluation (Table 4).
Following set of commands will give the required models.

```bash
# generate instances
python learn2branch/01_generate_instances.py $PROBLEM

# generate dataset
python 02_generate_dataset.py $PROBLEM

# Train models
for seed in 0 1 2
do
  # train GNN
  python 03_train_gcnn_torch.py $PROBLEM -s $seed

  # train film
  python 03_train_hybrid.py $PROBLEM -m film -s $seed # (A)
  python 03_train_hybrid.py $PROBLEM -m film --distilled  -s $seed # (B)

  # train COMP
  python learn2branch/03_train_competitor.py $PROBLEM -m extratrees --hybrid_data_structure -s $seed
  python learn2branch/03_train_competitor.py $PROBLEM -m svmrank --hybrid_data_structure -s $seed
  python learn2branch/03_train_competitor.py $PROBLEM -m lambdamart --hybrid_data_structure -s $seed
done

# Train more
# Add AT to the best performing film model out of (A) and (B) (on validation set). Assuming that (A) is the best ...
for seed in 0 1 2
do
  for beta in 0.1 0.01 0.001 0.0001
  do
    python 03_train_hybrid.py $PROBLEM -m film -s $seed --at ED --beta_at $beta
    python 03_train_hybrid.py $PROBLEM -m film -s $seed --at MHE --beta_at $beta
  done
done

# At this point, we call the best performing model (on validation set) as C
# for independent set only; we regularize film parameters
if [ "$PROBLEM" == "indset" ];; then
  for l2 in 1.0 0.1 0.01
  do
    # assuming MHE at 0.1 gives C for independent set
    python 03_train_hybrid.py $PROBLEM -s $seed -m film --at MHE --beta_at 0.1 --l2 $l2
    python 03_train_gcnn_torch.py $PROBLEM -s $seed --l2 $l2
  done

  # test on medium validation set
  python 04_test_gcnn_torch.py $PROBLEM  --test_path data/samples/indset/1000_4/mediumvalid
  python 04_test_hybrid.py $PROBLEM  --model_string film --test_path data/samples/indset/1000_4/mediumvalid
fi

# Test models

python 04_test_gcnn_torch.py $PROBLEM

python learn2branch/04_test.py $PROBLEM --no_gnn --ml_comp_brancher svmrank_khalil --hybrid_data_structure
python learn2branch/04_test.py $PROBLEM --no_gnn --ml_comp_brancher lambdamark_khalil --hybrid_data_structure
python learn2branch/04_test.py $PROBLEM --no_gnn --ml_comp_brancher extratrees_gcnn_agg --hybrid_data_structure

python 04_test_hybrid.py $PROBLEM --model_string film


# Call the best performing film model after this test as C.
python 04_test_hybrid.py $PROBLEM --model_string film

# evaluate using C

# GNN-CPU (replace baseline_torch with best l2 regularized model for independent set)
python 05_evaluate_gcnn_torch.py $PROBLEM -g -1 --model_name baseline_torch # GNN-CPU

# COMP
python learn2branch/05_evaluate.py $PROBLEM --ml_comp_brancher use_best_performing_ml_competitor_folder_name --time_limit 2700 --no_gnn --hybrid_data_structure -g -1

# FiLM (use the name of the folder for C)
python 05_evaluate_hybrid.py $PROBLEM -g -1 --model_string use_best_performing_model_folder_name_C

# internal branchers
python learn2branch/05_evaluate.py $PROBLEM --internal_brancher pscost --time_limit 2700 --no_gnn -g -1 --hybrid_data_structure # PB
python learn2branch/05_evaluate.py $PROBLEM --internal_brancher relpscost --time_limit 2700 --no_gnn  -g -1 --hybrid_data_structure # RPB
python learn2branch/05_evaluate.py $PROBLEM --internal_brancher fullstrong --time_limit 2700 --no_gnn  -g -1 --hybrid_data_structure # FSB




```
