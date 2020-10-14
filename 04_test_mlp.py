import os
import sys
import importlib
import argparse
import csv
import numpy as np
import time
import pickle
import pathlib
import gzip

import tensorflow as tf
import torch

import utilities

from utilities_mlp import MLPDataset as Dataset
from utilities_mlp import load_batch

def process(model, dataloader, top_k):
    """
    Executes only a forward pass of model over the dataset and computes accuracy

    Parameters
    ----------
    model : model.BaseModel
        A base model, which may contain some model.PreNormLayer layers.
    dataloader : torch.utils.data.DataLoader
        Dataset to use for training the model.
    top_k : list
        list of `k` (int) to estimate for accuracy using these many candidates

    Return
    ------
    mean_kacc : np.array
        computed accuracy for `top_k` candidates
    """

    mean_kacc = np.zeros(len(top_k))

    n_samples_processed = 0
    for batch in dataloader:
        cand_features, n_cands, best_cands, cand_scores, weights  = map(lambda x:x.to(device), batch)
        batched_states = (cand_features)
        batch_size = n_cands.shape[0]
        weights /= batch_size # sum loss

        with torch.no_grad():
            logits = model(batched_states)  # eval mode
            logits = model.pad_output(logits, n_cands)  # apply padding now

        true_scores = model.pad_output(torch.reshape(cand_scores, (1, -1)), n_cands)
        true_bestscore = torch.max(true_scores, dim=-1, keepdims=True).values
        true_scores = true_scores.cpu().numpy()
        true_bestscore = true_bestscore.cpu().numpy()

        kacc = []
        for k in top_k:
            pred_top_k = torch.topk(logits, k=k).indices.cpu().numpy()
            pred_top_k_true_scores = np.take_along_axis(true_scores, pred_top_k, axis=1)
            kacc.append(np.mean(np.any(pred_top_k_true_scores == true_bestscore, axis=1)))
        kacc = np.asarray(kacc)

        mean_kacc += kacc * batch_size
        n_samples_processed += batch_size

    mean_kacc /= n_samples_processed

    return mean_kacc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    parser.add_argument(
        '-m', '--model_string',
        help='searches for this string in respective trained_models folder',
        type=str,
        default='',
    )
    parser.add_argument(
        '--model_name',
        help='searches for this model_name in respective trained_models folder',
        type=str,
        default='',
    )
    parser.add_argument(
        '--test_path',
        help='if given, searches for samples in this path',
        type=str,
        default='',
    )
    args = parser.parse_args()

    ### HYPER PARAMETERS ###
    seeds = [0, 1, 2]
    test_batch_size = 128
    top_k = [1, 3, 5, 10]
    num_workers = 5

    problem_folders = {
        'setcover': '500r_1000c_0.05d',
        'cauctions': '100_500',
        'facilities': '100_100_5',
        'indset': '750_4',
    }

    problem_folder = problem_folders[args.problem]
    resultdir = "test_results"
    os.makedirs(resultdir, exist_ok=True)
    result_file = f"{resultdir}/{args.problem}_test_{time.strftime('%Y%m%d-%H%M%S')}.csv"

    ### MODELS TO TEST ###
    if args.model_string != "":
        models_to_test = [y for y in pathlib.Path(f"trained_models/{args.problem}").iterdir() if args.model_string in y.name and 'mlp_' in y.name]
        assert len(models_to_test) > 0, f"no model matched the model_string: {args.model_string}"
    elif args.model_name != "":
        model_path = pathlib.Path(f"trained_models/{args.problem}/{args.model_name}")
        assert model_path.exists(), f"path: {model_path} doesn't exist"
        assert 'mlp_' in model_path.name, f"only tests mlp models. model_path doesn't look like its mlp: {model_path}"
        models_to_test = [model_path]
    else:
        models_to_test = [y for y in pathlib.Path(f"trained_models/{args.problem}").iterdir() if 'mlp_' in y.name]
        assert len(models_to_test) > 0, f"no model matched the model_string: {args.model_string}"


    ### NUMPY / TORCH SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = torch.device("cpu")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    ### SET-UP DATASET ###
    problem_folder = f"data/samples/{args.problem}/{problem_folders[args.problem]}/test"
    if args.test_path:
        problem_folder = args.test_path

    test_files = list(pathlib.Path(problem_folder).glob('sample_*.pkl'))
    test_files = [str(x) for x in test_files]
    test_data = Dataset(test_files)
    test_data = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size,
                            shuffle = False, num_workers = num_workers, collate_fn = load_batch)

    print(f"{len(test_files)} test samples")

    evaluated_policies = [['mlp', model_path] for model_path in models_to_test]

    fieldnames = [
        'problem',
        'policy',
        'seed',
    ] + [
        f'acc@{k}' for k in top_k
    ]

    with open(result_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for model_type, model_path in evaluated_policies:
            print(f"{model_type}:{model_path.name}...")
            for seed in seeds:
                rng = np.random.RandomState(seed)

                policy = {}
                policy['name'] = model_path.name
                policy['type'] = model_type

                # load model
                best_params = str(model_path / f"{seed}/best_params.pkl")
                sys.path.insert(0, os.path.abspath(f"models/mlp"))
                import model
                importlib.reload(model)
                del sys.path[0]
                policy['model'] = model.Policy()
                policy['model'].restore_state(best_params)
                policy['model'].to(device)

                test_kacc = process(policy['model'], test_data, top_k)
                print(f"  {seed} " + " ".join([f"acc@{k}: {100*acc:4.1f}" for k, acc in zip(top_k, test_kacc)]))

                writer.writerow({
                    **{
                        'problem':args.problem,
                        'policy': f"{policy['type']}:{policy['name']}",
                        'seed': seed,
                    },
                    **{
                        f'acc@{k}': test_kacc[i] for i, k in enumerate(top_k)
                    },
                })
                csvfile.flush()
