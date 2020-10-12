"""
File adapted from https://github.com/ds4dm/learn2branch
"""
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
from utilities_gcnn_torch import GCNNDataset as Dataset, load_batch_gcnn as load_batch

def process(model, dataloader, top_k, optimizer=None):
    """
    Executes a forward and backward pass of model over the dataset.

    Parameters
    ----------
    model : model.BaseModel
        A base model, which may contain some model.PreNormLayer layers.
    dataloader : torch.utils.data.DataLoader
        Dataset to use for training the model.
    top_k : list
        list of `k` (int) to estimate for accuracy using these many candidates
    optimizer :  torch.optim
        optimizer to use for SGD. No gradient computation takes place if its None.

    Return
    ------
    mean_loss : np.float
        mean loss of model on data in dataloader
    mean_kacc : np.array
        computed accuracy for `top_k` candidates
    """
    mean_loss = 0
    mean_kacc = np.zeros(len(top_k))

    n_samples_processed = 0
    for batch in dataloader:
        c, ei, ev, v, n_cs, n_vs, n_cands, cands, best_cands, cand_scores, weights = map(lambda x:x.to(device), batch)
        batched_states = (c, ei, ev, v, n_cs, n_vs)
        batch_size = n_cs.shape[0]
        weights /= batch_size # sum loss

        with torch.no_grad():
            _, logits = model(batched_states)  # eval mode
            logits = torch.unsqueeze(torch.gather(input=torch.squeeze(logits, 0), dim=0, index=cands), 0)  # filter candidate variables
            logits = model.pad_output(logits, n_cands)  # apply padding now
            loss = _loss_fn(logits, best_cands, weights)

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

        mean_loss += loss.detach_().item() * batch_size
        mean_kacc += kacc * batch_size
        n_samples_processed += batch_size

    mean_loss /= n_samples_processed
    mean_kacc /= n_samples_processed

    return mean_loss, mean_kacc

def _loss_fn(logits, labels, weights):
    loss = torch.nn.CrossEntropyLoss(reduction='none')(logits, labels)
    return torch.sum(loss * weights)


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
        '--test_path',
        help='if given, searches for samples in this path',
        type=str,
        default='',
    )
    args = parser.parse_args()

    ### HYPER PARAMETERS ###
    seeds = [0,1,2]
    gcnn_models = ['baseline_torch']
    other_models = []
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

    os.makedirs("test_results", exist_ok=True)
    result_file = f"test_results/{args.problem}_GCNN_test_{time.strftime('%Y%m%d-%H%M%S')}.csv"

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

    evaluated_policies = [['gcnn', model] for model in gcnn_models] + \
            [['ml-competitor', model] for model in other_models]

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
        for policy_type, policy_name in evaluated_policies:
            print(f"{policy_type}:{policy_name}...")
            for seed in seeds:
                rng = np.random.RandomState(seed)

                policy = {}
                policy['name'] = policy_name
                policy['type'] = policy_type

                if policy['type'] == 'gcnn':
                    # load model
                    sys.path.insert(0, os.path.abspath(f"models/{policy['name']}"))
                    import model
                    importlib.reload(model)
                    del sys.path[0]
                    policy['model'] = model.GCNPolicy()
                    policy['model'].restore_state(f"trained_models/{args.problem}/{policy['name']}/{seed}/best_params.pkl")
                    policy['model'].to(device)

                test_loss, test_kacc = process(policy['model'], test_data, top_k)
                print(f"  {seed} " + " ".join([f"acc@{k}: {100*acc:4.1f}" for k, acc in zip(top_k, test_kacc)]))

                writer.writerow({
                    **{
                        'problem':args.problem,
                        'policy': f"{policy['type']}:{policy['name']} (1.0)",
                        'seed': seed,
                    },
                    **{
                        f'acc@{k}': test_kacc[i] for i, k in enumerate(top_k)
                    },
                })
                csvfile.flush()
