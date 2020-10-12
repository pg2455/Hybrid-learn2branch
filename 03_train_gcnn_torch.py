"""
File adapted from https://github.com/ds4dm/learn2branch
"""
import os
import importlib
import argparse
import sys
import pathlib
import pickle
import numpy as np
from time import strftime
from shutil import copyfile
import gzip

import torch

import utilities
from utilities import log

from utilities_gcnn_torch import GCNNDataset as Dataset
from utilities_gcnn_torch import load_batch_gcnn as load_batch

def pretrain(model, dataloader):
    """
    Pre-normalizes a model (i.e., PreNormLayer layers) over the given samples.

    Parameters
    ----------
    model : model.BaseModel
        A base model, which may contain some model.PreNormLayer layers.
    dataloader : torch.utils.data.DataLoader
        Dataset to use for pre-training the model.
    Return
    ------
    number of PreNormLayer layers processed.
    """
    model.pre_train_init()
    i = 0
    while True:
        for batch in dataloader:
            c, ei, ev, v, n_cs, n_vs, n_cands, cands, best_cands, cand_scores, weights = map(lambda x:x.to(device), batch)
            batched_states = (c, ei, ev, v, n_cs, n_vs)

            if not model.pre_train(batched_states):
                break

        res = model.pre_train_next()
        if res is None:
            break
        else:
            layer = res

        i += 1

    return i

def process(model, dataloader, top_k, optimizer=None):
    """
    Pre-normalizes a model (i.e., PreNormLayer layers) over the given samples.

    Parameters
    ----------
    model : model.BaseModel
        A base model, which may contain some model.PreNormLayer layers.
    dataloader : torch.utils.data.DataLoader
        Dataset to use for training the model.
    top_k : list
        list of `k` (int) to estimate for accuracy using these many candidates
    optimizer :  torch.optim
        optimizer to use for SGD

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

        if optimizer:
            optimizer.zero_grad()
            _, logits = model(batched_states)  # eval mode
            logits = torch.unsqueeze(torch.gather(input=torch.squeeze(logits, 0), dim=0, index=cands), 0)  # filter candidate variables
            logits = model.pad_output(logits, n_cands)  # apply padding now
            loss = _loss_fn(logits, best_cands, weights)
            loss.backward()
            optimizer.step()
        else:
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
        '-m', '--model',
        help='GCNN model to be trained.',
        type=str,
        default='baseline_torch',
    )
    parser.add_argument(
        '-s', '--seed',
        help='Random generator seed.',
        type=utilities.valid_seed,
        default=0,
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--data_path',
        help='name of the folder where train and valid folders are present. Assumes `data/samples` as default.',
        type=str,
        default="",
    )
    parser.add_argument(
        '--l2',
        help='value of l2 regularizer',
        type=float,
        default=0.0
    )
    args = parser.parse_args()

    ### HYPER PARAMETERS ###
    max_epochs = 500
    epoch_size = 312
    batch_size = 32
    pretrain_batch_size = 128
    valid_batch_size = 128
    lr = 0.001
    patience = 15
    early_stopping = 30
    top_k = [1, 3, 5, 10]
    train_sample_limit = 150000
    valid_sample_limit = 30000
    num_workers = 10

    problem_folders = {
        'setcover': '500r_1000c_0.05',
        'cauctions': '100_500',
        'facilities': '100_100_5',
        'indset': '750_4',
    }

    problem_folder = problem_folders[args.problem]

    running_dir = f"trained_models/{args.problem}/{args.model}/{args.seed}"
    os.makedirs(running_dir)

    ### LOG ###
    logfile = os.path.join(running_dir, 'log.txt')

    log(f"max_epochs: {max_epochs}", logfile)
    log(f"epoch_size: {epoch_size}", logfile)
    log(f"batch_size: {batch_size}", logfile)
    log(f"pretrain_batch_size: {pretrain_batch_size}", logfile)
    log(f"valid_batch_size : {valid_batch_size }", logfile)
    log(f"lr: {lr}", logfile)
    log(f"patience : {patience }", logfile)
    log(f"early_stopping : {early_stopping }", logfile)
    log(f"top_k: {top_k}", logfile)
    log(f"problem: {args.problem}", logfile)
    log(f"gpu: {args.gpu}", logfile)
    log(f"seed: {args.seed}", logfile)
    log(f"l2 {args.l2}", logfile)

    ### NUMPY / TORCH SETUP ###
    if args.gpu == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = torch.device("cpu")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = f'{args.gpu}'
        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(rng.randint(np.iinfo(int).max))

    ### SET-UP DATASET ###
    dir = f'data/samples/{args.problem}/{problem_folder}'
    if args.data_path:
        dir = f"{args.data_path}/{args.problem}/{problem_folder}"

    train_files = list(pathlib.Path(f'{dir}/train').glob('sample_*.pkl'))
    valid_files = list(pathlib.Path(f'{dir}/valid').glob('sample_*.pkl'))

    log(f"{len(train_files)} training samples", logfile)
    log(f"{len(valid_files)} validation samples", logfile)

    train_files = [str(x) for x in train_files]
    valid_files = [str(x) for x in valid_files]

    valid_data = Dataset(valid_files)
    valid_data = torch.utils.data.DataLoader(valid_data, batch_size=valid_batch_size,
                            shuffle = False, num_workers = num_workers, collate_fn = load_batch)

    pretrain_files = [f for i, f in enumerate(train_files) if i % 10 == 0]
    pretrain_data = Dataset(pretrain_files)
    pretrain_data = torch.utils.data.DataLoader(pretrain_data, batch_size=pretrain_batch_size,
                            shuffle = False, num_workers = num_workers, collate_fn = load_batch)

    ### MODEL LOADING ###
    sys.path.insert(0, os.path.abspath(f'models/{args.model}'))
    import model
    importlib.reload(model)
    model = model.GCNPolicy()
    del sys.path[0]
    model.to(device)

    ### TRAINING LOOP ###
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=patience, verbose=True)

    best_loss = np.inf
    for epoch in range(max_epochs + 1):
        log(f"EPOCH {epoch}...", logfile)

        # TRAIN
        if epoch == 0:
            n = pretrain(model=model, dataloader=pretrain_data)
            log(f"PRETRAINED {n} LAYERS", logfile)
        else:
            epoch_train_files = rng.choice(train_files, epoch_size * batch_size, replace=True)
            train_data = Dataset(epoch_train_files)
            train_data = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                    shuffle = False, num_workers = num_workers, collate_fn = load_batch)
            train_loss, train_kacc = process(model, train_data, top_k, optimizer)
            log(f"TRAIN LOSS: {train_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, train_kacc)]), logfile)

        # TEST
        valid_loss, valid_kacc = process(model, valid_data, top_k, None)
        log(f"VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)

        if valid_loss < best_loss:
            plateau_count = 0
            best_loss = valid_loss
            model.save_state(os.path.join(running_dir, 'best_params.pkl'))
            log(f"  best model so far", logfile)
        else:
            plateau_count += 1
            if plateau_count % early_stopping == 0:
                log(f"  {plateau_count} epochs without improvement, early stopping", logfile)
                break
            if plateau_count % patience == 0:
                lr *= 0.2
                log(f"  {plateau_count} epochs without improvement, decreasing learning rate to {lr}", logfile)
        scheduler.step(valid_loss)
    model.restore_state(os.path.join(running_dir, 'best_params.pkl'))
    valid_loss, valid_kacc = process(model, valid_data, top_k, None)
    log(f"BEST VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)
