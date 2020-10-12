
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

import tensorflow as tf
import torch

import utilities
from utilities import log

from utilities_mlp import MLPDataset as Dataset
from utilities_mlp import load_batch

def process(model, dataloader, top_k, optimizer=None, ROOT_WT = 0.0):
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
        cand_features, n_cands, best_cands, cand_scores, weights  = map(lambda x:x.to(device), batch)
        batched_states = (cand_features)
        batch_size = n_cands.shape[0]
        weights /= batch_size # sum loss

        if optimizer:
            optimizer.zero_grad()
            logits = model(batched_states)  # eval mode
            logits = model.pad_output(logits, n_cands)  # apply padding now
            loss = _loss_fn(logits, best_cands, weights)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(batched_states)  # eval mode
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
        '-s', '--seed',
        help='Random generator seed.',
        type=utilities.valid_seed,
        default=0,
    )
    parser.add_argument(
        '-g', '--gpu',
        help='CUDA GPU id (-1 for CPU).',
        type=int,
        default=-1,
    )
    parser.add_argument(
        '--data_path',
        help='name of the folder where train and valid folders are present. Assumes `data/samples` as default.',
        type=str,
        default="data/samples",
    )
    parser.add_argument(
        '-w','--node_weights',
        help='weighing scheme for loss',
        choices=['sigmoidal_decay', 'exponential_decay', 'linear_decay', 'constant', 'quadratic_decay', ''],
        default = "sigmoidal_decay"
    )
    args = parser.parse_args()

    ### HYPER PARAMETERS ###
    max_epochs = 1000
    epoch_size = 312
    batch_size = 32
    accum_steps = 1 # step() is called after  batch_size * accum_steps samples
    pretrain_batch_size = 128
    valid_batch_size = 128
    lr = 0.001
    patience = 15
    early_stopping = 30
    top_k = [1, 3, 5, 10]
    num_workers = 5

    if args.problem == "facilities":
        lr = 0.005 # for faster convergence

    problem_folders = {
        'setcover': '500r_1000c_0.05d',
        'cauctions': '100_500',
        'facilities': '100_100_5',
        'indset': '750_4',
    }

    modeldir = "mlp"
    if args.node_weights != "":
        modeldir = f"{modeldir}_{args.node_weights}"
    running_dir = f"trained_models/{args.problem}/{modeldir}/{args.seed}"
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
    log(f"seed {args.seed}", logfile)
    log(f"node weights: {args.node_weights}", logfile)

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
    problem_folder = problem_folders[args.problem]
    train_files = list(pathlib.Path(f"{args.data_path}/{args.problem}/{problem_folder}/train").glob('sample_*.pkl'))
    valid_files = list(pathlib.Path(f"{args.data_path}/{args.problem}/{problem_folder}/valid").glob('sample_*.pkl'))

    log(f"{len(train_files)} training samples", logfile)
    log(f"{len(valid_files)} validation samples", logfile)

    train_files = [str(x) for x in train_files]
    valid_files = [str(x) for x in valid_files]

    valid_data = Dataset(valid_files, args.node_weights)
    valid_data = torch.utils.data.DataLoader(valid_data, batch_size=valid_batch_size,
                            shuffle = False, num_workers = num_workers, collate_fn = load_batch)

    ### MODEL LOADING ###
    sys.path.insert(0, os.path.abspath(f'models/mlp'))
    import model
    importlib.reload(model)
    model = model.Policy()
    del sys.path[0]
    model.to(device)

    ### TRAINING LOOP ###
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=patience, verbose=True)

    best_loss, best_acc = np.inf, -1
    for epoch in range(max_epochs + 1):
        log(f"EPOCH {epoch}...", logfile)

        # TRAIN
        if epoch == 0:
            # no pretraining
            pass
        else:
            epoch_train_files = rng.choice(train_files, epoch_size * batch_size * accum_steps, replace=True)
            train_data = Dataset(epoch_train_files, args.node_weights)
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
