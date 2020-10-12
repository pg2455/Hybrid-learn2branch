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
from utilities import log, _loss_fn, _distillation_loss, _compute_root_loss

from utilities_hybrid import HybridDataset as Dataset, load_batch

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
            root_g, node_g, node_attr = [map(lambda x:x if x is None else x.to(device) , y) for y in batch]
            root_c, root_ei, root_ev, root_v, root_n_cs, root_n_vs, *_ = root_g
            g_c, g_ei, g_ev, g_v, g_n_cs, g_n_vs, candss = node_g
            cand_features, n_cands, best_cands, cand_scores, weights = node_attr

            batched_states = (root_c, root_ei, root_ev, root_v, root_n_cs, root_n_vs, candss, cand_features, None)

            if not model.pre_train(batched_states):
                break

        res = model.pre_train_next()
        if res is None:
            break
        else:
            layer = res

        i += 1

    return i

def process(model, teacher, dataloader, top_k, optimizer=None):
    """
    Executes a forward and backward pass of model over the dataset.

    Parameters
    ----------
    model : model.BaseModel
        A base model, which may contain some model.PreNormLayer layers.
    teacher : model.BaseModel
        A pretrained model when args.no_e2e is True, and an expert model when it is True.
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
    accum_iter = 0
    for batch in dataloader:
        root_g, node_g, node_attr = [map(lambda x:x if x is None else x.to(device) , y) for y in batch]
        root_c, root_ei, root_ev, root_v, root_n_cs, root_n_vs, root_cands, root_n_cands = root_g
        node_c, node_ei, node_ev, node_v, node_n_cs, node_n_vs, candss = node_g
        cand_features, n_cands, best_cands, cand_scores, weights  = node_attr
        cands_root_v = None

        # use teacher
        with torch.no_grad():
            if teacher is not None:
                if args.no_e2e:
                    root_v, _ = teacher((root_c, root_ei, root_ev, root_v, root_n_cs, root_n_vs))
                    cands_root_v = root_v[candss]

                # KD - get soft targets
                if args.distilled:
                    _, soft_targets = teacher((node_c, node_ei, node_ev, node_v, node_n_cs, node_n_vs))
                    soft_targets = torch.unsqueeze(torch.gather(input=torch.squeeze(soft_targets, 0), dim=0, index=candss), 0)
                    soft_targets = model.pad_output(soft_targets, n_cands)  # apply padding now

        batched_states = (root_c, root_ei, root_ev, root_v, root_n_cs, root_n_vs, candss, cand_features, cands_root_v)
        batch_size = n_cands.shape[0]
        weights /= batch_size # sum loss

        if optimizer:
            optimizer.zero_grad()
            var_feats, logits, film_parameters = model(batched_states)  # eval mode
            logits = model.pad_output(logits, n_cands)  # apply padding now

            # node loss
            if args.distilled:
                loss = _distillation_loss(logits, soft_targets, best_cands, weights, T, alpha)
            else:
                loss = _loss_fn(logits, best_cands, weights)

            # AT loss
            if args.at != "":
                loss  += args.beta_at * _compute_root_loss(args.at, model, var_feats, root_n_vs, root_cands, root_n_cands, batch_size, root_cands_separation)

            # regularization
            if (
                args.l2 > 0
                and film_parameters is not None
            ):
                beta_norm = (1-film_parameters[:, :, 0]).norm()
                gamma_norm = film_parameters[:, :, 1].norm()
                loss += args.l2 * (beta_norm + gamma_norm)

            loss.backward()
            accum_iter += 1
            if accum_iter % accum_steps == 0:
                optimizer.step()
                accum_iter = 0
        else:
            with torch.no_grad():
                var_feats, logits, film_parameters = model(batched_states)  # eval mode
                logits = model.pad_output(logits, n_cands)  # apply padding now

                # node loss
                if args.distilled:
                    loss = _distillation_loss(logits, soft_targets, best_cands, weights, T, alpha)
                else:
                    loss = _loss_fn(logits, best_cands, weights)

                # AT loss
                if args.at != "":
                    loss  += args.beta_at * _compute_root_loss(args.at, model, var_feats, root_n_vs, root_cands, root_n_cands, batch_size, root_cands_separation)

                # regularization
                if (
                    args.l2 > 0
                    and film_parameters is not None
                ):
                    beta_norm = (1-film_parameters[:, :, 0]).norm()
                    gamma_norm = film_parameters[:, :, 1].norm()
                    loss += args.l2 * (beta_norm + gamma_norm)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    parser.add_argument(
        '-m', '--model',
        help='model to be trained.',
        type=str,
        default='film',
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
        help='name of the folder',
        type=str,
        default="data/samples/",
    )
    parser.add_argument(
        '--no_e2e',
        help='if training is with a pretrained GCNN.',
        action="store_true"
    )
    parser.add_argument(
        '--distilled',
        help='if distillation should be used',
        action="store_true"
    )
    parser.add_argument(
        '--at',
        help='type of auxiliary task',
        type=str,
        default='',
        choices=['ED', 'MHE', '']
    )
    parser.add_argument(
        '--beta_at',
        help='weight for at loss function',
        type=float,
        default=0,
    )
    parser.add_argument(
        '--l2',
        help='regularization film weights',
        type=float,
        default=0.0
    )
    args = parser.parse_args()

    if (
        args.model in ['concat', 'film']
        and args.no_e2e
    ):
        args.model = f"{args.model}-pre"

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
    teacher_model = "baseline_torch" # used only if args.distilled or args.no_e2e is True
    T = 2 # used only if args.distilled is True
    alpha = 0.9 # used only if args.distilled is True

    root_cands_separation=False
    if args.problem == "facilities":
        # facilities have larger problem size (LPs have 10000 variables)
        # these settings are chosen so that training is feasible in considerable time (about 6-12 hours)
        lr = 0.005
        epoch_size=312*3
        batch_size = 16
        accum_steps = 2
        patience=10
        early_stopping=20
        pretrain_batch_size = 64
        valid_batch_size = 32
        root_cands_separation=True
        num_workers=7

    problem_folders = {
        'setcover': '500r_1000c_0.05d',
        'cauctions': '100_500',
        'facilities': '100_100_5',
        'indset': '750_4',
    }

    # DIRECTORY NAMING
    modeldir = f"{args.model}"
    if args.distilled:
        modeldir = f"{args.model}_distilled"

    if args.at != "":
        modeldir = f"{modeldir}_{args.at}_{args.beta_at}"

    if args.l2 > 0:
        modeldir = f"{modeldir}_l2_{args.l2}"

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
    log(f"e2e: {not args.no_e2e}", logfile)
    log(f"KD: {args.distilled}", logfile)
    log(f"AT: {args.at} beta={args.beta_at}", logfile)
    log(f"l2: {args.l2}", logfile)

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

    valid_data = Dataset(valid_files, args.data_path)
    valid_data = torch.utils.data.DataLoader(valid_data, batch_size=valid_batch_size,
                            shuffle = False, num_workers = num_workers, collate_fn = load_batch)

    pretrain_files = [f for i, f in enumerate(train_files) if i % 10 == 0]
    pretrain_data = Dataset(pretrain_files, args.data_path)
    pretrain_data = torch.utils.data.DataLoader(pretrain_data, batch_size=pretrain_batch_size,
                            shuffle = False, num_workers = num_workers, collate_fn = load_batch)

    ### MODEL LOADING ###
    sys.path.insert(0, os.path.abspath(f'models/{args.model}'))
    import model
    importlib.reload(model)
    distilled_model = model.Policy()
    del sys.path[0]
    distilled_model.to(device)

    ### TEACHER MODEL LOADING ###
    teacher = None
    if (
        args.distilled
        or args.no_e2e
    ):
        sys.path.insert(0, os.path.abspath(f'models/{teacher_model}'))
        import model
        importlib.reload(model)
        teacher = model.GCNPolicy()
        del sys.path[0]
        teacher.restore_state(f"trained_models/{args.problem}/{teacher_model}/{args.seed}/best_params.pkl")
        teacher.to(device)
        teacher.eval()

    model = distilled_model

    ### TRAINING LOOP ###
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=patience, verbose=True)
    best_loss = np.inf
    for epoch in range(max_epochs + 1):
        log(f"EPOCH {epoch}...", logfile)

        if (
            epoch == 0
            and not args.no_e2e
        ):
            n = pretrain(model=model, dataloader=pretrain_data)
            log(f"PRETRAINED {n} LAYERS", logfile)
        else:
            # bugfix: tensorflow's shuffle() seems broken...
            epoch_train_files = rng.choice(train_files, epoch_size * batch_size * accum_steps, replace=True)
            train_data = Dataset(epoch_train_files, args.data_path)
            train_data = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                    shuffle = False, num_workers = num_workers, collate_fn = load_batch)
            train_loss, train_kacc = process(model, teacher, train_data, top_k, optimizer)
            log(f"TRAIN LOSS: {train_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, train_kacc)]), logfile)

        # TEST
        valid_loss, valid_kacc = process(model, teacher, valid_data, top_k, None)
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
    valid_loss, valid_kacc = process(model, teacher, valid_data, top_k, None)
    log(f"BEST VALID LOSS: {valid_loss:0.3f} " + "".join([f" acc@{k}: {acc:0.3f}" for k, acc in zip(top_k, valid_kacc)]), logfile)
