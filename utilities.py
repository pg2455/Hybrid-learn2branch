import datetime
import numpy as np
# import scipy.sparse as sp
import pyscipopt as scip
import pickle
import gzip
import torch, wandb
import torch.nn.functional as F
from learn2branch.utilities import log, init_scip_params, extract_state, valid_seed, compute_extended_variable_features, \
                        preprocess_variable_features, extract_khalil_variable_features

def _preprocess(state, mode='min-max-1'):
    if mode == "min-max-1":
        return preprocess_variable_features(state, interaction_augmentation=False, normalization=True)
    elif mode == "min-max-2":
        state -= state.min(axis=0, keepdims=True)
        max_val = state.max(axis=0, keepdims=True)
        max_val[max_val == 0] = 1
        state = 2 * state/max_val - 1
        state[:,-1] = 1 # bias
        return state


def _loss_fn(logits, labels, weights):
    loss = torch.nn.CrossEntropyLoss(reduction='none')(logits, labels)
    return torch.sum(loss * weights)


def _compute_root_loss(inputs):
    """
    Computes losses due to auxiliary task imposed on root GCNN.

    Parameters
    ----------
    inputs : tuple
        contains the type of separation at root GCNN, pytorch model, variable features at root,
            root_cands_separation (whether the separation loss is to be computed for root variables only),
            and rest of the output from load_batch_gcnn_minimal

    Return
    ------
    (np.float): loss value
    """

    signal_type, model, var_feats, root_cands_separation, root_n_vs, root_cands, root_n_cands, batch_size = inputs

    if root_cands_separation:
        # compute separation loss only for candidates at root
        n_vs = root_n_cands
        var_feats =  model.pad_features(var_feats[root_cands], root_n_cands)
    else:
        n_vs = root_n_vs
        var_feats = model.pad_features(var_feats, root_n_vs)

    n_pairs = n_vs ** 2
    A = torch.matmul(var_feats, var_feats.transpose(2,1)) # dot products
    mask = torch.zeros_like(A)
    for i,nv in enumerate(n_vs):
        mask[i, nv:, :] = 1.0
        mask[i, :, nv:] = 1.0
        mask[i, torch.arange(nv), torch.arange(nv)] = 1.0
    mask = mask.type(torch.bool)

    if signal_type == "mhe":
        D = torch.sqrt(2 * (1 - A) + 1e-3) ** -1 - 1/2
    elif signal_type == "distance-squared":
        D = 4 - 2 * (1 - A)
    else:
        raise ValueError(f"Unknown signal for auxiliary task: {signal_type}")

    D[mask] = 0.0
    root_loss = 0.5 * D.sum(axis=[1,2])/n_pairs
    root_loss = torch.mean(root_loss)

    return root_loss


def distillation(logits, teacher_scores, labels, weights, T, alpha):
    """
    Implements distillation loss.
    """
    p = F.log_softmax(logits/T, dim=-1)
    q = F.softmax(teacher_scores/T, dim=-1)
    l_kl = F.kl_div(p, q, reduction="none") * (T**2)
    l_kl = torch.sum(torch.sum(l_kl, dim=-1) * weights)
    l_ce = torch.nn.CrossEntropyLoss(reduction='none')(logits, labels)
    l_ce = torch.sum(l_ce * weights)
    return l_kl * alpha + l_ce * (1. - alpha)
