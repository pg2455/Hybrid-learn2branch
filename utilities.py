import numpy as np
import torch
import torch.nn.functional as F
from learn2branch.utilities import log, init_scip_params, extract_state, valid_seed, compute_extended_variable_features, \
                        preprocess_variable_features, extract_khalil_variable_features

def _preprocess(state, mode='min-max-1'):
    """
    Implements preprocessing of `state`.

    Parameters
    ----------
    state : np.array
        2D array of features. rows are variables and columns are features.

    Return
    ------
    (np.array) : same shape as state but with transformed variables
    """
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
    """
    Cross-entropy loss
    """
    loss = torch.nn.CrossEntropyLoss(reduction='none')(logits, labels)
    return torch.sum(loss * weights)


def _compute_root_loss(separation_type, model, var_feats, root_n_vs, root_cands, root_n_cands, batch_size, root_cands_separation=False):
    """
    Computes losses due to auxiliary task imposed on root GCNN.

    Parameters
    ----------
    separation_type : str
        Type of separation to compute at root node's variable features
    model : model.BaseModel
        A base model, which may contain some model.PreNormLayer layers.
    var_feats : torch.tensor
        (2D) variable features at the root node
    root_n_vs : torch.tensor
        (1D) number of variables per sample
    root_cands : torch.tensor
        (1D) candidates variables (strong branching) at the root node
    root_n_cands : torch.tensor
        (1D) number of root candidate variables per sample
    batch_size : int
        number of samples
    root_cands_separation : bool
        True if separation is to be computed only between candidate variables at the root node. Useful for larger problems like Capacitated Facility Location.

    Return
    ------
    (np.float): loss value
    """

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

    if separation_type == "MHE":
        D = torch.sqrt(2 * (1 - A) + 1e-3) ** -1 - 1/2
    elif separation_type == "ED":
        D = 4 - 2 * (1 - A)
    else:
        raise ValueError(f"Unknown signal for auxiliary task: {signal_type}")

    D[mask] = 0.0
    root_loss = 0.5 * D.sum(axis=[1,2])/n_pairs
    root_loss = torch.mean(root_loss)

    return root_loss


def _distillation_loss(logits, teacher_scores, labels, weights, T, alpha):
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


def _get_model_type(model_name):
    """
    Returns the name of the model to which `model_name` belongs

    Parameters
    ----------
    model_name : str
        name of the model

    Return
    ------
    (str) : name of the folder to which this model belongs
    """
    if "concat" in model_name:
        if "-pre" in model_name:
            return "concat-pre"
        return "concat"

    if "hybridsvm-film" in model_name:
        return "hybridsvm-film"

    if "hybridsvm" in model_name:
        return "hybridsvm"

    if "film" in model_name:
        if "-pre" in model_name:
            return "film-pre"
        return "film"

    raise ValueError(f"Unknown model_name:{model_name}")
