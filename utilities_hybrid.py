import pickle
import gzip
import numpy as np
import torch

import utilities
import pathlib
from functools import lru_cache

@lru_cache(maxsize=2**14)
def _get_root_state(filename):
    """
    Extracts states from the sample at `filename`

    Parameters
    ----------
    filename : str
        filepath of the sample

    Return
    ------
    sample_state : tuple
         output of `utilities.extract_state`
    sample_cands : np.array
        indices of variables which were chosen as potential candidates for strong branching
    sample_action : int
        index of variable which was chosen for branching
    cand_scores : np.array
        strong branching scrores for `sample_cands`
    """
    filename = filename.replace('tmp', '')
    with gzip.open(filename, 'rb') as f:
        sample_state, _, sample_cands, sample_action, cand_scores = pickle.load(f)['root_state']
    return sample_state, sample_cands, sample_action, cand_scores


class HybridDataset(torch.utils.data.Dataset):
    def __init__(self, sample_files, rootdir = "data/samples/", weighing_scheme="sigmoidal_decay"):
        self.sample_files = sample_files
        self.rootdir = rootdir
        self.weighing_scheme = weighing_scheme

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, index):
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)

        # root data
        if sample['type'] == "root":
            sample_state, _, root_cands, root_action, root_cand_scores = sample['root_state']
        else:
            root_filename = sample['root_state']
            sample_state, root_cands, root_action,  root_cand_scores = _get_root_state(root_filename)

        c,e,v = sample_state
        root_c = c['values']
        root_ei = e['indices']
        root_ev = e['values']
        root_v = v['values']

        # data for gcnn
        obss, target, obss_feats, _ = sample['obss']
        v, gcnn_c_feats, gcnn_e = obss
        gcnn_v_feats = v[:, :19] # gcnn features

        # target
        sample_cand_scores = obss_feats['scores']
        sample_cands = np.where(sample_cand_scores != -1)[0]
        cand_scores = sample_cand_scores[sample_cands]
        sample_action = np.where(sample_cands == target)[0][0]

        # data for mlp
        v_feats = v[sample_cands]
        v_feats = utilities._preprocess(v_feats, mode='min-max-2')

        weight = obss_feats['depth']/sample['max_depth'] if sample['max_depth'] else 1.0
        if self.weighing_scheme == "linear_decay":
            m = np.exp(-0.5) - 1
            c = 1
            weight = weight * m + c
        elif self.weighing_scheme == "sigmoidal_decay":
            weight = (1 + np.exp(-0.5))/(1 + np.exp(weight - 0.5))
        elif self.weighing_scheme == "exponential_decay":
            weight = np.exp(weight * -0.5)
        elif self.weighing_scheme == "quadratic_decay":
            weight = (np.exp(-0.5) - 1) * weight ** 2 + 1
        elif self.weighing_scheme == "constant":
            weight = 1.0
        else:
            raise ValueError(f"Unknown weighing scheme:{self.weighing_scheme}")

        node_g = [gcnn_c_feats, gcnn_e['indices'], gcnn_e['values'], gcnn_v_feats]
        node_attr = [v_feats, sample_action, sample_cands, cand_scores, weight]
        root_g = [root_c, root_ei, root_ev, root_v, root_cands, root_action, root_cand_scores]
        return root_g, node_g, node_attr


def load_batch(sample_batch):
    sample_batch = [list(zip(*x)) for x in list(zip(*sample_batch))]
    if len(sample_batch) == 3:
        root_g, node_g, node_attr = sample_batch
        root_c, root_ei, root_ev, root_v, root_cands, root_action, _ = root_g
        root_n_cands = torch.as_tensor(np.array([cds.shape[0] for cds in root_cands]), dtype=torch.int32)
        root_g_states = load_batch_gcnn_minimal((root_c, root_ei, root_ev, root_v, root_cands))
        root_g_states += [root_n_cands]
    else:
        node_g, node_attr = sample_batch
        root_g_states = [None] * 6

    node_c, node_ei, node_ev, node_v = node_g
    cand_featuress, sample_actions, sample_cands, cand_scoress, weights = node_attr

    node_g = load_batch_gcnn_minimal((node_c, node_ei, node_ev, node_v, sample_cands))
    n_cands = [cds.shape[0] for cds in sample_cands]

    # convert to numpy arrays
    cand_featuress = np.concatenate(cand_featuress, axis=0)
    cand_scoress = np.concatenate(cand_scoress, axis=0)
    n_cands = np.array(n_cands)
    best_actions = np.array(sample_actions)
    weights = np.array(weights)

    # convert to tensors
    cand_featuress = torch.as_tensor(cand_featuress, dtype=torch.float32)
    cand_scoress = torch.as_tensor(cand_scoress, dtype=torch.float32)
    n_cands = torch.as_tensor(n_cands, dtype=torch.int32)
    best_actions = torch.as_tensor(sample_actions, dtype=torch.long)
    weights = torch.as_tensor(weights, dtype=torch.float32)

    node_attr = [cand_featuress, n_cands, best_actions, cand_scoress, weights]

    return root_g_states, node_g, node_attr


def load_batch_gcnn_minimal(batch):
    """
    Loads data for GCNN excluding scores. It is used for unpacking root state.

    Parameters
    ----------
    batch : tuple
        a tuple of constraint features, edge indices, edge features, variable features and candidates at the root

    Return
    ------
    (list): a list of above features with their proper format to be used in training models.
    """
    c_features, e_indices, e_features, v_features, candss = batch

    n_cs_per_sample = [c.shape[0] for c in c_features]
    n_vs_per_sample = [v.shape[0] for v in v_features]

    # concatenate samples in one big graph
    c_features = np.concatenate(c_features, axis=0)
    v_features = np.concatenate(v_features, axis=0)
    e_features = np.concatenate(e_features, axis=0)
    # edge indices have to be adjusted accordingly
    cv_shift = np.cumsum([
            [0] + n_cs_per_sample[:-1],
            [0] + n_vs_per_sample[:-1]
        ], axis=1)
    e_indices = np.concatenate([e_ind + cv_shift[:, j:(j+1)]
        for j, e_ind in enumerate(e_indices)], axis=1)

    # candidate indices as well
    if candss is not None:
        candss = np.concatenate([cands + shift
            for cands, shift in zip(candss, cv_shift[1])])
        candss = torch.as_tensor(candss, dtype=torch.long)

    # convert to tensors
    c_features = torch.as_tensor(c_features, dtype=torch.float32)
    e_indices = torch.as_tensor(e_indices, dtype=torch.long)
    e_features = torch.as_tensor(e_features, dtype=torch.float32)
    v_features = torch.as_tensor(v_features, dtype=torch.float32)
    n_cs_per_sample = torch.as_tensor(n_cs_per_sample, dtype=torch.int32)
    n_vs_per_sample = torch.as_tensor(n_vs_per_sample, dtype=torch.int32)

    return [c_features, e_indices, e_features, v_features, n_cs_per_sample, n_vs_per_sample, candss]
