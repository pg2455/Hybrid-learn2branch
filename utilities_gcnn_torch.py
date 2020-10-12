import pickle
import gzip
import numpy as np
import torch

class GCNNDataset(torch.utils.data.Dataset):
    def __init__(self, sample_files, weighted=False):
        self.sample_files = sample_files
        self.weighted = weighted

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, index):
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)

        weight = 1.0
        if "root" in self.sample_files[index]:
            sample_state, _, sample_cands, sample_action, cand_scores = sample['root_state']
            c,e,v = sample_state
            c_feats = c['values']
            v_feats = v['values']
        else:
            obss, target, obss_feats, _ = sample['obss']
            v,c_feats,e = obss
            v_feats = v[:, :19] # gcnn features
            sample_cand_scores = obss_feats['scores']
            sample_cands = np.where(sample_cand_scores != -1)[0]
            cand_scores = sample_cand_scores[sample_cands]
            sample_action = np.where(sample_cands == target)[0][0]
            if self.weighted:
                weight = obss_feats['depth']/sample['max_depth']
                weight = (1 + np.exp(-0.5))/(1 + np.exp(weight - 0.5))

        return c_feats, e['indices'], e['values'], v_feats, sample_cands, sample_action, cand_scores, weight

def load_batch_gcnn(sample_batch):
    c_features, e_indices, e_features, v_features, candss, cand_choices, cand_scoress, weights = zip(*sample_batch)

    n_cs_per_sample = [c.shape[0] for c in c_features]
    n_vs_per_sample = [v.shape[0] for v in v_features]
    n_cands_per_sample = [cds.shape[0] for cds in candss]

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
    candss = np.concatenate([cands + shift
        for cands, shift in zip(candss, cv_shift[1])])

    cand_choices = np.array(cand_choices)
    cand_scoress = np.concatenate(cand_scoress, axis=0)
    weights = np.array(weights)

    # convert to tensors
    c_features = torch.as_tensor(c_features, dtype=torch.float32)
    e_indices = torch.as_tensor(e_indices, dtype=torch.long)
    e_features = torch.as_tensor(e_features, dtype=torch.float32)
    v_features = torch.as_tensor(v_features, dtype=torch.float32)
    n_cs_per_sample = torch.as_tensor(n_cs_per_sample, dtype=torch.int32)
    n_vs_per_sample = torch.as_tensor(n_vs_per_sample, dtype=torch.int32)
    candss = torch.as_tensor(candss, dtype=torch.long)
    cand_choices = torch.as_tensor(cand_choices, dtype=torch.long)
    cand_scoress = torch.as_tensor(cand_scoress, dtype=torch.float32)
    n_cands_per_sample = torch.as_tensor(n_cands_per_sample, dtype=torch.int32)
    weights = torch.as_tensor(weights, dtype=torch.float32)

    return [c_features, e_indices, e_features, v_features, n_cs_per_sample, n_vs_per_sample, n_cands_per_sample, candss, cand_choices, cand_scoress, weights]
