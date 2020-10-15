import pickle
import gzip
import numpy as np
import torch

import utilities

class MLPDataset(torch.utils.data.Dataset):
    def __init__(self, sample_files, weighing_scheme="sigmoidal_decay"):
        self.sample_files = sample_files
        self.weighing_scheme = weighing_scheme if weighing_scheme != "" else "constant"

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, index):
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)

        is_root = "root" in self.sample_files[index]

        obss, target, obss_feats, _ = sample['obss']
        v, _, _ = obss
        sample_cand_scores = obss_feats['scores']
        sample_cands = np.where(sample_cand_scores != -1)[0]

        v_feats = v[sample_cands]
        v_feats = utilities._preprocess(v_feats, mode='min-max-2')

        cand_scores = sample_cand_scores[sample_cands]
        sample_action = np.where(sample_cands == target)[0][0]

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
            raise ValueError(f"Unknown value for node weights: {self.weighing_scheme}")

        return  v_feats, sample_action, cand_scores, weight

def load_batch(sample_batch):
    cand_featuress, sample_actions, cand_scoress, weights = list(zip(*sample_batch))

    n_cands = [cds.shape[0] for cds in cand_featuress]

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

    return cand_featuress, n_cands, best_actions, cand_scoress, weights
