import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BaseModel(torch.nn.Module):
    def initialize_parameters(self):
        for l in self.modules():
            if isinstance(l, torch.nn.Linear):
                torch.nn.init.orthogonal_(l.weight.data, gain=1)
                if l.bias is not None:
                    torch.nn.init.constant_(l.bias.data, 0)

    def _initialize_parameters(self):
        for l in self.modules():
            if isinstance(l, torch.nn.Linear):
                torch.nn.init.xavier_normal_(l.weight.data)
                if l.bias is not None:
                    torch.nn.init.constant_(l.bias.data, 0)

    def pretrain_init(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer):
                module.start_updates()

    def pretrain(self, state):
        with torch.no_grad():
            try:
                self.forward(state)
                return False
            except PreNormException:
                return True

    def pretrain_next(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer) \
                    and module.waiting_updates and module.received_updates:
                module.stop_updates()
                return module
        return None

    def save_state(self, filepath):
        torch.save(self.state_dict(), filepath)

    def restore_state(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))

class MLP(nn.Module):
    def __init__(self, in_size, out_size, activation):
        super(MLP, self).__init__()

        self.out = nn.Linear(in_size, out_size, bias=True)
        self.activation = activation

    def forward(self, input, betagamma):
        x = self.activation(self.out(input))
        x = x * betagamma[:, 0] + betagamma[:, 1]
        return x

class Policy(BaseModel):
    def __init__(self):
        super(Policy, self).__init__()

        self.n_input_feats = 92
        self.root_emb_size = 64
        self.ff_size = 256
        self.n_layers = 3

        self.activation = torch.nn.LeakyReLU()

        # FILM GENEARTOR
        self.film_generator = nn.Sequential(
            nn.LayerNorm(self.root_emb_size),
            nn.Linear(self.root_emb_size, self.root_emb_size),
            self.activation,
            nn.Linear(self.root_emb_size, self.n_layers * self.ff_size * 2)
        )

        # OUTPUT
        self.network = nn.ModuleList([
            MLP(self.n_input_feats, self.ff_size, self.activation),
            MLP(self.ff_size, self.ff_size, self.activation),
            MLP(self.ff_size, self.ff_size, self.activation)
        ])
        self.out = nn.Linear(self.ff_size, 1, bias=False)

        self.initialize_parameters()

    @staticmethod
    def pad_output(output, n_vars_per_sample, pad_value=-1e8):
        n_vars_max = torch.max(n_vars_per_sample)

        output = torch.split(
            tensor=output,
            split_size_or_sections=n_vars_per_sample.tolist(),
            dim=1,
        )

        output = torch.cat([
            F.pad(x,
                pad=[0, n_vars_max - x.shape[1], 0, 0],
                mode='constant',
                value=pad_value)
            for x in output
        ], dim=0)

        return output

    def forward(self, inputs):
        """
        Implements forward pass of the model

        Parameters
        ----------
        root_c : torch.tensor
            constraint features at the root node
        root_ei : torch.tensor
            indices to represent constraint-variable edges of the root node
        root_ev : torch.tensor
            edge features of the root node
        root_v : torch.tensor
            variable features at the root node
        root_n_cs : torch.tensor
            number of constraints per sample
        root_n_vs : torch.tensor
            number of variables per sample
        candss : torch.tensor
            candidate variable (strong branching candidates) indices at the root node
        cand_feats : torch.tensor
            candidate variable (strong branching candidates) features at a local node
        cand_root_feats : torch.tensor
            candidate root variable features at the root node

        Return
        ------
        root_var_feats : torch.tensor
            variable features computed from root gcnn (only if applicable)
        logits : torch.tensor
            output logits at the current node
        parameters : torch.tensor
            film-parameters to compute these logits (only if applicable)
        """
        cand_feats, cand_root_feats = inputs[-2:]

        film_parameters = self.film_generator(cand_root_feats)
        film_parameters = film_parameters.view(-1, self.n_layers, 2, self.ff_size)

        x = cand_feats
        for n, subnet in enumerate(self.network):
            x = subnet(x, film_parameters[:, n])

        output = self.out(x)
        output = torch.reshape(output, [1, -1])
        return None, output, film_parameters

    def get_params(self, root_feats):
        """
        Returns parameters/variable representations inferred at the root node.

        Parameters
        ----------
        root_feats : torch.tensor
            variable embeddings as computed by the root node GNN

        Returns
        -------
        (torch.tensor): variable representations / parameters as inferred from root gcnn and to be used else where in the tree.
        """
        film_parameters = self.film_generator(root_feats)
        return film_parameters.view(-1, self.n_layers, 2, self.ff_size)

    def predict(self, cand_feats, film_parameters):
        """
        Predicts score for each candindate represented by cand_feats

        Parameters
        ----------
        cand_feats : torch.tensor
            (2D) representing input features of variables at any node in the tree
        film_parameters : torch.tensor
            (2D) parameters that are used to module MLP outputs. Same size as cand_feats.

        Returns
        -------
        (torch.tensor) : (1D) a score for each candidate
        """
        x = cand_feats
        for n, subnet in enumerate(self.network):
            x = subnet(x, film_parameters[:, n])
        output = self.out(x)
        output = torch.reshape(output, [1, -1])
        return output
