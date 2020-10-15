import torch
import torch.nn as nn
import torch.nn.functional as F

class PreNormException(Exception):
    pass


class PreNormLayer(nn.Module):
    """
    Our pre-normalization layer, whose purpose is to normalize an input layer
    to zero mean and unit variance to speed-up and stabilize GCN training. The
    layer's parameters are aimed to be computed during the pre-training phase.
    """
    def __init__(self, n_units, shift=True, scale=True):
        super(PreNormLayer, self).__init__()
        assert shift or scale

        if shift:
            self.register_buffer(f"shift", torch.zeros((n_units,), dtype=torch.float32))
        else:
            self.shift = None

        if scale:
            self.register_buffer(f"scale", torch.ones((n_units,), dtype=torch.float32))
        else:
            self.scale = None

        self.n_units = n_units
        self.waiting_updates = False
        self.received_updates = False

    def forward(self, input):
        if self.waiting_updates:
            self.update_stats(input)
            self.received_updates = True
            raise PreNormException

        if self.shift is not None:
            input = input + self.shift

        if self.scale is not None:
            input = input * self.scale

        return input

    def start_updates(self):
        """
        Initializes the pre-training phase.
        """
        self.avg = 0
        self.var = 0
        self.m2 = 0
        self.count = 0
        self.waiting_updates = True
        self.received_updates = False

    def update_stats(self, input):
        """
        Online mean and variance estimation. See: Chan et al. (1979) Updating
        Formulae and a Pairwise Algorithm for Computing Sample Variances.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
        """
        assert self.n_units == 1 or input.shape[-1] == self.n_units, f"Expected input dimension of size {self.n_units}, got {input.shape[-1]}."

        input = input.reshape([-1, self.n_units])
        sample_avg = torch.mean(input, dim=0)
        sample_var = torch.mean((input - sample_avg) ** 2, dim=0)
        sample_count = input.numel() / self.n_units

        delta = sample_avg - self.avg

        self.m2 = self.var * self.count + sample_var * sample_count + delta ** 2 * self.count * sample_count / (
                self.count + sample_count)

        self.count += sample_count
        self.avg += delta * sample_count / self.count
        self.var = self.m2 / self.count if self.count > 0 else 1

    def stop_updates(self):
        """
        Ends pre-training for that layer, and fixes the layers's parameters.
        """

        assert self.count > 0
        if self.shift is not None:
            self.shift = - self.avg

        if self.scale is not None:
            self.var = torch.where(torch.eq(self.var, 0.0), torch.ones_like(self.var), self.var) # NaN check trick
            self.scale = 1 / torch.sqrt(self.var)

        del self.avg, self.var, self.m2, self.count
        self.waiting_updates = False


class BipartiteGraphConvolution(nn.Module):
    """
    Partial bipartite graph convolution (either left-to-right or right-to-left).
    """

    def __init__(self, emb_size, activation, initializer, right_to_left=False):
        super(BipartiteGraphConvolution, self).__init__()

        self.emb_size = emb_size
        self.activation = activation
        self.initializer = initializer
        self.right_to_left = right_to_left
        self.edge_nfeats = 1

        # feature layers
        self.feature_module_left = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=True),
            self.activation
        )

        self.feature_module_edge = nn.Sequential(
            nn.Linear(self.edge_nfeats, self.emb_size, bias=False),
            self.activation
        )

        self.feature_module_right = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=False),
            self.activation
        )

        self.feature_module_final = nn.Sequential(
            PreNormLayer(1, shift=False), # normalize after summation trick
            self.activation,
            nn.Linear(self.emb_size, self.emb_size, bias=True)
        )

        self.post_conv_module = nn.Sequential(
            PreNormLayer(1, shift=False), # normalize after summation trick
        )

        # output_layers
        self.output_module = nn.Sequential(
            nn.Linear(self.emb_size + self.emb_size, self.emb_size, bias=True),
            self.activation,
            nn.Linear(self.emb_size, self.emb_size, bias=True)
        )

    def forward(self, inputs):
        """
        Perfoms a partial graph convolution on the given bipartite graph.

        Inputs
        ------
        left_features: 2D float tensor
            Features of the left-hand-side nodes in the bipartite graph
        edge_indices: 2D int tensor
            Edge indices in left-right order
        edge_features: 2D float tensor
            Features of the edges
        right_features: 2D float tensor
            Features of the right-hand-side nodes in the bipartite graph
        scatter_out_size: 1D int tensor
            Output size (left_features.shape[0] or right_features.shape[0], unknown at compile time)
        """
        left_features, edge_indices, edge_features, right_features, scatter_out_size = inputs
        device = left_features.device

        if self.right_to_left:
            scatter_dim = 0
            prev_features = left_features
        else:
            scatter_dim = 1
            prev_features = right_features

        # compute joint features
        joint_features = self.feature_module_final(
            self.feature_module_left(left_features)[edge_indices[0]] +
            self.feature_module_edge(edge_features) +
            self.feature_module_right(right_features)[edge_indices[1]]
        )

        # perform convolution
        conv_output = torch.zeros([scatter_out_size, self.emb_size], device=device).index_add_(0, edge_indices[scatter_dim], joint_features)
        conv_output = self.post_conv_module(conv_output)

        # apply final module
        output = self.output_module(torch.cat([
            conv_output,
            prev_features
        ], axis=1))

        return output


class GCNPolicy(nn.Module):
    """
    Our bipartite Graph Convolutional neural Network (GCN) model.
    """

    def __init__(self):
        super(GCNPolicy, self).__init__()

        self.emb_size = 64
        self.cons_nfeats = 5
        self.edge_nfeats = 1
        self.var_nfeats = 19

        self.activation = nn.ReLU()
        self.initializer = lambda x: torch.nn.init.orthogonal_(x, gain=1)

        # CONSTRAINT EMBEDDING
        self.cons_embedding = nn.Sequential(
            PreNormLayer(n_units=self.cons_nfeats),
            nn.Linear(self.cons_nfeats, self.emb_size, bias=True),
            self.activation,
            nn.Linear(self.emb_size, self.emb_size, bias=True),
            self.activation
        )

        # EDGE EMBEDDING
        self.edge_embedding = nn.Sequential(
            PreNormLayer(self.edge_nfeats),
        )

        # VARIABLE_EMBEDDING
        self.var_embedding = nn.Sequential(
            PreNormLayer(n_units=self.var_nfeats),
            nn.Linear(self.var_nfeats, self.emb_size, bias=True),
            self.activation,
            nn.Linear(self.emb_size, self.emb_size, bias=True),
            self.activation
        )

        # GRAPH CONVOLUTIONS
        self.conv_v_to_c = BipartiteGraphConvolution(self.emb_size, self.activation, self.initializer, right_to_left=True)
        self.conv_c_to_v = BipartiteGraphConvolution(self.emb_size, self.activation, self.initializer)

        # self.initialize_parameters()

    def forward(self, inputs):
        """
        Accepts stacked mini-batches, i.e. several bipartite graphs aggregated
        as one. In that case the number of variables per samples has to be
        provided, and the output consists in a padded dense tensor.

        Parameters
        ----------
        inputs: list of tensors
            Model input as a bipartite graph. May be batched into a stacked graph.

        Inputs
        ------
        constraint_features: 2D float tensor
            Constraint node features (n_constraints x n_constraint_features)
        edge_indices: 2D int tensor
            Edge constraint and variable indices (2, n_edges)
        edge_features: 2D float tensor
            Edge features (n_edges, n_edge_features)
        variable_features: 2D float tensor
            Variable node features (n_variables, n_variable_features)
        n_cons_per_sample: 1D int tensor
            Number of constraints for each of the samples stacked in the batch.
        n_vars_per_sample: 1D int tensor
            Number of variables for each of the samples stacked in the batch.
        """
        constraint_features, edge_indices, edge_features, variable_features, n_cons_per_sample, n_vars_per_sample = inputs
        n_cons_total = torch.sum(n_cons_per_sample)
        n_vars_total = torch.sum(n_vars_per_sample)

        # EMBEDDINGS
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # GRAPH CONVOLUTIONS
        constraint_features = self.conv_v_to_c((
            constraint_features, edge_indices, edge_features, variable_features, n_cons_total))
        constraint_features = self.activation(constraint_features)

        variable_features = self.conv_c_to_v((
            constraint_features, edge_indices, edge_features, variable_features, n_vars_total))
        variable_features = self.activation(variable_features)

        return variable_features


class BaseModel(nn.Module):
    def initialize_parameters(self):
        for l in self.modules():
            if isinstance(l, torch.nn.Linear):
                self.initializer(l.weight.data)
                if l.bias is not None:
                    torch.nn.init.constant_(l.bias.data, 0)

    def pre_train_init(self):
        for module in self.modules():
            if isinstance(module, PreNormLayer) or module._get_name() == "PreNormLayer":
                module.start_updates()

    def pre_train(self, state):
        with torch.no_grad():
            try:
                self.forward(state)
                return False
            except PreNormException:
                return True

    def pre_train_next(self):
        for module in self.modules():
            if (isinstance(module, PreNormLayer) or module._get_name() == "PreNormLayer") \
                    and module.waiting_updates and module.received_updates:
                module.stop_updates()
                return module
        return None

    def save_state(self, filepath):
        torch.save(self.state_dict(), filepath)

    def restore_state(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))


class Policy(BaseModel):
    def __init__(self):
        super(Policy, self).__init__()

        self.root_gcn = GCNPolicy()
        self.n_input_feats = 92
        self.root_emb_size = self.root_gcn.emb_size
        self.n_layers = 1

        self.activation = torch.nn.LeakyReLU()
        self.initializer = lambda x: torch.nn.init.orthogonal_(x, gain=1)

        # HYPERNET GENEARTOR
        self.weight_generator = nn.Sequential(
            nn.LayerNorm(self.root_emb_size),
            nn.Linear(self.root_emb_size, self.n_layers * self.n_input_feats * 3)
        )

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
        root_c, root_ei, root_ev, root_v, root_n_cs, root_n_vs, candss, cand_feats, _ = inputs

        variable_features = self.root_gcn((root_c, root_ei, root_ev, root_v, root_n_cs, root_n_vs))
        cand_root_feats = variable_features[candss]

        film_parameters = self.weight_generator(cand_root_feats)
        film_parameters = film_parameters.view(-1, 3, self.n_layers * self.n_input_feats)

        cand_feats = cand_feats.repeat(1, self.n_layers)
        film_feats = self.activation(film_parameters[:,0] * cand_feats + film_parameters[:,1])
        output = torch.sum(film_parameters[:,2] * film_feats, axis=-1)
        output = torch.reshape(output, [1, -1])
        return F.normalize(variable_features, p=2, dim=1), output, film_parameters

    def get_params(self, inputs):
        """
        Returns parameters/variable representations inferred at the root node.

        Parameters
        ----------
        inputs : torch.tensor
            inputs to be used by the root node GNN

        Returns
        -------
        (torch.tensor): variable representations / parameters as inferred from root gcnn and to be used else where in the tree.
        """
        variable_features = self.root_gcn(inputs)
        film_parameters = self.weight_generator(variable_features)
        return film_parameters.view(-1, 3, self.n_layers * self.n_input_feats)

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
        cand_feats = cand_feats.repeat(1, self.n_layers)
        film_feats = self.activation(film_parameters[:,0] * cand_feats + film_parameters[:,1])
        output = torch.sum(film_parameters[:,2] * film_feats, axis=-1)
        output = torch.reshape(output, [1, -1])
        return output
