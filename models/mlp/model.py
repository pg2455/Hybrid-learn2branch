import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(torch.nn.Module):
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

class Policy(Model):
    def __init__(self):
        super(Model, self).__init__()

        self.n_input_feats = 92
        self.ff_size = 256

        self.activation = torch.nn.ReLU()

        # OUTPUT
        self.output_module = nn.Sequential(
            nn.Linear(self.n_input_feats, self.ff_size, bias=True),
            self.activation,
            nn.Linear(self.ff_size, self.ff_size, bias=True),
            self.activation,
            nn.Linear(self.ff_size, self.ff_size, bias=True),
            self.activation,
            nn.Linear(self.ff_size, 1, bias=False)
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
        features = inputs
        output = self.output_module(features)
        output = torch.reshape(output, [1, -1])
        return output
