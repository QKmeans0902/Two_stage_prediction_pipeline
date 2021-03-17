import torch
from torch import nn


class AutoEncoderLayer(nn.Module):

    def __init__(self, input_dim, output_dim, training=True, xavier_init=True):
        super(AutoEncoderLayer, self).__init__()
        self.in_features = input_dim
        self.out_features = output_dim
        self.is_training = training
        self.encoder = nn.Linear(self.in_features, self.out_features, bias=True)
        self.decoder = nn.Linear(self.out_features, self.in_features, bias=True)
        if xavier_init:
            self.initialize_param()

    def forward(self, x):
        output = torch.sigmoid(self.encoder(x))
        if self.is_training:
            return torch.sigmoid(self.decoder(output))
        else:
            return output

    def initialize_param(self):
        nn.init.xavier_uniform_(self.encoder.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.decoder.weight, gain=nn.init.calculate_gain('sigmoid'))
        nn.init.constant_(self.encoder.bias, 0)
        nn.init.constant_(self.decoder.bias, 0)

    def get_hidden_output(self, x):
        self.is_training = False
        return self(x)


class FeatureReduction(nn.Module):

    def __init__(self, in_dim, h_units, m_weights, m_biases, training=True):
        super(FeatureReduction, self).__init__()
        self.model = nn.ModuleList()
        self.is_training = training
        for n in range(len(h_units)):
            if n == 0:
                fc = nn.Linear(in_dim, h_units[n], bias=True)
            else:
                fc = nn.Linear(h_units[n - 1], h_units[n], bias=True)
            fc.weight = nn.Parameter(m_weights[n], requires_grad=True)
            fc.bias = nn.Parameter(m_biases[n], requires_grad=True)
            self.model.append(fc)

    def forward(self, x):
        model_for_ldc = self.model[:-1]
        last_layer = self.model[-1]
        for fc in model_for_ldc:
            x = torch.sigmoid(fc(x))
        if self.is_training:
            return last_layer(x)
        else:
            return x

    def get_ldc_output(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)
        self.is_training = False
        return self(x)

    def get_nn_weights(self):
        nn_weights = []
        for fc in self.model:
            weight = fc.weight.detach().numpy()
            nn_weights.append(weight)
        return nn_weights

