import numpy as np

import torch
import torch.nn.functional as F
from torch import optim
from torch import nn

from Model import AutoEncoderLayer


def binarization(x_1, x_2, y_1):
    bool_train_y = y_1.astype(np.bool)
    for col in range(x_1.shape[1]):
        train_x_vec, test_x_vec = x_1[:, col], x_2[:, col]
        x_true, x_false = train_x_vec[bool_train_y], train_x_vec[np.logical_not(bool_train_y)]
        thres = (np.median(x_true) + np.median(x_false)) / 2
        x_1[:, col] = train_x_vec > thres
        x_2[:, col] = test_x_vec > thres
    return x_1.astype(np.float), x_2.astype(np.float)


def assert_tensor(x):
    if torch.is_tensor(x):
        return x
    else:
        return torch.tensor(x, dtype=torch.float32)


def sae_config(x, h_units):
    sae = nn.ModuleList()
    for n in range(len(h_units)):
        if n == 0:
            ae = AutoEncoderLayer(x.shape[1], h_units[n])
        else:
            ae = AutoEncoderLayer(h_units[n - 1], h_units[n])
        sae.append(ae)
    return sae


def sae_pretraining(x, h_units, n_train):
    total_ae_weights = []
    total_ae_biases = []
    for n in range(n_train):
        x_copy = x
        sae = sae_config(x_copy, h_units)
        ae_weights = []
        ae_biases = []
        for n_layer, layer in enumerate(sae):
            print('AutoEncoder:%i/%i' % (n_layer + 1, len(sae)))
            x_copy, weight, bias = ae_train(x_copy, layer)
            ae_weights.append(weight)
            ae_biases.append(bias)
        total_ae_weights.append(ae_weights)
        total_ae_biases.append(ae_biases)
    return total_ae_weights, total_ae_biases


def ae_train(x, ae, n_epochs=100, lr=0.002):
    optimizer = optim.Adam(ae.parameters(), lr=lr)
    x = assert_tensor(x)
    count, epoch = 0, 0
    while (epoch < n_epochs) and (count < 10):
        pred = ae(x)
        loss = F.mse_loss(pred, x)
        if loss < 0.01:
            count += 1
        else:
            count = 0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch += 1
        print('Epoch:', epoch, 'MSE_loss:', loss.item())
    weight, bias = ae.encoder.weight, ae.encoder.bias
    h_output = ae.get_hidden_output(x)
    return h_output.detach(), weight.detach(), bias.detach()


def median_init(nested_list):
    m_params = []
    n_train = len(nested_list)
    n_params = len(next(iter(nested_list)))
    for j in range(n_params):
        params = torch.stack([nested_list[i][j] for i in range(n_train)])
        m_param = torch.median(params, dim=0)[0]
        m_params.append(m_param)
    return m_params


def train_nn(model, optimizer, x, y, n_epochs=500):
    x = assert_tensor(x)
    y = assert_tensor(y)
    model.train()
    for epoch in range(n_epochs):
        nn_pred = torch.squeeze(model(x))
        loss = F.binary_cross_entropy_with_logits(nn_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model.get_nn_weights()


def nn_ldc(model, x):
    model.eval()
    ldc = model.get_ldc_output(x).detach().numpy()
    return ldc

