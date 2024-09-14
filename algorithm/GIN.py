import torch
import torch.nn as nn


class GIN(nn.Module):
    def __init__(self, label_adj, num_layers, in_features, out_features, hidden_features=[],
                 eps=0.0, train_eps=True, residual=True):
        super(GIN, self).__init__()

        self.label_adj = label_adj

        self.GINLayers = nn.ModuleList()

        if in_features != out_features:
            first_layer_res = False
        else:
            first_layer_res = True
        self.GINLayers.append(GINLayer(MLP(in_features, out_features, hidden_features),
                                       eps, train_eps, first_layer_res))
        for i in range(num_layers - 1):
            self.GINLayers.append(GINLayer(MLP(out_features, out_features, hidden_features),
                                           eps, train_eps, residual))

    def forward(self, input):
        for l in self.GINLayers:
            input = l(input, self.label_adj)
        return input


class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=[],nonlinearity = 'relu'):
        super(MLP, self).__init__()
        self.nonlinearity = nonlinearity
        self.fcs = nn.ModuleList()
        if hidden_features:
            in_dims = [in_features] + hidden_features
            out_dims = hidden_features + [out_features]
            for i in range(len(in_dims)):
                self.fcs.append(nn.Linear(in_dims[i], out_dims[i]))
                self.fcs.append(nn.LeakyReLU(0.1, inplace=True))
                # self.fcs.append(nn.Sigmoid())
                # self.fcs.append(nn.ReLU(inplace=True))
                # self.fcs.append(nn.Tanh())
                # self.fcs.append(nn.Dropout(p=0.1))
        else:
            self.fcs.append(nn.Linear(in_features, out_features))
            self.fcs.append(nn.Sigmoid())

        self.reset_parameters()

    def reset_parameters(self):
        for l in self.fcs:
            if l.__class__.__name__ == 'Linear':
                nn.init.kaiming_uniform_(l.weight, a=0.1,
                                         nonlinearity=self.nonlinearity)
                if self.nonlinearity == 'leaky_relu' or self.nonlinearity == 'relu':
                    nn.init.uniform_(l.bias, 0, 0.1)
                else:
                    nn.init.constant_(l.bias, 0.0)
            elif l.__class__.__name__ == 'BatchNorm1d':
                l.reset_parameters()

    def forward(self, input):
        for l in self.fcs:
            input = l(input)
        return input


class GINLayer(nn.Module):
    def __init__(self, mlp, eps=0.0, train_eps=True, residual=True):
        super(GINLayer, self).__init__()
        self.mlp = mlp
        self.initial_eps = eps
        self.residual = residual
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        self.reset_parameters()

    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.eps.data.fill_(self.initial_eps)

    def forward(self, input, adj):
        res = input

        # Aggregating neighborhood information
        neighs = torch.matmul(adj, res)

        # Reweighting the center node representation
        res = (1 + self.eps) * res + neighs

        # Updating node representations
        res = self.mlp(res)

        # Residual connection
        if self.residual:
            output = res + input
        else:
            output = res

        return output

def get_activator(para_activator: str = 's'):
    '''
    Parsing the specific char of activator.

    :param para_activator: specific char of activator.
    :return: Activator layer.
    '''
    if para_activator == 's':
        return nn.Sigmoid()
    elif para_activator == 'r':
        return nn.ReLU()
    elif para_activator == 'l':
        return nn.LeakyReLU()
    elif para_activator == 'e':
        return nn.ELU()
    elif para_activator == 'u':
        return nn.Softplus()
    elif para_activator == 'o':
        return nn.Softsign()
    else:
        return nn.Sigmoid()
