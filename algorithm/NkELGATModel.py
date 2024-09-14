import numpy as np
import torch
import torch.nn as nn

from NkELGATNet import NkELGATNet


class NkELGATModel(nn.Module):
    def __init__(self, para_train_data: np.ndarray = None,
                 para_train_target: np.ndarray = None,
                 para_k_label: int = 2,
                 para_parallel_layer_nodes: list = [],
                 para_local_activators: str = "s" * 100,
                 para_alpha: float = 0.5,
                 para_gin_layer_num: int = 2,
                 para_gin_out_features: int = 32,
                 para_gin_layer_nodes: list = [],
                 para_learning_rate: float = 0.001,
                 para_loops: int = 500,
                 para_gin_activators: str = "s" * 100
                 ):
        super().__init__()

        self.device = torch.device("cuda")
        self.train_target = torch.from_numpy(para_train_target).float().to(self.device)
        self.alpha = para_alpha
        self.loops = para_loops
        self.label_adj = self.get_adjacency_matrix(para_train_target).to(self.device)

        self.model = NkELGATNet(para_train_data, para_train_target,
                                para_k_label, para_parallel_layer_nodes, para_local_activators,
                                self.label_adj, para_gin_layer_num, para_train_target.shape[1], para_gin_out_features,
                                para_gin_layer_nodes)

        self.optimizer = torch.optim.Adam(self.model.get_config_optim(), lr=para_learning_rate)
        self.loss_function = nn.CrossEntropyLoss().to(self.device)
        self.to(self.device)

        pass

    def fit(self):
        loss_local = []
        loss_global = []
        loss_list = []
        temp_loops = self.loops
        for epoch in range(temp_loops):
            temp_loss_local, temp_local_output, temp_res = self.model.fit();
            temp_loss_global = self.loss_function(temp_res, self.train_target)
            temp_loss = self.alpha * temp_loss_local + (1 - self.alpha) * temp_loss_global

            loss_local.append(temp_loss_local.item())
            loss_global.append(temp_loss_global.item())
            loss_list.append(temp_loss.item())

            self.optimizer.zero_grad()
            temp_loss.backward()
            self.optimizer.step()
        return temp_loss;

    def predict(self, para_test_data):
        temp_local_output = self.model.predict(para_test_data);

        return temp_local_output

    def get_adjacency_matrix(self, para_train_target):
        """
        根据条件概率计算邻接矩阵: A_{ij} =  1/2[P(l_i | l_j) + P(l_j | l_i)] 对角线元素值为0
        @return: res
        """
        temp_labels = torch.from_numpy(para_train_target).float().to(self.device)
        # compute label correlation matrix
        adj = torch.matmul(temp_labels.t(), temp_labels)
        y_sum = torch.sum(temp_labels.t(), dim=1, keepdim=True)
        y_sum[y_sum < 1e-6] = 1e-6
        adj = adj / y_sum
        adj = (adj + adj.t()) * 0.5

        q = adj.size(0)
        for i in range(q):
            adj[i, i] = 0

        return adj