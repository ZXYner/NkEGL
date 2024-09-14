import torch
import torch.nn as nn

from NkEL import NkEL
from GIN import GIN


class NkELGATNet(nn.Module):
    def __init__(self,
                 para_train_data,
                 para_train_target,
                 para_k_label=2,
                 para_parallel_layer_nodes=[],
                 para_local_activators="ssss",
                 para_label_adj=[],
                 para_gin_layer_num: int = 2,
                 para_in_features: int = 0,
                 para_out_features: int = 0,
                 para_gin_layer_nodes=[],
                 ):
        super().__init__()
        self.device = torch.device("cuda")
        self.NkEL_model = NkEL(para_train_data,
                               para_train_target,
                               para_k_label,
                               para_parallel_layer_nodes,
                               para_local_activators).to(self.device)
        self.GIN_model = GIN(para_label_adj, para_gin_layer_num, para_in_features, para_out_features,
                             para_gin_layer_nodes).to(self.device)
        self.cls_conv = nn.Conv1d(para_in_features, para_in_features, para_out_features, groups=para_in_features)
        pass

    def get_config_optim(self):
        return [{'params': self.NkEL_model.parameters()},
                {'params': self.GIN_model.parameters()},
                {'params': self.cls_conv.parameters()}
                ]

    def fit(self):
        temp_local_output, temp_loss_local = self.NkEL_model.fit();
        temp_local_output = torch.from_numpy(temp_local_output).float().to(self.device)
        temp_num_instance = temp_local_output.shape[0]
        temp_num_label = temp_local_output.shape[1]
        temp_input = temp_local_output.unsqueeze(-1).expand(-1, -1, temp_num_label)
        temp_input = temp_input.transpose(1, 2)
        temp_input = self.GIN_model(temp_input)
        temp_input = self.cls_conv(temp_input).squeeze(2)
        return temp_loss_local, temp_local_output, temp_input

    def predict(self, para_input):
        temp_local_output = self.NkEL_model.predict(para_input);
        temp_local_output = torch.from_numpy(temp_local_output).float().to(self.device)

        return temp_local_output
