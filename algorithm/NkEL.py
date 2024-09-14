import numpy as np
import torch
import torch.nn as nn


class NkEL(nn.Module):
    def __init__(self, para_train_data: np.array = None,
                 para_train_target: np.array = None,
                 para_k_label: int = 3,
                 para_nkel_parallel_nodes: list = [32],
                 para_activators: str = "s" * 100
                 ):
        super().__init__()
        self.train_data = para_train_data
        self.train_target = para_train_target
        self.k_label = para_k_label

        self.label_select = None
        self.label_embedding_num = np.zeros(self.train_target.shape[1])
        self.train_target_loss = None
        self.device = torch.device("cuda")

        self.get_nearest_label_subset()


        self.parallel_model = nn.ModuleList()
        for _ in range(len(self.label_select)):
            self.parallel_model.append(ParallelAnn(para_nkel_parallel_nodes, para_activators))

        self.loss_function = nn.MSELoss().to(self.device)
        self.soft_max = nn.Softmax(dim=1)
        pass

    def get_euclidean_dis(self):
        """
        Get the euclidean distance matrix between labels.

        @return: Distance matrix.
        """
        temp_data = self.train_target.T
        sum_x = np.sum(np.square(temp_data), 1)
        dis = np.add(np.add(-2 * np.dot(temp_data, temp_data.T), sum_x).T, sum_x)
        return np.sqrt(dis)

    def get_nearest_label_subset(self):
        """
        Construct the labelsets according to the distance.
        """
        print("select the nearest")
        distance_matrix = self.get_euclidean_dis()
        # distance_matrix = self.get_jaccard_dis()
        temp_label = np.size(distance_matrix, 0)
        result = []
        temp_select_list = []
        temp_train_target_list = []
        temp_label_index_length = len(str(temp_label))
        for i in range(temp_label):
            temp_distance_index = (np.argsort(distance_matrix[i])).tolist()
            temp_list = [i]
            temp_distance_index.remove(i)
            temp_index = 0
            while len(temp_list) < self.k_label:
                index_ = temp_distance_index[temp_index]
                if index_ not in temp_list:
                    temp_list.append(index_)
                temp_index += 1
            temp_list = (-1 * np.sort(-1 * np.array(temp_list))).tolist()
            temp_str = ''.join("0" * (temp_label_index_length - len(str(_))) + str(_) for _ in temp_list)
            # print(temp_list, "temp_sum", temp_str)
            if temp_str not in temp_select_list:
                temp_select_list.append(temp_str)
                result.append(temp_list)
                self.label_embedding_num[temp_list] += 1
                temp_train_target = self.transform_label_class(temp_list)
                temp_train_target_list.append(temp_train_target)
        self.label_select = np.array(result)
        temp_train_target_array = np.array(temp_train_target_list[0])
        if len(np.argwhere(np.array(self.label_embedding_num) == 0)) > 0:
            print("Some wrong with ", np.argwhere(np.array(self.label_embedding_num) == 0))
        for i in range(len(temp_train_target_list) - 1):
            temp_train_target_array = np.append(temp_train_target_array, temp_train_target_list[i + 1], axis=0)
            pass
        self.train_target_loss = torch.from_numpy(temp_train_target_array).float().to(self.device)
        pass

    def transform_label_class(self, para_list=[]):
        """
        Transform label value to class

        @param para_list: The labelset
        @return: The binary class value
        """
        temp_select = self.train_target[:, para_list]
        temp_instance_num, temp_label_num = temp_select.shape
        result_matrix = np.zeros((temp_instance_num, 2 ** temp_label_num), dtype=int)
        for i in range(temp_instance_num):
            temp_class = 0
            for j in range(temp_label_num):
                temp_class += temp_select[i][j] * (2 ** j)
            result_matrix[i][temp_class] = 1
            pass
        return result_matrix.tolist()

    def forward(self, para_input: np.ndarray = None):
        temp_input = torch.from_numpy(para_input).float().to(self.device)
        temp_parallel_output = [model(temp_input) for model in self.parallel_model]
        self.parallel_output = temp_parallel_output
        temp_output = temp_parallel_output[0]
        for i in range(len(temp_parallel_output) - 1):
            temp_output = torch.cat((temp_output, temp_parallel_output[i + 1]), 0)
        return temp_output

    # def one_round_train(self):
    #     """
    #     A round traning process.
    #
    #     @param para_input: Training data feature matrix.
    #     @return: Loss value.
    #     """
    #
    #     # Compute prediction error
    #     temp_out = self(self.train_data)
    #     temp_loss = self.loss_function(temp_out, self.train_target_loss)
    #
    #     # Backpropagation
    #     self.optimizer.zero_grad()
    #     temp_loss.backward()
    #     self.optimizer.step()
    #
    #     return temp_loss.item()

    def fit(self):
        temp_out = self(self.train_data)
        temp_loss = self.loss_function(temp_out, self.train_target_loss)
        temp_label_result = self.get_label_result(self.train_data.shape[0])

        return temp_label_result, temp_loss

    def predict(self, para_input):
        self(para_input)
        temp_label_result = self.get_label_result(para_input.shape[0])
        return temp_label_result

    # def get_label_result(self, para_instance_num: int = 0):
    #     temp_parallel_output = self.parallel_output
    #     temp_label_subsets_num = len(self.label_select)
    #     temp_target_shape = (para_instance_num, self.train_target.shape[1])
    #     result_matrix = np.zeros(temp_target_shape)
    #     para_instance_num = self.train_data.shape[0]
    #
    #     for i in range(temp_label_subsets_num):
    #         temp_predict_result = temp_parallel_output[i]
    #         temp_label_subset = self.label_select[i]
    #         temp_predict_prob_result = self.soft_max(temp_predict_result).to(self.device).cpu().detach().numpy()
    #         temp_class_label_matrix = transform_class_label(self.k_label)
    #         temp_predict = np.zeros((para_instance_num, self.k_label))
    #         temp_result_matrix = np.zeros(temp_target_shape)
    #
    #         for j in range(self.k_label):
    #             temp_index = temp_class_label_matrix[j]
    #             for k in temp_index:
    #
    #                 temp_predict[:, j] += temp_predict_prob_result[:, k]
    #             temp_label = temp_label_subset[j]
    #             temp_embedding_num = self.label_embedding_num[temp_label]
    #             temp_result_matrix[:, temp_label] = temp_predict[:, j] / temp_embedding_num
    #             #   求各个网络的预测值的和
    #         result_matrix += temp_result_matrix
    #     return result_matrix

    def get_label_result(self, para_instance_num: int = 0):
        temp_parallel_output = self.parallel_output
        temp_label_subsets_num = len(self.label_select)
        temp_target_shape = (para_instance_num, self.train_target.shape[1])
        result_matrix = np.zeros(temp_target_shape)

        for i in range(temp_label_subsets_num):
            temp_predict_result = temp_parallel_output[i]
            temp_label_subset = self.label_select[i]
            temp_predict_prob_result = self.soft_max(temp_predict_result).to(self.device).cpu().detach().numpy()
            temp_class_label_matrix = transform_class_label(self.k_label)
            temp_predict = np.zeros((para_instance_num, self.k_label))
            temp_result_matrix = np.zeros(temp_target_shape)

            for j in range(self.k_label):
                temp_index = temp_class_label_matrix[j]
                for k in temp_index:
                    temp_predict[:, j] += temp_predict_prob_result[:, k]
                temp_label = temp_label_subset[j]
                temp_embedding_num = self.label_embedding_num[temp_label]
                temp_result_matrix[:, temp_label] = temp_predict[:, j] / temp_embedding_num
                #   求各个网络的预测值的和
            result_matrix += temp_result_matrix
        return result_matrix


class ParallelAnn(nn.Module):
    def __init__(self, para_parallel_layer_nodes=[], para_activators: str = "s" * 100):
        """
         Construct a parallel structure neural network

        @param para_parallel_layer_nodes: The nodes of hidden layer
        @param para_activators: Activator function string
        """
        super().__init__()
        temp_model = []
        for i in range(len(para_parallel_layer_nodes) - 1):
            temp_input = para_parallel_layer_nodes[i]
            temp_out = para_parallel_layer_nodes[i + 1]
            temp_linear_model = nn.Linear(temp_input, temp_out)
            temp_model.append(temp_linear_model)
            temp_model.append(get_activator(para_activators[i]))
            # temp_model.append(nn.Dropout(p=0.1))
            pass
        self.model = nn.Sequential(*temp_model)

        pass

    def forward(self, para_input: torch.tensor = None):
        temp_output = self.model(para_input)

        return temp_output

    pass


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


def transform_class_label(para_label_select_len: int = 0):
    """
    Transform class to labels.
    1 2 3     class
    0 0 0 --> 1
    1 0 0 --> 2
    0 1 0 --> 3
    1 1 0 --> 4
    0 0 1 --> 5
    1 0 1 --> 6
    0 1 1 --> 7
    1 1 1 --> 8
    So for the label 1 label we will get [2,4,6,8]
    Note: due to the characteristic of python, we start from class 0
    @param para_label_select_len: Value of k
    @return:A matrix
    """
    temp_matrix = []
    for i in range(2 ** para_label_select_len):
        temp_index_value = i
        tem_list = []
        for j in range(para_label_select_len):
            tem_list.append(temp_index_value % 2)
            temp_index_value //= 2
        temp_matrix.append(tem_list)
    temp_matrix = np.array(temp_matrix).transpose()

    result_matrix = []
    for i in range(para_label_select_len):
        temp_class_index = np.argwhere(temp_matrix[i] > 0).flatten().tolist()
        result_matrix.append(temp_class_index)
    return np.array(result_matrix)
