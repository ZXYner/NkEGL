import json
import os
import scipy.io as scio
import torch
from sklearn import preprocessing
from sklearn.model_selection import KFold

from NkELGATModel import NkELGATModel
from utils.metrics import *


class Demo:
    def __init__(self, para_file, para_output_file_path="", para_file_name="",
                 para_k_label=4, para_nodes=[], para_local_activators="ssss", para_alpha=0.5,
                 para_gin_layer_num: int = 2,
                 para_gin_out_features: int = 32,
                 para_gin_layer_nodes: list = [32],
                 para_learning_rate: float = 0.001,
                 para_loops: int = 500
                 ):
        assert scio.loadmat(para_file), print("读取文件失败")

        all_data = scio.loadmat(para_file)
        self.file_path = para_output_file_path
        self.file_name = para_file_name
        train_data = all_data['data']
        self.train_target = all_data['targets']
        min_max_scaler = preprocessing.MinMaxScaler()
        self.train_data = min_max_scaler.fit_transform(np.array(train_data))
        self.train_target[self.train_target < 0] = 0
        self.train_target_t = self.train_target.transpose()
        self.device = torch.device('cuda')
        self.k_label = para_k_label
        self.node_list = para_nodes
        self.local_activators = para_local_activators
        self.gin_layer_num: int = para_gin_layer_num
        self.gin_out_features: int = para_gin_out_features
        self.gin_layer_nodes: list = para_gin_layer_nodes
        self.learning_rate: float = para_learning_rate
        self.alpha = para_alpha
        self.loops = para_loops

    def k_fold_test(self):
        kf = KFold(n_splits=5, shuffle=True)

        peak_f1_list_0 = np.array([])
        ndcg_list_0 = np.array([])
        auc_list_0 = np.array([])
        ranking_loss_list_0 = np.array([])
        hamming_loss_list_0 = np.array([])
        one_error_list_0 = np.array([])
        coverage_list_0 = np.array([])

        temp_feature_num = self.train_data.shape[1]
        temp_parallel_nodes = [temp_feature_num]
        for i in self.node_list:
            temp_parallel_nodes.append(i)
        temp_parallel_nodes.append(2 ** self.k_label)
        for train_index, test_index in kf.split(self.train_data):
            temp_train_data = self.train_data[train_index, :]
            temp_test_data = self.train_data[test_index, :]
            temp_train_target = self.train_target[train_index, :]
            temp_test_target = self.train_target[test_index, :]

            nkegl_model = NkELGATModel(temp_train_data, temp_train_target,
                                      self.k_label, temp_parallel_nodes, self.local_activators, self.alpha,
                                      self.gin_layer_num, self.gin_out_features, self.gin_layer_nodes,
                                      self.learning_rate, self.loops).to(self.device)
            # train
            nkegl_model.fit()
            # predict
            temp_local_predict = nkegl_model.predict(temp_test_data)
            temp_local_predict = temp_local_predict.cpu().detach().numpy()

            peak_f1_value_0 = peak_f1_score(temp_test_target, temp_local_predict)
            ndcg_value_0 = ndcg(temp_test_target, temp_local_predict)
            auc_value_0 = micro_averaging_auc(temp_test_target, temp_local_predict)
            one_error_value_0 = one_error(temp_test_target, temp_local_predict)
            ranking_loss_value_0 = ranking_loss(temp_test_target, temp_local_predict)
            hamming_loss_value_0 = hamming_loss(temp_test_target, temp_local_predict)
            coverage_value_0 = coverage(temp_test_target, temp_local_predict)

            # 计算评价指标值
            peak_f1_list_0 = np.append(peak_f1_list_0, peak_f1_value_0)
            ndcg_list_0 = np.append(ndcg_list_0, ndcg_value_0)
            auc_list_0 = np.append(auc_list_0, auc_value_0)
            one_error_list_0 = np.append(one_error_list_0, one_error_value_0)
            ranking_loss_list_0 = np.append(ranking_loss_list_0, ranking_loss_value_0)
            hamming_loss_list_0 = np.append(hamming_loss_list_0, hamming_loss_value_0)
            coverage_list_0 = np.append(coverage_list_0, coverage_value_0)

        mean_peak_f1_0 = np.mean(peak_f1_list_0)
        std_peak_f1_0 = np.std(peak_f1_list_0)
        mean_ndcg_0 = np.mean(ndcg_list_0)
        std_ndcg_0 = np.std(ndcg_list_0)
        mean_auc_0 = np.mean(auc_list_0)
        std_auc_0 = np.std(auc_list_0)

        mean_one_error_0 = np.mean(one_error_list_0)
        std_one_error_0 = np.std(one_error_list_0)
        mean_ranking_loss_0 = np.mean(ranking_loss_list_0)
        std_ranking_loss_0 = np.std(ranking_loss_list_0)
        mean_hamming_loss_0 = np.mean(hamming_loss_list_0)
        std_hamming_loss_0 = np.std(hamming_loss_list_0)
        mean_coverage_0 = np.mean(coverage_list_0)
        std_coverage_0 = np.std(coverage_list_0)

        with open(self.file_path + self.file_name + ".txt", "w") as f:
            f.write("peak_f1_0: " + str(round(mean_peak_f1_0, 4)) + "  "
                    + str(round(std_peak_f1_0, 4)) + "\n")
            f.write("ndcg_0: " + str(round(mean_ndcg_0, 4)) + "  "
                    + str(round(std_ndcg_0, 4)) + "\n")
            f.write("auc_0: " + str(round(mean_auc_0, 4)) + "  "
                    + str(round(std_auc_0, 4)) + "\n")
            f.write("one_error_0: " + str(round(mean_one_error_0, 4)) + "  "
                    + str(round(std_one_error_0, 4)) + "\n")
            f.write("ranking_loss_0: " + str(round(mean_ranking_loss_0, 4)) + "  "
                    + str(round(std_ranking_loss_0, 4)) + "\n")
            f.write("hamming_loss_0: " + str(round(mean_hamming_loss_0, 4)) + "  "
                    + str(round(std_hamming_loss_0, 4)) + "\n")
            f.write("coverage_0: " + str(round(mean_coverage_0, 4)) + "  " + str(round(std_coverage_0, 4)) + "\n")



if __name__ == '__main__':

    lists = ['Emotions']
    # lists = ['Birds', 'CAL500', 'CHD_49',
    #          'Emotions', 'Enron', 'Flags', 'Foodtruck', 'Genbase',
    #          'Image', 'Medical', 'ReutersK500',
    #          'Scene']
    config_name = '/home/zxy/PycharmProjects/NkEL_GIN/configuration/config.json'
    assert os.path.exists(config_name), 'Config file is not accessible.'
    # open json
    with open(config_name) as f:
        cfg = json.load(f)['nkegl']
    output_file_path = '../result/'
    for file_name in lists:
        print("begin: ***************************************************", file_name)
        file_path = cfg[file_name]['fileName']
        k_label = cfg[file_name]['kLabel']
        hidden_layer_nodes = cfg[file_name]['hiddenLayerNumNodes']
        local_activator = cfg[file_name]['activator'] * 100
        alpha = cfg[file_name]['alpha']
        loops = cfg[file_name]['loops']
        lr = cfg[file_name]['lr']
        gin_layer_num: int = cfg[file_name]['ginLayerNum']
        gin_out_features: int = cfg[file_name]['ginOutFeature']
        gin_layer_nodes: list = cfg[file_name]['ginLayerNode']
        nkegl = Demo(file_path, output_file_path, file_name,
                      k_label, hidden_layer_nodes, local_activator, alpha,
                      gin_layer_num, gin_out_features, gin_layer_nodes, lr, loops)
        nkegl.k_fold_test()
        print("end: ***************************************************", file_name)
