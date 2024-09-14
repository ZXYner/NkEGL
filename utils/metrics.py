"""
The metrics of multi-label learning. "n" is the number of instances and "q" is the number of labels.

Args:
    target:   n × q, binary value 0, 1
    predict:  n × q, real-value in [0, 1]
"""
import numpy as np


def accuracy(para_target, para_predict, threshold=0.5):
    # count the number of instances and labels respectively
    n, q = len(para_target), len(para_target[0])

    # transform real value score into binary value for "predict" according to threshold
    predict = np.array(para_predict)
    predict[predict >= threshold] = 1
    predict[predict < threshold] = 0

    return np.sum(predict == para_target) / (n * q)


def average_precision(para_target, para_predict):
    # obtain matrix composed of rows that are not all 0
    row_sums = np.sum(para_target != 0, axis=1)
    nonzero_indices = np.nonzero(row_sums)[0]
    target = para_target[nonzero_indices, :]
    predict = para_predict[nonzero_indices, :]

    # count the number of instances and labels respectively
    n, q = len(target), len(target[0])

    # acquire the rank(bigger value has higher rank) of each predict value in each data point(row)
    rank = np.array(predict)
    rank_index = np.argsort(predict, 1)
    for row in range(n):
        rank[row][rank_index[row]] = range(q, 0, -1)

    # compute the rank score for each 1 in each data point
    rank_scores = np.zeros(n)
    rows, cols = np.nonzero(target)
    for row, col in zip(rows, cols):
        rank_scores[row] += np.sum((target[row, :] == 1) * (predict[row, :] >= predict[row, col])) / rank[row, col]

    # find the index of data point(row) which does not justly consist of zero
    num_ones = np.sum(target, 1)
    not_zero_rows = num_ones > 0

    return np.sum(rank_scores[not_zero_rows] / num_ones[not_zero_rows]) / n


def coverage(para_target, para_predict):
    # count the number of instances and labels respectively
    n, q = len(para_target), len(para_target[0])

    # acquire the rank(bigger value has higher rank) of each predict value in each data point(row)
    rank = np.array(para_predict)
    rank_index = np.argsort(para_predict, 1)
    for row in range(n):
        rank[row][rank_index[row]] = range(q, 0, -1)

    return np.sum(np.max(rank * para_target, 1) - 1) / (n * q)


def hamming_loss(para_target, para_predict, para_threshold=0.5):
    # count the number of instances and labels respectively
    n, q = len(para_target), len(para_target[0])

    # transform real value score into binary value for "predict" according to threshold
    predict = np.array(para_predict)
    predict[predict >= para_threshold] = 1
    predict[predict < para_threshold] = 0

    # compute the number of "one" for each data point(row)
    num_ones_tar = np.sum(para_target, 1)
    num_ones_pre = np.sum(predict, 1)

    # compute the number of joint "one" between "predict" and "target" for each point(row)
    num_ones_joint = np.sum(predict * para_target, 1)

    return np.sum(num_ones_tar + num_ones_pre - 2 * num_ones_joint) / (n * q)


def macro_averaging_auc(para_target, para_predict):
    # count the number of instances and labels respectively
    n, q = len(para_target), len(para_target[0])

    # count the number of right pair for each label (column)
    right_pair_cnt = np.zeros(q)
    rows, cols = np.nonzero(para_target)
    for row, col in zip(rows, cols):
        right_pair_cnt[col] += np.sum((para_target[:, col] == 0) * (para_predict[:, col] <= para_predict[row, col]))

    # find the index of label(column) which consists of both zero and one
    ones_cnt = np.sum(para_target, 0)
    not_trivial_vec_cols = (ones_cnt > 0) * (ones_cnt < n)

    # count the number of all pair for each label (column) which consists of both zero and one
    all_pair_cnt = ones_cnt[not_trivial_vec_cols] * (n - ones_cnt[not_trivial_vec_cols])

    return np.sum(right_pair_cnt[not_trivial_vec_cols] / all_pair_cnt) / q


def micro_averaging_auc(para_target, para_predict):
    # count the number of instances and labels respectively
    n, q = len(para_target), len(para_target[0])

    # count the number of right pair
    rows, cols = np.nonzero(para_target)
    num_right_pair = 0
    for row, col in zip(rows, cols):
        num_right_pair += np.sum((para_target == 0) * (para_predict <= para_predict[row, col]))

    # count the number of all pair
    num_ones = np.sum(para_target)
    num_zeros = n * q - num_ones
    num_all_pair = num_ones * num_zeros

    return num_right_pair / num_all_pair


def ndcg(para_target, para_predict):
    # flatten "predict" and "target" as vector
    predict = np.array(para_predict).reshape(-1)
    target = np.array(para_target).reshape(-1)

    # sort "target" in descending order of probability values in "predict"
    index = np.argsort(-predict)
    target = target[index]

    # compute DCG
    DCG, num_ele = 0, len(target)
    for i in range(num_ele):
        DCG += target[i] / np.log2(i + 2)

    # compute IDCG
    IDCG, num_ones = 0, np.sum(target)
    for i in range(num_ones):
        IDCG += 1 / np.log2(i + 2)

    return DCG / IDCG


def one_error(para_target, para_predict):
    # obtain matrix composed of rows that are not all 0
    row_sums = np.sum(para_target != 0, axis=1)
    nonzero_indices = np.nonzero(row_sums)[0]
    target = para_target[nonzero_indices, :]
    predict = para_predict[nonzero_indices, :]

    # count the number of instances
    n = len(target)

    # obtain the index of maximum element for each data point(row)
    index = np.argmax(predict, 1)

    return np.sum(1 - target[range(n), index]) / n


def peak_f1_score(para_target, para_predict):
    # count the number of instances and labels respectively
    n, q = len(para_target), len(para_target[0])

    # flatten "predict" and "target" as vector
    predict = np.array(para_predict).reshape(-1)
    target = np.array(para_target).reshape(-1)

    # acquire index of ascending ordered "predict"
    index = np.argsort(predict)

    # compute TP+FN
    TP_FN = np.sum(target)

    # compute all f1 scores according to different threshold
    num_ele = n * q
    f1_scores = np.zeros(num_ele)
    for i, ind in enumerate(index):
        TP = np.sum((predict >= predict[ind]) * (target == 1))
        P = TP / (num_ele - i)
        R = TP / TP_FN
        f1_scores[i] = 0 if (P + R) == 0 else (2 * P * R) / (P + R)

    return np.max(f1_scores)


def ranking_loss(para_target, para_predict):
    # count the number of instances and labels respectively
    n, q = len(para_target), len(para_target[0])

    # count the number of right pair for each data point (row)
    right_pair_cnt = np.zeros(n)
    rows, cols = np.nonzero(para_target)
    for row, col in zip(rows, cols):
        right_pair_cnt[row] += np.sum((para_target[row, :] == 0) * (para_predict[row, :] >= para_predict[row, col]))

    # find the index of data point(row) which consists of both zero and one
    ones_cnt = np.sum(para_target, 1)
    not_trivial_vec_rows = (ones_cnt > 0) * (ones_cnt < q)

    # count the number of all pair for each data point (row) which consists of both zero and one
    all_pair_cnt = ones_cnt[not_trivial_vec_rows] * (q - ones_cnt[not_trivial_vec_rows])

    return np.sum(right_pair_cnt[not_trivial_vec_rows] / all_pair_cnt) / n
