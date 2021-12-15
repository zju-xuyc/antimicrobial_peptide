import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from settings import MAX_MIC
import os


def get_dict():
    Letter_dict = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12,
                   'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20}
    return Letter_dict


def get_reverse_dict():
    reverse_dict = {1: 'A', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N',
                    13: 'P', 14: 'Q', 15: 'R', 16: 'S', 17: 'T', 18: 'V', 19: 'W', 20: 'Y'}
    return reverse_dict


def build_index(data, Letter_dict):
    """
    Preprocess
    负责词表的映射
    建立词向量
    input: data 序列
    output: 映射完的数据
    """
    data_process = []
    for i in range(len(data)):
        tmp = []
        for j in range(len(data[i])):
            tmp.append(Letter_dict[data[i][j]])
        data_process.append(tmp)
        # if len(tmp)<=2:
        #     print(data[i])
    return data_process


def df2list(data_filter, varible, label, l_type, ngram_num, log_num=2):
    """
    将dataframe转换为需要的list对象 格式为[["sequence"],["label"]],
    并将其中的完整list转换为n_gram的形式
    input:
        data_filter: 原始dataframe
        variable: sequence序列
        label: 抗菌性的结果
        n_gram: n_gram number
    output:
        all_data: 格式化并且分完词的数据
    """
    all_data = []
    for i in data_filter.iterrows():
        tmp = []
        if len(list(i[1][varible])) <= 50:
            tmp.append(create_ngram_list(i[1][varible], ngram_num))  # i[1][varible]是sequence
        else:
            tmp.append(create_ngram_list(i[1][varible][0:49], ngram_num))
        if log_num == 2:
            tmp.append(float(np.log2(float(i[1][label]))))
            tmp.append(i[1][l_type])
            tmp.append(len(tmp[0]))
        elif log_num == 10:
            tmp.append(float(np.log10(float(i[1][label]))))
            tmp.append(i[1][l_type])
            tmp.append(len(tmp[0]))
        else:
            tmp.append(i[1][label])
            tmp.append(i[1][l_type])
            tmp.append(len(tmp[0]))
        all_data.append(tmp)
    return all_data


def create_ngram_list(input_list, ngram_num):
    """
    建立n分词的列表
    input:
        input_list: 需要切分的list
        ngram_num:
    output:
        ngram_list: 切好的list
    """
    ngram_list = []
    if len(input_list) < ngram_num:
        ngram_list = [x for x in input_list]
    else:
        for i in range(len(input_list) - ngram_num + 1):
            ngram_list.append(input_list[i:i + ngram_num])
    return ngram_list


def user_func(batch):  # 两种padding的顺序
    """
    Torch.DataLoader
    输出转换后的数据
    [output, output_label,output_type, length]
    """
    output = []
    output_label = []
    output_type = []
    length = []
    batch = sorted(batch, key=lambda x: x[3], reverse=True)
    for i in batch:
        output.append(i[0])
        output_label.append(i[1])
        output_type.append(i[2])
        length.append(i[3])
    output = torch.tensor(output)
    output_label = torch.tensor(output_label)
    output_type = torch.tensor(output_type)
    length = torch.tensor(length)

    return output, output_label, output_type, length


def get_randomed_featured_data(data_path):
    data = pd.read_csv(data_path, encoding="utf8")
    try:
        data = data.drop(["Unnamed: 0"], axis=1)
    except:
        print(f"{data_path} no Unnamed: 0")
    # data = data.sample(frac=1).reset_index(drop=True)
    data_x = data.iloc[:, 1:-2].values
    data_y = np.log10(data.iloc[:, -2].values)
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)
    return x_train, x_test, y_train, y_test


def evaluate_customized(model, test_y, predict_y):
    """
    输出 topK_MSE
    model: 无用 单纯为了统一输入
    test_y: 测试标签值
    predict_y: 预测值
    return
    top20_mse,
    top60_mse,
    all_active_error, 正样例误差
    all_error         总误差
    """
    test_y = list(test_y)
    predict_y = list(predict_y)
    predict_matrix = [test_y, predict_y]  # 按label由小到大排序
    predict_matrix = sorted(list(map(list, zip(*predict_matrix))))  # 转置再排序
    predict_matrix = list(map(list, zip(*predict_matrix)))  # 转置回原始数据格式
    predict = predict_matrix[1]
    label = predict_matrix[0]
    top20_mse = np.mean([(actual - predicted) ** 2 for actual, predicted in zip(label[0:20], predict[0:20])])
    top60_mse = np.mean([(actual - predicted) ** 2 for actual, predicted in zip(label[0:60], predict[0:60])])

    all_active_error = np.mean([
        (actual - predicted) ** 2
        for actual, predicted in zip(label, predict)
        if actual < MAX_MIC - 0.01
    ])

    all_error = np.mean([(actual - predicted) ** 2 for actual, predicted in zip(label, predict)])

    return top20_mse, top60_mse, all_active_error, all_error


def get_train_xgb_classifier_data(train_data_path, test_data_path):
    train_data = pd.read_csv(train_data_path, encoding="utf8", index_col=0)
    test_data = pd.read_csv(test_data_path, encoding="utf8", index_col=0)
    x_train = train_data.iloc[:, 1:-2].values
    x_test = test_data.iloc[:, 1:-2].values
    y_train = train_data.iloc[:, -1].values
    y_test = test_data.iloc[:, -1].values
    # print(y_test)
    return x_train, x_test, y_train, y_test


def get_train_xgb_rank_featured_data(train_data_path, test_data_path):
    train_data = pd.read_csv(train_data_path, encoding="utf8", index_col=0)
    test_data = pd.read_csv(test_data_path, encoding="utf8", index_col=0)
    # 将测试数据按MIC值从小到大排序
    test_data.sort_values("MIC", inplace=True)
    x_train = train_data.iloc[:, 1:-2].values
    x_test = test_data.iloc[:, 1:-2].values

    y_train = np.log10(train_data.iloc[:, -2].values)
    y_test = np.log10(test_data.iloc[:, -2].values)
    # print(y_test)
    return x_train, x_test, y_train, y_test, test_data

def get_concat_rank_top_k(k):
    rank_path = "/home/xyc/peptide/predict_result/rank_result"
    rank_file_list = os.listdir(rank_path)
    sequences = []
    rank_results = []
    topk_seq = []
    rank_file_list.sort()
    for file in rank_file_list:
        contents = open(os.path.join(rank_path,file),"r").readlines()
        for content in contents:
            content = content.split(" ")
            sequence = content[0]
            rank_result = float(content[1][:-1])
            sequences.append(sequence)
            rank_results.append(rank_result)

    ind = np.argsort(rank_results)[:k]
    for i in ind:
        topk_seq.append(sequences[i])
        
    return topk_seq

if __name__ == "__main__":
    get_concat_rank_top_500()
