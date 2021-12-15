import torch
from torch.utils.data import Dataset
from utils import get_dict, build_index
import numpy as np
import pandas as pd

class PredictDataset(Dataset):
    def __init__(self, file):
        super().__init__()
        Letter_dict = get_dict()
        self.dataset = []
        for sequence in file:
            self.dataset.append([Letter_dict[i] for i in sequence.strip()])
        print("total_sample:", len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        length = len(self.dataset[item])
        text = self.dataset[item]
        return torch.tensor(text), torch.tensor(length)


class PeptideDataset(Dataset):

    def __init__(self, all_data):
        tmp_t = []
        tmp_s = []
        tmp_p = []
        tmp_l = []
        for i in range(len(all_data)):
            tmp_s.append(all_data[i][0])
            tmp_t.append(all_data[i][1])
            tmp_p.append(all_data[i][2])
            tmp_l.append(all_data[i][3])
        Letter_dict = get_dict()
        self.target = tmp_t
        self.sequence = build_index(tmp_s, Letter_dict)
        self.data_type = tmp_p
        self.length = tmp_l

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index):
        sequence = self.sequence[index]
        target = self.target[index]
        data_type = self.data_type[index]
        length = self.length[index]
        if len(sequence) < 50:                          
            npi = np.zeros((50 - len(sequence)), dtype=np.int)
            sequence.extend(npi)
        return torch.tensor(sequence), torch.tensor([target]), data_type, length


class FinetuneDataset(Dataset):

    def __init__(self, file_path, is_train=True):
        super().__init__()
        self.Letter_dict = get_dict()
        df = pd.read_csv(file_path)
        self.dataset = df.values
        num = len(df)
        if is_train:
            self.dataset = self.dataset[:int(0.8 * num)]
        else:
            self.dataset = self.dataset[int(0.8 * num):]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        length = len(self.dataset[item][0])
        label = np.log10(self.dataset[item][1])
        text = self.dataset[item][0]
        sequence = [self.Letter_dict[i] for i in text]
        if len(sequence) < 50:
            zero = np.zeros(50 - len(sequence))
            sequence.extend(zero)

        return torch.tensor(sequence), torch.tensor([label]), torch.tensor(length).long()