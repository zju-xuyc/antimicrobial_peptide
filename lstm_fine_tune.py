from dataset import FinetuneDataset
from models import LstmNet
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.backends import cudnn
import numpy as np
import random
from utils import get_dict


class FineTune():
    def __init__(self, param_path, train_file, save_parm_path):
        self.save_parm_path = save_parm_path
        self.net = LstmNet(embedding_dim, hidden_num, num_layer, bidirectional, dropout, get_dict())
        if torch.cuda.is_available():
            self.net.load_state_dict(torch.load(param_path))
        else:
            self.net.load_state_dict(torch.load(param_path))
        self.net.to(device)
        train_dataset = FinetuneDataset(train_file)
        self.train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False)

        test_dataset = FinetuneDataset(train_file, is_train=False)
        self.test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        self.loss_func = torch.nn.MSELoss()
        self.opt = torch.optim.SGD(self.net.parameters(), lr=learning_rate)

    def __call__(self, *args, **kwargs):
        best_loss = 1000
        for epoch in range(epoch_num):
            self.net.train()
            train_loss = 0
            for i, (text, label, length) in enumerate(self.train_dataloader):
                text, label = text.float().to(device), label.float().to(device)
                out = self.net(text, length)
                loss = self.loss_func(out, label)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                train_loss += loss.cpu().detach().item()
            avg_train_loss = train_loss / len(self.train_dataloader)

            self.net.eval()
            test_loss = 0
            for i, (text, label, length) in enumerate(self.test_dataloader):
                text, label = text.to(device), label.to(device)
                out = self.net(text, length)
                loss = self.loss_func(out, label)
                test_loss += loss.cpu().detach().item()
            avg_test_loss = test_loss / len(self.test_dataloader)
            print("epoch:", epoch, "train_loss:", avg_train_loss, "test_loss:", avg_test_loss)
            if avg_test_loss < best_loss:
                torch.save(self.net.state_dict(), self.save_parm_path)
                best_loss = avg_test_loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-ed", "--embedding_dim", type=int, default=50)
    parser.add_argument("-hn", "--hidden_num", type=int, default=128)
    parser.add_argument("-nl", "--num_layer", type=int, default=2)
    parser.add_argument("-bs", "--batch_size", type=int, default=16)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5)
    parser.add_argument("-dp", "--dropout", type=float, default=0.7)
    parser.add_argument("-bd", "--bidirectional", type=str, default="False")
    parser.add_argument("-e", "--epoch_num", type=int, default=10000)
    parser.add_argument("-lg", "--log_num", type=int, default=10)
    parser.add_argument("-sd", "--seed", type=int, default=500)
    parser.add_argument("-dn", "--device_num", type=str, default=0)
    parser.add_argument("-lpp", "--lstm_param_path", type=str, default="path_to_params")
    parser.add_argument("-tfp", "--train_file_path", type=str, default="path_to_incremental_data")
    parser.add_argument("-spp", "--save_parm_path", type=str, default="path_to_save_finetune_params")
    args = parser.parse_args()
    print(vars(args))

    device_num = args.device_num
    device = torch.device(f"cuda:{device_num}" if torch.cuda.is_available() else "cpu")
    embedding_dim = args.embedding_dim
    hidden_num = args.hidden_num
    num_layer = args.num_layer
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    dropout = args.dropout
    if args.bidirectional == "True":
        bidirectional = True
    else:
        bidirectional = False
    epoch_num = args.epoch_num
    seed = args.seed
    """
    fix random seeds
    """
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    train = FineTune(args.lstm_param_path, args.train_file_path, args.save_parm_path)
    train()
