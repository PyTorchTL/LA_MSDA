import codecs
import csv
import numpy as np
import torch


class MytenSorData(torch.utils.data.Dataset):
    # Initialize function, get data
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    # index is the index obtained by dividing the data according to the batchsize,
    # and finally the data and the corresponding labels are returned together
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # returns the data size and length, the purpose is to facilitate the division of DataLoader
    def __len__(self):
        return len(self.data)

