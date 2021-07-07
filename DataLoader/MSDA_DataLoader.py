import scipy
import torch
import torch.utils.data
import scipy.io as scio
from .MSDA_LoadTensorData import MytenSorData
import numpy as np
from sklearn.model_selection import train_test_split

# load the dataset, and split the training and testing data by configuring the auxiliary training data rate in the target domain

def load_data(root_path, file, batch_size, name_list, shape, testrate, cudadevice, kwargs):
    # get sample and label
    data = scio.loadmat(root_path + file)
    sample = data[name_list[0]]  # name_list[0] = 'data'
    label = data[name_list[1]].flatten()  # name_list[1] = 'label'
    # standardization
    sample = sample / np.tile(np.sum(sample, axis=1).reshape(-1, 1), [1, sample.shape[1]])
    sample = scipy.stats.mstats.zscore(sample)
    # reshape
    sample = np.reshape(sample, (-1, 1, shape[0], shape[1]))
    # Transpose
    sample = np.transpose(sample, (0, 1, 3, 2))
    # to tensor
    if (testrate != 0):
        sampleTrain, sampleTest, labelTrain, labelTest = train_test_split(sample, label, test_size=testrate)
        sampleTrain, labelTrain = torch.from_numpy(sampleTrain).to(cudadevice), torch.from_numpy(labelTrain).type(torch.long).to(cudadevice)
        sampleTest, labelTest = torch.from_numpy(sampleTest).to(cudadevice), torch.from_numpy(labelTest).type(torch.long).to(cudadevice)
        return [Data_prefetcher(
            torch.utils.data.DataLoader(MytenSorData(sampleTrain, labelTrain), batch_size=batch_size, shuffle=True,
                                        drop_last=True, **kwargs)),
            Data_prefetcher(torch.utils.data.DataLoader(MytenSorData(sampleTest, labelTest), batch_size=batch_size,
                                                        shuffle=True,
                                                        drop_last=True, **kwargs))]
    else:
        sample, label = torch.from_numpy(sample).to(cudadevice), torch.from_numpy(label).type(torch.long).to(cudadevice)
        return [Data_prefetcher(
            torch.utils.data.DataLoader(MytenSorData(sample, label), batch_size=batch_size, shuffle=True,
                                        drop_last=True, **kwargs)),
            Data_prefetcher(
            torch.utils.data.DataLoader(MytenSorData(sample, label), batch_size=batch_size, shuffle=True,
                                        drop_last=True, **kwargs))]

def load_multi_data(ga_root_path, ga_file_list, hc_root_path, hc_file_list, batch_size, kwargs):
    len_ga = len(ga_file_list)
    len_hc = len(hc_file_list)
    len_union = len_ga + len_hc
    sample = np.zeros((len_union * 490, 630)).astype('float32')
    label = np.zeros((len_union * 490)).astype('long')
    mem = 0
    for i, file in enumerate(ga_file_list):
        tmp = scio.loadmat(ga_root_path + file)
        sample[i * 490:i * 490 + 490, :] = tmp['D']
        label[i * 490:i * 490 + 490] = tmp['label'].flatten()
        mem = i
    mem += 1
    for j, file in enumerate(hc_file_list):
        tmp = scio.loadmat(hc_root_path + file)
        sample[(mem + j) * 490:(mem + j) * 490 + 490, :] = tmp['D']
        label[(mem + j) * 490:(mem + j) * 490 + 490] = tmp['label'].flatten()
    sample = np.reshape(sample, (-1, 1, 21, 30))
    # sample = np.transpose(sample, (0, 1, 3, 2))
    data_tensor = MytenSorData(sample, label)
    loader = torch.utils.data.DataLoader(data_tensor, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    return loader


class Data_prefetcher():
    def __init__(self, loader):
        self.baseloader = loader
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            # self.loader = iter(self.baseloader)
            # self.next_input, self.next_target = next(self.loader)
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()

    def preloadInfinite(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.loader = iter(self.baseloader)
            self.next_input, self.next_target = next(self.loader)
            # self.next_input = None
            # self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()

    ##Basic next function, stop at the end
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target

    ##The infinite next, the end and again
    def infiniteNext(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preloadInfinite()
        return input, target

    ##Get all the data
    def getAllData(self):

        return self.baseloader.dataset.data, self.baseloader.dataset.label


if __name__ == '__main__':

    cuda = True
    seed = 8

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    batch_size = 64
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
