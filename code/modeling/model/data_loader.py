import torch
import numpy as np
import os

from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):

    def __init__(self, data, labels, flatten=True):
        self.data = data
        self.labels = labels
        self.flatten = flatten

        super().__init__()

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.flatten:
            x = x.view(-1)

        return x, y


def get_loaders(config):
    DATA_PATH = "/home/ubuntu/Studio/BIS_Prediction/data/vitaldb"
    
    # train_x = torch.from_numpy(np.load(os.path.join(DATA_PATH, "train_EEG.npy"))).type(torch.FloatTensor)
    # train_y = torch.from_numpy(np.load(os.path.join(DATA_PATH, "train_label.npy")) / 100).type(torch.FloatTensor)
    
    train_x = torch.from_numpy(np.load(os.path.join(DATA_PATH, "valid_EEG.npy"))).type(torch.FloatTensor)
    train_y = torch.from_numpy(np.load(os.path.join(DATA_PATH, "valid_label.npy"))).type(torch.FloatTensor)
    
    valid_x = torch.from_numpy(np.load(os.path.join(DATA_PATH, "valid_EEG.npy"))).type(torch.FloatTensor)
    valid_y = torch.from_numpy(np.load(os.path.join(DATA_PATH, "valid_label.npy")) / 100).type(torch.FloatTensor)
    
    test_x = torch.from_numpy(np.load(os.path.join(DATA_PATH, "test_EEG.npy"))).type(torch.FloatTensor)
    test_y = torch.from_numpy(np.load(os.path.join(DATA_PATH, "test_label.npy")) / 100).type(torch.FloatTensor)

    flatten = True if config.model == 'fc' else False

    train_loader = DataLoader(
        dataset=MyDataset(train_x, train_y, flatten=flatten),
        batch_size=config.batch_size,
        shuffle=True,
    )
    
    valid_loader = DataLoader(
        dataset=MyDataset(valid_x, valid_y, flatten=flatten),
        batch_size=config.batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        dataset=MyDataset(test_x, test_y, flatten=flatten),
        batch_size=config.batch_size,
        shuffle=False,
    )

    return train_loader, valid_loader, test_loader
