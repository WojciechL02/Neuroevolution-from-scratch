import pandas as pd
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, size):
        table = []
        for _ in range(size//2):
            x = round(torch.rand(size=(1,)).item() * 8, 4)
            table.append([x, x**2 - 2*x + 1, 1])
        for _ in range(size//2):
            x = round(torch.rand(size=(1,)).item() * 8, 4)
            table.append([x, 2.71 ** x, 0])
        for _ in range(size//2):
            x = round(torch.rand(size=(1,)).item() * 8, 4)
            table.append([x, 0.9 * x**2 - 3*x + 2, 2])
        df=pd.DataFrame(table, columns=["x", "y", "class"])

        x=df.iloc[:,0:10].values
        y=df.iloc[:,10].values

        self.x_train=torch.tensor(x,dtype=torch.float32)
        self.y_train=torch.tensor(y,dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx],self.y_train[idx]


class KaggleDataset(Dataset):
    def __init__(self, x, y):
        self.x_train=torch.from_numpy(x).float()
        self.y_train=torch.tensor(y.to_numpy())

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]
