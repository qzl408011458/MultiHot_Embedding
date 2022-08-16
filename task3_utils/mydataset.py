
from torch.utils.data.dataset import Dataset
import torch

class myDataset(Dataset):
    def __init__(self, X_df, y_df):
        self.X = X_df.to(torch.float32)
        self.y = y_df

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {'data':  self.X[idx],
                'label': self.y[idx]}