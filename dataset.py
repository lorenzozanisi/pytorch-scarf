import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pipeline.Models import ITGDatasetDF



class ScarfDataset(ITGDatasetDF):
    def __init__(self, 
                data = pd.DataFrame,
                target_var = "stable_label",
                leading_flux = "efeetg_gb",
                gkmodel: str = "QLK15D"):
        super().__init__(df=data, leading_flux=leading_flux, target_var = target_var, gkmodel = gkmodel)
        
    def __getitem__(self, index):
        # the dataset must return a pair of samples: the anchor and a random one from the
        # dataset that will be used to corrupt the anchor
        random_idx = np.random.randint(0, len(self))
        random_sample = torch.tensor(self.data[random_idx], dtype=torch.float)
        sample = torch.tensor(self.data[index], dtype=torch.float)

        return sample, random_sample

    def __len__(self):
        return len(self.data)

    def to_dataframe(self):
        return pd.DataFrame(self.data, columns=self.columns)

    @property
    def shape(self):
        return self.data.shape


class ExampleDataset(Dataset):
    def __init__(self, data, target):
        self.target = data.loc[:,target].values
        self.data = data.values

    def __getitem__(self, index):
        # the dataset must return a pair of samples: the anchor and a random one from the
        # dataset that will be used to corrupt the anchor
        data_idx = self.data.index
        random_idx = np.random.randint(np.min(data_idx),np.max(data_idx), len(self))
        random_sample = torch.tensor(self.data[random_idx], dtype=torch.float)
        sample = torch.tensor(self.data[index], dtype=torch.float)

        return sample, random_sample

    def __len__(self):
        return len(self.data)

    def to_dataframe(self):
        return pd.DataFrame(self.data, columns=self.columns)

    @property
    def shape(self):
        return self.data.shape