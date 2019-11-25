import torch
from torch.utils import data
import numpy as np
import pickle


class UrbanSound8KDataset(data.Dataset):
    def __init__(self, dataset_path, mode):
        self.dataset = pickle.load(open(dataset_path, 'rb'))
        self.mode = mode

    def __getitem__(self, index):
        features = self.dataset[index]['features']
        chroma = features["chroma"]
        spec = features["spectral_contrast"]
        tonnetz = features["tonnetz"]
        if self.mode == 'LMC':
            # Edit here to load and concatenate the neccessary features to
            # create the LMC feature
            lm = features["logmelspec"]
            feature = np.concatenate((lm, chroma, spec, tonnetz))
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
        elif self.mode == 'MC':
            # Edit here to load and concatenate the neccessary features to
            # create the MC feature
            m = features["mfcc"]
            feature = np.concatenate((m, chroma, spec, tonnetz))
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)
        elif self.mode == 'MLMC':
            # Edit here to load and concatenate the neccessary features to
            # create the MLMC feature
            lm = features["logmelspec"]
            m = features["mfcc"]
            feature = np.concatenate((lm, m, chroma, spec, tonnetz))
            feature = torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)

        label = self.dataset[index]['classID']
        fname = self.dataset[index]['filename']
        return feature, label, fname

    def __len__(self):
        return len(self.dataset)
