import os
from scipy.ndimage import morphology
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class SaDataset(Dataset):

    def __init__(self, file_csv, transforms, subset, dataset_id, length):
        self.df = pd.read_csv(file_csv)
        self.df = self.df.loc[self.df["fold"].isin(subset)]
        self.df.reset_index(drop=True, inplace=True)
        self.files = self.df['path'].tolist()
        self.masks = self.df['source'].tolist()
        self.transforms = transforms
        self.file_sample, self.mask_sample = self._select_ds(dataset_id)
        if length == 0:
            length = len(self.file_sample)
        self.file_sample = self.file_sample[0:length]
        self.mask_sample = self.mask_sample[0:length]
        print(f'LIDC:{self.mask_sample.count(1)}, LNDB:{self.mask_sample.count(2)}, Inhouse:{self.mask_sample.count(3)}')

    def _select_ds(self, dataset_id):
        file_sample = []
        mask_sample = []
        for i in range(len(self.masks)):
            if self.masks[i] in dataset_id:
                file_sample.append(self.files[i])
                mask_sample.append(self.masks[i])
        return file_sample, mask_sample
    
    def _apply_transforms(self, data):
        for t in self.transforms:
            data = t(data)
        return data
    
    def _get_distance(self, mask):

        dt = morphology.distance_transform_edt(mask)
        dt /= (dt.max() + 1e-8)        
        dt_n = morphology.distance_transform_edt(1 - mask)
        dt_n /= (dt_n.max() + 1e-8)

        shape_information = (1 - dt) * mask + (dt_n - 1) * (1 - mask)
        sdm = 1 / (1 + np.exp(-shape_information / 10.))
        return sdm


    
    
    def __len__(self):
        return len(self.file_sample)
    

    def __getitem__(self, idx):
        data_path = self.file_sample[idx]
        dataset = self.mask_sample[idx]
        file_id = data_path.split("/")[-1].split(".npz")[0]
        raw_data = np.load(data_path)
        image = raw_data["img"]
        image = np.expand_dims(image, axis=0)
        mask = raw_data["seg"].astype(np.int16)
        distance_map = self._get_distance(mask)
        data = {
            "image": image,
            "mask": mask,
            "name": file_id,
            "dataset": dataset,
            "distance_map": distance_map
        }
        data = self._apply_transforms(data)
        return data

    @staticmethod
    def _collate_fn(samples):
        images = torch.stack([x["image"] for x in samples])
        masks = torch.stack([x["mask"] for x in samples])
        distance_maps = torch.stack([x["distance_map"] for x in samples])
        names = [x["name"]for x in samples]
        datasets = [x["dataset"]for x in samples]
        
        return {
            "images": images,
            "masks": masks,
            "names": names,
            "datasets": datasets,
            "distance_maps": distance_maps
        }

    @staticmethod
    def get_dataloader(dataset, batch_size, shuffle=False, num_workers=0):
        return DataLoader(dataset, batch_size, shuffle,
                          num_workers=num_workers, collate_fn=SaDataset._collate_fn, drop_last=False)
