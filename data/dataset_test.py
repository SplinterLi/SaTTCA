import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class SaDataset(Dataset):

    def __init__(self, file_csv, transforms, subset, dataset_id, length):
        self.data_base = '/data/storage/lizhihao/data/data'
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
        self.mapping = {0: [1, 1], 1: [0, 1], 2: [0, 0]}

    def _select_df(self, dataset_name, file_id):
        csv_path = f'{self.data_base}/{dataset_name}/small_norm_label66.csv'
        df = pd.read_csv(csv_path)
        temp_df = df.loc[(df['file_id'] == file_id)]

        return temp_df

    def _select_ds(self, dataset_id):
        file_sample = []
        mask_sample = []
        for i in range(len(self.masks)):
            if self.masks[i] in dataset_id:
                file_sample.append(self.files[i])
                mask_sample.append(self.masks[i])
        return file_sample, mask_sample

    def _gauss_click(self, center, bbox, diameter, size, sigma = 1, mu=1):
        (x_b, y_b ,z_b) = bbox
        (x_c, y_c, z_c) = center
        gauss_click = np.zeros(size)
        for i in range(x_c - int(x_b/2), x_c + int(x_b/2)):
            for j in range(y_c - int(y_b/2), y_c + int(y_b/2)):
                for k in range(z_c - int(z_b/2), z_c + int(z_b/2)):
                    dist_2 = (i-x_c) ** 2 +(j-y_c) ** 2 + (k-z_c) ** 2
                    if dist_2 < (diameter/2) ** 2:
                        gauss_click[i, j, k] = mu * np.exp(((- dist_2)/sigma ** 2))
        click = np.zeros(size)
        click[x_c, y_c, z_c] = mu
        return gauss_click, click
    
    def _sphere(self, predict, mask, rate):
        box = []
        for i in range(3):
            dims = self.mapping[i]
            length = predict.max(dims[0])[0].max(dims[1])[0].sum(1)
            box.append(length)
        if rate != 0:
            diameter = min(box) * rate
        else:
            diameter = 1
        shape = predict.shape
        sphere = self._dia2mask(diameter, shape)
        if mask != 0:
            sphere_real = sphere * mask
            fake_rate = (sphere - sphere_real) / sphere
            return sphere_real, fake_rate
        else:
            return sphere

        
    
    def _apply_transforms(self, data):
        for t in self.transforms:
            data = t(data)
        return data

    def __len__(self):
        return len(self.file_sample)


    def __getitem__(self, idx):
        data_path = self.file_sample[idx]
        dataset = self.mask_sample[idx]
        file_id = data_path.split("/")[-1].split(".npz")[0]
        raw_data = np.load(data_path)
        center = (32, 48, 48)
        bbox = (64, 96, 96)
        size = (64, 96, 96)
        image = raw_data["img"]
        image = np.expand_dims(image, axis=0)
        _, click =  self._gauss_click(center, bbox, 36, size, sigma=5, mu=1)
        mask = raw_data["seg"].astype(np.int16)
        click = np.expand_dims(click, axis=0)
        data = {
            "image": image,
            "mask": mask,
            "click":click,
            "name": file_id,
            "dataset": dataset
        }
        data = self._apply_transforms(data)
        return data

    @staticmethod
    def _collate_fn(samples):
        images = torch.stack([x["image"] for x in samples])
        masks = torch.stack([x["mask"] for x in samples])
        clicks = torch.stack([x["click"] for x in samples])
        names = [x["name"]for x in samples]
        datasets = [x["dataset"]for x in samples]
        return {
            "images": images,
            "masks": masks,
            "clicks":clicks,
            "names": names,
            "datasets": datasets
        }

    @staticmethod
    def get_dataloader(dataset, batch_size, shuffle=False, num_workers=0):
        return DataLoader(dataset, batch_size, shuffle,
                          num_workers=num_workers, collate_fn=SaDataset._collate_fn, drop_last=False)
