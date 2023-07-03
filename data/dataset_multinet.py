import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class MSDataset(Dataset):

    def __init__(self, data_base, file_csv, transforms, subset, dataset_id, length, input_shape):
        self.input_shape = input_shape
        self.data_base = data_base
        self.df = pd.read_csv(file_csv)
        self.df = self.df.loc[self.df["fold"].isin(subset)]
        self.df.reset_index(drop=True, inplace=True)
        self.files = self.df['path'].tolist()
        self.codes = self.df['coding'].astype(str).tolist()
        self.masks = self.df['source'].tolist()
        self.sizes = self.df['size'].tolist()
        self.transforms = transforms
        self.file_sample, self.code_sample, self.mask_sample = self._select_ds(dataset_id)
        if length == 0:
            length = len(self.file_sample)
        self.file_sample = self.file_sample[0:length]
        self.code_sample = self.code_sample[0:length]
        self.mask_sample = self.mask_sample[0:length]
        print(f'LIDC:{self.mask_sample.count(1)}, LNDB:{self.mask_sample.count(2)}, Inhouse:{self.mask_sample.count(3)}')

    def _select_ds(self, dataset_id):
        file_sample = []
        code_sample = []
        mask_sample = []
        for i in range(len(self.masks)):
            if self.masks[i] in dataset_id:
                file_sample.append(self.files[i])
                code_sample.append(self.codes[i])
                mask_sample.append(self.masks[i])
        return file_sample, code_sample, mask_sample
    
    def _crop_by_size(self, image, mask, click):
        image_list = []
        mask_list = []
        click_list = []
        for input_shape in self.input_shape:
            z0, y0, x0 = image.shape
            (z, y, x) = input_shape
            image_list.append(image[int(z0/2 - z/2) : int(z0/2 + z/2), 
                                    int(y0/2 - y/2) : int(y0/2 + y/2), 
                                    int(x0/2 - x/2) : int(x0/2 + x/2)])
            mask_list.append(mask[int(z0/2 - z/2) : int(z0/2 + z/2), 
                                    int(y0/2 - y/2) : int(y0/2 + y/2), 
                                    int(x0/2 - x/2) : int(x0/2 + x/2)])
            click_list.append(click[int(z0/2 - z/2) : int(z0/2 + z/2), 
                        int(y0/2 - y/2) : int(y0/2 + y/2), 
                        int(x0/2 - x/2) : int(x0/2 + x/2)])
        return image_list, mask_list, click_list
    
    def _gauss_click(self, center, bbox, diameter, size, sigma = 1, mu=1):
        (x_b, y_b ,z_b) = bbox
        (x_c, y_c, z_c) = center
        gauss_click = np.zeros(size)
        for i in range(int(x_c) - int(x_b/2), int(x_c) + int(x_b/2)):
            for j in range(int(y_c) - int(y_b/2), int(y_c) + int(y_b/2)):
                for k in range(int(z_c) - int(z_b/2), int(z_c) + int(z_b/2)):
                    dist_2 = (i-x_c) ** 2 +(j-y_c) ** 2 + (k-z_c) ** 2
                    if dist_2 < (diameter/2) ** 2:
                        gauss_click[i, j, k] = mu * np.exp(((- dist_2)/sigma ** 2))
        click = np.zeros(size)
        click[int(x_c), int(y_c), int(z_c)] = mu
        return gauss_click, click
    
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
        center = tuple(np.array(self.input_shape)/2)
        bbox = self.input_shape
        size = self.input_shape
        raw_data = np.load(data_path)
        _, click =  self._gauss_click(center, bbox, 36, size, sigma=5, mu=1)
        image = raw_data["img"]
        mask = raw_data["seg"].astype(np.int16)
        image_list, mask_list = self._crop_by_size(image, mask, click)
        for i in range(len(image_list)):
            image_list[i] = np.expand_dims(image_list[i], axis=0)
        data = {
            "image": image_list,
            "mask": mask_list,
            "name": file_id,
            "dataset": dataset
        }
        data = self._apply_transforms(data)
        return data

    @staticmethod
    def _collate_fn(samples):
        images = []
        masks = []
        for i in range(3):   
            images.append(torch.stack([x["image"][i] for x in samples]))
            masks.append(torch.stack([x["mask"][i] for x in samples])) 
        names = [x["name"]for x in samples]
        datasets = [x["dataset"]for x in samples]
        return {
            "images": images,
            "masks": masks,
            "names": names,
            "datasets": datasets
        }

    @staticmethod
    def get_dataloader(dataset, batch_size, shuffle=False, num_workers=0):
        return DataLoader(dataset, batch_size, shuffle,
                          num_workers=num_workers, collate_fn=MSDataset._collate_fn, drop_last=True)
