import random
from re import T

import SimpleITK as sitk
import numpy as np
import torch
import torch.nn.functional as F

class Window:

    def __init__(self, window_min, window_max):
        self.window_min = window_min
        self.window_max = window_max

    def __call__(self, data):
        for i in range(len(data["image"])):
            data['image'][i] = np.clip(data['image'][i], self.window_min, self.window_max)
        return data


class MinMaxNormalize:

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, data):
        for i in range(len(data["image"])):
            data["image"][i] = (data["image"][i] - self.min_val)\
                / (self.max_val - self.min_val)
        return data


class LabelTrans:

    def __call__(self, data):
        data["mask"] = torch.tensor(data["mask"][np.newaxis, ...], dtype=torch.float)
        if "distance_map" in data.keys():    
            data["distance_map"] = torch.tensor(data["distance_map"][np.newaxis, ...], dtype=torch.float)
        data['image'] = torch.tensor(data['image'], dtype=torch.float)        
        return data

class ClickTrans:
    
    def __call__(self, data):        
        data['click'] = torch.tensor(data['click'], dtype=torch.float)        
        return data


class RandomFlip:

    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, data):
        if random.random() < self.p:
            axis_flipped = random.randint(0, 2)
            data["image"] = np.ascontiguousarray(np.flip(data["image"],
                axis=axis_flipped))

        return data
    
    
class RandomZoom:
    
    def __init__(self, mode, scale, bais):
        self.mode = mode
        self.scale = scale
        self.t = bais
    
    def _operator(self, ksize):
        return torch.nn.AvgPool3d(kernel_size=ksize, stride=1, padding=int((ksize-1)/ 2))
        
    
    def __call__(self, data):
        if self.mode == 'erode':
            # kernel_size = self.scale - int(random.random() * self.scale / 2) * 2
            kernel_size = self.scale
            erode_operator = self._operator(kernel_size)
            data["mask"] = torch.round(erode_operator(data["mask"]) - self.t)
            if data["mask"].sum() == 0:
                if data["mask"].dim() == 4:
                    data["mask"][:, 32, 48, 48] = 1
                else:
                    data["mask"][:, :, 32, 48, 48] = 1
        if self.mode == 'dilate':
            # kernel_size = self.scale - int(random.random() * self.scale / 2) * 2
            kernel_size = self.scale
            dilate_operator = self._operator(kernel_size)
            data["mask"] = torch.round(dilate_operator(data["mask"]) + self.t)           
        return data
            