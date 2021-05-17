# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch.utils.data import Dataset

from fastreid.data.data_utils import read_image


class AttrDataset(Dataset):
    """Image Person Attribute Dataset"""

    def __init__(self, img_items, transform, attr_dict):
        self.img_items = img_items
        self.transform = transform
        self.attr_dict = attr_dict

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path, labels = self.img_items[index]
        img = read_image(img_path)

        if self.transform is not None: img = self.transform(img)

        # labels = torch.as_tensor(labels)
        for i,label in enumerate(labels):
            labels[i]=torch.as_tensor(labels[i])

        return {
            "images": img,
            "targets1": labels[0],
            "targets2": labels[1],
            "img_paths": img_path,
        }

    @property
    def num_classes(self):
        attr_dict_len=[]
        for single_attr_dict in self.attr_dict:
            attr_dict_len.append(len(single_attr_dict))
        return attr_dict_len
        # return len(self.attr_dict)

    @property
    def sample_weights(self):
        sample_weights=[]
        for single_num_classes in self.num_classes:
            single_sample_weights = torch.zeros(single_num_classes, dtype=torch.float32)
            sample_weights.append(single_sample_weights)
        # sample_weights = torch.zeros(self.num_classes, dtype=torch.float32)
        for _, attr in self.img_items:
            for i,single_attr in enumerate(attr):
                sample_weights[i] += torch.as_tensor(single_attr)
            # sample_weights += torch.as_tensor(attr)
        # sample_weights /= len(self)
        for single_sample_weights in sample_weights:
            single_sample_weights /= len(self)
        return sample_weights
