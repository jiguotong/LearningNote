#!/usr/bin/python
# -*- encoding: utf-8 -*-
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, rootpth, mode='train', *args, **kwargs):
        super(MyDataset, self).__init__(*args, **kwargs)
        self.mode = mode
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.images_path = os.path.join(current_dir, rootpth, 'input_png')
        self.labels_path = os.path.join(current_dir, rootpth, 'label_png')
        self.images, self.labels = [], []

        files_name_list = os.listdir(self.images_path)
        for file_name in files_name_list:
            self.images.append(os.path.join(self.images_path, file_name))
            self.labels.append(os.path.join(self.labels_path, file_name))

        # 数据预处理
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),      # 包含 浮点化、归一化、通道重排
            transforms.Normalize(0.5, 1),
        ])

        self.label_transform = transforms.Compose([
            transforms.PILToTensor()  # 保持原数不变
        ])

    def __getitem__(self, idx):
        impth = self.images[idx]
        lbpth = self.labels[idx]
        img = Image.open(impth).convert('RGB')
        label = Image.open(lbpth)

        img = self.image_transform(img)
        label = self.label_transform(label).to(torch.int64)

        return img, label


    def __len__(self):
        return len(self.images)