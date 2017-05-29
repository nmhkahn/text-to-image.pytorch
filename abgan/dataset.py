import os
import random
import pickle
import numpy as np
import scipy.misc as misc
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

class VQADataset(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 output_shape=[64, 64],
                 train=True):
        self.train = train
        self.dataset_dir = dataset_dir
        self.output_shape = output_shape

        if not len(output_shape) in [2, 3]:
            raise ValueError("[*] output_shape must be [H,W] or [C,H,W]")

        if self.train:
            _file = open(os.path.join(dataset_dir, "train.pkl"), "rb")
            self.train_data = pickle.load(_file, encoding="latin1")
            _file.close()
        else:
            _file = open(os.path.join(dataset_dir, "test.pkl"), "rb")
            self.test_data = pickle.load(_file, encoding="latin1")
            _file.close()

        self.transform = transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text
        if self.train:
            path  = os.path.join(self.dataset_dir, self.train_data[index]["path"])
            im = Image.open(path).convert("RGB").resize(self.output_shape)
            
            wrong_index = random.choice(
                [x for x in range(self.__len__()) if x != index])

            right_embed = self.train_data[index]["embedding"]
            wrong_embed = self.train_data[wrong_index]["embedding"]
            right_embed_mean = torch.FloatTensor(np.mean(right_embed, axis=0))
            wrong_embed_mean = torch.FloatTensor(np.mean(wrong_embed, axis=0))
            
            return self.transform(im), right_embed_mean, wrong_embed_mean
        else:
            path  = os.path.join(self.dataset_dir, self.test_data[index]["path"])
            im = Image.open(path).convert("RGB").resize(self.output_shape)

            embed = self.test_data[index]["embedding"]
            mean_embed = torch.FloatTensor(np.mean(embed, axis=0))

            return self.transform(im), mean_embed

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
