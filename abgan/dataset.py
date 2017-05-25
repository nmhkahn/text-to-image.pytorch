import csv
import numpy as np
import scipy.misc as misc
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

class VQADataset(data.Dataset):
    def __init__(self, file_path, output_shape=[128, 128]):
        if not len(output_shape) in [2, 3]:
            raise ValueError("output shape must be [H,W] or [C,H,W]")

        with open(file_path) as _file:
            reader = csv.reader(_file)
            self.data = [row[0] for row in reader]

        self.output_shape = output_shape
        self.transform = transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        im_path = "dataset/" + self.data[index]
        im = misc.imread(im_path, mode="RGB")
        im = misc.imresize(im, self.output_shape)
        return self.transform(im)

    def __len__(self):
        return len(self.data)
