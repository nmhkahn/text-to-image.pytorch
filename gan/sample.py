import os
import pickle
import argparse
import numpy as np
import torch
import torchvision
from torch.autograd import Variable

from dataset import VQADataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda",
                        action="store_true")
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="dataset/")
    parser.add_argument("--model_dir",
                        type=str,
                        default="model/stage1")
    parser.add_argument("--sample_dir",
                        type=str,
                        default="sample/stage1")
    return parser.parse_args()


def sample(indices):
    # TODO: temporary only load stage1 (add argparse)
    from stage1.model import Generator

    generator = Generator()
    dataset = VQADataset(config.dataset_dir, train=config.is_train)

    ims, embeds = [], []
    _file = open("{}/captions.txt".format(config.sample_dir), "w")
    for idx in indices:
        im, embed, caption = dataset[idx]
        ims.append(im)
        embeds.append(embed)

        _file.write("index: {}\n".format(idx))
        for c in caption:
            _file.write(c+"\n")
        _file.write("\n")
    _file.close()

    ims    = torch.stack(ims, 0)
    embeds = torch.stack(embeds, 0)
    noise  = Variable(torch.randn(len(indices), 100))

    if config.cuda:
        noise = noise.cuda()
        embeds = Variable(embeds).cuda()
    else:
        embeds = Variable(embeds)

    embeds = embeds.view(len(indices), -1)
    fake_ims = generator(noise, embeds)

    torchvision.utils.save_image(ims,
                                 "{}/real.png".format(config.sample_dir),
                                 normalize=True)
    torchvision.utils.save_image(fake_ims.data,
                                 "{}/fake.png".format(config.sample_dir),
                                 normalize=True)


def main(config):
    random_index = np.arange(16, 32)
    sample(random_index)


if __name__ == "__main__":
    config = parse_args()
    config.is_train = False
    main(config)
