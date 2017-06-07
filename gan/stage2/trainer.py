import os
import glob
import time
import numpy as np
import torch
import torch.utils
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader

from gan.dataset import VQADataset
from gan.stage2.model import *

class Trainer():
    def __init__(self, config):
        self.config = config
        self.start_epoch = 0

        self.stage1_generator = Stage1Generator()
        self.stage2_generator = Stage2Generator()
        self.stage2_discriminator = Stage2Discriminator()

        print(self.stage2_generator)
        print(self.stage2_discriminator)

        self.bce_loss_fn = nn.BCELoss()
        self.opt_g = torch.optim.Adam(self.stage2_generator.parameters(),
            lr=config.lr, betas=(config.beta1, config.beta2))
        self.opt_d = torch.optim.Adam(self.stage2_discriminator.parameters(),
            lr=config.lr, betas=(config.beta1, config.beta2))

        self.dataset = VQADataset(config.dataset_dir,
                                  output_shape=[256, 256],
                                  train=config.is_train)
        self.data_loader = DataLoader(self.dataset,
                                      batch_size=config.batch_size,
                                      num_workers=4,
                                      shuffle=True, drop_last=True)

        self.ones = Variable(torch.ones(config.batch_size), requires_grad=False)
        self.zeros = Variable(torch.zeros(config.batch_size), requires_grad=False)

        if config.cuda:
            self.stage1_generator     = self.stage1_generator.cuda()
            self.stage2_generator     = self.stage2_generator.cuda()
            self.stage2_discriminator = self.stage2_discriminator.cuda()
            self.bce_loss_fn = self.bce_loss_fn.cuda()
            self.ones        = self.ones.cuda()
            self.zeros       = self.zeros.cuda()

    def fit(self):
        config = self.config
        num_steps_per_epoch = len(self.data_loader)

        for epoch in range(self.start_epoch, config.max_epochs):
            for step, (real_im, right_text, wrong_text) in enumerate(self.data_loader):
                t1 = time.time()

                if config.cuda:
                    real_im    = Variable(real_im).cuda()
                    right_text = Variable(right_text).cuda()
                    wrong_text = Variable(wrong_text).cuda()
                else:
                    real_im    = Variable(real_im)
                    right_text = Variable(right_text)
                    wrong_text = Variable(wrong_text)

                # generate fake image
                noise = Variable(torch.randn(config.batch_size, 100))
                noise = noise.cuda() if config.cuda else noise
                fake_im_stage1 = self.stage1_generator(noise, right_text)
                fake_im_stage2 = self.stage2_generator(fake_im_stage1, right_text)

                # train the discriminator
                D_real  = self.stage2_discriminator(real_im, right_text)
                D_wrong = self.stage2_discriminator(real_im, wrong_text)
                D_fake  = self.stage2_discriminator(fake_im_stage2.detach(), right_text)

                loss_real  = self.bce_loss_fn(D_real, self.ones)
                loss_wrong = self.bce_loss_fn(D_wrong, self.zeros)
                loss_fake  = self.bce_loss_fn(D_fake, self.zeros)

                loss_disc = loss_real + 0.5*(loss_fake + loss_wrong)
                loss_disc.backward()
                self.opt_d.step()
                self._reset_gradients()

                # train the generator
                D_fake  = self.stage2_discriminator(fake_im_stage2, right_text)
                loss_gen = self.bce_loss_fn(D_fake, self.ones)
                loss_gen.backward()
                self.opt_g.step()
                self._reset_gradients()

                t2 = time.time()

                if (step+1) % 100 == 0 or (step+1) == num_steps_per_epoch:
                    steps_remain = num_steps_per_epoch-step+1 + \
                        (config.max_epochs-epoch+1)*num_steps_per_epoch
                    eta = int((t2-t1)*steps_remain)

                    print("[{}/{}][{}/{}] Loss_D: {:.4f} Loss_G: {:.4f}, ETA: {} second"
                          .format(epoch+1, config.max_epochs,
                                  step+1, num_steps_per_epoch,
                                  loss_disc.data[0], loss_gen.data[0], eta))

            torchvision.utils.save_image(fake_im_stage2.data,
                "sample/fake_stage2_{}.png".format(epoch+1), normalize=True)
            torchvision.utils.save_image(fake_im_stage1.data,
                "sample/fake_stage1_{}.png".format(epoch+1), normalize=True)

            torch.save(self.stage2_generator.state_dict(),
                       "{}/stage2_generator_{}.pth"
                       .format(config.model_dir, epoch+1))
            torch.save(self.stage2_discriminator.state_dict(),
                       "{}/stage2_discriminator_{}.pth"
                       .format(config.model_dir, epoch+1))

    def load_stage1(self, directory):
        paths = glob.glob(os.path.join(directory, "*.pth"))
        gen_path  = [path for path in paths if "generator" in path][0]
        self.stage1_generator.load_state_dict(torch.load(gen_path))

        print("Load pretrained stage1 [{}]".format(gen_path))

    def _reset_gradients(self):
        self.stage2_generator.zero_grad()
        self.stage2_discriminator.zero_grad()
