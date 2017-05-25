import os
import glob
import time
import torch
import torch.utils
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import VQADataset
from model import Generator, Discriminator

class Trainer():
    def __init__(self, config):
        self.generator = Generator()
        self.discriminator = Discriminator()

        print(self.generator)
        print(self.discriminator)

        self.loss_fn = nn.BCELoss()
        self.opt_g = torch.optim.Adam(self.generator.parameters(),
            lr=config.lr, betas=(config.beta1, config.beta2))
        self.opt_d = torch.optim.Adam(self.discriminator.parameters(),
            lr=config.lr, betas=(config.beta1, config.beta2))

        self.dataset = VQADataset(config.file_path)
        self.data_loader = DataLoader(self.dataset,
                                      batch_size=config.batch_size,
                                      num_workers=4,
                                      shuffle=True, drop_last=True)

        self.label_real = Variable(torch.ones(config.batch_size))
        self.label_fake = Variable(torch.zeros(config.batch_size))
        self.noise = Variable(torch.randn(config.batch_size, 128))

        if config.cuda:
            self.generator     = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()
            self.loss_fn       = self.loss_fn.cuda()
            self.label_real    = self.label_real.cuda()
            self.label_fake    = self.label_fake.cuda()

        self.config = config
        self.start_epoch = 0

    def fit(self):
        config = self.config
        num_steps_per_epoch = len(self.data_loader)

        for epoch in range(self.start_epoch, config.max_epochs):
            for step, batch_im in enumerate(self.data_loader):
                t1 = time.time()

                batch_im = batch_im.cuda() if config.cuda else batch_im
                batch_im = Variable(batch_im)

                # Train the discriminator
                logits_real = self.discriminator(batch_im)
                loss_real = self.loss_fn(logits_real, self.label_real)

                noise = Variable(torch.randn(config.batch_size, 128))
                noise = noise.cuda() if config.cuda else noise

                gen_fake  = self.generator(noise)
                logits_fake = self.discriminator(gen_fake.detach())
                loss_fake = self.loss_fn(logits_fake, self.label_fake)

                loss_disc = loss_real + loss_fake
                loss_disc.backward()
                self.opt_d.step()

                self._reset_gradients()

                # Train the generator
                noise = Variable(torch.randn(config.batch_size, 128))
                noise = noise.cuda() if config.cuda else noise

                logits_fake = self.discriminator(gen_fake)
                loss_gen = self.loss_fn(logits_fake, self.label_real)
                loss_gen.backward()
                self.opt_g.step()

                self._reset_gradients()

                t2 = time.time()

                if (step+1) % 300 == 0 or (step+1) == num_steps_per_epoch:
                    steps_remain = num_steps_per_epoch-step+1 + \
                        (config.max_epochs-epoch+1)*num_steps_per_epoch
                    eta = int((t2-t1)*steps_remain)

                    print("[{}/{}][{}/{}] Loss_D: {:.4f} Loss_G: {:.4f}, ETA: {} second"
                          .format(epoch+1, config.max_epochs,
                                  step+1, num_steps_per_epoch,
                                  loss_disc.data[0], loss_gen.data[0], eta))

                    torchvision.utils.save_image(gen_fake.data,
                        "sample/fake_{}_{}.png".format(epoch+1, step+1), normalize=True)
                    torchvision.utils.save_image(batch_im.data,
                        "sample/real_{}_{}.png".format(epoch+1, step+1), normalize=True)

            torch.save(self.generator.state_dict(),
                       "model/generator_{}.pth".format(epoch+1))
            torch.save(self.discriminator.state_dict(),
                       "model/discriminator_{}.pth".format(epoch+1))

    def generate(self, filename, noise):
        # TODO check shapes
        noise = Variable(torch.from_numpy(noise))

        if self.config.cuda:
            noise = noise.cuda()

        gen = self.generator(noise)
        torchvision.utils.save_image(gen.data, "sample/{}.png".format(filename))

    def load(self, directory):
        paths = glob.glob(os.path.join(directory, "*.pth"))
        gen_path  = [path for path in paths if "generator" in path][0]
        disc_path = [path for path in paths if "discriminator" in path][0]

        self.generator.load_state_dict(torch.load(gen_path))
        self.discriminator.load_state_dict(torch.load(disc_path))

        self.start_epoch = int(gen_path.split(".")[0].split("_")[-1])
        print("Load pretrained [{}, {}]".format(gen_path, disc_path))

    def _reset_gradients(self):
        self.generator.zero_grad()
        self.discriminator.zero_grad()
