import math
import torch
from torch import optim
from models import *
from models.types_ import Tensor
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils 
from torch.utils.data import DataLoader
from terrain_loader import TerrainDataset


class Pix2pixExperiment(pl.LightningModule):

    def __init__(self, gen_model,disc_model,
                 params: dict) -> None:
        super(Pix2pixExperiment, self).__init__()

        self.params = params
        self.gen_model = gen_model#(self.params['in_channels'],self.params['out_channels'])
        self.disc_model = disc_model#(self.params['in_channels'])
        self.loss = Loss()
        self.curr_device = None
        self.hold_graph = False
        
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input_img,  **kwargs) -> Tensor:
        gen_output = self.gen_model(input_img)
        # disc_real_output = self.disc_model(input_img, dem)
        return gen_output #, disc_real_output, disc_gen_output

    @staticmethod
    def set_requires_grad(nets, requires_grad = False):

        """
        Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """

        if not isinstance(nets, list): nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        input_img, dem = batch
        self.curr_device = input_img.device

        if optimizer_idx==0:
            self.gen_output = self.forward(input_img)
            with torch.no_grad():
                disc_gen_output = self.disc_model(input_img, self.gen_output)
            total_gen_loss, l1_rec_loss, gan_loss = self.loss.generator_loss(disc_gen_output,self.gen_output,dem)
            train_losses = {'total_gen_loss':total_gen_loss,
                    'l1_rec_loss':l1_rec_loss,
                    'gan_loss':gan_loss,
                    'loss': total_gen_loss}
            self.logger.experiment.log({key: val.item() for key, val in train_losses.items() if key!='loss'})

        elif optimizer_idx==1:
            disc_real_output = self.disc_model(input_img, dem)
            disc_gen_output = self.disc_model(input_img, self.gen_output.detach())
            total_disc_loss = self.loss.discriminator_loss(disc_real_output,disc_gen_output)

            train_losses = {
                    'total_disc_loss':total_disc_loss,
                    'loss':total_disc_loss}
            self.logger.experiment.log({key: val.item() for key, val in train_losses.items() if key!='loss'})

        return train_losses

    def validation_step(self, batch, batch_idx ):
        input_img, dem = batch
        self.curr_device = input_img.device
        self.curr_device = dem.device

        gen_output = self.forward(input_img)
        disc_gen_output = self.disc_model(input_img, gen_output)
        disc_real_output = self.disc_model(input_img, dem)

        total_gen_loss, l1_rec_loss, gan_loss = self.loss.generator_loss(disc_gen_output,gen_output,dem)

        total_disc_loss = self.loss.discriminator_loss(disc_real_output,disc_gen_output)

        val_losses = {'total_gen_loss':total_gen_loss,
                    'l1_rec_loss':l1_rec_loss,
                    'gan_loss':gan_loss,
                    'total_disc_loss':total_disc_loss,}
                    # 'loss': total_gen_loss}

        self.logger.experiment.log({key: val.item() for key, val in val_losses.items() if key!='loss'})

        return val_losses


    def test_step(self,batch,batch_idx):
        input_img, dem = batch
        self.curr_device = input_img.device
        self.curr_device = dem.device

        gen_output = self.forward(input_img)
        disc_gen_output = self.disc_model(input_img, gen_output)
        disc_real_output = self.disc_model(input_img, dem)

        total_gen_loss, l1_rec_loss, gan_loss = self.loss.generator_loss(disc_gen_output,gen_output,dem)

        total_disc_loss = self.loss.discriminator_loss(disc_real_output,disc_gen_output)

        val_losses = {'total_gen_loss':total_gen_loss,
                    'l1_rec_loss':l1_rec_loss,
                    'gan_loss':gan_loss,
                    'total_disc_loss':total_disc_loss,}
                    # 'loss': total_gen_loss}

        self.logger.experiment.log({key: val.item() for key, val in val_losses.items() if key!='loss'})

        return val_losses

    def validation_epoch_end(self, outputs):
        total_gen_loss = torch.stack([x['total_gen_loss'] for x in outputs]).mean()
        l1_rec_loss = torch.stack([x['l1_rec_loss'] for x in outputs]).mean()
        gan_loss = torch.stack([x['gan_loss'] for x in outputs]).mean()
        total_disc_loss = torch.stack([x['total_disc_loss'] for x in outputs]).mean()

        tensorboard_logs = {'total_gen_loss': total_gen_loss, 'l1_rec_loss': l1_rec_loss, 
                            'gan_loss': gan_loss, 'total_disc_loss': total_disc_loss}
        
        self.sample_images()
        return {'val_loss': total_gen_loss, 'log': tensorboard_logs}

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.gen_model(test_input)
        vutils.save_image(recons.data,
                          f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                          f"recons_{self.logger.name}_{self.current_epoch}.png",
                          normalize=True,
                          nrow=5)
        
        vutils.save_image(test_label.data,
                          f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                          f"real_img_{self.logger.name}_{self.current_epoch}.png",
                          normalize=True,
                          nrow=5)

        del test_input, recons #, samples

    def configure_optimizers(self):

        optims = []
        scheds = []

        gen_optimizer = optim.Adam(self.gen_model.parameters(),
                               lr=self.params['gen_learning_rate'],
                               betas = (self.params['beta1'],self.params['beta2']),
                               weight_decay=self.params['weight_decay'])
        optims.append(gen_optimizer)

        disc_optimizer = optim.Adam(self.disc_model.parameters(),
                               lr=self.params['disc_learning_rate'],
                               betas = (self.params['beta1'],self.params['beta2']),
                               weight_decay=self.params['weight_decay'])

        optims.append(disc_optimizer)

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    @data_loader
    def train_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'terrain':
            dataset = TerrainDataset(root = self.params['data_path'],
                                     train=True,
                                     transform = transform,
                                     hide_green=self.params['hide_green'],
                                     norm=self.params['norm'])
        else:
            raise ValueError('Undefined dataset type')

        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size= self.params['batch_size'],
                          num_workers=self.params['n_workers'],
                          shuffle = True,
                          drop_last=True)


    @data_loader
    def test_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'celeba':
            self.sample_dataloader =  DataLoader(CelebA(root = self.params['data_path'],
                                                        split = "test",
                                                        transform=transform,
                                                        download=False),
                                                 batch_size= 25,
                                                 shuffle = True,
                                                 drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        elif self.params['dataset'] == 'terrain':
            dataset = TerrainDataset(root = self.params['data_path'],
                                    train=False,
                                    transform = transform,
                                    hide_green=self.params['hide_green'])
            self.sample_dataloader = DataLoader(dataset,
                                                batch_size= 25,
                                                shuffle = True,
                                                drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        else:
            raise ValueError('Undefined dataset type')

        return self.sample_dataloader



    @data_loader
    def val_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'terrain':
            dataset = TerrainDataset(root = self.params['data_path'],
                                    train=False,
                                    transform = transform,
                                    hide_green=self.params['hide_green'],
                                    norm=self.params['norm'])
            self.sample_dataloader = DataLoader(dataset,
                                                batch_size= 25,
                                                num_workers=self.params['n_workers'],
                                                shuffle = False,
                                                drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        else:
            raise ValueError('Undefined dataset type')

        return self.sample_dataloader

    def data_transforms(self):

        if self.params['dataset'] == 'terrain':
            transform = None
        else:
            raise ValueError('Undefined dataset type')
        return transform

