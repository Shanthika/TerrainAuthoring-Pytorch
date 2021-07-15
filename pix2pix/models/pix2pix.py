import torch.nn as nn
import torch.nn.functional as F
import torch
 
##############################
#      U-NET Generator       #
##############################


class DownSample(nn.Module):
    def __init__(self, in_size, out_size, k_size=4, apply_batchnorm=True):
        super(DownSample, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size=k_size, stride=2, padding=(1,1), bias=False)]
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.3, True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UpSample(nn.Module):
    def __init__(self, in_size, out_size, k_size=4, dropout=False):
        super(UpSample, self).__init__()
        layers = [nn.ConvTranspose2d(in_size, out_size, kernel_size=k_size, stride=2, padding=(1,1), bias=False)        ]
        layers.append(nn.BatchNorm2d(out_size))
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU(True))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()

        self.down1 = DownSample(in_channels, 64, apply_batchnorm=False)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)
        self.down4 = DownSample(256, 512)
        self.down5 = DownSample(512, 512)
        self.down6 = DownSample(512, 512)
        self.down7 = DownSample(512, 512)
        self.down8 = DownSample(512, 512)

        self.up1 = UpSample(512, 512, dropout=0.5)
        self.up2 = UpSample(1024, 512, dropout=0.5)
        self.up3 = UpSample(1024, 512, dropout=0.5)
        self.up4 = UpSample(1024, 512)
        self.up5 = UpSample(1024, 256)
        self.up6 = UpSample(512, 128)
        self.up7 = UpSample(256, 64)

        self.final = nn.Sequential(
            # nn.Upsample(scale_factor=2),
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=(1,1), bias=True),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        # print(d1.shape)
        d2 = self.down2(d1)
        # print(d2.shape)
        d3 = self.down3(d2)
        # print(d3.shape)
        d4 = self.down4(d3)
        # print(d4.shape)
        d5 = self.down5(d4)
        # print(d5.shape)
        d6 = self.down6(d5)
        # print(d6.shape)
        d7 = self.down7(d6)
        # print(d7.shape)
        d8 = self.down8(d7)
        # print(d8.shape)
        u1 = self.up1(d8, d7)
        # print(u1.shape)
        u2 = self.up2(u1, d6)
        # print(u2.shape)
        u3 = self.up3(u2, d5)
        # print(u3.shape)
        u4 = self.up4(u3, d4)
        # print(u4.shape)
        u5 = self.up5(u4, d3)
        # print(u5.shape)
        u6 = self.up6(u5, d2)
        # print(u6.shape)
        u7 = self.up7(u6, d1)
        # print(u7.shape)
        return self.final(u7)


##############################
#        Discriminator       #
##############################

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            DownSample(in_channels*2, 64, apply_batchnorm=False),
            DownSample(64, 128 ),
            DownSample(128, 256 ),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=(1,1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.3, True),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=(1,1),bias=True)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


class Loss:
    def __init__(self, lambda_: int = 100):
        self.loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        self.lambda_ = lambda_

    def generator_loss(self, disc_gen_output, gen_output, gt):
        gan_loss = self.loss(disc_gen_output,torch.ones_like(disc_gen_output, requires_grad = False))
        l1_rec_loss  = self.l1_loss(gt, gen_output)
        total_gen_loss = gan_loss +  l1_rec_loss * self.lambda_  
        return total_gen_loss, l1_rec_loss, gan_loss
    
    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss(disc_real_output, torch.ones_like (disc_real_output, requires_grad = False))
        generated_loss = self.loss(disc_generated_output, torch.zeros_like(disc_generated_output, requires_grad = False))
        total_disc_loss = (real_loss + generated_loss)
        return total_disc_loss
    

