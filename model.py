import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
import einops
from einops.layers.torch import Rearrange
from einops import repeat


#############################
class CGenerator(nn.Module):
    def __init__(self, num_classes=10, nz=100):
        super().__init__()

        self.nz = nz
        self.num_classes = num_classes

        # ConvNet Output Size Calculator https://asiltureli.github.io/Convolution-Layer-Calculator/
        # TODO
        self.main = nn.Sequential(
            nn.Linear(self.nz+self.num_classes, 256*7*7, bias=False),                                                                # -> 256*7*7
            nn.BatchNorm1d(num_features=256*7*7),
            nn.LeakyReLU(),
            Rearrange('b (c h w) -> b c h w', c=256, h=7, w=7),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=1, padding=2, bias=False),  # -> 128 x 7 x 7
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),   # -> 64 x 14 x 14
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False),     # -> 1 x 28 x 28
            nn.Tanh(),
        )
    
    def forward(self, z, c):
        # TODO
        # 노이즈의 차원(nz)에 조건(10차원)을 붙여 입력으로 준다.
        z = torch.cat([z, c], dim=1)
        return self.main(z)



class CDiscriminator(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.num_classes = num_classes

        # ConvNet Output Size Calculator https://asiltureli.github.io/Convolution-Layer-Calculator/
        # TODO
        # DCGAN에서 입력이 1 * 28 * 28이었다면
        # CDCGAN에서는 조건 10* 28 * 28 을 추가해서
        # 11 * 28 * 28 이다. 따라서 맨 처음 Conv2d에서 채널 수를 바꿔준다.
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=1 + self.num_classes, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),      # -> 64 x 14 x 14
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),    # -> 128 x 7 x 7
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(128*7*7, 1, bias=False),                                                              # -> 1
            nn.Sigmoid(),
        )
        
    def forward(self, x, c):
        # TODO
        # 판별자에 조건을 추가하기 위해서는 생성자와 다른 방법이 필요
        # 판별자 입력 x는 1 * 28 * 28 이므로 
        # One-hot 10차원 벡터 -> 10 * 28 * 28
        # 총 11 * 28 * 28이 들어가도록 한다.
        # einpos 기능으로 shpae을 10 * 1 * 1로 만든 후 28*28로 반복.
        c_broadcast = repeat(c,'b c -> b c h w', h=28, w=28) # 10 * 28 * 28
        x = torch.cat([x,c_broadcast], dim=1)
        return self.main(x)
    


class CDCGAN(L.LightningModule):
    def __init__(self, num_classes=10, nz=100, lr=0.0002, beta1=0.5):
        super().__init__()
        self.automatic_optimization = False

        self.num_classes = num_classes
        self.nz = nz
        self.lr = lr
        self.beta1 = beta1
        self.save_hyperparameters()

        self.generator = CGenerator(nz=self.nz, num_classes=self.num_classes)
        self.discriminator = CDiscriminator(num_classes=self.num_classes)

    def forward(self, z, c):
        return  self.generator(z,c)
    
    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
    
    def training_step(self, batch, batch_idx):
        x, c = batch
        batch_size = x.size(0)


        opt_g, opt_d = self.optimizers()

        # sample noises
        z = torch.randn(batch_size, self.nz, device=self.device)

        # one-hot condition
        c = F.one_hot(c, num_classes=self.num_classes).float()

        # generator_step
        # TODO
        self.toggle_optimizer(opt_g)

        fake = self(z, c)
        pred_fake = self.discriminator(fake, c)
        
        loss_G = self.adversarial_loss(pred_fake, torch.ones_like(pred_fake))
        self.log("loss_G", loss_G, prog_bar=True)
        self.manual_backward(loss_G)
        opt_g.step()
        opt_g.zero_grad()

        self.untoggle_optimizer(opt_g)


        # discriminator_step
        # TODO
        self.toggle_optimizer(opt_d)

        real = x
        z = torch.randn(batch_size, self.nz, device=self.device)
        fake = self(z, c).detach()
        
        pred_real = self.discriminator(real, c)
        pred_fake = self.discriminator(fake, c)

        loss_D_real = self.adversarial_loss(pred_real, torch.ones_like(pred_real))
        loss_D_fake = self.adversarial_loss(pred_fake, torch.zeros_like(pred_fake))
        loss_D = (loss_D_real + loss_D_fake) / 2
        self.log("loss_D", loss_D, prog_bar=True)
        self.manual_backward(loss_D)
        opt_d.step()
        opt_d.zero_grad()

        self.untoggle_optimizer(opt_d)


    def configure_optimizers(self):
        opt_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        return [opt_G, opt_D], []

