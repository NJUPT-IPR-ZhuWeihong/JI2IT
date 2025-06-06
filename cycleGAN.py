import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from BDSP_Face import *



os.environ["CUDA_VISIBLE_DEVICES"]="0"

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(2020)

#cycleGAN+BDSP_1
parser = argparse.ArgumentParser()

parser.add_argument("--experiment_name", type=str, default="Megaface10", help="name of the experiment")

parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=401, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="F:\ITS 2022\Megaface\Me_New\CycleGAN", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=200, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=6, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight") 
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
parser.add_argument("--lambda_dsp", type=float, default=5.0, help="dsp loss weight ")
opt = parser.parse_args()
print(opt)



# Create sample and checkpoint directories
# Create sample and checkpoint directories
os.makedirs("images/%s" % (opt.experiment_name), exist_ok=True)
os.makedirs("saved_models/%s" % ( opt.experiment_name), exist_ok=True)
os.makedirs("test/%s" % ( opt.experiment_name), exist_ok=True)
# Losses
criterion_GAN = torch.nn.MSELoss()#该统计参数是预测数据和原始数据对应点误差的平方和的均值
criterion_cycle = torch.nn.L1Loss()#它是把目标值  与模型输出（估计值）做绝对值得到的误差。
criterion_identity = torch.nn.L1Loss()#它是把目标值 与模型输出（估计值）  做绝对值得到的误差。
criterion_BDSP = torch.nn.L1Loss()#它是把目标值 与模型输出（估计值）  做绝对值得到的误差。

cuda = torch.cuda.is_available()

input_shape = (opt.channels, opt.img_height, opt.img_width)

# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
#G_AB = Generator()
#G_BA = Generator()

D_A = Discriminator(input_shape)
D_B = Discriminator(input_shape)

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()
    criterion_BDSP.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G_AB.load_state_dict(torch.load("F:\ITS 2022\Part I\cycleGAN\cycleGAN_6\saved_models\Megaface01\G_AB_180.pth"))
    G_BA.load_state_dict(torch.load("F:\ITS 2022\Part I\cycleGAN\cycleGAN_6\saved_models\Megaface01\G_BA_180.pth"))
    D_A.load_state_dict(torch.load("F:\ITS 2022\Part I\cycleGAN\cycleGAN_6\saved_models\Megaface01\D_A_180.pth"))
    D_B.load_state_dict(torch.load("F:\ITS 2022\Part I\cycleGAN\cycleGAN_6\saved_models\Megaface01\D_B_180.pth"))
else:
#    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
transforms_ = [
    transforms.Resize(int(opt.img_height), Image.BICUBIC),
    #transforms.RandomCrop((opt.img_height, opt.img_width)),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# Training data loader
#"data/%s" % opt.dataset_name
dataloader = DataLoader(
    ImageDataset('%s'%opt.dataset_name, transforms_=transforms_, unaligned=False),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=0,
)
# Test data loader
val_dataloader = DataLoader(
    ImageDataset('%s'%opt.dataset_name, transforms_=transforms_, unaligned=False, mode="test"),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=0,
)


def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    G_AB.eval()
    G_BA.eval()
    real_A = Variable(imgs["A"].type(Tensor))
    fake_B = G_AB(real_A)
    real_B = Variable(imgs["B"].type(Tensor))
    fake_A = G_BA(real_B)
    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=True)
    real_B = make_grid(real_B, nrow=5, normalize=True)
    fake_A = make_grid(fake_A, nrow=5, normalize=True)
    fake_B = make_grid(fake_B, nrow=5, normalize=True)
    # Arange images along y-axis
    image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
    save_image(image_grid, "images/%s/%s.png" % ( opt.experiment_name, batches_done), normalize=False)


# ----------
#  Training
# ----------

prev_time = time.time()
#训练批次循环
for epoch in range(opt.epoch, opt.n_epochs):
    #对train集数据加载，获得其索引和图片

    for i, batch in enumerate(dataloader):
        

        # Set model input
        #A和B分别是油画和真实图片
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))



        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        #这两个是生成器，此处是训练模式
        G_AB.train()
        G_BA.train()

        #经典四步走 
        optimizer_G.zero_grad()

        #id loss
        loss_id_A = criterion_identity(G_BA(real_A), real_A)#身份损失是生成图片与真实图片之间的L1Loss()
        loss_id_B = criterion_identity(G_AB(real_B), real_B)
        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)
        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2


        #BDSP loss
        loss_BDSP_A = criterion_BDSP(BDSP_Face(fake_B), BDSP_Face(real_A))
        loss_BDSP_B = criterion_BDSP(BDSP_Face(fake_A), BDSP_Face(real_B))
        loss_BDSP = (loss_BDSP_A + loss_BDSP_B) / 2

        # Cycle loss
        recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        #额外BDSP_Face loss
        loss_BDSP_RA = criterion_BDSP(BDSP_Face(recov_A), BDSP_Face(real_A))
        loss_BDSP_RB = criterion_BDSP(BDSP_Face(recov_B), BDSP_Face(real_B))
        loss_BDSP_R = (loss_BDSP_RA + loss_BDSP_RB) / 2
      

        # Total loss
        loss_G = loss_GAN*1.1  + opt.lambda_cyc * loss_cycle + opt.lambda_id * loss_identity + opt.lambda_dsp*loss_BDSP  + opt.lambda_dsp*loss_BDSP_R

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------
        #训练判别器A
        optimizer_D_A.zero_grad()

        # Real loss
        #真实油画图片图片判别结果与全一差距
        loss_real = criterion_GAN(D_A(real_A), valid)#MSE
        # Fake loss (on batch of previously generated samples)
        #生成伪油画图片与虚假图片的差距
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
        # Total loss
        #计算损失
        loss_D_A = (loss_real + loss_fake) / 2
        #更新参数
        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------
        #训练判别器B
        optimizer_D_B.zero_grad()

        # Real loss
        #真实现实图片与全一集之间的差距
        loss_real = criterion_GAN(D_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f, BDSP_1: %f, BDSP_2: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                loss_G.item(),
                loss_GAN.item(),
                loss_cycle.item(),
                loss_identity.item(),
                loss_BDSP.item()*5,
                loss_BDSP_R.item()*5,
                time_left,
            )
        )

         #If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)


    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()


    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0 and epoch>=200:
        # Save model checkpoints
        torch.save(G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % ( opt.experiment_name, epoch))
        torch.save(G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (opt.experiment_name, epoch))
        torch.save(D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (opt.experiment_name, epoch))
        torch.save(D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (opt.experiment_name, epoch))


#torch.save(G_AB.state_dict(), "saved_models/%s/%s/G_AB_%d.pth" % (opt.dataset_name, opt.experiment_name, epoch))
#torch.save(G_BA.state_dict(), "saved_models/%s/%s/G_BA_%d.pth" % (opt.dataset_name, opt.experiment_name, epoch))
#torch.save(D_A.state_dict(), "saved_models/%s/%s/D_A_%d.pth" % (opt.dataset_name, opt.experiment_name, epoch))
#torch.save(D_B.state_dict(), "saved_models/%s/%s/D_B_%d.pth" % (opt.dataset_name, opt.experiment_name, epoch))

