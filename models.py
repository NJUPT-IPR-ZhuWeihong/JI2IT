import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]#channels = 3

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(channels),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features#in_features = 64

        # Downsampling
        for _ in range(2):
            out_features *= 2#out_features=128,256
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features


        # Residual blocks  （256， 32， 32）
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(channels), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)


        #+		input_shape	(3, 128, 128)	tuple
        #+		00	ReflectionPad2d((3, 3, 3, 3))	ReflectionPad2d
        #+		01	Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1))	Conv2d
        #+		02	InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)	InstanceNorm2d
        #+		03	ReLU(inplace=True)	ReLU
        #+		04	Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))	Conv2d
        #+		05	InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)	InstanceNorm2d
        #+		06	ReLU(inplace=True)	ReLU
        #+		07	Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))	Conv2d
        #+		08	InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)	InstanceNorm2d
        #+		09	ReLU(inplace=True)	ReLU
        #+		10	ResidualBlock(
        #      (block): Sequential(
        #        (0): ReflectionPad2d((1, 1, 1, 1))
        #        (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        #        (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        #        (3): ReLU(inplace=True)
        #        (4): ReflectionPad2d((1, 1, 1, 1))
        #        (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
        #        (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)

        #  )
        #)	ResidualBlock


    def forward(self, x):
        return self.model(x)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)

def pad_tensor(input):
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 16

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div / 2)
            pad_bottom = int(height_div - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.data.shape[2], input.data.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom


def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:, :, pad_top: height - pad_bottom, pad_left: width - pad_right]
######## EnlightenGAN Start #######################
class Generator(nn.Module):
    def __init__(self,norm_layer=nn.InstanceNorm2d):
        super(Generator, self).__init__()
        p = 1

        self.conv11 = nn.Conv2d(3, 32, 3, padding=p)

        self.LReLU1_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_1 = norm_layer(32)

        self.conv1_2 = nn.Conv2d(32, 32, 3, stride=1, padding=p)
        self.LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_2 = norm_layer(32)
        self.max_pool1 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=p)
        self.LReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_1 = norm_layer(64)

        self.conv2_2 = nn.Conv2d(64, 64, 3, stride=1, padding=p)
        self.LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_2 = norm_layer(64)
        self.max_pool2 = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=p)
        self.LReLU3_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_1 = norm_layer(128)

        self.conv3_2 = nn.Conv2d(128, 128, 3, stride=1, padding=p)
        self.LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_2 = norm_layer(128)
        self.max_pool3 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=p)
        self.LReLU4_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_1 = norm_layer(256)

        self.conv4_2 = nn.Conv2d(256, 256, 3, stride=1, padding=p)
        self.LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_2 = norm_layer(256)
        self.max_pool4 = nn.MaxPool2d(2)

        self.conv5_1 = nn.Conv2d(256, 512, 3, padding=p)
        self.LReLU5_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn5_1 = norm_layer(512)

        self.res1 = ResidualBlock(512)
        self.res2 = ResidualBlock(512)
        self.res3 = ResidualBlock(512)
        self.res4 = ResidualBlock(512)

        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=p)
        self.LReLU5_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn5_2 = norm_layer(512)

        # self.deconv5 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.deconv5 = nn.Conv2d(512, 256, 3, padding=p)
        self.conv6_1 = nn.Conv2d(512, 256, 3, padding=p)
        self.LReLU6_1 = nn.LeakyReLU(0.2, inplace=True)

        self.bn6_1 = norm_layer(256)
        self.conv6_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU6_2 = nn.LeakyReLU(0.2, inplace=True)

        self.bn6_2 = norm_layer(256)

        # self.deconv6 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.deconv6 = nn.Conv2d(256, 128, 3, padding=p)
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=p)
        self.LReLU7_1 = nn.LeakyReLU(0.2, inplace=True)

        self.bn7_1 = norm_layer(128)
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU7_2 = nn.LeakyReLU(0.2, inplace=True)

        self.bn7_2 = norm_layer(128)

        # self.deconv7 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.deconv7 = nn.Conv2d(128, 64, 3, padding=p)
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding=p)
        self.LReLU8_1 = nn.LeakyReLU(0.2, inplace=True)

        self.bn8_1 = norm_layer(64)
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU8_2 = nn.LeakyReLU(0.2, inplace=True)

        self.bn8_2 = norm_layer(64)

        # self.deconv8 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.deconv8 = nn.Conv2d(64, 32, 3, padding=p)
        self.conv9_1 = nn.Conv2d(64, 32, 3, padding=p)
        self.LReLU9_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn9_1 = norm_layer(32)

        self.conv9_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU9_2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv10 = nn.Conv2d(32, 3, 1)
        self.tanh = nn.Tanh()

    def forward(self, input):

        input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(input)
        x = self.bn1_1(self.LReLU1_1(self.conv11(input)))

        conv1 = self.bn1_2(self.LReLU1_2(self.conv1_2(x)))  # 128 128 32
        x = self.max_pool1(conv1)

        x = self.bn2_1(self.LReLU2_1(self.conv2_1(x)))
        conv2 = self.bn2_2(self.LReLU2_2(self.conv2_2(x)))  # 64 64 64
        x = self.max_pool2(conv2)

        x3 = self.bn3_1(self.LReLU3_1(self.conv3_1(x)))
        conv3 = self.bn3_2(self.LReLU3_2(self.conv3_2(x3)))  # 32 32 128
        x = self.max_pool3(conv3)

        x4 = self.bn4_1(self.LReLU4_1(self.conv4_1(x)))  # 32 32 256
        conv4 = self.bn4_2(self.LReLU4_2(self.conv4_2(x4)))  # 16 16 256
        x = self.max_pool4(conv4)

        x = self.bn5_1(self.LReLU5_1(self.conv5_1(x)))  # 16,16,512

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)

        x = self.bn5_2(self.LReLU5_2(self.conv5_2(x)))

        conv5 = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)  # [512,32,32]
        up6 = torch.cat([self.deconv5(conv5), conv4], 1)
        x = self.bn6_1(self.LReLU6_1(self.conv6_1(up6)))
        x = F.dropout2d(x)
        conv6 = self.bn6_2(self.LReLU6_2(self.conv6_2(x)))

        conv6 = F.interpolate(conv6, scale_factor=2, mode='bilinear', align_corners=True)
        up7 = torch.cat([self.deconv6(conv6), conv3], 1)
        x = self.bn7_1(self.LReLU7_1(self.conv7_1(up7)))
        x = F.dropout2d(x)
        conv7 = self.bn7_2(self.LReLU7_2(self.conv7_2(x)))

        conv7 = F.interpolate(conv7, scale_factor=2, mode='bilinear', align_corners=True)
        up8 = torch.cat([self.deconv7(conv7), conv2], 1)
        x = self.bn8_1(self.LReLU8_1(self.conv8_1(up8)))
        x = F.dropout2d(x)
        conv8 = self.bn8_2(self.LReLU8_2(self.conv8_2(x)))

        conv8 = F.interpolate(conv8, scale_factor=2, mode='bilinear', align_corners=True)
        up9 = torch.cat([self.deconv8(conv8), conv1], 1)
        x = self.bn9_1(self.LReLU9_1(self.conv9_1(up9)))
        conv9 = self.LReLU9_2(self.conv9_2(x))
        latent = self.conv10(conv9)
        output = latent + input
        output = self.tanh(output)
        output = pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)
        return output



if __name__ == '__main__':
    gan = GeneratorResNet()
    x=torch.randn(8,3,128,128)
    g = gan(x)
    print(gan)
    print(g.shape)
    dis = Discriminator((3,128,128))
    d = dis(x)
    print(dis)
    print(d.shape)