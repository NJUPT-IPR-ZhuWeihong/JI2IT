from models import *

import torch
import os
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid
import numpy as np
import cv2
from tqdm import tqdm


#
transform = transforms.Compose([
    # transforms.Resize((128, 128), Image.BICUBIC),   #注意设置图片大小
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])




def loader_Image(path):
    '''
    path:图片路径
    函数说明：读取图片，并按照训练时的方式转换到tensor变量形式
    '''
    img = Image.open(path)  #加载图片为PIL格式
    #img_shape = np.array(img.size)  #获取图片格式
    #img = img.convert('RGB')  #将图片转为RGB
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def test_on_images(floder_path, save_path, cycle_G_AB_path):
    '''
    floder_path:测试图片所在的文件夹地址
    save_path:保存生成图片文件夹的地址
    generator_path：生成器保存的模型参数
    '''
    for e in np.arange(400, 401, 10):

        #创建保存文件夹
        #os.makedirs('test/Megaface10/model_{}'.format(e), exist_ok=True)
        #os.makedirs(save_path, exist_ok=True)
        #GPU OR CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        cycle_G_AB = GeneratorResNet((3, 128, 128), 6).to(device)
        #cycle_G_BA = GeneratorResNet((3, 128, 128), 6).to(device)

        #cycle_G_AB = Generator().to(device)

        Tensor = torch.cuda.FloatTensor
        #cycle_G_AB.load_state_dict(torch.load('saved_models/Megaface10/G_BA_{}.pth'.format(e)))
        cycle_G_AB.load_state_dict(torch.load(cycle_G_AB_path))
        cycle_G_AB.eval()

        #加载测试图片
        for ImgName in tqdm(os.listdir(floder_path)):
            temp_img_path = os.path.join(floder_path, ImgName)  #图片路径
            img_var = Variable(loader_Image(temp_img_path).type(Tensor))  #输入的低分辨率图片
            with torch.no_grad():
                gen_img = cycle_G_AB(img_var)

                #保存图片
                gen_img = make_grid(gen_img, nrow=1, normalize=True)
                #save_image(gen_img, os.path.join('test/Megaface10/model_{}'.format(e), ImgName), normalize=False)
                save_image(gen_img, os.path.join(save_path, ImgName), normalize=False)#保存图片的路径
                #print("{} have done!".format(ImgName))
        print("epoch {} have done!".format(e))


def test_on_floders(floder_path, save_path, cycle_G_AB_path):
    # 创建保存文件夹
    os.makedirs(save_path, exist_ok=True)
    # GPU OR CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cycle_G_AB = GeneratorResNet((3, 128, 128), 6).to(device)
    Tensor = torch.cuda.FloatTensor

    cycle_G_AB.load_state_dict(torch.load(cycle_G_AB_path))
    cycle_G_AB.eval()
    for folder in os.listdir(floder_path):
        os.makedirs(os.path.join(save_path, folder), exist_ok=True)
        # 加载测试图片
        for ImgName in os.listdir(os.path.join(floder_path, folder)):
            temp_img_path = os.path.join(floder_path, folder, ImgName)  # 图片路径
            img_var = Variable(loader_Image(temp_img_path).type(Tensor))  # 输入的低分辨率图片
            with torch.no_grad():
                gen_img = cycle_G_AB(img_var)
                # 保存图片
                gen_img = make_grid(gen_img, nrow=1, normalize=True)
                save_image(gen_img, os.path.join(save_path, folder, ImgName), normalize=False)
        print("{} have done!".format(folder))


# cycleGAN+BDSP最好模型  BDSP(10, 0.8) epoch = 160
# fid_value = 71.20918603140751
testImage_path = r'CFF10'
saveImage_path = r'results'
cycle_G_AB_path = "saved_models/G_BA_160.pth"
# cycle_G_AB_path = "saved_models/CelebA/G_BA_130.pth"
test_on_images(testImage_path, saveImage_path, cycle_G_AB_path)
#test_on_floders(testImage_path, saveImage_path, cycle_G_AB_path)

