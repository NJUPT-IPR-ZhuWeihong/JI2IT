# Joint Image-to-Image Translation for Traffic Monitoring Driver Face Image Enhancement (JI2IT)
## Introduction
The real traffic monitoring driver face (TMDF) images are with complex multiple degradations, which decline
face recognition accuracy in real intelligent transportation systems (ITS). We are the first to propose joint image-to-image
(I2I) translation to enhance TMDF images of ITS. the experiments on TMDF (i.e., the brevity name of the face database collected from real ITS)
and Chinese famous face (CFF) databases, as well as CelebA and MegaFace databases, indicate that the proposed method can efficiently 
enhance TMDF images whose degradation variations are learned by FDSP-CG.

## Datasets
1. [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) is a publicly available dataset.
2. [MegaFace](https://megaface.cs.washington.edu/) is a publicly available dataset.
3. CFF is a private dataset and cannot be made public.
4. TMDF is also a private dataset and cannot be made public.

## Training
We provide the training codes for JI2IT.
You could improve it according to your own needs.
```
./gfpgan/train.py
```

### Configuration
Modify the related parameters (paths, loss weights, training steps, and etc.) in the config yaml files.
```
./options/train_restormgan.yml
```

## Testing
### Pre-trained models
Please download our pre-trained models via the following links [Baiduyun (extracted code: 1a2b)]() 
[Google Drive](). 
Place the downloaded pre-trained model in the following path。
```
./experiments/pretrained_models
```

```
./inference_gfpgan.py
```

## Citation
If you find this work useful for your research, please cite our paper
```
@article{hu2023joint,
  title={Joint image-to-image translation for traffic monitoring driver face image enhancement},
  author={Hu, Chang-Hui and Liu, Yu and Xu, Lin-Tao and Jing, Xiao-Yuan and Lu, Xiao-Bo and Yang, Wan-Kou and Liu, Pan},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={24},
  number={8},
  pages={7961--7973},
  year={2023},
  publisher={IEEE}
}
```

## Contact
If you have any questions, please feel free to contact the authors via 994628118@qq.com.
