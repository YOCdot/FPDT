# ---*--- File Information ---*---
# @File       :  integration.py
# @Date&Time  :  2022-09-07, 10:42:57
# @Project    :  tfpd
# @Platform   :  Apple ARM64
# @Software   :  PyCharm
# @Author     :  yoc
# @Email      :  yyyyyoc@hotmail.com


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

from typing import List
from torch import Tensor

from models.convnext import LayerNorm, enc_proj_tiny, enc_proj_small, enc_proj_base


class ScaleBlock(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int, stride: int,
                 eps: float = 1e-6, drop_path: float = 0., layer_scale_init_value: float = 1e-6,
                 data_format="channels_first"):
        super().__init__()
        self.dwconv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=kernel_size, stride=stride,
                                groups=in_dim)
        self.norm = LayerNorm(normalized_shape=in_dim, eps=eps, data_format=data_format)

        # self.pwconv1 = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1)
        # self.pwconv1 = nn.Linear(in_features=in_dim, out_features=out_dim)

        self.pwconv = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.act = nn.GELU()

        # self.pwconv2 = nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=1, stride=1)
        # self.pwconv2 = nn.Linear(in_features=out_dim, out_features=out_dim)

        # TODO: 这两个模块要想办法添加到网络里去
        # self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_dim)),
        #                           requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):

        x = self.dwconv(x)
        x = self.norm(x)
        x = self.act(x)

        x = torch.permute(x, (0, 2, 3, 1))
        # print("per: {}".format(x.size()))
        x = self.pwconv(x)
        # print("pwconv: {}".format(x.size()))
        x = torch.permute(x, (0, 3, 1, 2))
        # print("per: {}".format(x.size()))

        # if self.gamma is not None:
        #     # x = x.permute(0, 2, 3, 1)
        #     print("gamma: {}, x: {}".format(self.gamma.size(), x.size()))
        #     # exit()
        #     x = self.gamma * x
        # x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        # x = input_ + self.drop_path(x)

        return x


class Pyramid(nn.Module):

    def __init__(self, encoder: nn.Module,
                 out_dim: int = 192, kernel_list: list = [3, 5, 7, 9], dim_list: list = [96, 192, 384, 768]):
        super().__init__()

        '''encoding-projection'''

        # ConvNeXt encoder
        self.encoder = encoder

        # dim-projection
        dim_list.reverse()
        self.scale_convs = nn.ModuleList([
            ScaleBlock(in_dim=in_dim, out_dim=out_dim, kernel_size=ks, stride=ks) for in_dim, ks in zip(dim_list, kernel_list)
        ])

        self.downsampling = nn.ModuleList([
            nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, stride=1) for in_dim, out_dim in zip(dim_list[:-1], dim_list[1:])
        ])

    def forward(self, x):

        feature_maps = self.encoder(x)

        if (feature_maps[0].size())[-1] and (feature_maps[0].size())[-2] != 160:  # check dim
            raise ValueError("size mismatch!")
        else:
            feature_maps.reverse()
            # top-to-bottom
            for i, ds in enumerate(self.downsampling):
                feature_maps[i+1] = feature_maps[i+1] + ds(
                    F.interpolate(input=feature_maps[i],
                                  size=(feature_maps[i].shape[-2]*2, feature_maps[i].shape[-1]*2),
                                  mode="nearest")
                )

        # down-sampling
        for i, (m, c) in enumerate(zip(feature_maps, self.scale_convs)):
            # print("idx:{}, map:{}\nconv:{}.".format(i, m.size(), c))
            # print("idx:{}, map:{}.".format(i, m.size()))
            print(c(m).size())
            feature_maps[i] = c(m).flatten(2)
            # print("intermediate:{}, scaled:{}\n".format(m.size(), feature_maps[i].size()))

        # aggregation
        aggregated = torch.cat([fm for fm in feature_maps], dim=-1)
        print("aggregated:{}".format(aggregated.size()))

        return aggregated


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def pyramid_tiny(pretrained=None, **kwargs):

    if pretrained:
        encoder = enc_proj_tiny(pretrained=pretrained, **kwargs)
        pyramid = Pyramid(encoder=encoder, out_dim=192, kernel_list=[3, 5, 7, 9])
        print(f"Pyramidal structure complete!: {get_parameter_number(pyramid)}")
    else:
        print("not using pre-train!")
        encoder = enc_proj_tiny(pretrained=None, **kwargs)
        pyramid = Pyramid(encoder=encoder, out_dim=192, kernel_list=[3, 5, 7, 9])

    return pyramid


def pyramid_small(pretrained=None, **kwargs):

    if pretrained:
        encoder = enc_proj_small(pretrained=pretrained, **kwargs)
        pyramid = Pyramid(encoder=encoder, out_dim=384, kernel_list=[3, 5, 7, 9])
        print(f"Pyramidal structure complete!: {get_parameter_number(pyramid)}")
    else:
        print("not using pre-train!")
        encoder = enc_proj_small(pretrained=None, **kwargs)
        pyramid = Pyramid(encoder=encoder, out_dim=384, kernel_list=[3, 5, 7, 9])

    return pyramid


def pyramid_base(pretrained=None, **kwargs):

    if pretrained:
        encoder = enc_proj_base(pretrained=pretrained, **kwargs)
        pyramid = Pyramid(encoder=encoder, out_dim=768, kernel_list=[3, 5, 7, 9])
        print(f"Pyramidal structure complete!: {get_parameter_number(pyramid)}")
    else:
        print("not using pre-train!")
        encoder = enc_proj_base(pretrained=None, **kwargs)
        pyramid = Pyramid(encoder=encoder, out_dim=768, kernel_list=[3, 5, 7, 9])

    return pyramid


if __name__ == '__main__':

    import time
    starting = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    import cv2
    import torchvision.transforms as transforms
    img = cv2.imread('../train_image.jpg')
    # INPUT-384
    trans384 = transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize((384, 384))])  # (480, 640, 3) -> (3, 224, 224)
    input384 = trans384(img)  # tensor数据格式是torch(C,H,W)
    input384 = input384.unsqueeze(0)
    input384_b = input384
    input384 = torch.cat((input384, input384_b), dim=0)
    input384 = input384.to(device)

    # INPUT-640
    trans640 = transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize((640, 640))])  # (480, 640, 3) -> (3, 224, 224)
    input640 = trans640(img)  # tensor数据格式是torch(C,H,W)
    input640 = input640.unsqueeze(0)
    input640_b = input640
    input640 = torch.cat((input640, input640_b), dim=0)
    input640 = input640.to(device)

    # mod = enc_proj_tiny(pretrained=True)
    # print(mod)

    # pyramid_model = pyramid_tiny(pretrained=True)
    pyramid_model = pyramid_tiny(pretrained="/Users/iyoc/ProjectFiles/ConvNeXtWeights/convnext_tiny_22k_1k_384.pth")
    # pyramid_model = pyramid_tiny(pretrained="/home/ubuntu/allusers/z1t/yoc/convnext_tiny_22k_1k_384.pth")
    # pyramid_model = pyramid_small(pretrained="/mnt/DataDisk/yoc/convnext_small_22k_1k_384.pth")
    pyramid_model = pyramid_model.to(device)

    # y_hat, feature_maps = mod(input384)
    # y_hat, feature_maps = mod(input640)
    # print(y_hat.size())

    # print(pyramid_model)
    f_m = pyramid_model(input640)

    ending = time.time()
    print("time using: {:.2f}s".format(ending - starting))


