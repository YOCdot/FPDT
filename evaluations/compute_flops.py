# ---*--- File Information ---*---
# @File       :  __init__.py
# @Date&Time  :  2022-07-18, 01:02:30
# @Project    :  flops
# @Platform   :  Apple Silicon (arm64)
# @Software   :  PyCharm
# @Author     :  yoc
# @Email      :  yyyyyoc@hotmail.com


import cv2

import torch
import torchvision.transforms as transforms

import thop
from thop import clever_format

from detr import build_detr
from ssd import build_ssd300
from yolos import build_yolos
from yolov3 import build_yolov3
from fpdt import build_fpdt


def test_img_2_tensor(img_size=672, img_path='./train_image.jpg'):
    img = cv2.imread(img_path)
    trans = transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize((img_size, img_size))])  # (480, 640, 3) -> (3, 224, 224)
    ipt = trans(img)  # tensor数据格式是torch(C,H,W)
    ipt = ipt.unsqueeze(0)

    o_ipt = None
    for i in range(2):
        if i == 0:
            o_ipt = ipt
        else:
            o_ipt = torch.cat((o_ipt, ipt), dim=0)
    print(o_ipt.size())
    return o_ipt


def main(*args, **kwargs):
    print(kwargs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Computing device: {device}.")
    torch.cuda.empty_cache()

    input300 = test_img_2_tensor(300).to(device)
    input400 = test_img_2_tensor(400).to(device)
    input640 = test_img_2_tensor(640).to(device)
    input672 = test_img_2_tensor(672).to(device)

    metric_dict = {}
    for model_name in kwargs.keys():
        print(model_name)
        kwargs[model_name].to(device)
        # if (model_name == "detr-r101") or (model_name == "yolos-small"):
        #     flops, params = thop.profile(kwargs[model_name], inputs=(input400,))
        # elif (model_name == "yolov3-tiny") or (model_name == "yolov3-base"):
        #     flops, params = thop.profile(kwargs[model_name], inputs=(input640,))
        # elif model_name == "ssd-300":
        #     flops, params = thop.profile(kwargs[model_name], inputs=(input300,))
        # else:
        # if kwargs[model_name] == "fpdt-tiny" or "fpdt-small":
        flops, params = thop.profile(kwargs[model_name], inputs=(input640,))
        # else:
        #     flops, params = thop.profile(kwargs[model_name], inputs=(input672,))
            # raise ValueError("no image size matched!")
        flops, params = clever_format([flops, params], "%.3f")
        metric_dict[model_name] = {f"Model: {model_name}, Parameters: {params}, FLOPs: {flops}."}
        # print(f"Model: {model_name}, FLOPs: {flops}, Parameters: {params}")

    for k in metric_dict.keys():
        print(metric_dict[k])


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':

    # model_list = [
    #     build_detr(backbone="resnet34"),
    #     build_detr(backbone="resnet101"),
    #     build_ssd300(backbone="resnet50"),
    #     build_tssd(model_name="tiny"),
    #     build_tssd(model_name="small"),
    #     build_yolos(model_name="tiny"),
    #     build_yolos(model_name="small"),
    #     build_yolov3(model_name="tiny"),
    #     build_yolov3(model_name="base")
    # ]

    model_dict = {
        # "fpdt-tiny": build_fpdt(model_name="tiny"),
        "fpdt-small": build_fpdt(model_name="small"),
        "detr-r34": build_detr(backbone="resnet34"),
        "detr-r101": build_detr(backbone="resnet101"),
        "detr-r152": build_detr(backbone="resnet152"),
        "ssd-300": build_ssd300(backbone="resnet50"),
        # "t-ssd-tiny": build_tssd(model_name="tiny"),
        # "t-ssd-small": build_tssd(model_name="small"),
        "yolos-tiny": build_yolos(model_name="tiny"),
        "yolos-small": build_yolos(model_name="small"),
        "yolov3-tiny": build_yolov3(model_name="tiny"),
        "yolov3-base": build_yolov3(model_name="base")
    }

    # model_dict = {"detr-r34": "resnet34",
    #               "detr-r101": "resnet101",
    #               "ssd": "resnet50",
    #               "tssd-tiny": "tiny",
    #               "tssd-small": "small",
    #               "yolos-tiny": "tiny",
    #               "yolos-small": "small",
    #               "yolov3-tiny": "tiny",
    #               "yolov3-base": "base"}

    main(**model_dict)
