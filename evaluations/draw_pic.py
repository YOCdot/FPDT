# ---*--- File Information ---*---
# @File       :  draw_pic.py
# @Date&Time  :  2022-07-19, 20:21:38
# @Project    :  flops
# @Platform   :  Apple Silicon (arm64)
# @Software   :  PyCharm
# @Author     :  yoc
# @Email      :  yyyyyoc@hotmail.com


import os
from pathlib import Path
import argparse
import imagesize
import cv2
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from einops import rearrange

from tssd import build

coco2017_cat_id = {
    "1": "person", "2": "bicycle", "3": "car", "4": "motorcycle", "5": "airplane",
    "6": "bus", "7": "train", "8": "truck", "9": "boat", "10": "traffic light",
    "11": "fire hydrant", "13": "stop sign", "14": "parking meter", "15": "bench",
    "16": "bird", "17": "cat", "18": "dog", "19": "horse", "20": "sheep",
    "21": "cow", "22": "elephant", "23": "bear", "24": "zebra", "25": "giraffe",
    "27": "backpack", "28": "umbrella",
    "31": "handbag", "32": "tie", "33": "suitcase", "34": "frisbee", "35": "skis",
    "36": "snowboard", "37": "sports ball", "38": "kite", "39": "baseball bat", "40": "baseball glove",
    "41": "skateboard", "42": "surfboard", "43": "tennis", "44": "bottle",
    "46": "wine glass", "47": "cup", "48": "fork", "49": "knife", "50": "spoon",
    "51": "bowl", "52": "banana", "53": "apple", "54": "sandwich", "55": "orange",
    "56": "broccoli", "57": "carrot", "58": "hot dog", "59": "pizza", "60": "donut",
    "61": "cake", "62": "chair", "63": "couch", "64": "potted plant", "65": "bed",
    "67": "dining table", "70": "toilet",
    "72": "tv", "73": "laptop", "74": "mouse", "75": "remote",
    "76": "keyboard", "77": "cell phone", "78": "microwave", "79": "oven", "80": "toaster",
    "81": "sink", "82": "refrigerator", "84": "book", "85": "clock",
    "86": "vase", "87": "scissors", "88": "teddy bear", "89": "hair drier", "90": "toothbrush",
}

img_choice = {
    "train2017": ["000000198947.jpg", "000000199943.jpg", "000000200376.jpg", "000000201508.jpg", "000000204162.jpg",
                  "000000205791.jpg", "000000231126.jpg", "000000241407.jpg", "000000246064.jpg", "000000257711.jpg",
                  "000000258727.jpg", "000000264019.jpg", "000000265950.jpg", "000000000081.jpg", "000000231126.jpg",
                  "000000205791.jpg"],
    "val2017": ["000000091495.jpg", "000000481413.jpg", "000000491725.jpg", "000000509008.jpg", "000000509719.jpg",
                "000000516173.jpg", "000000517523.jpg", "000000479953.jpg", "000000477955.jpg", "000000328337.jpg"]
}


def get_model_args(args: argparse.Namespace):
    model_parser = argparse.ArgumentParser("Draw info during detecting.", add_help=False)

    model_parser.add_argument("--model_name", default=args.model_scale, type=str)
    model_parser.add_argument("--device", default=args.device, type=str)
    model_parser.add_argument("--return_attn", default=False if not args.return_attn else args.return_attn, type=bool)

    model_parser.add_argument("--dataset_file", default="coco", type=str, help="dataset type")
    model_parser.add_argument("--det_token_num", default=100, type=int)
    model_parser.add_argument("--set_cost_class", default=1, type=int)
    model_parser.add_argument("--set_cost_bbox", default=5, type=int)
    model_parser.add_argument("--set_cost_giou", default=2, type=int)
    model_parser.add_argument("--bbox_loss_coef", default=5, type=int)
    model_parser.add_argument("--giou_loss_coef", default=2, type=int)
    model_parser.add_argument("--eos_coef", default=0.1, type=float)
    model_parser.add_argument("--aux_loss", default=False, type=bool)

    model_parser.add_argument("--encoder_pt", default=None)
    model_parser.add_argument("--neck_pt", default=None)

    model_args = model_parser.parse_args()

    return model_args


def load_model(draw_args: argparse.Namespace):
    model_args = get_model_args(args=draw_args)
    model, _, post_processor = build(model_args)

    if draw_args.weight_root is not None:
        checkpoint = torch.load(draw_args.weight_root, map_location="cpu")
        model.load_state_dict(state_dict=checkpoint["model"], strict=True)
        print(f"{draw_args.model_scale} weights loaded strictly!")

    return model, post_processor


def load_img(draw_args: argparse.Namespace, expand: bool = False):
    class Normalize(object):
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def __call__(self, image, target=None):
            image = torchvision.transforms.functional.normalize(image, mean=self.mean, std=self.std)
            return image

    original_size = imagesize.get(draw_args.image_path)
    img_name = draw_args.image_path.split("/")[-2] + "-" + draw_args.image_path.split("/")[-1]

    img_cv2 = cv2.imread(draw_args.image_path)

    trans = transforms.Compose([transforms.ToTensor(),
                                # (480, 640, 3) -> (3, 224, 224)
                                transforms.Resize((draw_args.warp_size, draw_args.warp_size)),
                                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_tensor = trans(img_cv2)  # tensor数据格式是torch(C,H,W)
    img_tensor = img_tensor.unsqueeze(0)

    if expand is True:
        out_tensor = None
        for i in range(expand):
            if i == 0:
                out_tensor = img_tensor
            else:
                out_tensor = torch.cat((out_tensor, img_tensor), dim=0)
        return out_tensor

    print(f"Image {img_name} was warped from {original_size} to ({draw_args.warp_size}, {draw_args.warp_size})!")

    return img_tensor, img_cv2, original_size, img_name


def draw_boxes_cv2(output, orig_size, img):
    boxes, logits = output["pred_boxes"].squeeze(0), output['pred_logits'].squeeze(0)
    for box, label in zip(boxes, logits):
        label = torch.argmax(label, dim=-1).item()
        if label != 91:
            label = coco2017_cat_id[str(label)]
            box = convert_cxcybwbh_to_ltxyrbxy(cxcybwbh=box, original_wh=orig_size)
            box_color = (0, 0, 255)
            # b-box
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=box_color, thickness=2)

            # label text
            labelSize = cv2.getTextSize(label + '0', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(img, (int(box[0]), int(box[1]) + 2),
                          (int(box[0]) + labelSize[0], int(box[1]) + labelSize[1] + 3),
                          color=box_color,
                          thickness=-1
                          )
            cv2.putText(img, label, (int(box[0]), int(box[1]) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        thickness=2)
            print(f"Box coord: (ltx-{box[0]:.2f}, lty-{box[1]:.2f}, rbx-{box[2]:.2f}, rby-{box[3]:.2f})  |  "
                  f"Category: ({label}).")


def convert_cxcybwbh_to_ltxyrbxy(cxcybwbh: torch.Tensor, original_wh: tuple):
    """To match opencv format, convert from (cx, cy, bw, bh) to (ltx, lty, rbx, rby), and get the real coordinate,
    originates from left top of image"""
    cx, cy, bw, bh = cxcybwbh
    ltx = ((cx - 0.5 * bw) * original_wh[0]).item()
    lty = ((cy - 0.5 * bh) * original_wh[1]).item()
    rbx = ((cx + 0.5 * bw) * original_wh[0]).item()
    rby = ((cy + 0.5 * bh) * original_wh[1]).item()
    return [ltx, lty, rbx, rby]


def convert_cxcybwbh_to_lbxywh(cxcybwbh: torch.Tensor, original_wh: tuple):
    """To match matplotlib format, convert from (cx, cy, bw, bh) to (lbx, lby, w, h), and get the real coordinate,
    originates from left bottom of image"""
    cx, cy, bw, bh = cxcybwbh
    rbx = ((cx - 0.5 * bw) * original_wh[0]).item()
    rby = ((cy - 0.5 * bh) * original_wh[1]).item()
    w = (bw * original_wh[0]).item()
    h = (bh * original_wh[1]).item()
    return [rbx, rby, w, h]


def draw_img_with_boxes(output, orig_size, img_path):
    fig, ax = plt.subplots(1, 1)
    img = plt.imread(img_path)

    boxes, logits = output["pred_boxes"].squeeze(0), output['pred_logits'].squeeze(0)
    for box, label in zip(boxes, logits):
        label = torch.argmax(label, dim=-1).item()
        if label != 91:
            box = convert_cxcybwbh_to_lbxywh(cxcybwbh=box, original_wh=orig_size)
            label = coco2017_cat_id[str(label)]
            print(f"Coordinate: ({box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}, {box[3]:.2f})  |  "
                  f"Category: ({label}).")

            # bbox
            box_rect = matplotlib.patches.Rectangle(xy=(box[0], box[1]),
                                                    width=box[2],
                                                    height=box[3],
                                                    edgecolor='r',
                                                    facecolor='none',
                                                    linewidth=1)
            ax.add_patch(box_rect)
            # category
            plt.text(x=box_rect.xy[0] + 5, y=box_rect.xy[1] - 10, s=label,
                     color='white',
                     fontweight='bold',
                     backgroundcolor='r',
                     fontsize=8)
    ax.imshow(img)


def draw_box_and_cat(output, orig_size, ax):
    boxes, logits = output["pred_boxes"].squeeze(0), output['pred_logits'].squeeze(0)
    for box, label in zip(boxes, logits):
        label = torch.argmax(label, dim=-1).item()
        if label != 91:
            box = convert_cxcybwbh_to_lbxywh(cxcybwbh=box, original_wh=orig_size)
            label = coco2017_cat_id[str(label)]
            print(f"Coordinate: ({box[0]:.2f}, {box[1]:.2f}, {box[2]:.2f}, {box[3]:.2f})  |  "
                  f"Category: ({label}).")

            # bbox
            box_rect = matplotlib.patches.Rectangle(xy=(box[0], box[1]),
                                                    width=box[2],
                                                    height=box[3],
                                                    edgecolor='r',
                                                    facecolor='none',
                                                    linewidth=2)
            ax.add_patch(box_rect)
            # category
            ax.text(x=box_rect.xy[0] + 4, y=box_rect.xy[1] + 9, s=label,
                    color='white',
                    fontweight="bold",
                    backgroundcolor="r",
                    fontsize=8)


def draw_attn_mask(map_list, original_size, ax):
    len_list = len(map_list)
    print(f"{len_list} different attention map scales.")

    attn_map = None

    for idx, m in enumerate(map_list):

        e = int(m.shape[1] ** .5)
        m = rearrange(m, "b (e1 e2) d -> b d e1 e2", e1=e, e2=e)
        m = F.interpolate(m, size=(original_size[1], original_size[0]), mode="bicubic", align_corners=False)
        m = torch.squeeze(m)

        max_value, _max_idx = torch.max(m, dim=0)

        max_value = max_value.cpu().detach().numpy()

        if attn_map is None:
            attn_map = max_value
        else:
            attn_map += max_value

    sns.heatmap(data=attn_map,
                square=True, cbar=False, alpha=0.6, cmap="binary",
                xticklabels=False, yticklabels=False,
                ax=ax)


def draw_predictions(inf_model: nn.Module,
                     attn_model: nn.Module,
                     draw_args: argparse.Namespace,
                     save_path: str = None):
    """original | box | attention-mask | det-token"""

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
    img = plt.imread(draw_args.image_path)
    for ax in axes:
        ax.imshow(img)
        ax.set_xticklabels("")
        ax.set_yticklabels("")
    axes[0].set_title(f"Input Image", size=25)
    axes[1].set_title(f"Predicted Boxes", size=25)
    axes[2].set_title(f"Attention Map", size=25)

    img_tensor, _, orig_size, img_name = load_img(draw_args=draw_args)
    img_tensor = img_tensor.to(draw_args.device)

    inf_output = inf_model(img_tensor)
    # TODO: draw bonding box on image
    draw_box_and_cat(output=inf_output, orig_size=orig_size, ax=axes[1])

    attn_output = attn_model(img_tensor)
    bf_pool, af_pool = attn_output["before"], attn_output["after"]

    # # TODO: draw attention heatmap with image
    draw_attn_mask(bf_pool, original_size=orig_size, ax=axes[2])

    fig.tight_layout()  # 调整整体空白
    plt.subplots_adjust(wspace=.1)  # 调整子图间距
    if save_path is not None:
        plt.savefig(save_path + "/" + img_name)
    plt.show()


def draw_pos_det(attn_model: nn.Module,
                 draw_args: argparse.Namespace,
                 save_path: str = None):
    img_name = load_img(draw_args=draw_args)[-1]

    fig = plt.figure(1, figsize=(20, 10))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(223)
    ax3 = plt.subplot(122)

    det_tokens = attn_model.backbone.det_token
    pos_embed = attn_model.backbone.pos_embed

    det_tokens = det_tokens.squeeze(0)

    det_tokens = rearrange(det_tokens, "l d -> d l")

    det_maxes, _det_idx = torch.max(det_tokens, dim=0)
    det_maxes = det_maxes.unsqueeze(0)
    det_maxes = det_maxes.cpu().detach().numpy()
    # Draw a heatmap with the numeric values in each cell
    ax1.set_title("det_token", size=25)
    sns.heatmap(data=det_maxes, square=False, cmap="YlOrRd", annot=False, ax=ax1, cbar=False)

    det_pos = pos_embed[:, -100:, :].squeeze(0)
    det_pos = rearrange(det_pos, "l d -> d l")
    det_pos_max, _det_pos_idx = torch.max(det_pos, dim=0)
    det_pos_max = det_pos_max.unsqueeze(0)
    det_pos_max = det_pos_max.cpu().detach().numpy()
    ax2.set_title("Positional Encoding of det_token", size=25)
    sns.heatmap(data=det_pos_max, square=False, cmap="YlGn", annot=False, ax=ax2, cbar=False)

    patch_pos = pos_embed[:, 1:-100, :].squeeze(0)
    patch_pos = rearrange(patch_pos, "(e1 e2) d -> d e1 e2", e1=30, e2=30)
    patch_maxes, _patch_idx = torch.max(patch_pos, dim=0)
    patch_maxes = patch_maxes.cpu().detach().numpy()
    ax3.set_title("Positional Encoding of patches", size=25)
    sns.heatmap(data=patch_maxes, square=True, cmap="YlGn", ax=ax3)

    fig.tight_layout()  # 调整整体空白
    plt.subplots_adjust(wspace=.2, hspace=0.3)  # 调整子图间距
    if save_path is not None:
        plt.savefig(save_path + "/" + img_name)
    plt.show()


def draw_single_img(inf_model: nn.Module,
                    attn_model: nn.Module,
                    draw_args: argparse.Namespace):
    # FIXME: 模型可视化
    draw_pos_det(attn_model=attn_model, draw_args=draw_args)
    # FIXME: 画图
    draw_predictions(inf_model=inf_model, attn_model=attn_model, draw_args=draw_args)


def draw_multi_images(draw_args: argparse.Namespace,
                      inf_model: nn.Module, attn_model: nn.Module,
                      img_root: str = "/Users/iyoc/ProjectFiles/PythonProjects/coco",
                      img_register: dict = img_choice,
                      save_path: str = None):

    # FIXME: 模型可视化
    draw_pos_det(attn_model=attn_model, draw_args=draw_args, save_path=save_path)

    # FIXME: 画图
    for file_name in img_register["train2017"]:
        draw_args.image_path = img_root + "/train2017/" + file_name
        draw_predictions(inf_model=inf_model, attn_model=attn_model, draw_args=draw_args, save_path=save_path)
    for file_name in img_register["val2017"]:
        draw_args.image_path = img_root + "/val2017/" + file_name
        draw_predictions(inf_model=inf_model, attn_model=attn_model, draw_args=draw_args, save_path=save_path)


def main(draw_args):
    # voc2012 - 2 sheeps
    # draw_args.image_path = "/Users/iyoc/ProjectFiles/ModelComparison/SSD/VOCdevkit/VOC2012/JPEGImages/2007_000175.jpg"
    # voc2012 - 1 sheep
    # draw_args.image_path = "/Users/iyoc/ProjectFiles/ModelComparison/SSD/VOCdevkit/VOC2012/JPEGImages/2007_000676.jpg"

    draw_args.return_attn = False
    inf_model, inf_postprocessor = load_model(draw_args=draw_args)
    inf_model = inf_model.to(draw_args.device)
    inf_model.eval()

    draw_args.return_attn = True
    attn_model, _ = load_model(draw_args=draw_args)
    attn_model = attn_model.to(draw_args.device)
    attn_model.eval()

    draw_args.multi_draw = True

    if draw_args.multi_draw:
        draw_args.img_root = "/Users/iyoc/ProjectFiles/PythonProjects/coco"
        draw_multi_images(draw_args=draw_args, inf_model=inf_model, attn_model=attn_model,
                          img_root=draw_args.img_root, save_path=None)
    else:
        draw_single_img(inf_model=inf_model, attn_model=attn_model, draw_args=draw_args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Draw info during detecting.", add_help=False)
    parser.add_argument("--model_scale", default="small", type=str,
                        help="The scale supported for T-SSD.")
    parser.add_argument("--weight_root", default="./", type=str,
                        help="Root path of pre-train weight.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Computing device.")
    parser.add_argument("--image_path", default="./example-dog.jpg", type=str,
                        help="Root path of image to be inferences.")
    parser.add_argument("--warp_size", default=672, type=int,
                        help="The input size you want.")
    parser.add_argument("--return_attn", default=False, type=bool,
                        help="Whether return the attention maps.")
    args = parser.parse_args()
    print(f"Computing Device: {args.device}")

    # local weights
    if args.model_scale == "tiny":
        args.weight_root = "/Users/iyoc/ProjectFiles/ModelComparison/TransSSD/T-SSD-Tiny/tiny_outputs-ep201/checkpoint.pth"
        # args.weight_root = "/home/yoc/codes/TSSD-Ti.pth"
    elif args.model_scale == "small":
        args.weight_root = "/Users/iyoc/ProjectFiles/ModelComparison/TransSSD/T-SSD-Small/small_outputs/checkpoint.pth"
        # args.weight_root = "/home/yoc/codes/TSSD-S.pth"
    else:
        raise ValueError(f"No {args.model_scale} scale!")

    main(args)
