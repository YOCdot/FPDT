# ---*--- File Information ---*---
# @File       :  transformer.py
# @Date&Time  :  2022-06-29, 18:33:47
# @Project    :  ctd
# @Device     :  Apple M1 Max
# @Software   :  PyCharm
# @Author     :  yoc
# @Email      :  yyyyyoc@hotmail.com


import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from typing import List

# from models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# import torch.utils.checkpoint as checkpoint
# from timm.models.vision_transformer import VisionTransformer as ViT


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, keep_dim=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.keep_shape = keep_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # TODO: relaxed size constraints
        # B, C, H, W = x.shape
        # assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        # assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."

        x = self.proj(x)
        if self.keep_shape:
            return x  # (b, d, h, w)
        else:
            x = x.flatten(2).transpose(1, 2)  # BDHW -> BPD
            return x


class ConvTention(nn.Module):

    def __init__(self, embed_dim=128, unfold_kernel_size=3, unfold_stride=1, num_heads=1, tensor_size=40,
                 norm_eps=None, dropout=0., unfold_dilation=1, unfold_padding=0):
        super().__init__()

        self.embed_dim = embed_dim
        self.unfold_kernel_size = unfold_kernel_size
        self.unfold_stride = unfold_stride
        self.num_heads = num_heads
        self.in_size = tensor_size
        self.win_patches = unfold_kernel_size ** 2

        self.out_size = int(
            (self.in_size + 2 * unfold_padding - unfold_dilation * (unfold_kernel_size - 1) - 1) / unfold_stride + 1
        )

        """
        takes in (b, dim*p_win, total_wins), unfold: (b, dim, h, w) -> (b, dim*p_win, total_wins)
        dim*p_win = self.embed_dim * (self.unfold_kernel_size**2)
        total_wins = int((H - (self.unfold_kernel_size - 1) - 1) / self.unfold_stride + 1)
                        *
                     int((w - (self.unfold_kernel_size - 1) - 1) / self.unfold_stride + 1)
        num_pos = kernel_size ** 2 + 1
        """
        # total_wins = self.out_size ** 2
        # num_pos = unfold_kernel_size ** 2 + 1
        self.unfold = nn.Unfold(kernel_size=unfold_kernel_size, stride=unfold_stride,
                                dilation=unfold_dilation, padding=unfold_padding)
        # 没有改变维度，所以是 self.in_size
        self.fold = nn.Fold(output_size=self.in_size, kernel_size=unfold_kernel_size, stride=unfold_stride,
                            dilation=unfold_dilation,
                            padding=unfold_padding)  # size & stride as 1 because of win_fet

        # TODO: lots of alternatives, such as linformer/performer
        # self.attn = SelfAttention(dim=embed_dim, heads=self.num_heads, dropout=dropout)
        self.attn = Attention(dim=embed_dim, num_heads=self.num_heads, attn_drop=dropout, proj_drop=dropout)

        self.pre_norm = nn.LayerNorm(self.embed_dim, eps=norm_eps)
        self.post_norm = nn.LayerNorm(self.embed_dim, eps=norm_eps)

    def forward(self, x):
        """
        Input shape: [batch_size, embed_dim, h, w]
        Output shape: [Batch_size, embed_dim(or scale_val), h_out, w_out]
        """

        B, D, H, W = x.shape
        # batch_size = x.shape[0]

        # unfold
        x = self.unfold(x)  # (b, dim, h, w) -> (b, dim*len_win, num_wins)

        x = rearrange(x, "b (dim win) num_wins -> (b num_wins) win dim", dim=self.embed_dim)

        x = self.pre_norm(x)
        x = self.attn(x)
        x = self.post_norm(x)
        x = rearrange(x, "(b num_wins) win dim -> b (dim win) num_wins", b=B)  # (b, num_wins, win, dim)
        x = self.fold(x)  # (batch, embed_dim*win_size, num_wins) -> (b, dim, h, w)

        return x


class EncodingModule(nn.Module):
    def __init__(self, initial_dim=3, img_size=640, patch_size=16, patched_dim=384, post_drop=0.0, norm_eps=None,
                 kernel_list=[4, 6, 6, 7, 8], stride_list=[1, 1, 1, 1, 1], head_list=[6, 6, 6, 6, 6],
                 size_reduce=False):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size

        self.patched_size = img_size // patch_size
        self.size_reduce = size_reduce

        ct_size = self.patched_size
        ct_list = torch.jit.annotate(List[nn.Module], [])
        proj_list = torch.jit.annotate(List[nn.Module], [])
        for idx, (kernel_size, stride, heads) in enumerate(zip(kernel_list, stride_list, head_list)):
            a_ct = ConvTention(embed_dim=patched_dim, unfold_kernel_size=kernel_size,
                               unfold_stride=stride,
                               num_heads=heads, tensor_size=ct_size,
                               norm_eps=norm_eps,
                               # size_reduce=size_reduce
                               )
            ct_list.append(a_ct)

            if self.size_reduce:
                a_proj = nn.Linear(in_features=ct_size ** 2, out_features=a_ct.out_size ** 2)
                proj_list.append(a_proj)
                ct_size = a_ct.out_size  # FIXME: 每次的输出都相同，没有改变整体的 embedding 长度

        self.ct = nn.ModuleList(ct_list)
        self.proj = nn.ModuleList(proj_list)
        self.dropout = nn.Dropout(p=post_drop)

        self.out_size = self.ct[-1].out_size if self.size_reduce else self.ct[-1].in_size
        self.out_dim = self.ct[-1].embed_dim
        self.out_heads = self.ct[-1].num_heads

        self.num_patches = int(self.ct[-1].out_size ** 2)

    def forward(self, x):

        if self.size_reduce:
            for layer, proj in zip(self.ct, self.proj):
                x = self.dropout(x)
                x = layer(x)
                x = rearrange(x, "b d h w -> b d (h w)")
                x = proj(x)
                x = self.dropout(x)
                x = rearrange(x, "b d (h w) -> b d h w", h=layer.out_size, w=layer.out_size)
        else:
            for layer in self.ct:
                x = layer(x)
                x = self.dropout(x)
        x = x.flatten(2).transpose(1, 2)  # (b, d, h, w) -> (b, h*w, d)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer & DeiT"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,  # embed_layer=PatchEmbed,
                 name="tiny",
                 norm_layer=None, act_layer=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            # weight_init: (str): weight init scheme
        """
        super().__init__()

        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_size = patch_size

        # self.patch_embed = embed_layer(
        #     img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
        #     keep_dim=True  # TODO: 保持形状
        # )
        # num_patches = self.patch_embed.num_patches

        if name == "tiny":
            num_patches = 600
        elif name == "small":
            num_patches = 800
        elif name == "base":
            num_patches = 1000
        else:
            raise ValueError(f"{name} not exists!")

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # TODO: Classifier head(s)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # self.head_dist = None
        # if distilled:
        #     self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        if distilled:
            trunc_normal_(self.dist_token, std=.02)

        # self.name = name
        # if name == "tiny":
        #     self.down_sample_1 = nn.Linear(in_features=384, out_features=192)
        #     self.down_sample_2 = nn.Linear(in_features=768, out_features=192)
        #     self.down_sample_3 = nn.Linear(in_features=768, out_features=192)
        # elif name == "small":
        #     self.up_sample = nn.Linear(in_features=192, out_features=384)
        #     self.down_sample_1 = nn.Linear(in_features=768, out_features=384)
        #     self.down_sample_2 = nn.Linear(in_features=768, out_features=384)
        #
        # elif name == "base":
        #     self.up_sample_1 = nn.Linear(in_features=256, out_features=768)
        #     self.up_sample_2 = nn.Linear(in_features=512, out_features=768)
        #     self.down_sample_1 = nn.Linear(in_features=1024, out_features=768)
        #     self.down_sample_2 = nn.Linear(in_features=1024, out_features=768)
        # else:
        #     raise ValueError(f"No {name} sampling!")

        self.apply(self._init_weights)

    def get_encoder(self, encoder: nn.Module):
        self.encoder = encoder

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def init_tokens(self, len_tokens, num_det):
        """TODO: 将所有旧的 pos 转化为新大小: 224 -> 640"""

        # 初始化 det_token
        self.det_token_num = num_det
        self.det_token = trunc_normal_(nn.Parameter(torch.zeros(1, num_det, self.embed_dim)), std=0.2)
        det_pos = trunc_normal_(nn.Parameter(torch.zeros(1, num_det, self.embed_dim)), std=0.2)

        cls_pos = self.pos_embed[:, 0, :]  # 提取出 cls 的 pos_embed
        cls_pos = cls_pos[:, None]  # (b, dim) -> (b, 1, dim)

        patch_pos = self.pos_embed[:, 1:, :]  # 提取出 patch 的 pos_embed
        patch_pos = patch_pos.transpose(1, 2)  # (b, p, d) -> (b, d, p)
        patch_pos = F.interpolate(patch_pos,
                                  size=len_tokens,
                                  mode='linear',
                                  align_corners=False)
        patch_pos = patch_pos.transpose(1, 2)
        # (b, 1 + patch + det_token_num, dim)
        self.pos_embed = torch.nn.Parameter(torch.cat((cls_pos, patch_pos, det_pos), dim=1))

    def pos_interpolation(self, pos_embed, actual_len=600):

        cls_pos = pos_embed[:, 0, :]
        cls_pos = cls_pos[:, None]  # (b, dim) -> (b, 1, dim)
        det_pos = pos_embed[:, -self.det_token_num:, :]

        patch_pos = pos_embed[:, 1:-self.det_token_num, :]
        patch_pos = patch_pos.transpose(1, 2)
        patch_pos = F.interpolate(patch_pos,
                                  size=actual_len,
                                  mode='linear',
                                  align_corners=False)
        patch_pos = patch_pos.transpose(1, 2)
        pos_embed = torch.cat((cls_pos, patch_pos, det_pos), dim=1)

        return pos_embed

    def forward_features(self, x):
        batch_size, input_img_size = x.shape[0], (x.shape[2], x.shape[3])

        # out_list = self.encoder(x)
        # if self.name == "tiny":
        #     token_1 = out_list[0]
        #     token_2 = self.down_sample_1(out_list[1])
        #     token_3 = self.down_sample_2(out_list[2])
        #     token_4 = self.down_sample_3(out_list[3])
        #     x = torch.cat((token_1, token_2, token_3, token_4), dim=1)  # B, L, D
        # elif self.name == "small":
        #     token_1 = self.up_sample(out_list[0])
        #     token_2 = out_list[1]
        #     token_3 = self.down_sample_1(out_list[2])
        #     token_4 = self.down_sample_2(out_list[3])
        #     x = torch.cat((token_1, token_2, token_3, token_4), dim=1)  # B, L, D
        # elif self.name == "base":
        #     token_1 = self.up_sample_1(out_list[0])
        #     token_2 = self.up_sample_2(out_list[1])
        #     token_3 = self.down_sample_1(out_list[2])
        #     token_4 = self.down_sample_2(out_list[3])
        #     x = torch.cat((token_1, token_2, token_3, token_4), dim=1)  # B, L, D
        # else:
        #     raise ValueError("Wrong sampling!")

        x = self.encoder(x)  # B, D, L
        x = x.transpose(1, 2)  # B, D, L

        # 检查 self.patch_pos 的 h, w 是否和传入相同
        if (self.pos_embed.shape[1] - 1 - self.det_token_num) != (x.shape[1]):
            pos = self.pos_interpolation(pos_embed=self.pos_embed, actual_len=x.shape[1])
            print("warning: interpolate!")
        else:
            pos = self.pos_embed

        cls = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        det = self.det_token.expand(batch_size, -1, -1)

        if self.dist_token is None:
            # x = torch.cat((cls_token, x), dim=1)
            x = torch.cat((cls, x, det), dim=1)
        else:
            x = torch.cat((cls, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + pos)
        x = self.blocks(x)
        x = self.norm(x)

        if self.dist_token is None:
            # return x[:, 0]
            return x[:, -self.det_token_num:, :]
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        return x


def deit_tiny_patch16_224(pretrained=None, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), name="tiny", **kwargs)
    # kernel_list = [4, 6, 6, 7, 8]
    # stride_list = [1, 1, 1, 1, 1]
    # head_list = [3, 3, 4, 4, 6]
    # encoder = EncodingModule(initial_dim=3, img_size=700, patch_size=16, patched_dim=192,  # post_drop=0.1,
    #                          kernel_list=kernel_list, norm_eps=1e-6, stride_list=stride_list, head_list=head_list,
    #                          # size_reduce=True
    #                          )
    if pretrained:
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
        #     map_location="cpu", check_hash=True
        # )
        # model.load_state_dict(checkpoint["model"])
        checkpoint = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        print("\nDeiT-Tiny loaded!")
    return model  # , encoder


def deit_small_patch16_224(pretrained=None, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), name="small", **kwargs)
    # kernel_list = [4, 6, 6, 7, 8]
    # stride_list = [1, 1, 1, 1, 1]
    # head_list = [3, 4, 4, 6, 6]
    # encoder = EncodingModule(initial_dim=3, img_size=700, patch_size=16, patched_dim=384,  # post_drop=0.1,
    #                          kernel_list=kernel_list, norm_eps=1e-6, stride_list=stride_list, head_list=head_list,
    #                          # size_reduce=True
    #                          )
    if pretrained:
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
        #     map_location="cpu", check_hash=True
        # )
        # model.load_state_dict(checkpoint["model"])
        checkpoint = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        print("\nDeiT-Small loaded!")
    return model  # , encoder


def deit_base_patch16_224(pretrained=None, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), name="base", **kwargs)
    # kernel_list = [4, 6, 6, 7, 8]
    # stride_list = [1, 1, 1, 1, 1]
    # head_list = [3, 4, 6, 6, 6]
    # encoder = EncodingModule(initial_dim=3, img_size=700, patch_size=16, patched_dim=768, post_drop=0.1,
    #                          kernel_list=kernel_list, norm_eps=1e-6, stride_list=stride_list, head_list=head_list,
    #                          # size_reduce=True
    #                          )
    if pretrained:
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
        #     map_location="cpu", check_hash=True
        # )
        # model.load_state_dict(checkpoint["model"])
        checkpoint = torch.load(pretrained, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        print("\nDeiT-Base loaded!")
    return model  # , encoder


if __name__ == "__main__":
    import torchvision.transforms as transforms
    import cv2

    # TODO: deit 加载测试
    # deit = deit_small_patch16_224(pretrained="../pt_weights/deit_small_patch16_224-cd65a155.pth")

    img = cv2.imread('../train_image.jpg')
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((640, 640))])  # (480, 640, 3) -> (3, 224, 224)
    input_tensor = trans(img)  # tensor数据格式是torch(C,H,W)
    # print(input_tensor.size())
    input_tensor = input_tensor.unsqueeze(0)
    print("input:", input_tensor.size())

    # output_tensor = deit(input_tensor)
    encode = EncodingModule()
    output_tensor = encode(input_tensor)
    print(output_tensor.size())

    # img = cv2.imread('/Users/iyoc/Desktop/论文图/train_image.jpg')
    # trans = transforms.Compose([transforms.ToTensor(),
    #                             transforms.Resize((640, 640))])  # (480, 640, 3) -> (3, 224, 224)
    # input_tensor = trans(img)  # tensor数据格式是torch(C,H,W)
    # input_tensor = input_tensor.unsqueeze(0)
    # print(f"Encoding:")
    # em = EncodingModule()
    # encoded = em(input_tensor)
    # print(encoded.size())
