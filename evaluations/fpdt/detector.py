# ---*--- File Information ---*---
# @File       :  detector.py
# @Date&Time  :  2022-05-10, 13:18:08
# @Project    :  ctd
# @Device     :  Apple M1 Max
# @Software   :  PyCharm
# @Author     :  yoc
# @Email      :  yyyyyoc@hotmail.com


import torch
import torch.nn as nn
import torch.nn.functional as F

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from models.matcher import build_matcher

from models.transformer import deit_tiny_patch16_224, deit_small_patch16_224, deit_base_patch16_224
# from models.encoder import encoder_tiny_672_192, encoder_small_672_384, encoder_base_768_768
from models.pyramid import pyramid_tiny, pyramid_small, pyramid_base

from functools import partial


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class Detector(nn.Module):
    def __init__(self, backbone: nn.Module, encoder: nn.Module, token_len=600, sample_name="tiny",
                 det_token_num=100, num_classes=91):
        super().__init__()

        self.backbone = backbone
        self.backbone.get_encoder(encoder)
        self.backbone.init_tokens(len_tokens=token_len, num_det=det_token_num)

        self.class_embed = MLP(self.backbone.embed_dim, self.backbone.embed_dim, num_classes + 1, 3)
        self.bbox_embed = MLP(self.backbone.embed_dim, self.backbone.embed_dim, 4, 3)

    def forward(self, samples: NestedTensor):

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        x = self.backbone(samples.tensors)
        # print("after pyramid: {}".format(x.size()))

        outputs_class = self.class_embed(x)
        outputs_coord = self.bbox_embed(x).sigmoid()
        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        print(f"Heads: {get_parameter_number(self.class_embed), get_parameter_number(self.bbox_embed)}")
        return out

    def forward_return_attention(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        attention = self.backbone(samples.tensors, return_attention=True)
        return attention


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


def tfpd_tiny(encoder_pt=None, deit_pt=None, det_token_num=100, neck_pt=None, num_classes=91):
    # encoder = encoder_tiny_672_192(pretrain=encoder_pt)
    encoder = pyramid_tiny(pretrained=encoder_pt)
    neck = deit_tiny_patch16_224(pretrained=deit_pt)
    detector = Detector(encoder=encoder, backbone=neck,
                        # token_len=sum(encoder.num_list),  # TODO token_len
                        token_len=510,
                        det_token_num=det_token_num, num_classes=num_classes, sample_name="tiny")

    if neck_pt:
        checkpoint = torch.load(neck_pt, map_location="cpu")
        cls_pos_weight = checkpoint["model"]['backbone.pos_embed'][:, 0, :].unsqueeze(0)
        det_pos_weight = checkpoint["model"]['backbone.pos_embed'][:, -det_token_num:, :]
        patch_pos_weight = checkpoint["model"]['backbone.pos_embed'][:, 1:-det_token_num, :]
        patch_pos_weight = F.interpolate(patch_pos_weight.transpose(1, 2),
                                         # size=sum(encoder.num_list),  # TODO token_len
                                         size=510,
                                         mode='linear',
                                         align_corners=False).transpose(1, 2)
        checkpoint["model"]['backbone.pos_embed'] = torch.cat((cls_pos_weight, patch_pos_weight, det_pos_weight), dim=1)
        detector.load_state_dict(state_dict=checkpoint["model"], strict=False)
        print("tiny neck weights loaded!")
    print("TFPD-Tiny Successfully Built")

    return detector


def tfpd_small(encoder_pt=None, deit_pt=None, det_token_num=100, neck_pt=None, num_classes=91):
    # encoder = encoder_small_672_384(pretrain=encoder_pt)
    encoder = pyramid_small(pretrained=encoder_pt)
    neck = deit_small_patch16_224(pretrained=deit_pt)
    detector = Detector(encoder=encoder, backbone=neck,
                        # token_len=sum(encoder.num_list),  # TODO token_len
                        token_len=510,
                        det_token_num=det_token_num, num_classes=num_classes, sample_name="small")

    if neck_pt:
        checkpoint = torch.load(neck_pt, map_location="cpu")
        cls_pos_weight = checkpoint["model"]['backbone.pos_embed'][:, 0, :].unsqueeze(0)
        det_pos_weight = checkpoint["model"]['backbone.pos_embed'][:, -det_token_num:, :]
        patch_pos_weight = checkpoint["model"]['backbone.pos_embed'][:, 1:-det_token_num, :]
        patch_pos_weight = F.interpolate(patch_pos_weight.transpose(1, 2),
                                         # size=sum(encoder.num_list),  # TODO token_len
                                         size=510,
                                         mode='linear',
                                         align_corners=False).transpose(1, 2)
        checkpoint["model"]['backbone.pos_embed'] = torch.cat((cls_pos_weight, patch_pos_weight, det_pos_weight), dim=1)
        detector.load_state_dict(state_dict=checkpoint["model"], strict=False)
        print("small neck weights loaded!")
    print("TFPD-Small Successfully Built")

    return detector


def tfpd_base(encoder_pt=None, deit_pt=None, det_token_num=100, neck_pt=None, num_classes=91):
    # encoder = encoder_base_768_768(pretrain=encoder_pt)
    encoder = pyramid_base(pretrained=encoder_pt)
    neck = deit_base_patch16_224(pretrained=deit_pt)
    print(f"base neck: {get_parameter_number(neck)}")
    detector = Detector(encoder=encoder, backbone=neck,
                        # token_len=sum(encoder.num_list),
                        token_len=510,
                        det_token_num=det_token_num, num_classes=num_classes, sample_name="base")

    if neck_pt:
        checkpoint = torch.load(neck_pt, map_location="cpu")
        cls_pos_weight = checkpoint["model"]['backbone.pos_embed'][:, 0, :].unsqueeze(0)
        det_pos_weight = checkpoint["model"]['backbone.pos_embed'][:, -det_token_num:, :]
        patch_pos_weight = checkpoint["model"]['backbone.pos_embed'][:, 1:-det_token_num, :]
        patch_pos_weight = F.interpolate(patch_pos_weight.transpose(1, 2),
                                         # size=sum(encoder.num_list),
                                         size=510,
                                         mode='linear',
                                         align_corners=False).transpose(1, 2)
        checkpoint["model"]['backbone.pos_embed'] = torch.cat((cls_pos_weight, patch_pos_weight, det_pos_weight), dim=1)
        detector.load_state_dict(state_dict=checkpoint["model"], strict=False)
        print("base neck weights loaded!")
    print("SwinSSD-Base Successfully Built")

    return detector


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    print("\n* ---------- Model Information ---------- *")

    if args.model_name == "tiny":
        model = tfpd_tiny(encoder_pt=args.encoder_pt, deit_pt=None, det_token_num=args.det_token_num,
                          neck_pt=args.neck_pt, num_classes=num_classes)
    elif args.model_name == 'small':
        model = tfpd_small(encoder_pt=args.encoder_pt, deit_pt=None, det_token_num=args.det_token_num,
                           neck_pt=args.neck_pt, num_classes=num_classes)
    elif args.model_name == "base":
        model = tfpd_base(encoder_pt=args.encoder_pt, deit_pt=None, det_token_num=args.det_token_num,
                          neck_pt=args.neck_pt, num_classes=num_classes)
    else:
        raise ValueError(f"{args.model_name} does not exist!")

    print(
        "\tParameters Total: {:.2f}M ({:.2f}M trainable)!".format(
            get_parameter_number(model)['Total'] / 1_000_000,
            get_parameter_number(model)['Trainable'] / 1_000_000)
    )
    print("* --------------------------------------- *\n")
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.requires_grad}")

    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    # TODO this is a hack
    # if args.aux_loss:
    #     aux_weight_dict = {}
    #     for i in range(args.dec_layers - 1):
    #         aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
    #     weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def build_fpdt(model_name="tiny"):

    class Param:
        def __init__(self, m_name=model_name):
            # self.pre_trained = "/Users/iyoc/ProjectFiles/DeiTWeights/deit_tiny_patch16_224-a1311bcf.pth"
            self.dropout = 0.1
            self.device = "cpu"
            self.dataset_file = "coco"
            # self.model_name = "tiny"
            # self.model_name = "small"
            self.model_name = m_name
            self.det_token_num = 100
            self.set_cost_class = 1
            self.set_cost_bbox = 5
            self.set_cost_giou = 2
            self.bbox_loss_coef = 5
            self.giou_loss_coef = 2
            self.bbox_loss_coef = 5
            self.eos_coef = 0.1
            self.aux_loss = False
            # self.batch_size = 2
            if self.model_name == "tiny":
                # self.encoder_pt = "/home/ubuntu/allusers/z1t/yoc/convnext_tiny_22k_1k_384.pth"
                # self.neck_pt = "/home/ubuntu/allusers/z1t/yoc/yolos_ti.pth"
                # self.encoder_pt = "/Users/iyoc/ProjectFiles/SwinWeights/swin_tiny_patch4_window7_224_22kto1k_finetune.pth"
                # self.neck_pt = "/Users/iyoc/ProjectFiles/YolosWeights/yolos_ti.pth"
                # self.encoder_pt = "/Users/iyoc/ProjectFiles/ConvNeXtWeights/convnext_tiny_22k_1k_384.pth"
                self.neck_pt = None
                self.encoder_pt = None
            elif self.model_name == "small":
                # self.neck_pt = "/Users/iyoc/ProjectFiles/YolosWeights/yolos_s_300_pre.pth"
                # self.encoder_pt = "/Users/iyoc/ProjectFiles/SwinWeights/swin_small_patch4_window7_224_22kto1k_finetune.pth"
                # self.neck_pt = "/mnt/DataDisk/yoc/yolos_s_300_pre.pth"
                # self.encoder_pt = "/mnt/DataDisk/yoc/convnext_small_22k_1k_384.pth"
                # self.neck_pt = "/Users/iyoc/ProjectFiles/YolosWeights/yolos_s_300_pre.pth"
                # self.encoder_pt = "/Users/iyoc/ProjectFiles/ConvNeXtWeights/convnext_small_22k_1k_384.pth"
                self.neck_pt = None
                self.encoder_pt = None
            elif self.model_name == "base":
                self.neck_pt = "/Users/iyoc/ProjectFiles/YolosWeights/yolos_base.pth"
                self.encoder_pt = "/Users/iyoc/ProjectFiles/ConvNeXtWeights/convnext_base_22k_1k_384.pth"
                # self.encoder_pt = "/Users/iyoc/ProjectFiles/SwinWeights/swin_base_patch4_window12_384_22kto1k.pth"

    num_classes = 91

    args = Param()

    model, c_, p_ = build(args)

    return model


if __name__ == "__main__":
    """
    这里的 `num_classes` 有些误导性。
    它其实对应于 `max_obj_id + 1`，其中 `max_obj_id` 是数据集中的最大类别索引id。
    例如，COCO数据集的 `max_obj_id`是90，所以我们传递给 `mun_classes` 的值是91。
    至于另一个例子，如果某一数据集只有单一的类别索引id为1，那么应给 `num_classes` 传递值为2（max_obj_id + 1）
    更多如下细节请查看讨论：
        https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223

    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    """


    class param:
        def __init__(self):
            # self.pre_trained = "/Users/iyoc/ProjectFiles/DeiTWeights/deit_tiny_patch16_224-a1311bcf.pth"
            self.dropout = 0.1
            self.device = "cpu"
            self.dataset_file = "coco"
            # self.model_name = "tiny"
            self.model_name = "small"
            # self.model_name = "base"
            self.det_token_num = 100
            self.set_cost_class = 1
            self.set_cost_bbox = 5
            self.set_cost_giou = 2
            self.bbox_loss_coef = 5
            self.giou_loss_coef = 2
            self.bbox_loss_coef = 5
            self.eos_coef = 0.1
            self.aux_loss = False
            self.batch_size = 2
            if self.model_name == "tiny":
                # self.encoder_pt = "/home/ubuntu/allusers/z1t/yoc/convnext_tiny_22k_1k_384.pth"
                # self.neck_pt = "/home/ubuntu/allusers/z1t/yoc/yolos_ti.pth"
                # self.encoder_pt = "/Users/iyoc/ProjectFiles/SwinWeights/swin_tiny_patch4_window7_224_22kto1k_finetune.pth"
                # self.neck_pt = "/Users/iyoc/ProjectFiles/YolosWeights/yolos_ti.pth"
                # self.encoder_pt = "/Users/iyoc/ProjectFiles/ConvNeXtWeights/convnext_tiny_22k_1k_384.pth"
                self.neck_pt = None
                self.encoder_pt = None
            elif self.model_name == "small":
                # self.neck_pt = "/Users/iyoc/ProjectFiles/YolosWeights/yolos_s_300_pre.pth"
                # self.encoder_pt = "/Users/iyoc/ProjectFiles/SwinWeights/swin_small_patch4_window7_224_22kto1k_finetune.pth"
                # self.neck_pt = "/mnt/DataDisk/yoc/yolos_s_300_pre.pth"
                # self.encoder_pt = "/mnt/DataDisk/yoc/convnext_small_22k_1k_384.pth"
                # self.neck_pt = "/Users/iyoc/ProjectFiles/YolosWeights/yolos_s_300_pre.pth"
                # self.encoder_pt = "/Users/iyoc/ProjectFiles/ConvNeXtWeights/convnext_small_22k_1k_384.pth"
                self.neck_pt = None
                self.encoder_pt = None
            elif self.model_name == "base":
                self.neck_pt = "/Users/iyoc/ProjectFiles/YolosWeights/yolos_base.pth"
                self.encoder_pt = "/Users/iyoc/ProjectFiles/ConvNeXtWeights/convnext_base_22k_1k_384.pth"
                # self.encoder_pt = "/Users/iyoc/ProjectFiles/SwinWeights/swin_base_patch4_window12_384_22kto1k.pth"

            # weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
            # weight_dict['loss_giou'] = args.giou_loss_coef
            # weight_dict = {'loss_ce': 1, 'loss_bbox': 5}
            # weight_dict['loss_giou'] = 2


    class Samples:
        def __init__(self, tensor):
            self.tensors = tensor


    import torchvision.transforms as transforms
    import cv2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "mps" if torch.backends.mps.is_available() else "cpu"
    # device = "mps" if (getattr(torch.backends, "mps",
    #                            None) is not None) and torch.backends.mps.is_available() else "cpu"
    # device = "cpu"
    print(device)
    torch.cuda.empty_cache()

    # 测试
    img = cv2.imread('../train_image.jpg')
    # INPUT-672
    trans672 = transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize((672, 672))])  # (480, 640, 3) -> (3, 224, 224)
    input672 = trans672(img)  # tensor数据格式是torch(C,H,W)
    input672 = input672.unsqueeze(0)
    input672_b = input672
    input672 = torch.cat((input672, input672_b), dim=0)
    input672 = input672.to(device)
    imgs672 = Samples(input672)
    # INPUT-768
    trans768 = transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize((768, 768))])  # (480, 640, 3) -> (3, 224, 224)
    input768 = trans768(img)  # tensor数据格式是torch(C,H,W)
    input768 = input768.unsqueeze(0)
    input768_b = input768
    input768 = torch.cat((input768, input768_b), dim=0)
    input768 = input768.to(device)
    imgs768 = Samples(input768)

    # INPUT-640
    trans640 = transforms.Compose([transforms.ToTensor(),
                                   transforms.Resize((640, 640))])  # (480, 640, 3) -> (3, 224, 224)
    input640 = trans640(img)  # tensor数据格式是torch(C,H,W)
    input640 = input640.unsqueeze(0)
    input640_b = input640
    input640 = torch.cat((input640, input640_b), dim=0)
    input640 = input640.to(device)

    import time
    starting = time.time()

    arg = param()
    model, criterion, postprocessor = build(args=arg)
    model = model.to(device)
    criterion = criterion.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print(get_parameter_number(model))

    targets = [
        {
            "boxes": torch.tensor([
                [0.3942, 0.9196, 0.1344, 0.1604],
                [0.8893, 0.4365, 0.1069, 0.3341],
                [0.4273, 0.4169, 0.1050, 0.3663],
                [0.5437, 0.2772, 0.0999, 0.2787],
                [0.6574, 0.2706, 0.1333, 0.3888],
                [0.3846, 0.5219, 0.3462, 0.5660],
                [0.9578, 0.3083, 0.0844, 0.3477],
                [0.7949, 0.2354, 0.0786, 0.3156],
                [0.3868, 0.1809, 0.1230, 0.3528],
                [0.1376, 0.2884, 0.0946, 0.3424],
                [0.4834, 0.1767, 0.0900, 0.3132],
                [0.7449, 0.3355, 0.0861, 0.1863],
                [0.8696, 0.3026, 0.1189, 0.3489],
                [0.2315, 0.2468, 0.1145, 0.3314]
            ], dtype=torch.float32, device=device),
            "labels": torch.tensor([41, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   dtype=torch.int64,
                                   device=device)
        },
        {
            "boxes": torch.tensor([
                [0.3942, 0.9196, 0.1344, 0.1604],
                [0.8893, 0.4365, 0.1069, 0.3341],
                [0.4273, 0.4169, 0.1050, 0.3663],
                [0.5437, 0.2772, 0.0999, 0.2787],
                [0.6574, 0.2706, 0.1333, 0.3888],
                [0.3846, 0.5219, 0.3462, 0.5660],
                [0.9578, 0.3083, 0.0844, 0.3477],
                [0.7949, 0.2354, 0.0786, 0.3156],
                [0.3868, 0.1809, 0.1230, 0.3528],
                [0.1376, 0.2884, 0.0946, 0.3424],
                [0.4834, 0.1767, 0.0900, 0.3132],
                [0.7449, 0.3355, 0.0861, 0.1863],
                [0.8696, 0.3026, 0.1189, 0.3489],
                [0.2315, 0.2468, 0.1145, 0.3314]
            ], dtype=torch.float32, device=device),
            "labels": torch.tensor([41, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   dtype=torch.int64,
                                   device=device)
        }
    ]

    if arg.model_name == "tiny" or arg.model_name == "small":
        # outputs = model(imgs672)
        outputs = model(input640)

    elif arg.model_name == "base":
        # outputs = model(imgs768)
        outputs = model(input640)

    for k, v in outputs.items():
        print(f"{k}: {v.size()}")
    loss_dict = criterion(outputs, targets)  # 计算损失字典
    weight_dict = criterion.weight_dict
    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)  # 计算总体损失
    print(f"total loss: {losses}")
    optim.zero_grad()
    print(f"start back-prop...")
    losses.backward()
    print(f"back-prop successfully finished!")

    ending = time.time()

    print("Total time: {:.2f}".format(ending - starting))

    for name, param in model.named_parameters():
        if param.grad is None:
            print(name)
