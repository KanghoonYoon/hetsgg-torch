
import torch
from torch import nn

from hetsgg.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):


    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.cfg = cfg.clone()
        self.backbone = build_backbone(cfg) # ResNet
        self.rpn = build_rpn(cfg, self.backbone.out_channels) # 256
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None, logger=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets, logger)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training: # True
            losses = {}
            losses.update(detector_losses)
            if not self.cfg.MODEL.RELATION_ON: # True
                # During the relationship training stage, the rpn_head should be fixed, and no loss. 
                losses.update(proposal_losses)
            return losses

        return result
