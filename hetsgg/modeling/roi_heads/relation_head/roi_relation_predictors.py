import torch
from math import sqrt

from torch import nn

from hetsgg.config import cfg
from hetsgg.data import get_dataset_statistics
from hetsgg.modeling import registry
from hetsgg.modeling.roi_heads.relation_head.classifier import build_classifier
from hetsgg.modeling.roi_heads.relation_head.model_kern import (
    to_onehot,
)

from hetsgg.modeling.utils import cat
from hetsgg.structures.boxlist_ops import squeeze_tensor
from .model_motifs import FrequencyBias

from hetsgg.modeling.roi_heads.relation_head.model_HetSGG import HetSGG
from hetsgg.modeling.roi_heads.relation_head.model_HetSGGplus import HetSGGplus_Context

from .rel_proposal_network.loss import (
    RelAwareLoss,
)
from .utils_relation import obj_prediction_nms



@registry.ROI_RELATION_PREDICTOR.register("HetSGG_Predictor")
class HetSGG_Predictor(nn.Module):
    def __init__(self, config, in_channels):
        super(HetSGG_Predictor, self).__init__()
        self.num_obj_cls = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES # Duplicate
        self.num_rel_cls = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES # Duplicate
        self.use_bias = cfg.MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS
        self.rel_aware_model_on = config.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON
        self.use_obj_recls_logits = config.MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS

        self.rel_aware_loss_eval = None

        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            self.mode = "predcls" if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL else "sgcls"
        else:
            self.mode = "sgdet"
            
        self.obj_recls_logits_update_manner = (
            cfg.MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_MANNER
        )

        self.n_reltypes = cfg.MODEL.ROI_RELATION_HEAD.HETSGG.NUM_RELATION
        self.n_dim = cfg.MODEL.ROI_RELATION_HEAD.HETSGG.H_DIM
        self.n_ntypes = int(sqrt(self.n_reltypes))
        self.obj2rtype = {(i, j): self.n_ntypes*j+i for j in range(self.n_ntypes) for i in range(self.n_ntypes)}

        self.rel_classifier = build_classifier(self.num_rel_cls, self.num_rel_cls) # Linear Layer
        self.obj_classifier = build_classifier(self.num_obj_cls, self.num_obj_cls)

        self.context_layer = HetSGG(config, in_channels)

        assert in_channels is not None
        if self.use_bias:
            statistics = get_dataset_statistics(config)
            self.freq_bias = FrequencyBias(config, statistics)
            self.freq_lambda = nn.Parameter(
                torch.Tensor([1.0]), requires_grad=False
            )  # hurt performance when set learnable


    def init_classifier_weight(self):
        self.obj_classifier.reset_parameters()
        for i in self.n_reltypes:
            self.rel_classifier[i].reset_parameters()


    def forward(
        self,
        inst_proposals,
        rel_pair_idxs,
        rel_labels,
        rel_binarys,
        roi_features,
        union_features,
        logger=None,
        is_training=True,
    ):
        obj_feats, rel_feats = self.context_layer(roi_features, union_features, inst_proposals, rel_pair_idxs, rel_binarys, logger, is_training) # GNN
    
        if self.mode == "predcls":
            obj_labels = cat([proposal.get_field("labels") for proposal in inst_proposals], dim=0)
            refined_obj_logits = to_onehot(obj_labels, self.num_obj_cls)
        else:
            refined_obj_logits = self.obj_classifier(obj_feats)

        rel_cls_logits = self.rel_classifier(rel_feats)

        num_objs = [len(b) for b in inst_proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)
        obj_pred_logits = cat([each_prop.get_field("predict_logits") for each_prop in inst_proposals], dim=0)

        if self.use_obj_recls_logits:
            boxes_per_cls = cat([proposal.get_field("boxes_per_cls") for proposal in inst_proposals], dim=0)  # comes from post process of box_head
            # here we use the logits refinements by adding
            if self.obj_recls_logits_update_manner == "add":
                obj_pred_logits = refined_obj_logits + obj_pred_logits
            if self.obj_recls_logits_update_manner == "replace":
                obj_pred_logits = refined_obj_logits
            refined_obj_pred_labels = obj_prediction_nms(
                boxes_per_cls, obj_pred_logits, nms_thresh=0.5
            )
            obj_pred_labels = refined_obj_pred_labels
        else:
            obj_pred_labels = cat([each_prop.get_field("pred_labels") for each_prop in inst_proposals], dim=0)
        
        if self.use_bias:
            obj_pred_labels = obj_pred_labels.split(num_objs, dim=0)
            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_pred_labels):
                pair_preds.append(torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1))
            pair_pred = cat(pair_preds, dim=0)
            rel_cls_logits = rel_cls_logits + self.freq_bias.index_with_labels(pair_pred.long())

        obj_pred_logits = obj_pred_logits.split(num_objs, dim=0)
        rel_cls_logits = rel_cls_logits.split(num_rels, dim=0)
        add_losses = {}

        return obj_pred_logits, rel_cls_logits, add_losses



def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)




@registry.ROI_RELATION_PREDICTOR.register("HetSGGplus_Predictor")
class HetSGGplus_Predictor(nn.Module):
    def __init__(self, config, in_channels):
        super(HetSGGplus_Predictor, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.FREQUENCY_BAIS

        # mode
        if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = "predcls"
            else:
                self.mode = "sgcls"
        else:
            self.mode = "sgdet"

        assert in_channels is not None
        self.pooling_dim = cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.input_dim = in_channels
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.GRAPH_HIDDEN_DIM

        self.split_context_model4inst_rel = (
            config.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.SPLIT_GRAPH4OBJ_REL
        )
        if self.split_context_model4inst_rel:
            self.obj_context_layer = HetSGGplus_Context(
                config,
                self.input_dim,
                hidden_dim=self.hidden_dim,
                num_iter=config.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.GRAPH_ITERATION_NUM,
            )
            self.rel_context_layer = HetSGGplus_Context(
                config,
                self.input_dim,
                hidden_dim=self.hidden_dim,
                num_iter=config.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.GRAPH_ITERATION_NUM,
            )
        else:
            self.context_layer = HetSGGplus_Context(
                config,
                self.input_dim,
                hidden_dim=self.hidden_dim,
                num_iter=config.MODEL.ROI_RELATION_HEAD.BGNN_MODULE.GRAPH_ITERATION_NUM,
            )

        self.rel_feature_type = config.MODEL.ROI_RELATION_HEAD.EDGE_FEATURES_REPRESENTATION

        self.use_obj_recls_logits = config.MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS
        self.obj_recls_logits_update_manner = (
            config.MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_MANNER
        )
        assert self.obj_recls_logits_update_manner in ["replace", "add"]

        # post classification
        self.rel_classifier = build_classifier(self.hidden_dim, self.num_rel_cls)
        self.obj_classifier = build_classifier(self.hidden_dim, self.num_obj_cls)

        self.rel_aware_model_on = config.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.SET_ON

        if self.rel_aware_model_on:
            self.rel_aware_loss_eval = RelAwareLoss(config)

        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        # freq
        if self.use_bias:
            statistics = get_dataset_statistics(config)
            self.freq_bias = FrequencyBias(config, statistics)
            self.freq_lambda = nn.Parameter(
                torch.Tensor([1.0]), requires_grad=False
            )  # hurt performance when set learnable

        self.init_classifier_weight()

        # for logging things
        self.forward_time = 0

    def init_classifier_weight(self):
        self.rel_classifier.reset_parameters()
        self.obj_classifier.reset_parameters()

    def start_preclser_relpn_pretrain(self):
        self.context_layer.set_pretrain_pre_clser_mode()

    def end_preclser_relpn_pretrain(self):
        self.context_layer.set_pretrain_pre_clser_mode(False)

    def forward(
        self,
        inst_proposals,
        rel_pair_idxs,
        rel_labels,
        rel_binarys,
        roi_features,
        union_features,
        logger=None,
        is_training=None
    ):
        """

        :param inst_proposals:
        :param rel_pair_idxs:
        :param rel_labels:
        :param rel_binarys:
            the box pairs with that match the ground truth [num_prp, num_prp]
        :param roi_features:
        :param union_features:
        :param logger:

        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """


        obj_feats, rel_feats, pre_cls_logits, relatedness = self.context_layer(
            roi_features, union_features, inst_proposals, rel_pair_idxs, rel_binarys, logger
        )

        if relatedness is not None:
            for idx, prop in enumerate(inst_proposals):
                prop.add_field("relness_mat", relatedness[idx])

        if self.mode == "predcls":
            obj_labels = cat(
                [proposal.get_field("labels") for proposal in inst_proposals], dim=0
            )
            refined_obj_logits = to_onehot(obj_labels, self.num_obj_cls)
        else:
            refined_obj_logits = self.obj_classifier(obj_feats)

        rel_cls_logits = self.rel_classifier(rel_feats)

        num_objs = [len(b) for b in inst_proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)
        obj_pred_logits = cat(
            [each_prop.get_field("predict_logits") for each_prop in inst_proposals], dim=0
        )

        # using the object results, update the pred label and logits
        if self.use_obj_recls_logits:
            if (self.mode == "sgdet") | (self.mode =="sgcls"):
                boxes_per_cls = cat(
                    [proposal.get_field("boxes_per_cls") for proposal in inst_proposals], dim=0
                )  # comes from post process of box_head
                # here we use the logits refinements by adding
                if self.obj_recls_logits_update_manner == "add":
                    obj_pred_logits = refined_obj_logits + obj_pred_logits
                if self.obj_recls_logits_update_manner == "replace":
                    obj_pred_logits = refined_obj_logits
                refined_obj_pred_labels = obj_prediction_nms(
                    boxes_per_cls, obj_pred_logits, nms_thresh=0.5
                )
                obj_pred_labels = refined_obj_pred_labels
            # else:
            #     _, obj_pred_labels = refined_obj_logits[:, 1:].max(-1)
        else:
            obj_pred_labels = cat(
                [each_prop.get_field("pred_labels") for each_prop in inst_proposals], dim=0
            )

        if self.use_bias:
            obj_pred_labels = obj_pred_labels.split(num_objs, dim=0)
            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_pred_labels):
                pair_preds.append(
                    torch.stack((obj_pred[pair_idx[:, 0]], obj_pred[pair_idx[:, 1]]), dim=1)
                )
            pair_pred = cat(pair_preds, dim=0)
            rel_cls_logits = (
                rel_cls_logits
                + self.freq_lambda * self.freq_bias.index_with_labels(pair_pred.long())
            )

        obj_pred_logits = obj_pred_logits.split(num_objs, dim=0)
        rel_cls_logits = rel_cls_logits.split(num_rels, dim=0)

        add_losses = {}
        ## pre clser relpn supervision
        if pre_cls_logits is not None and self.training:
            rel_labels = cat(rel_labels, dim=0)
            for iters, each_iter_logit in enumerate(pre_cls_logits):
                if len(squeeze_tensor(torch.nonzero(rel_labels != -1))) == 0:
                    loss_rel_pre_cls = None
                else:
                    loss_rel_pre_cls = self.rel_aware_loss_eval(each_iter_logit, rel_labels)

                add_losses[f"pre_rel_classify_loss_iter-{iters}"] = loss_rel_pre_cls

        return obj_pred_logits, rel_cls_logits, add_losses