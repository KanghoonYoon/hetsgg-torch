import ipdb
import numpy.random as npr
import torch

from hetsgg.modeling.utils import cat
from hetsgg.structures.boxlist_ops import boxlist_iou


class RelationSampling(object):
    def __init__(
            self,
            fg_thres,
            require_overlap,
            num_sample_per_gt_rel,
            batch_size_per_image,
            positive_fraction,
            max_proposal_pairs,
            use_gt_box,
            test_overlap,
    ):
        self.fg_thres = fg_thres
        self.require_overlap = require_overlap
        self.num_sample_per_gt_rel = num_sample_per_gt_rel
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.use_gt_box = use_gt_box
        self.max_proposal_pairs = max_proposal_pairs
        self.test_overlap = test_overlap

    def prepare_test_pairs(self, device, proposals):
        rel_pair_idxs = []
        for p in proposals:
            n = len(p)
            cand_matrix = torch.ones((n, n), device=device) - torch.eye(n, device=device)
            if (not self.use_gt_box) and self.test_overlap:
                cand_matrix = cand_matrix.byte() & boxlist_iou(p, p).gt(0).byte()
            idxs = torch.nonzero(cand_matrix).view(-1, 2)
            if len(idxs) > self.max_proposal_pairs:
                pairs_qualities = p.get_field("pred_scores")
                pairs_qualities = pairs_qualities[idxs[:, 0]] * pairs_qualities[idxs[:, 1]]
                select_idx = torch.sort(pairs_qualities, descending=True)[-1][: self.max_proposal_pairs]
                idxs = idxs[select_idx]

            if len(idxs) > 0:
                rel_pair_idxs.append(idxs)
            else:
                rel_pair_idxs.append(torch.zeros((1, 2), dtype=torch.int64, device=device))
        return rel_pair_idxs

    def gtbox_relsample(self, proposals, targets):
        assert self.use_gt_box
        num_pos_per_img = int(self.batch_size_per_image * self.positive_fraction)
        rel_idx_pairs = []
        rel_labels = []
        rel_sym_binarys = []
        for img_id, (proposal, target) in enumerate(zip(proposals, targets)):
            device = proposal.bbox.device
            num_prp = proposal.bbox.shape[0]

            assert proposal.bbox.shape[0] == target.bbox.shape[0]
            tgt_rel_matrix = target.get_field("relation")  # [tgt, tgt]
            tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
            assert tgt_pair_idxs.shape[1] == 2
            tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
            tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
            tgt_rel_labs = tgt_rel_matrix[tgt_head_idxs, tgt_tail_idxs].contiguous().view(-1)


            locating_match_stat = torch.ones((len(proposal)), device=device)
            proposal.add_field("locating_match", locating_match_stat)


            binary_rel = torch.zeros((num_prp, num_prp), device=device).long()
            binary_rel[tgt_head_idxs, tgt_tail_idxs] = 1
            binary_rel[tgt_tail_idxs, tgt_head_idxs] = 1
            rel_sym_binarys.append(binary_rel)

            rel_possibility = torch.ones((num_prp, num_prp), device=device).long() - torch.eye(num_prp,
                                                                                               device=device).long()
            rel_possibility[tgt_head_idxs, tgt_tail_idxs] = 0
            tgt_bg_idxs = torch.nonzero(rel_possibility > 0)

            if tgt_pair_idxs.shape[0] > num_pos_per_img:
                perm = torch.randperm(tgt_pair_idxs.shape[0], device=device)[:num_pos_per_img]
                tgt_pair_idxs = tgt_pair_idxs[perm]
                tgt_rel_labs = tgt_rel_labs[perm]
            num_fg = min(tgt_pair_idxs.shape[0], num_pos_per_img)

            num_bg = self.batch_size_per_image - num_fg
            perm = torch.randperm(tgt_bg_idxs.shape[0], device=device)[:num_bg]
            tgt_bg_idxs = tgt_bg_idxs[perm]

            img_rel_idxs = torch.cat((tgt_pair_idxs, tgt_bg_idxs), dim=0)
            img_rel_labels = torch.cat((tgt_rel_labs.long(), torch.zeros(tgt_bg_idxs.shape[0], device=device).long()),
                                       dim=0).contiguous().view(-1)

            rel_idx_pairs.append(img_rel_idxs)
            rel_labels.append(img_rel_labels)

        return proposals, rel_labels, rel_idx_pairs, rel_sym_binarys

    def detect_relsample(self, proposals, targets):
        # corresponding to rel_assignments function in neural-motifs
        """
        The input proposals are already processed by subsample function of box_head,
        in this function, we should only care about fg box, and sample corresponding fg/bg relations
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])  contain fields: labels, predict_logits
            targets (list[BoxList]) contain fields: labels
        """
        self.num_pos_per_img = int(self.batch_size_per_image * self.positive_fraction)
        rel_idx_pairs = []
        rel_labels = []
        rel_labels_all = []
        rel_sym_binarys = []
        for img_id, (proposal, target) in enumerate(zip(proposals, targets)):
            device = proposal.bbox.device
            prp_box = proposal.bbox
            prp_lab = proposal.get_field("labels").long()
            tgt_box = target.bbox
            tgt_lab = target.get_field("labels").long()
            tgt_rel_matrix = target.get_field("relation")  # [tgt, tgt]

            # IoU matching for object detection results
            ious = boxlist_iou(target, proposal)  # [tgt, prp]
            is_match = (tgt_lab[:, None] == prp_lab[None]) & (ious > self.fg_thres)  # [tgt, prp]
            locating_match = (ious > self.fg_thres).nonzero(as_tuple = False)  # [tgt, prp]
            locating_match_stat = torch.zeros((len(proposal)), device=device)
            locating_match_stat[locating_match[:, 1]] = 1
            proposal.add_field("locating_match", locating_match_stat)

            # Proposal self IoU to filter non-overlap
            prp_self_iou = boxlist_iou(proposal, proposal)  # [prp, prp]
            if self.require_overlap and (not self.use_gt_box): # False ,  False
                rel_possibility = (prp_self_iou > 0) & (prp_self_iou < 1)  # not self & intersect
            else:
                num_prp = prp_box.shape[0]
                rel_possibility = torch.ones((num_prp, num_prp), device=device).long() - torch.eye(num_prp, device=device).long()
            # only select relations between fg proposals
            rel_possibility[prp_lab == 0] = 0
            rel_possibility[:, prp_lab == 0] = 0

            img_rel_triplets, corrsp_gt_rel_idx, binary_rel = self.motif_rel_fg_bg_sampling(device, tgt_rel_matrix,
                                                                         ious, is_match, rel_possibility,
                                                                         proposal.get_field('pred_scores'))
            
            if target.has_field("relation_non_masked"):
                rel_map = target.get_field("relation_non_masked")
                gt_rel_idx = torch.nonzero(rel_map != 0)
                fg_gt_rel_pair_idx = gt_rel_idx[corrsp_gt_rel_idx[corrsp_gt_rel_idx >= 0]]
                bg_size = len(corrsp_gt_rel_idx) - len(torch.nonzero(corrsp_gt_rel_idx >= 0))
                fg_labels = rel_map[fg_gt_rel_pair_idx[:, 0].contiguous().view(-1),
                                    fg_gt_rel_pair_idx[:, 1].contiguous().view(-1)].long()
                bg_labels = torch.zeros((bg_size), device=device, dtype=torch.long)
                rel_labels_all.append(torch.cat((fg_labels, bg_labels), dim=0))

            rel_idx_pairs.append(img_rel_triplets[:, :2])
            rel_labels.append(img_rel_triplets[:, 2])
            rel_sym_binarys.append(binary_rel)

        if len(rel_labels_all) == 0:
            rel_labels_all = rel_labels
        
        return proposals, rel_labels, rel_labels_all, rel_idx_pairs, rel_sym_binarys

    def motif_rel_fg_bg_sampling(self, device, tgt_rel_matrix, ious, is_match, rel_possibility, proposals_quality):
        """
        prepare to sample fg relation triplet and bg relation triplet
        the motifs sampling method only sampled the relation pairs whose boxes are overlapping with the
        ground truth

        tgt_rel_matrix: # [number_target, number_target]
        ious:           # [number_target, num_proposal]
        is_match:       # [number_target, num_proposal]
        rel_possibility:# [num_proposal, num_proposal]

        return:
            the sampled relation labels [num_rel_proposal, 3]
            binary_relatedness: the box pairs with that match the ground truth
                                [num_prp, num_prp]

        """

        tgt_pair_idxs = torch.nonzero(tgt_rel_matrix != 0)

        assert tgt_pair_idxs.shape[1] == 2
        tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1) # subject
        tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1) # object
        tgt_rel_labs = tgt_rel_matrix[tgt_head_idxs, tgt_tail_idxs].contiguous().view(-1)

        num_tgt_rels = tgt_rel_labs.shape[0]
        # generate binary prp mask
        num_prp = is_match.shape[-1]
        binary_prp_head = is_match[tgt_head_idxs]  # num_tgt_rel, num_prp (matched prp head)
        binary_prp_tail = is_match[tgt_tail_idxs]  # num_tgt_rel, num_prp (matched prp tail)
        binary_rel_matrixs = torch.zeros((num_prp, num_prp), device=device).long()

        fg_rel_triplets = []
        
        corrsp_gt_rel_idx = []

        for i in range(num_tgt_rels):
            # generate binary prp mask
            bi_match_head = torch.nonzero(binary_prp_head[i] > 0)
            bi_match_tail = torch.nonzero(binary_prp_tail[i] > 0)

            num_bi_head = bi_match_head.shape[0]
            num_bi_tail = bi_match_tail.shape[0]
            if num_bi_head > 0 and num_bi_tail > 0:
                bi_match_head = bi_match_head.view(1, num_bi_head).expand(num_bi_tail, num_bi_head).contiguous()
                bi_match_tail = bi_match_tail.view(num_bi_tail, 1).expand(num_bi_tail, num_bi_head).contiguous()
                binary_rel_matrixs[bi_match_head.view(-1), bi_match_tail.view(-1)] = 1
                binary_rel_matrixs[bi_match_tail.view(-1), bi_match_head.view(-1)] = 1

            tgt_head_idx = int(tgt_head_idxs[i])
            tgt_tail_idx = int(tgt_tail_idxs[i])
            tgt_rel_lab = int(tgt_rel_labs[i])
            prp_head_idxs = torch.nonzero(is_match[tgt_head_idx]).squeeze(1)
            prp_tail_idxs = torch.nonzero(is_match[tgt_tail_idx]).squeeze(1)
            num_match_head = prp_head_idxs.shape[0]
            num_match_tail = prp_tail_idxs.shape[0]
            if num_match_head <= 0 or num_match_tail <= 0: 
                continue
            prp_head_idxs = prp_head_idxs.view(-1, 1).expand(num_match_head, num_match_tail).contiguous().view(-1)
            prp_tail_idxs = prp_tail_idxs.view(1, -1).expand(num_match_head, num_match_tail).contiguous().view(-1)
            valid_pair = prp_head_idxs != prp_tail_idxs
            if valid_pair.sum().item() <= 0:
                continue
            # remove self-pair
            # remove selected pair from rel_possibility
            prp_head_idxs = prp_head_idxs[valid_pair]
            prp_tail_idxs = prp_tail_idxs[valid_pair]
            rel_possibility[prp_head_idxs, prp_tail_idxs] = 0
            fg_labels = torch.tensor([tgt_rel_lab] * prp_tail_idxs.shape[0], dtype=torch.int64, device=device) \
                             .view(-1, 1)
            
            fg_rel_i = cat((prp_head_idxs.view(-1, 1), prp_tail_idxs.view(-1, 1), fg_labels), dim=-1).to(torch.int64)
            if fg_rel_i.shape[0] > self.num_sample_per_gt_rel: # 4
                ious_score = (ious[tgt_head_idx, prp_head_idxs] * ious[tgt_tail_idx, prp_tail_idxs]).view(
                    -1).detach().cpu().numpy()
                ious_score = ious_score / ious_score.sum()
                perm = npr.choice(ious_score.shape[0], p=ious_score, size=self.num_sample_per_gt_rel, replace=False)
                fg_rel_i = fg_rel_i[perm]
            if fg_rel_i.shape[0] > 0:
                fg_rel_triplets.append(fg_rel_i)

            corrsp_gt_rel_idx.extend([i,] * fg_rel_i.shape[0])

        if len(fg_rel_triplets) == 0: 
            fg_rel_triplets = torch.zeros((0, 3), dtype=torch.int64, device=device)
        else:
            fg_rel_triplets = cat(fg_rel_triplets, dim=0).to(torch.int64)
            if fg_rel_triplets.shape[0] > self.num_pos_per_img: # 250
                perm = torch.randperm(fg_rel_triplets.shape[0], device=device)[:self.num_pos_per_img]
                fg_rel_triplets = fg_rel_triplets[perm]

        bg_rel_inds = torch.nonzero(rel_possibility > 0).view(-1, 2)
        bg_rel_labs = torch.zeros(bg_rel_inds.shape[0], dtype=torch.int64, device=device)
        bg_rel_triplets = cat((bg_rel_inds, bg_rel_labs.view(-1, 1)), dim=-1).to(torch.int64)

        num_neg_per_img = min(self.batch_size_per_image - fg_rel_triplets.shape[0], bg_rel_triplets.shape[0])
        if bg_rel_triplets.shape[0] > 0:
            pairs_qualities = proposals_quality[bg_rel_triplets[:, 0]] * proposals_quality[bg_rel_triplets[:, 1]]
            _, sorted_idx = torch.sort(pairs_qualities, dim=0, descending=True)
            bg_rel_triplets = bg_rel_triplets[sorted_idx][: int(num_neg_per_img * 2.0)]
            perm = torch.randperm(bg_rel_triplets.shape[0], device=device)[:num_neg_per_img]
            bg_rel_triplets = bg_rel_triplets[perm]
        else:
            bg_rel_triplets = torch.zeros((0, 3), dtype=torch.int64, device=device)

        if fg_rel_triplets.shape[0] == 0 and bg_rel_triplets.shape[0] == 0:
            _, idx = torch.sort(proposals_quality, descending=True)
            bg_rel_triplets = torch.zeros((2, 3), dtype=torch.int64, device=device)
            for i in range(2):
                bg_rel_triplets[i, 0] = 0
                bg_rel_triplets[i, 1] = 0
                bg_rel_triplets[i, 2] = 0
        
        corrsp_gt_rel_idx.extend([-1, ] * bg_rel_triplets.shape[0])
        corrsp_gt_rel_idx = torch.Tensor(corrsp_gt_rel_idx).long().to(device)

        return cat((fg_rel_triplets, bg_rel_triplets), dim=0), corrsp_gt_rel_idx, binary_rel_matrixs


def make_roi_relation_samp_processor(cfg):
    samp_processor = RelationSampling(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_RELATION_HEAD.REQUIRE_BOX_OVERLAP,
        cfg.MODEL.ROI_RELATION_HEAD.NUM_SAMPLE_PER_GT_REL,
        cfg.MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE,
        cfg.MODEL.ROI_RELATION_HEAD.POSITIVE_FRACTION,
        cfg.MODEL.ROI_RELATION_HEAD.MAX_PROPOSAL_PAIR,
        cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX,
        cfg.TEST.RELATION.REQUIRE_OVERLAP,
    )

    return samp_processor
