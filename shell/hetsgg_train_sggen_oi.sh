export CUDA_VISIBLE_DEVICES="4"
export num_gpu=2
export use_multi_gpu=false
export use_obj_refine=False 
export task='sggen'

export config=oi_v6 # oi_v4, oi_v6
export output_dir="checkpoints/${task}-HetSGGPredictor-${config}"

export path_faster_rcnn=''

if $use_multi_gpu;then
    python -m torch.distributed.launch --master_port 10023 --nproc_per_node=${num_gpu} tools/relation_train_net.py --config-file "configs/relHetSGG_${config}.yaml" \
        SOLVER.IMS_PER_BATCH 18 \
        TEST.IMS_PER_BATCH 12 \
        OUTPUT_DIR ${output_dir} \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
        MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE ${use_obj_refine} \
        MODEL.PRETRAINED_DETECTOR_CKPT ${path_faster_rcnn}
else
    python  tools/relation_train_net.py --config-file "configs/relHetSGG_${config}.yaml" \
        SOLVER.IMS_PER_BATCH 9 \
        TEST.IMS_PER_BATCH 6 \
        OUTPUT_DIR ${output_dir} \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
        MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE ${use_obj_refine} \
        MODEL.PRETRAINED_DETECTOR_CKPT ${path_faster_rcnn}
fi



