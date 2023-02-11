export CUDA_VISIBLE_DEVICES="0"
export num_gpu=1
export use_multi_gpu=false
export use_obj_refine=True
export task='sgcls'

export model_config="relHetSGG_vg" # relHetSGG_vg, relHetSGGp_vg
export output_dir="checkpoints/${task}-HetSGGPredictor-vg"

export path_faster_rcnn=''


if $use_multi_gpu;then
    # Multi GPU -sgcls Task
    python -m torch.distributed.launch --master_port 10032 --nproc_per_node=$num_gpu tools/relation_train_net.py --config-file "configs/${model_config}.yaml" \
        SOLVER.IMS_PER_BATCH 9 \
        TEST.IMS_PER_BATCH 6 \
        OUTPUT_DIR ${output_dir} \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE ${use_obj_refine} \
        MODEL.PRETRAINED_DETECTOR_CKPT ${path_faster_rcnn}
else
    # Single GPU
    python  tools/relation_train_net.py --config-file "configs/${model_config}.yaml" \
        SOLVER.IMS_PER_BATCH 9 \
        TEST.IMS_PER_BATCH 6 \
        OUTPUT_DIR ${output_dir} \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE ${use_obj_refine} \
        MODEL.PRETRAINED_DETECTOR_CKPT ${path_faster_rcnn}
fi