export CUDA_VISIBLE_DEVICES="6"
export num_gpu=1
export use_multi_gpu=false
export use_obj_refine=False
export task='predcls'

export REPEAT_FACTOR=0.13
export INSTANCE_DROP_RATE=1.6

export model_config="relHetSGGp_vg" # relHetSGG_vg, relHetSGGp_vg
export output_dir="checkpoints/${task}-HetSGGPredictor-vg"

# export path_faster_rcnn='' # Put faster r-cnn path

if $use_multi_gpu;then
    python -m torch.distributed.launch --master_port 10029 --nproc_per_node=${num_gpu} tools/relation_train_net.py \
        --config-file "configs/${model_config}.yaml" \
        SOLVER.IMS_PER_BATCH 18 \
        TEST.IMS_PER_BATCH 12 \
        OUTPUT_DIR ${output_dir} \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.REPEAT_FACTOR ${REPEAT_FACTOR} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.INSTANCE_DROP_RATE ${INSTANCE_DROP_RATE} \
        MODEL.PRETRAINED_DETECTOR_CKPT ${path_faster_rcnn}
else
    python tools/relation_train_net.py --config-file "configs/${model_config}.yaml" \
        SOLVER.IMS_PER_BATCH 9 \
        TEST.IMS_PER_BATCH 6 \
        OUTPUT_DIR ${output_dir} \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.REL_OBJ_MULTI_TASK_LOSS ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.OBJECT_CLASSIFICATION_REFINE ${use_obj_refine} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.REPEAT_FACTOR ${REPEAT_FACTOR} \
        MODEL.ROI_RELATION_HEAD.DATA_RESAMPLING_PARAM.INSTANCE_DROP_RATE ${INSTANCE_DROP_RATE}
        # MODEL.PRETRAINED_DETECTOR_CKPT ${path_faster_rcnn}
fi
