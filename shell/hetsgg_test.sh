export CUDA_VISIBLE_DEVICES="6"
export num_gpu=2
export use_multi_gpu=false
export task='sgcls'

export test_list=('0045000') # checkpoint

export save_result=False
export output_dir="/checkpoints/" # Please input the checkpoint directory

if $use_multi_gpu;then
    for name in ${test_list[@]}
    do
        python -m torch.distributed.launch --master_port 10025 --nproc_per_node=${num_gpu} tools/relation_test_net.py --config-file "${output_dir}/config.yml" \
            TEST.IMS_PER_BATCH 16 \
            TEST.SAVE_RE/SULT ${save_result} \
            OUTPUT_DIR ${output_dir} \
            MODEL.WEIGHT "${output_dir}/model_${name}.pth"
    done
else
    for name in ${test_list[@]}
    do
        python tools/relation_test_net.py --config-file "${output_dir}/config.yml"  \
            TEST.IMS_PER_BATCH 8 \
            TEST.SAVE_RE/SULT ${save_result} \
            OUTPUT_DIR ${output_dir} \
            MODEL.WEIGHT "${output_dir}/model_${name}.pth"
    done
fi