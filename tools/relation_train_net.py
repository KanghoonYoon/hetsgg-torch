import os
import sys
sys.path.append(os.getcwd())
import argparse
import datetime
import os
import random
import time
import numpy as np
import torch
import pickle

from hetsgg.config import cfg
from hetsgg.data import make_data_loader
from hetsgg.engine.inference import inference
from hetsgg.engine.trainer import reduce_loss_dict
from hetsgg.modeling.detector import build_detection_model
from hetsgg.solver import make_lr_scheduler
from hetsgg.solver import make_optimizer
from hetsgg.utils.checkpoint import DetectronCheckpointer
from hetsgg.utils.checkpoint import clip_grad_norm
from hetsgg.utils import visualize_graph as vis_graph
from hetsgg.utils.collect_env import collect_env_info
from hetsgg.utils.comm import synchronize, get_rank, all_gather
from hetsgg.utils.logger import setup_logger, debug_print, TFBoardHandler_LEVEL
from hetsgg.utils.metric_logger import MetricLogger
from hetsgg.utils.miscellaneous import mkdir, save_config
from hetsgg.utils.global_buffer import save_buffer


try:
    from apex import amp
except ImportError:
    raise ImportError("Use APEX for multi-precision via apex.amp")

SEED = 666

torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.set_num_threads(6)

torch.autograd.set_detect_anomaly(False)

SHOW_COMP_GRAPH = False

torch.cuda.set_device(0)


def show_params_status(model):
    """
    Prints parameters of a model
    """
    st = {}
    strings = []
    total_params = 0
    trainable_params = 0
    for p_name, p in model.named_parameters():

        if not ("bias" in p_name.split(".")[-1] or "bn" in p_name.split(".")[-1]):
            st[p_name] = ([str(x) for x in p.size()], np.prod(p.size()), p.requires_grad)
        total_params += np.prod(p.size())
        if p.requires_grad:
            trainable_params += np.prod(p.size())
    for p_name, (size, prod, p_req_grad) in st.items():
        strings.append(
            "{:<80s}: {:<16s}({:8d}) ({})".format(
                p_name, "[{}]".format(",".join(size)), prod, "grad" if p_req_grad else "    "
            )
        )
    strings = "\n".join(strings)
    return (
        f"\n{strings}\n ----- \n \n"
        f"      trainable parameters:  {trainable_params/ 1e6:.3f}/{total_params / 1e6:.3f} M \n "
    )


def train(
    cfg,
    local_rank,
    distributed,
    logger,
):
    global SHOW_COMP_GRAPH

    debug_print(logger, "Start initializing dataset & dataloader")

    arguments = {}
    arguments["iteration"] = 0
    output_dir = cfg.OUTPUT_DIR



    debug_print(logger, "prepare training")
    model = build_detection_model(cfg)
    model.train()
    debug_print(logger, "end model construction")


    eval_modules = (
        model.rpn,
        model.backbone,
        model.roi_heads.box,
    )
    train_modules = ()
    rel_pn_module_ref = []
   

    fix_eval_modules(eval_modules) 
    set_train_modules(train_modules)

    if model.roi_heads.relation.rel_pn is not None:
        rel_on_module = (model.roi_heads.relation.rel_pn,)
    else:
        rel_on_module = None

    logger.info("trainable models:")

    slow_heads = []


    except_weight_decay = []


    load_mapping = {
        "roi_heads.relation.box_feature_extractor": "roi_heads.box.feature_extractor",
        "roi_heads.relation.rel_pair_box_feature_extractor": "roi_heads.box.feature_extractor",
        "roi_heads.relation.union_feature_extractor.feature_extractor": "roi_heads.box.feature_extractor",
    }

    print("load model to GPU")
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    num_batch = cfg.SOLVER.IMS_PER_BATCH # 12

    optimizer = make_optimizer(
        cfg,
        model,
        logger,
        slow_heads=slow_heads, 
        slow_ratio=2.5,
        rl_factor=float(num_batch),
        except_weight_decay=except_weight_decay,
    )

    scheduler = make_lr_scheduler(cfg, optimizer, logger)
    debug_print(logger, "end optimizer and shcedule")

    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = "O1" if use_mixed_precision else "O0"
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk, custom_scheduler=True
    )
    checkpoint = None
    if cfg.MODEL.PRETRAINED_DETECTOR_CKPT != "":
        checkpointer.load(
            cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False, load_mapping=load_mapping
        ) 
    else:
        checkpoint =  checkpointer.load(
            cfg.MODEL.WEIGHT,
            with_optim=False,
        )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    debug_print(logger, "end load checkpointer")

    if cfg.MODEL.ROI_RELATION_HEAD.RE_INITIALIZE_CLASSIFIER:
        model.roi_heads.relation.predictor.init_classifier_weight()

    rel_model_ref = model.roi_heads.relation

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    debug_print(logger, "end distributed")

    train_data_loader = make_data_loader(
        cfg,
        mode="train",
        is_distributed=distributed,
        start_iter=checkpoint['iteration'] if checkpoint else 0,
    )
    val_data_loaders = make_data_loader(
        cfg,
        mode="val",
        is_distributed=distributed,
    )

    debug_print(logger, "end dataloader")

    pre_clser_pretrain_on = False

    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(train_data_loader)
    start_iter = checkpoint['iteration'] if checkpoint else 0
    start_training_time = time.time()
    end = time.time()
    model.train()

    print_first_grad = True
    for iteration, (images, targets, _) in enumerate(train_data_loader, start_iter):
        if any(len(target) < 1 for target in targets):
            logger.error(
                f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}"
            )
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        model.train()
        fix_eval_modules(eval_modules)

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        loss_dict = model(images, targets, logger=logger)

        losses = sum(loss for loss in loss_dict.values())

        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        meters.update(loss=losses_reduced, **loss_dict_reduced)
        optimizer.zero_grad()

        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()


        if not SHOW_COMP_GRAPH and get_rank() == 0:
            try:
                g = vis_graph.visual_computation_graph(
                    losses, model.named_parameters(), cfg.OUTPUT_DIR, "total_loss-graph"
                )
                g.render()
                for name, ls in loss_dict_reduced.items():
                    g = vis_graph.visual_computation_graph(
                        losses, model.named_parameters(), cfg.OUTPUT_DIR, f"{name}-graph"
                    )
                    g.render()
            except:
                logger.info("print computational graph failed")

            SHOW_COMP_GRAPH = True

        verbose = (
            iteration % cfg.SOLVER.PRINT_GRAD_FREQ
        ) == 0 or print_first_grad  
        print_first_grad = False
        clip_grad_norm(
            [(n, p) for n, p in model.named_parameters() if p.requires_grad],
            max_norm=cfg.SOLVER.GRAD_NORM_CLIP,
            logger=logger,
            verbose=verbose,
            clip=True,
        )

        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        elapsed_time = str(datetime.timedelta(seconds=int(end - start_training_time)))

        if (
            iteration
            in [
                cfg.MODEL.ROI_RELATION_HEAD.RELATION_PROPOSAL_MODEL.FIX_MODEL_AT_ITER,
            ]
            and rel_on_module is not None
        ):
            logger.info("fix the rel pn module")
            fix_eval_modules(rel_pn_module_ref)

        if pre_clser_pretrain_on:
            if iteration == STOP_ITER:
                logger.info("pre clser pretraining ended.")
                m2opt.roi_heads.relation.predictor.end_preclser_relpn_pretrain()
                pre_clser_pretrain_on = False

        if iteration % 30 == 0:
            logger.log(TFBoardHandler_LEVEL, (meters.meters, iteration))

            logger.log(
                TFBoardHandler_LEVEL,
                ({"curr_lr": float(optimizer.param_groups[0]["lr"])}, iteration),
            )
            # save_buffer(output_dir)

        if iteration % cfg.LOSS_PERIOD== 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "\ninstance name: {instance_name}\n" "elapsed time: {elapsed_time}\n",
                        "eta: {eta}\n",
                        "iter: {iter}/{max_iter}\n",
                        "{meters}",
                        "lr: {lr:.6f}\n",
                        "max mem: {memory:.0f}\n",
                    ]
                ).format(
                    instance_name=cfg.OUTPUT_DIR[len("checkpoints/") :],
                    eta=eta_string,
                    elapsed_time=elapsed_time,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    max_iter=max_iter,
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            if pre_clser_pretrain_on:
                logger.info("relness module pretraining..")

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

        val_result_value = None  
        if cfg.SOLVER.TO_VAL and iteration % cfg.SOLVER.VAL_PERIOD == 0:

            logger.info("Start validating")
            val_result = run_val(cfg, model, val_data_loaders, distributed, logger)
            val_result_value = val_result[1]
            if get_rank() == 0:
                for each_ds_eval in val_result[0]:
                    for each_evalator_res in each_ds_eval[1]:
                        logger.log(TFBoardHandler_LEVEL, (each_evalator_res, iteration))

        if cfg.SOLVER.SCHEDULE.TYPE == "WarmupReduceLROnPlateau":
            scheduler.step(val_result_value, epoch=iteration)
            if scheduler.stage_count >= cfg.SOLVER.SCHEDULE.MAX_DECAY_STEP:
                logger.info("Trigger MAX_DECAY_STEP at iteration {}.".format(iteration))
                break
        else:
            scheduler.step()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    return model


def fix_eval_modules(eval_modules):
    for module in eval_modules:
        if module is None:
            continue

        for _, param in module.named_parameters():
            param.requires_grad = False


def set_train_modules(modules):
    for module in modules:
        for _, param in module.named_parameters():
            param.requires_grad = True


def run_val(cfg, model, val_data_loaders, distributed, logger):
    if distributed:
        model = model.module
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations",)
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes",)

    dataset_names = cfg.DATASETS.VAL
    val_result = []
    for dataset_name, val_data_loader in zip(dataset_names, val_data_loaders):
        dataset_result = inference(
            cfg,
            model,
            val_data_loader,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=None,
            logger=logger,
        )
        synchronize()
        val_result.append(dataset_result)

    val_values = []
    for each in val_result:
        if isinstance(each, tuple):
            val_values.append(each[0])
    # support for multi gpu distributed testing
    # send evaluation results to each process
    gathered_result = all_gather(torch.tensor(val_values).cpu())
    gathered_result = [t.view(-1) for t in gathered_result]
    gathered_result = torch.cat(gathered_result, dim=-1).view(-1)
    valid_result = gathered_result[gathered_result >= 0]
    val_result_val = float(valid_result.mean())

    del gathered_result, valid_result
    return val_result, val_result_val


def run_test(cfg, model, distributed, logger):
    if distributed:
        model = model.module
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations",)
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, mode="test", is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(
        output_folders, dataset_names, data_loaders_val
    ):
        inference(
            cfg,
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            logger=logger,
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Relation Detection Training")
    parser.add_argument(
        "--config-file",
        default="configs/relHetSGG_vg.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    cfg.set_new_allowed(True)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.set_new_allowed(True)

    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("hetsgg", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))


    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()

    output_config_path = os.path.join(cfg.OUTPUT_DIR, "config.yml")
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    model = train(cfg, args.local_rank, args.distributed, logger)

    if not args.skip_test:
        run_test(cfg, model, args.distributed, logger)


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "8"

    main()
