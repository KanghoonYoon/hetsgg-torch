import bisect
import copy
import logging
import os
import pickle
from collections import Counter

import numpy as np
import torch
import torch.utils.data
from matplotlib import pyplot as plt
from tqdm import tqdm

from hetsgg.config import cfg
from hetsgg.utils.comm import get_world_size, is_main_process, synchronize
from hetsgg.utils.imports import import_file
from hetsgg.utils.miscellaneous import save_labels

from . import datasets as D
from . import samplers
from .collate_batch import BatchCollator, BBoxAugCollator
from .transforms import build_transforms


# by Jiaxin
def get_dataset_statistics(cfg):
    """
    get dataset statistics (e.g., frequency bias) from training data
    will be called to help construct FrequencyBias module
    """
    logger = logging.getLogger(__name__)
    logger.info('-' * 100)
    logger.info('get dataset statistics...')
    paths_catalog = import_file(
        "hetsgg.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_names = cfg.DATASETS.TRAIN

    data_statistics_name = ''.join(dataset_names) + '_statistics'
    save_file = os.path.join(cfg.OUTPUT_DIR, "{}.cache".format(data_statistics_name))

    if os.path.exists(save_file):
        logger.info('Loading data statistics from: ' + str(save_file))
        logger.info('-' * 100)
        return torch.load(save_file, map_location=torch.device("cpu"))

    statistics = []
    for dataset_name in dataset_names:
        data = DatasetCatalog.get(dataset_name, cfg)
        factory = getattr(D, data["factory"])
        args = data["args"]
        dataset = factory(**args)
        if "VG_stanford" in dataset_name:
            get_dataset_distribution(dataset, dataset_name)
            logger.info(f"Data Length : {len(dataset)}")
        statistics.append(dataset.get_statistics())
    logger.info('finish')

    assert len(statistics) == 1
    result = {
        'fg_matrix': statistics[0]['fg_matrix'],
        'pred_dist': statistics[0]['pred_dist'],
        'obj_classes': statistics[0]['obj_classes'],  # must be exactly same for multiple datasets
        'rel_classes': statistics[0]['rel_classes'],
        'att_classes': statistics[0]['att_classes'],
    }
    logger.info('Save data statistics to: ' + str(save_file))
    logger.info('-' * 100)
    torch.save(result, save_file)
    return result


def get_dataset_distribution(train_data, dataset_name):
    """save relation frequency distribution after the sampling etc processing
    the data distribution that model will be trained on it

    Args:
        train_data ([type]): [description]
        dataset_name ([type]): [description]
    """
    # 
    if is_main_process():
        print("Get relation class frequency distribution on dataset.")
        pred_counter = Counter()
        for i in tqdm(range(len(train_data))):
            tgt_rel_matrix = train_data.get_groundtruth(i, inner_idx=False).get_field("relation")
            tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
            tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
            tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)
            tgt_rel_labs = tgt_rel_matrix[tgt_head_idxs, tgt_tail_idxs].contiguous().view(-1).numpy()
            for each in tgt_rel_labs:
                pred_counter[each] += 1

        with open(os.path.join(cfg.OUTPUT_DIR, "pred_counter.pkl"), 'wb') as f:
            pickle.dump(pred_counter, f)

        from hetsgg.data.datasets.visual_genome import HEAD, TAIL, BODY
        
        head = HEAD
        body = BODY
        tail = TAIL

        count_sorted = []
        counter_name = []
        cate_set = []
        cls_dict = train_data.ind_to_predicates

        for idx, name_set in enumerate([head, body, tail]):
            # sort the cate names accoding to the frequency
            part_counter = []
            for name in name_set:
                part_counter.append(pred_counter[name])
            part_counter = np.array(part_counter)
            sorted_idx = np.flip(np.argsort(part_counter))

            # reaccumulate the frequency in sorted index
            for j in sorted_idx:
                name = name_set[j]
                cate_set.append(idx)
                counter_name.append(cls_dict[name])
                count_sorted.append(pred_counter[name])

        count_sorted = np.array(count_sorted)

        fig, axs_c = plt.subplots(1, 1, figsize=(16, 5), tight_layout=True)
        palate = ['r', 'g', 'b']
        color = [palate[idx] for idx in cate_set]
        axs_c.bar(counter_name, count_sorted, color=color)
        axs_c.grid()
        plt.xticks(rotation=-60)
        axs_c.set_ylim(0, 50000)
        fig.set_facecolor((1, 1, 1))

        save_file = os.path.join(cfg.OUTPUT_DIR, "rel_freq_dist.png")
        fig.savefig(save_file, dpi=300)
    synchronize()


def build_dataset(cfg, dataset_list, transforms, dataset_catalog, is_train=True):
    """
    Arguments:
        dataset_list (list[str]): Contains the names of the datasets, i.e.,
            coco_2014_trian, coco_2014_val, etc
        transforms (callable): transforms to apply to each (image, target) sample
        dataset_catalog (DatasetCatalog): contains the information on how to
            construct a dataset.
        is_train (bool): whether to setup the dataset for training or testing
    """
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasets = []
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name, cfg)
        factory = getattr(D, data["factory"])
        args = data["args"]
        # for COCODataset, we want to remove images without annotations
        # during training
        if data["factory"] == "COCODataset":
            args["remove_images_without_annotations"] = is_train
        if data["factory"] == "PascalVOCDataset":
            args["use_difficult"] = not is_train
        args["transforms"] = transforms
        # make dataset from factory
        # print("build dataset with args:")
        # pprint(args)
    
        dataset = factory(**args)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return [dataset]


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        if hasattr(dataset, "idx_list"):
            i = dataset.idx_list[i]
        img_info = dataset.img_info[i]
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
        dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0
):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_loader(cfg, mode='train', is_distributed=False, start_iter=0):
    assert mode in {'train', 'val', 'test'}
    num_gpus = get_world_size()
    is_train = mode == 'train'
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
                images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
                images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0


    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    paths_catalog = import_file(
        "hetsgg.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    if mode == 'train':
        dataset_list = cfg.DATASETS.TRAIN
    elif mode == 'val':
        dataset_list = cfg.DATASETS.VAL
    else:
        dataset_list = cfg.DATASETS.TEST

    # If bbox aug is enabled in testing, simply set transforms to None and we will apply transforms later
    transforms = None if not is_train and cfg.TEST.BBOX_AUG.ENABLED else build_transforms(cfg, is_train)
    datasets = build_dataset(cfg, dataset_list, transforms, DatasetCatalog, is_train)

    if is_train:
        # save category_id to label name mapping
        save_labels(datasets, cfg.OUTPUT_DIR)

    data_loaders = []
    for dataset in datasets:
        # print('============')
        # print(len(dataset))
        # print(images_per_gpu)
        # print('============')
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter
        )
        collator = BBoxAugCollator() if not is_train and cfg.TEST.BBOX_AUG.ENABLED else \
            BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
        data_loaders.append(data_loader)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders




