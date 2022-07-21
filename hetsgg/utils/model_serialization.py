import logging
from collections import OrderedDict

import torch


def align_and_update_state_dicts(model_state_dict, loaded_state_dict, load_mapping):
    logger = logging.getLogger(__name__)
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))

    mapped_current_keys = current_keys.copy()
    for i, key in enumerate(mapped_current_keys):
        for source_key, target_key in load_mapping.items():
            if source_key in key:
                mapped_current_keys[i] = key.replace(source_key, target_key)
                logger.info("MAPPING {} in current model to {} in loaded model.".format(key, mapped_current_keys[i]))

    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in mapped_current_keys for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    # remove indices that correspond to no-match
    idxs[max_match_size == 0] = -1

    # used for logging
    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "REMATCHING! {: <{}} loaded from {: <{}} of shape {}"
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            key = current_keys[idx_new]
            logger.info("NO-MATCHING of current module: {} of shape {}".format(key, 
                                    tuple(model_state_dict[key].shape)))
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        model_state_dict[key] = loaded_state_dict[key_old]
        if ((not key.startswith('module.'))  and key != key_old) or (key.startswith('module.') and key[7:] != key_old):
            logger.info(
                log_str_template.format(
                    key,
                    max_size,
                    key_old,
                    max_size_loaded,
                    tuple(loaded_state_dict[key_old].shape),
                )
            )
    print('Mapping All')


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def load_state_dict(model, loaded_state_dict, load_mapping):
    model_state_dict = model.state_dict()
    # if the state_dict comes from a model that was wrapped in a
    # DataParallel or DistributedDataParallel during serialization,
    # remove the "module" prefix before performing the matching
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    align_and_update_state_dicts(model_state_dict, loaded_state_dict, load_mapping)

    # use strict loading
    model.load_state_dict(model_state_dict, strict=False)
