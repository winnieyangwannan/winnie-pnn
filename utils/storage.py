import csv
import os
import torch
import logging
import sys

import utils


def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def get_storage_dir():
    if "RL_STORAGE" in os.environ:
        return os.environ["RL_STORAGE"]
    return "storage"


def get_model_dir(env_name, num_columns, transfer):
    return os.path.join(get_storage_dir(), env_name, 'columns_' + str(num_columns), transfer)


def get_status_path(model_dir, seed):
    return os.path.join(model_dir, str(seed) + '_status.pt')


def get_status(model_dir, seed):
    path = get_status_path(model_dir, seed)
    return torch.load(path)


def save_status(status, model_dir, seed):
    path = get_status_path(model_dir, seed)
    utils.create_folders_if_necessary(path)
    torch.save(status, path)


def get_vocab(model_dir, seed):
    return get_status(model_dir, seed)["vocab"]


def get_model_state(model_dir,seed):
    return get_status(model_dir, seed)["model_state"]


def get_txt_logger(model_dir):
    path = os.path.join(model_dir, "log.txt")
    utils.create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger()


def get_csv_logger(model_dir):
    csv_path = os.path.join(model_dir, "log.csv")
    utils.create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)

# for pnn
def pnn_load_state_dict(pnn, col_idx, path):
    status = torch.load(path)

    ac_sd = status['model_state']

    ac_keys = list(ac_sd.keys())

    ac_keys = [k for k in ac_keys]

    keys_new = list(pnn.base.columns[col_idx].state_dict().keys())
    keys_old = ac_keys

    new_dict = {k1: ac_sd[k2] for k1, k2 in zip(keys_new, keys_old)}
    pnn.base.columns[col_idx].load_state_dict(new_dict)

