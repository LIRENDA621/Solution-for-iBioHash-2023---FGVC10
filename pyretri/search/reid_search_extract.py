# -*- coding: utf-8 -*-


import os

import torch

import argparse
import importlib

from pyretri.config import get_defaults_cfg
from pyretri.datasets import build_folder, build_loader
from pyretri.models import build_model
from pyretri.extract import build_extract_helper

from pyretri.models.backbone import ft_net


def load_datasets():
    data_json_dir = "/home/songrenjie/projects/RetrievalToolBox/new_data_jsons/"
    datasets = {
        "market_gallery": os.path.join(data_json_dir, "market_gallery.json"),
        "market_query": os.path.join(data_json_dir, "market_query.json"),
        "duke_gallery": os.path.join(data_json_dir, "duke_gallery.json"),
        "duke_query": os.path.join(data_json_dir, "duke_query.json"),
    }
    for data_path in datasets.values():
        assert os.path.exists(data_path), "non-exist dataset path {}".format(data_path)
    return datasets


def parse_args():
    parser = argparse.ArgumentParser(description='A tool box for deep learning-based image retrieval')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
    parser.add_argument('--save_path', '-sp', default=None, type=str, help="the save path for feature")
    parser.add_argument("--search_modules", "-sm", default="", type=str, help="name of search module's directory")
    args = parser.parse_args()

    return args


def main():

    # init args
    args = parse_args()

    # init retrieval pipeline settings
    cfg = get_defaults_cfg()

    # load search space
    datasets = load_datasets()
    pre_processes = importlib.import_module("{}.pre_process_dict".format(args.search_modules)).pre_processes
    models = importlib.import_module("{}.extract_dict".format(args.search_modules)).models
    extracts = importlib.import_module("{}.extract_dict".format(args.search_modules)).extracts

    # search in an exhaustive way
    for data_name, data_args in datasets.items():
        for pre_proc_name, pre_proc_args in pre_processes.items():
            if 'market' in data_name:
                model_name = 'market_res50'
            elif 'duke' in data_name:
                model_name = 'duke_res50'

            feature_full_name = data_name + "_" + pre_proc_name + "_" + model_name
            print(feature_full_name)

            if os.path.exists(os.path.join(args.save_path, feature_full_name)):
                print("[Search Extract]: config exists...")
                continue

            # load retrieval pipeline settings
            cfg.datasets.merge_from_other_cfg(pre_proc_args)
            cfg.model.merge_from_other_cfg(models[model_name])
            cfg.extract.merge_from_other_cfg(extracts[model_name])

            # build dataset and dataloader
            dataset = build_folder(data_args, cfg.datasets)
            dataloader = build_loader(dataset, cfg.datasets)

            # build model
            model = build_model(cfg.model)

            # build helper and extract features
            extract_helper = build_extract_helper(model, cfg.extract)
            extract_helper.do_extract(dataloader, save_path=os.path.join(args.save_path, feature_full_name))


if __name__ == '__main__':
    main()
