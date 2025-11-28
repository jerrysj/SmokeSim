import argparse
import glob, os, sys
from pathlib import Path
from os import PathLike

import pickle as pkl

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.append('/home/project/GY/aaai_2025/third_party/OpenPCDet')

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class KittiDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError


        prefix = os.path.splitext(os.path.basename(self.sample_file_list[index]))[0]
        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict



def run_kitti_model_and_save(
    cfg_file:  PathLike,
    data_path: PathLike,
    ckpt:      PathLike,
    ext:       str,
    out_dir:   PathLike,
    save_name: str,
):
    cfg_from_yaml_file(cfg_file, cfg)

    logger = common_utils.create_logger()
    dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(data_path), ext=ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    all_resuls = []
    with torch.no_grad():
        for idx, data_dict in enumerate(dataset):
            logger.info(f'adv sample with index:  {idx + 1}\t will be detection')
            data_dict = dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            frame_results = {}
            frame_results['pred_boxes']  = pred_dicts[0]['pred_boxes'].cpu().numpy()
            frame_results['pred_scores'] = pred_dicts[0]['pred_scores'].cpu().numpy()
            frame_results['pred_labels'] = pred_dicts[0]['pred_labels'].cpu().numpy()
            frame_results['frame_id']    = data_dict['frame_id']
            all_resuls.append(frame_results)

    out_seq_path = os.path.join(out_dir, f'{save_name}_det_result.pkl')
    logger.info(f'Detection done. Write the result to: {out_seq_path}')
    with open(out_seq_path, 'wb') as f:
        pkl.dump(all_resuls, f)


class NuscenesDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 5)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError


        prefix = os.path.splitext(os.path.basename(self.sample_file_list[index]))[0]
        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def run_nusc_model_and_save(
    cfg_file:  PathLike,
    data_path: PathLike,
    ckpt:      PathLike,
    ext:       str,
    out_dir:   PathLike,
    save_name: str,
):
    cfg_from_yaml_file(cfg_file, cfg)

    logger = common_utils.create_logger()
    dataset = NuscenesDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(data_path), ext=ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    all_resuls = []
    with torch.no_grad():
        for idx, data_dict in enumerate(dataset):
            logger.info(f'adv sample with index:  {idx + 1}\t will be detection')
            data_dict = dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            frame_results = {}
            frame_results['pred_boxes']  = pred_dicts[0]['pred_boxes'].cpu().numpy()
            frame_results['pred_scores'] = pred_dicts[0]['pred_scores'].cpu().numpy()
            frame_results['pred_labels'] = pred_dicts[0]['pred_labels'].cpu().numpy()
            frame_results['frame_id']    = data_dict['frame_id']
            all_resuls.append(frame_results)

    out_seq_path = os.path.join(out_dir, f'{save_name}_det_result.pkl')
    logger.info(f'Detection done. Write the result to: {out_seq_path}')
    with open(out_seq_path, 'wb') as f:
        pkl.dump(all_resuls, f)

