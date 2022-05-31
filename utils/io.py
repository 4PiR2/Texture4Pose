import glob
import json
import os.path
import re
from typing import Optional, Union

import cv2
import torch


def read_json_file(path: str) -> Union[dict, list]:
    with open(path, 'r') as f:
        return json.load(f)


def read_img_file(path: str, dtype: torch.dtype = torch.float, device: Union[torch.device, str] = None) -> torch.Tensor:
    """
    :return: [1, 3(RGB), H, W] \in [0, 1]
    """
    return torch.tensor(cv2.imread(path, cv2.IMREAD_COLOR)[None, ..., ::-1].copy(), dtype=dtype,
                        device=parse_device(device)).permute(0, 3, 1, 2) / 255.


def read_depth_img_file(path: str, dtype: torch.dtype = torch.float, device: Union[torch.device, str] = None) \
        -> torch.Tensor:
    """
    :return: [1, 1, H, W]
    """
    return torch.tensor(cv2.imread(path, cv2.IMREAD_ANYDEPTH)[None, None].astype('float32'), dtype=dtype,
                        device=parse_device(device))


def parse_device(device: Union[torch.device, str] = None) -> Union[torch.device, str]:
    """
    :return: device if device is not None else 'cpu'
    """
    return device if device is not None else 'cpu'


def find_lightning_ckpt_path(root: str = '.', version: int = None, epoch: int = None, step: int = None) \
        -> Optional[str]:
    paths = []
    # pattern = re.compile(r'/version_(\d+)/checkpoints/epoch=(\d+)-step=(\d+)\.ckpt$')
    for path in glob.iglob(os.path.join(root, '**', '*.ckpt'), recursive=True):
        # m = pattern.search(path)
        # v, e, s = m.group(1), m.group(2), m.group(3)
        # paths.append([v, e, s, path])
        if path.endswith('lask.ckpt'):
            return path
        paths.append(path)
    # if version is not None:
    #     paths = filter(lambda path: path[0] == version, paths)
    # if epoch is not None:
    #     paths = filter(lambda path: path[1] == epoch, paths)
    # if step is not None:
    #     paths = filter(lambda path: path[2] == step, paths)
    paths.sort(reverse=True)
    # paths = [path[-1] for path in paths]
    return paths[0] if len(paths) else None
