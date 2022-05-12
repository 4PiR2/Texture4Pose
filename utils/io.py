import json
from typing import Union

import cv2
import torch


def read_json_file(path: str) -> Union[dict, list]:
    with open(path, 'r') as f:
        return json.load(f)


def read_img_file(path: str, dtype: torch.dtype = torch.float, device: Union[torch.device, str] = None) -> torch.Tensor:
    """
    :return: [1, 3(RGB), H, W] \in [0, 1]
    """
    return torch.tensor(cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1].copy(), dtype=dtype,
                        device=parse_device(device)).permute(2, 0, 1)[None] / 255.


def read_depth_img_file(path: str, dtype: torch.dtype = torch.float, device: Union[torch.device, str] = None) \
        -> torch.Tensor:
    """
    :return: [1, 1, H, W]
    """
    return torch.tensor(cv2.imread(path, cv2.IMREAD_ANYDEPTH), dtype=dtype, device=parse_device(device))[None, None]


def parse_device(device: Union[torch.device, str] = None) -> Union[torch.device, str]:
    """
    :return: device if device is not None else 'cpu'
    """
    return device if device is not None else 'cpu'
