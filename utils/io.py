import json
from typing import Union

import torch


def read_json_file(path: str) -> Union[dict, list]:
    with open(path, 'r') as f:
        return json.load(f)


def parse_device(device: Union[torch.device, str] = None) -> Union[torch.device, str]:
    """
    :return: device if device is not None else 'cpu'
    """
    return device if device is not None else 'cpu'
