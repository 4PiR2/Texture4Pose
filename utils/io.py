import glob
import json
import numpy as np
import os.path
from PIL import Image, ExifTags
import pillow_heif
import re
from typing import Optional, Union

import cv2
import torch


pillow_heif.register_heif_opener()


def read_json_file(path: str) -> Union[dict, list]:
    with open(path, 'r') as f:
        return json.load(f)


def _rotate_pil_im(im: Image, orientation: int = 1) -> Image:
    if orientation == 2:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    elif orientation == 3:
        im = im.transpose(Image.ROTATE_180)
    elif orientation == 4:
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
    elif orientation == 5:
        im = im.transpose(Image.ROTATE_90).transpose(Image.FLIP_TOP_BOTTOM)
    elif orientation == 6:
        im = im.transpose(Image.ROTATE_270)
    elif orientation == 7:
        im = im.transpose(Image.ROTATE_270).transpose(Image.FLIP_TOP_BOTTOM)
    elif orientation == 8:
        im = im.transpose(Image.ROTATE_90)
    return im


def _fetch_exif_data(exif_data, gps: bool = False):
    # https://www.awaresystems.be/imaging/tiff/tifftags/privateifd.html
    tag_ids, contents = list(exif_data.keys()), list(exif_data.values())
    tags = [(ExifTags.TAGS if not gps else ExifTags.GPSTAGS).get(tag_id, None) for tag_id in tag_ids]
    for ifd_id in [0x8769, 0x8825, 0xA005]:
        if ifd_id in tag_ids:
            tag_ids_ifd, tags_ifd, contents_ifd = _fetch_exif_data(exif_data.get_ifd(ifd_id), ifd_id == 0x8825)
            tag_ids += tag_ids_ifd
            tags += tags_ifd
            contents += contents_ifd
    return tag_ids, tags, contents


def imread(filename: str, size: tuple[int, int] = None, opencv_bgr: bool = True):
    im = Image.open(filename)
    try:
        exif = im.getexif()
        orientation = int(exif[274])
    except (KeyError, AttributeError, TypeError, IndexError):
        orientation = None
    if (filename.split('.')[-1]).lower() == 'heic':
        orientation = im.info['original_orientation']
        if orientation is not None:
            if orientation == 3:  # 180 deg
                im = _rotate_pil_im(im, 3)
            elif orientation == 6:  # 270 deg
                im = _rotate_pil_im(im, 8)
            elif orientation == 8:  # 90 deg
                im = _rotate_pil_im(im, 6)
        tag_ids, tags, contents = _fetch_exif_data(exif)
        for tag_id, tag, content in zip(tag_ids, tags, contents):
            print(f'{hex(tag_id)}\t{str(tag):25}: {content}')
    if size is not None:
        im.thumbnail(size, Image.ANTIALIAS)
    img = np.array(im)
    if opencv_bgr:
        img = img[..., ::-1]
    return img


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
