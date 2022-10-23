import io
from typing import Union
import xml.etree.ElementTree as ET

import exifread
import numpy as np
from PIL import Image
import pyheif
from pyheif import HeifContainer, HeifDepthImage, HeifAuxiliaryImage, HeifTopLevelImage, UndecodedHeifFile, HeifFile
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import plot_scene
import skimage.transform
import torch


def load_heif_file(filename: str) -> list[HeifFile]:
    container: HeifContainer = pyheif.open_container(filename, apply_transformations=False, convert_hdr_to_8bit=False)
    images: dict[int, Union[UndecodedHeifFile, HeifFile]] = {}
    top_level_image: HeifTopLevelImage
    for top_level_image in container.top_level_images:
        images[top_level_image.id] = top_level_image.image
        # print('top_level_image', top_level_image.id, top_level_image.is_primary)
        depth_image: HeifDepthImage = top_level_image.depth_image
        if depth_image is not None:
            images[depth_image.id] = depth_image.image
            # print('depth_image', depth_image.id)
        auxiliary_image: HeifAuxiliaryImage
        for auxiliary_image in top_level_image.auxiliary_images:
            images[auxiliary_image.id] = auxiliary_image.image
            # print('auxiliary_image', auxiliary_image.id, auxiliary_image.type)
    for v in images.values():
        v.load()
    return list(images.values())


def heifobj2pilobj(heifobj: HeifFile, orientation: int) -> Image.Image:
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

    im = Image.frombytes(heifobj.mode, heifobj.size, heifobj.data, 'raw', heifobj.mode, heifobj.stride)
    return _rotate_pil_im(im, orientation)


def xml2dict(node: ET.Element) -> Union[dict, str]:
    if len(node) == 0:
        return node.text
    result = {}
    for child in node:
        if child.tag not in result:
            result[child.tag] = [xml2dict(child)]
        else:
            result[child.tag].append(xml2dict(child))
    return result


def parse_meta_data(depth_map_metadata_str: str):
    metadata: dict = xml2dict(ET.fromstring(depth_map_metadata_str)[0][0])
    int_min = int(metadata['{http://ns.apple.com/pixeldatainfo/1.0/}IntMinValue'][0])
    int_max = int(metadata['{http://ns.apple.com/pixeldatainfo/1.0/}IntMaxValue'][0])
    float_min = float(metadata['{http://ns.apple.com/pixeldatainfo/1.0/}FloatMinValue'][0])
    float_max = float(metadata['{http://ns.apple.com/pixeldatainfo/1.0/}FloatMaxValue'][0])
    cam_K = metadata['{http://ns.apple.com/depthData/1.0/}IntrinsicMatrix'][0]\
                    ['{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Seq'][0]\
                    ['{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li']
    cam_K = np.array([float(i) for i in cam_K]).reshape([3, 3], order='F')
    cam_K_reference_width = int(metadata['{http://ns.apple.com/depthData/1.0/}IntrinsicMatrixReferenceWidth'][0])
    cam_K_reference_height = int(metadata['{http://ns.apple.com/depthData/1.0/}IntrinsicMatrixReferenceHeight'][0])
    return int_min, int_max, float_min, float_max, cam_K, cam_K_reference_width, cam_K_reference_height


filename = 'test_images/IMG_0979.HEIC'
rgb_image, depth_map, hdr_gain_map = load_heif_file(filename)
for metadata in rgb_image.metadata:
    if metadata['type'] == 'Exif':
        exifdata: dict = exifread.process_file(io.BytesIO(metadata['data'][6:]), details=True)
orientation: int = exifdata['Image Orientation'].values[0]
depth_map_metadata_str: str = depth_map.metadata[0]['data'].decode('utf-8')

int_min, int_max, float_min, float_max, cam_K, *_ = parse_meta_data(depth_map_metadata_str)
depth_map = np.array(heifobj2pilobj(depth_map, orientation))[..., 0]
depth_map = (float_max - float_min) / (int_max - int_min) * (depth_map - int_min) + float_min
depth_map = 1. / depth_map

rgb_image = np.array(heifobj2pilobj(rgb_image, orientation))
H, W, _ = rgb_image.shape
coord_2d = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
coord_2d = np.stack([*coord_2d, np.ones_like(coord_2d[0])], axis=-1)
coord_3d = np.linalg.solve(cam_K, coord_2d[..., None])[..., 0] * skimage.transform.resize(depth_map, (H, W))[..., None]

point_cloud = Pointclouds(points=torch.tensor(coord_3d.reshape([1, -1, 3])),
                          features=torch.tensor(rgb_image.reshape([1, -1, 3]) / 255.))

plot_scene({'Pointcloud': {'person': point_cloud}}).show()

_ = 0
