import os

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import utils.image_2d
import utils.io


w, h = 640, 480

x1, y1 = 0, 0
x2, y2 = 1000 * .6, 250 * .6
x3, y3 = 150 * .6, 800 * .6

u1, v1 = 320, 128
u2, v2 = 320 - 128, 128 * (1 + 3 ** .5)
u3, v3 = 320 + 128, 128 * (1 + 3 ** .5)

fname = 'definition_of_uv_texture.png'

print(os.getcwd())


def barycentric_coordinates(x, y):
    denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    l1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denominator
    l2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denominator
    l3 = 1. - (l1 + l2)
    return l1, l2, l3


x, y = np.meshgrid(np.arange(w), np.arange(h))
l1, l2, l3 = barycentric_coordinates(x, y)

mask = (l1 >= 0.) & (l2 >= 0.) & (l3 >= 0.)

u = l1 * u1 + l2 * u2 + l3 * u3
v = l1 * v1 + l2 * v2 + l3 * v3

texture = utils.io.read_img_file('/data/coco/train2017/000000000009.jpg').double()

u = u * 2. / w - 1.
v = v * 2. / h - 1.
face_img = F.grid_sample(texture, torch.tensor([u, v]).permute(1, 2, 0)[None])
# utils.image_2d.visualize(face_img)
face_img = face_img[0].permute(1, 2, 0).detach().cpu().numpy()
color = np.round(np.concatenate([face_img * 255., np.full_like(face_img[..., :1], 255)], axis=-1)).astype(np.uint8)
color[~mask, -1] = 0


def save_image(data, filename):
    sizes = np.shape(data)
    fig = plt.figure()
    fig.set_size_inches(1, sizes[0] / sizes[1], forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data)
    plt.savefig(filename, transparent=True, dpi=sizes[1])
    plt.show()
    plt.close()


save_image(color, fname)

_ = 0
