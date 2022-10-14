import os

import numpy as np
from matplotlib import pyplot as plt


w, h = 1000, 800

x1, y1 = 0, 0
x2, y2 = 1000, 250
x3, y3 = 150, 800

r1, g1, b1 = 255, 255, 0
r2, g2, b2 = 0, 255, 255
r3, g3, b3 = 255, 0, 255


fname = 'definition_of_vertex_texture.png'

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

r = l1 * r1 + l2 * r2 + l3 * r3
g = l1 * g1 + l2 * g2 + l3 * g3
b = l1 * b1 + l2 * b2 + l3 * b3

color = np.round(np.stack([r, g, b, np.full_like(r, 255)], axis=-1)).astype(np.uint8)
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
