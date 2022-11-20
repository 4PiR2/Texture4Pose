import os
import pickle

import matplotlib
import matplotlib.colors as colors
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from utils.tb_reader import read_tb_log

# https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html


def heatmap(data, row_ticklabels, col_ticklabels, ax=None,
            x_label=None, y_label=None, title=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_ticklabels
        A list or array of length M with the labels for the rows.
    col_ticklabels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, norm=colors.LogNorm(vmin=0.05, vmax=1.0), **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    cbar_ticks = [.05, .1, .2, .4, .8]
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f'{i:.2f}'.lstrip('0') for i in cbar_ticks])

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_ticklabels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_ticklabels)

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        # ax.set_ylabel(y_label, rotation=0)
        ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title, fontsize=10)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2, axis='y')
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def save_pkl(oid: int, is_siren: bool = True):
    if is_siren:
        root_dir = f'/data/{oid}_siren_omega_scan'
    else:
        root_dir = f'/data/{oid}_cb_cycle_scan'

    versions = [int(d.split('_')[-1]) for d in os.listdir(root_dir)]
    versions.sort()

    event_log_paths = []

    for version in versions:
        version_root_dir = os.path.join(root_dir, f'version_{version}')
        list_dir = os.listdir(version_root_dir)
        for d in list_dir:
            if d.startswith('events.out.tfevents.'):
                event_log_paths.append(os.path.join(version_root_dir, d))

    values = []
    for event_log_path in event_log_paths:
        df = read_tb_log(event_log_path)
        v = df.loc[df['metric'] == 'val_metric']['value'].to_numpy()
        values.append(v)

    values = np.array(values)
    with open(f'plots/freq_study/{oid}_{"siren" if is_siren else "cb"}_val_metric.pkl', 'wb') as f:
        pickle.dump(values, f)


def load_pkl(oid: int, is_siren: bool = True):
    with open(f'plots/freq_study/{oid}_{"siren" if is_siren else "cb"}_val_metric.pkl', 'rb') as f:
        return pickle.load(f)


# oid = 105
# is_siren = True
# data = load_pkl(oid, is_siren)

data_101 = load_pkl(101, False)
data_104 = load_pkl(104, False)
data_105 = load_pkl(105, False)
data_0 = np.zeros_like(data_101[:1])

data = np.concatenate([data_101[:5], data_104[:5], data_105[:5]])

fig, ax = plt.subplots()
fig.set_size_inches(5, 5)

im, cbar = heatmap(
    data,
    # [f'{2 ** i}' if i >= 0 else f'1/{2 ** -i}' for i in [6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6]],
    [f'Sphere {2 ** i:02}'.replace('0', '  ') for i in range(5)] + [f'H. Cylinder {2 ** i:02}'.replace('0', '  ') for i in range(5)] + [f'Sphericon {2 ** i:02}'.replace('0', '  ') for i in range(5)],
    [500 * i for i in range(1, 11)],
    ax=ax,
    x_label=None,
    # y_label='Parameter $\omega_*$',
    y_label='Parameter $\omega_\mathrm{cb}$',
    title='Optimization Steps',
    cmap="Oranges",
    cbarlabel='3D Loss $L^\mathrm{3D}$'
)
texts = annotate_heatmap(im, valfmt=lambda x, _: f'{x:.2f}'.lstrip('0'))

fig.patch.set_alpha(0.)
ax.patch.set_alpha(0.)
fig.tight_layout()
# plt.savefig(f'plots/freq_study/{oid}_{"siren" if is_siren else "cb"}.pdf', bbox_inches='tight', pad_inches=.1)
# plt.savefig(f'plots/freq_study/{oid}_{"siren" if is_siren else "cb"}.svg', bbox_inches='tight', pad_inches=.1)
plt.savefig(f'plots/freq_study/cb.pdf', bbox_inches='tight', pad_inches=.1, transparent=True)
plt.savefig(f'plots/freq_study/cb.svg', bbox_inches='tight', pad_inches=.1, transparent=True)
plt.show()
a = 0
