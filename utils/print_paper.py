from typing import Union

from matplotlib import pyplot as plt
import numpy as np
import torch


def meter2px(meter: float, dpi: int = 72, rounding: bool = True) -> Union[int, float]:
    px = meter * dpi / .0254
    return round(px) if rounding else px


def px2meter(px: Union[int, float], dpi: int = 72) -> float:
    return px * .0254 / dpi


def get_paper_size(unit_or_dpi: Union[str, int], paper_size: str = 'a4') -> tuple[Union[int, float], Union[int, float]]:
    if paper_size.lower() == 'a4':
        w, h = 210, 297
    elif paper_size.lower() == 'a3':
        w, h = 297, 420
    else:
        raise NotImplementedError
    if unit_or_dpi == 'mm':
        return w, h
    elif unit_or_dpi == 'inch':
        return w / 25.4, h / 25.4
    elif isinstance(unit_or_dpi, int):
        return meter2px(w * 1e-3, unit_or_dpi), meter2px(h * 1e-3, unit_or_dpi)


def get_ax_paper(dpi: int = 72, paper_size: str = 'a4') -> plt.Axes:
    fig = plt.figure(figsize=get_paper_size('inch', paper_size), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_axis_off()
    return ax


def tensor_to_ndarray_rgba(img: Union[torch.Tensor, np.ndarray]):
    if isinstance(img, torch.Tensor):
        if img.dim() == 4:
            img = img[0]
        elif img.dim() == 2:
            img = img.expand(3, -1, -1)
        img = img.permute(1, 2, 0).detach().cpu().numpy()
    elif isinstance(img, np.ndarray):
        if img.ndim == 2:
            img = img[..., None].repeat(3, axis=-1)
        img[..., :3] = img[..., 2::-1]  # BGR2RGB
    if img.dtype != np.uint8:
        img = (img * 255.).astype('uint8')
    if img.shape[-1] < 4:
        img = np.concatenate([img, np.full_like(img[..., :1], 255)], axis=-1)
    return img


def plt_save_pdf(path: str, ax: plt.Axes = None, fig: plt.Figure = None):
    if ax is not None:
        fig = ax.get_figure()
    if fig is not None:
        fig.savefig(path, format='pdf')
    else:
        plt.savefig(path, format='pdf')


def print_tensor_to_paper_pdf(img: Union[torch.Tensor, np.ndarray], path: str, interpolation: str = 'antialiased',
                              dpi: int = 72, paper_size: str = 'a4'):
    # interpolation: Supported values are
    # 'none', 'antialiased', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite',
    # 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos', 'blackman'.
    img = tensor_to_ndarray_rgba(img)
    h, w = img.shape[:2]
    W, H = get_paper_size(dpi, paper_size)
    im = np.full([H, W, 4], 255, dtype=np.uint8)
    im[(H - h) // 2:(H + h) // 2, (W - w) // 2:(W + w) // 2] = img
    ax = get_ax_paper(dpi, paper_size)
    ax.imshow(im, interpolation=interpolation)
    # Matplotlib's linewidth is in points, 1 inch = 72 point, default dpi = 72, 1 point = 0.352778 mm
    # https://stackoverflow.com/questions/57657419/how-to-draw-a-figure-in-specific-pixel-size-with-matplotlib
    plt_save_pdf(path, ax)
    plt.close(ax.get_figure())


def make_grid(img: torch.Tensor, size: tuple[int, int], margin: float = .05):
    *C, H, W = img.shape
    n_h, n_w = size
    margin_h, margin_w = round(H * margin), round(W * margin)
    img_grid = torch.ones(*C, H * n_h + margin_h * (n_h - 1), W * n_w + margin_w * (n_w - 1))
    for i in range(n_h):
        for j in range(n_w):
            img_grid[..., (H + margin_h) * i: (H + margin_h) * i + H, (W + margin_w) * j: (W + margin_w) * j + W] = img
    return img_grid
