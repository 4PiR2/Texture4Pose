import torch
import torchvision.transforms.functional as vF


def _down_sample(x: torch.Tensor, sigma: float = 1.) -> torch.Tensor:
    blurred = vF.gaussian_blur(x, kernel_size=[int(3.5 * sigma) * 2 + 1] * 2, sigma=[sigma] * 2)
    return blurred[..., ::2, :][..., ::2]


def _up_sample(x: torch.Tensor, sigma: float = 1.) -> torch.Tensor:
    *C, H, W = x.shape
    unpool = torch.zeros(*C, H * 2, W * 2, device=x.device)
    unpool[..., ::2, ::2] = x
    return vF.gaussian_blur(unpool * 4., kernel_size=[int(3.5 * sigma) * 2 + 1] * 2, sigma=[sigma] * 2)


def _gaussian_pyramids(x: torch.Tensor, n_layers: int, sigma: float = 1.) -> list[torch.Tensor]:
    layers = [x]
    for _ in range(n_layers):
        layers.append(_down_sample(layers[-1], sigma))
    return layers


def _laplacian_pyramids(x: torch.Tensor, n_layers: int, sigma: float = 1.) -> list[torch.Tensor]:
    gaussian_layers = _gaussian_pyramids(x, n_layers, sigma)
    layers = [gaussian_layers[i] - _up_sample(gaussian_layers[i+1], sigma) for i in range(n_layers)]
    layers.append(gaussian_layers[-1])
    return layers


def _reconstruct_from_laplacian_pyramids(layers: list[torch.Tensor], sigma: float = 1.) -> torch.Tensor:
    x = layers[-1]
    for i in range(len(layers) - 1)[::-1]:
        x = _up_sample(x, sigma) + layers[i]
    return x


def laplacian_blend(img_0: torch.Tensor, img_1: torch.Tensor, mask: torch.Tensor, n_layers: int, sigma: float = 1.):
    """
    https://becominghuman.ai/image-blending-using-laplacian-pyramids-2f8e9982077f
    https://docs.opencv.org/3.4/d4/d1f/tutorial_pyramids.html
    :param img_0: [..., H, W]
    :param img_1: [..., H, W]
    :param mask: [..., H, W]
    :param n_layers: int
    :param sigma: float
    :return: [..., H, W]
    """
    laplacian_layers_0 = _laplacian_pyramids(img_0, n_layers, sigma)
    laplacian_layers_1 = _laplacian_pyramids(img_1, n_layers, sigma)
    gaussian_layers_m = _gaussian_pyramids(mask.to(dtype=img_0.dtype), n_layers, sigma)
    layers = []
    for l0, l1, lm in zip(laplacian_layers_0, laplacian_layers_1, gaussian_layers_m):
        layers.append(l0 * (1. - lm) + l1 * lm)
    return _reconstruct_from_laplacian_pyramids(layers, sigma).clamp(min=0., max=1.)


if __name__ == '__main__':
    import utils.io
    from utils.image_2d import visualize

    im0 = utils.io.read_img_file('/data/coco/train2017/000000000009.jpg')[..., :256, :512]
    im1 = utils.io.read_img_file('/data/coco/train2017/000000000025.jpg')[..., :256, :512]
    mask = torch.zeros_like(im0)
    mask[..., :200] = 1.
    x = laplacian_blend(im0, im1, mask, 7, 1.)
    visualize(x)
