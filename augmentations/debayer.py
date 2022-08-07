# https://gist.github.com/SebastianGrans/888e576e0d39e2f968eae2f6fa05f2db
# This only works for BayerRG color filter arrays. I.e. RGGB
# Inspired by this: https://github.com/cheind/pytorch-debayer
# I just wanted a minimal implementation to understand it better.
# https://en.wikipedia.org/wiki/Demosaicing

import torch
import torch.nn.functional as F


def _get_bayer_img(img: torch.Tensor, permute_channel: bool = False):
    """
    [[R, G],
     [G, B]]
    # https://github.com/rasmushaugaard/surfemb/blob/53e1852433a3b2b84fedc7a3a01674fe1b6189cc/surfemb/data/tfms.py#L44
    :param img: [..., 3(RGB), H, W]
    :param permute_channel: bool
    :return: [..., 1, H, W], [3(RGB)]
    """
    device = img.device
    arange = torch.arange(3, device=device)
    if permute_channel:
        # permute channels before bayering/debayering to cover different bayer formats
        channel_idxs = torch.randperm(3, device=device)
    else:
        channel_idxs = arange
    channel_idxs_inv = torch.empty_like(channel_idxs)
    channel_idxs_inv[channel_idxs] = arange

    # assemble bayer image
    img_bayer = torch.empty_like(img[..., :1, :, :])  # [..., 1, H, W]
    img_bayer[..., ::2, ::2] = img[..., channel_idxs[0], ::2, ::2]
    img_bayer[..., 1::2, ::2] = img[..., channel_idxs[1], 1::2, ::2]
    img_bayer[..., ::2, 1::2] = img[..., channel_idxs[1], ::2, 1::2]
    img_bayer[..., 1::2, 1::2] = img[..., channel_idxs[2], 1::2, 1::2]
    return img_bayer, channel_idxs_inv


def _debayer(img_bayer: torch.Tensor) -> torch.Tensor:
    """
    demosaicking
    :param img_bayer: [N, 1, H, W]
    :return: [N, 3(RGB), H, W]
    """
    dtype = img_bayer.dtype
    device = img_bayer.device
    N, _, H, W = img_bayer.shape

    kernels = torch.tensor([
        [[[0., 0., 0.],
          [0., 1., 0.],
          [0., 0., 0.]]],
        [[[0., .25, 0.],
          [.25, 0, .25],
          [0., .25, 0.]]],
        [[[.25, 0., .25],
          [0., 0., 0.],
          [.25, 0., .25]]],
        [[[0., 0., 0.],
          [.5, 0., .5],
          [0., 0., 0.]]],
        [[[0., .5, 0.],
          [0., 0., 0.],
          [0., .5, 0.]]],
    ], dtype=dtype, device=device)  # [5, 1, 3, 3]

    indices = torch.tensor([[
        # dest channel r
        [[0, 3],
         [4, 2]],
        # dest channel g
        [[1, 0],
         [0, 1]],
        # dest channel b
        [[2, 4],
         [3, 0]],
    ]], device=device)  # [1, 3(RGB), 2, 2]
    channel_selection_map = indices.repeat(1, 1, H // 2, W // 2).expand(N, -1, -1, -1)  # [N, 3(RGB), H, W]

    # With a 3x3 kernel we need to pad the boundry. Here we pad
    # by simply replicating the pixels.
    # img_torch thus has the shape (1, 1, W+2, H+2)
    img_bayer = F.pad(img_bayer, (1, 1, 1, 1), mode='replicate')

    # Apply the convolution resulting in the output, c, to have
    # the shape (1, 5, W, H)
    img_convolved = F.conv2d(img_bayer, kernels)

    # Now we need to get the final image by, for each pixel,
    # select the appropriate channels from dimension 1 of the tensor.
    return torch.gather(img_convolved, -3, channel_selection_map).clamp(min=0., max=1.)  # [N, 3(RGB), H, W]


def debayer_aug(img: torch.Tensor, permute_channel: bool = True) -> torch.Tensor:
    """
    # debayer_method = np.random.choice((cv2.COLOR_BAYER_BG2BGR, cv2.COLOR_BAYER_BG2BGR_EA))
    # debayered = cv2.cvtColor(bayer, debayer_method)[..., channel_idxs_inv]
    :param img: [N, 3(RGB), H, W]
    :param permute_channel: bool
    :return: [N, 3(RGB), H, W]
    """
    img_bayer, channel_idxs_inv = _get_bayer_img(img, permute_channel)
    return _debayer(img_bayer)[..., channel_idxs_inv, :, :]
