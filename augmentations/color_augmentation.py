import torch

import utils.color


def match_background_mean(img: torch.Tensor, mask: torch.Tensor, blend_saturation: float = 1.,
                          blend_light: float = 1.) -> torch.Tensor:
    """

    :param img: [N, 3(RGB), H, W] \in [0, 1]
    :param mask: [N, 1, H, W]
    :param blend_saturation: float, 0. -> keep original, 1. -> fully adapted
    :param blend_light: float, 0. -> keep original, 1. -> fully adapted
    :return: [N, 3(RGB), H, W] \in [0, 1]
    """
    N, _, H, W = img.shape
    img_hsl = utils.color.rgb2hsl(img)  # [N, 3(HSL), H, W]
    mask_count = mask.sum(dim=[-2, -1])  # [N, 1]
    bg_mean = (img_hsl * ~mask).sum(dim=[-2, -1]) / (H * W - mask_count)  # [N, 3(HSL)]
    obj_mean = (img_hsl * mask).sum(dim=[-2, -1]) / mask_count  # [N, 3(HSL)]
    mean_diff = bg_mean - obj_mean
    mean_diff[:, 0] = 0.
    mean_diff[:, 1] *= blend_saturation
    mean_diff[:, 2] *= blend_light
    img_hsl_obj_adapted = ((img_hsl + mean_diff[..., None, None]) * mask + img_hsl * ~mask).clamp(min=0., max=1.)
    # mean values might not be the same after clamping
    return utils.color.hsl2rgb(img_hsl_obj_adapted)


def match_cdf(source: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    """
    skimage.exposure.match_histograms
    skimage.exposure.histogram_matching._match_cumulative_cdf
    source: [...]
    template: [...]
    return: [...]
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    src_values, src_unique_indices, src_counts = source.unique(return_inverse=True, return_counts=True)
    tmpl_values, tmpl_counts = template.unique(return_counts=True)

    # calculate normalized quantiles for each array
    src_quantiles = src_counts.cumsum(dim=0) / source.numel()
    tmpl_quantiles = tmpl_counts.cumsum(dim=0) / template.numel()

    def interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
        """
        np.interp
        :param x: [n], x
        :param xp: [N], x_ref
        :param fp: [N], y_ref
        :return: [n], y
        """
        idx_r = torch.searchsorted(xp, x)
        idx_l = (idx_r - 1).relu()
        idx_r = idx_r.clamp(max=len(xp) - 1)
        x_l, x_r = xp[idx_l], xp[idx_r]
        y_l, y_r = fp[idx_l], fp[idx_r]
        y = (y_r * (x - x_l) + y_l * (x_r - x)) / (x_r - x_l)
        m = y.isnan()
        y[m] = y_l[m]
        return y

    interp_values = interp(src_quantiles, tmpl_quantiles, tmpl_values)
    result = interp_values[src_unique_indices].reshape(source.shape)

    if source.requires_grad:
        def get_pdf(values: torch.Tensor, counts: torch.Tensor, bins: int = None) -> (torch.Tensor, torch.Tensor):
            if bins is None:
                # exact calculation
                dy = counts / counts.sum()
                dx = values.diff(prepend=torch.full_like(values[:1], -torch.inf))
                pdf = dy / dx
                bin_edges = torch.cat([torch.full_like(values[:1], -torch.inf), values], dim=0)
            else:
                # smoothed
                device = values.device
                dtype = values.dtype
                if device == torch.device('cpu'):
                    # torch.histogram does not support CUDA https://github.com/pytorch/pytorch/issues/69519
                    pdf, bin_edges = values.histogram(bins=bins, weight=counts.to(dtype=dtype), density=True)
                else:
                    values = values.repeat_interleave(counts)
                    pdf = values.histc(bins=bins) * (bins / ((values[-1] - values[0]) * len(values) + 1e-6))
                    bin_edges = torch.linspace(float(values[0]), float(values[-1]), bins + 1, dtype=dtype,
                                               device=device)
            return pdf, bin_edges

        n_bins = 100
        src_hist, src_bin_edges = get_pdf(src_values, src_counts, n_bins)
        tmpl_hist, tmpl_bin_edges = get_pdf(tmpl_values, tmpl_counts, n_bins)
        src_pdf = interp(src_values, (src_bin_edges[:-1] + src_bin_edges[1:]) * .5, src_hist)
        interp_pdf = interp(interp_values, (tmpl_bin_edges[:-1] + tmpl_bin_edges[1:]) * .5, tmpl_hist)
        derivative = (src_pdf / interp_pdf).clamp(max=100.)  # prevent div 0 error or gradient explosion

        # import matplotlib.pyplot as plt
        # plt.plot(src_values.detach().cpu().numpy(), src_quantiles.detach().cpu().numpy(), label='src')
        # plt.plot(tmpl_values.detach().cpu().numpy(), tmpl_quantiles.detach().cpu().numpy(), label='tmpl')
        # plt.plot(interp_values.detach().cpu().numpy(), src_quantiles.detach().cpu().numpy(), label='interp')
        # plt.legend()
        # plt.show()
        # plt.plot(src_values.detach().cpu().numpy(), src_pdf.detach().cpu().numpy(), label='src')
        # plt.plot(src_values.detach().cpu().numpy(), interp_pdf.detach().cpu().numpy())
        # plt.plot(interp_values.detach().cpu().numpy(), interp_pdf.detach().cpu().numpy(), label='interp')
        # plt.legend()
        # plt.show()
        # plt.plot(src_values.detach().cpu().numpy(), derivative.detach().cpu().numpy())
        # plt.show()

        D = derivative[src_unique_indices].reshape(source.shape)
        result = result.detach() + D.detach() * source - (D * source).detach()

    return result


def match_background_histogram(img: torch.Tensor, mask: torch.Tensor, blend_saturation: float = 1.,
                               blend_light: float = 1., p: float = 1.) -> torch.Tensor:
    """
    make the object's histogram the same as background's
    https://albumentations.ai/docs/api_reference/full_reference/#albumentations.augmentations.domain_adaptation.HistogramMatching
    https://en.wikipedia.org/wiki/Histogram_equalization
    :param img: [N, 3(RGB), H, W] \in [0, 1]
    :param mask: [N, 1, H, W]
    :param blend_saturation: float, 0. -> keep original, 1. -> fully adapted
    :param blend_light: float, 0. -> keep original, 1. -> fully adapted
    :param p: float
    :return: [N, 3(RGB), H, W] \in [0, 1]
    """

    def per_image(img_hsl: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """

        :param img_hsl: [3(HSL), H, W]
        :param m: [H, W]
        :return: [3(HSL), H, W]
        """
        def per_channel(img_channel: torch.Tensor) -> torch.Tensor:
            """
            :param img_channel: [H, W]
            :return: [H, W]
            """
            img_channel = img_channel.clone()
            img_channel[m] = match_cdf(img_channel[m], img_channel[~m])
            return img_channel

        img_hsl = img_hsl.clone()
        if blend_saturation and torch.rand(1) <= p:
            img_hsl[1] = per_channel(img_hsl[1]) * blend_saturation + img_hsl[1] * (1. - blend_saturation)
        if blend_light and torch.rand(1) <= p:
            img_hsl[2] = per_channel(img_hsl[2]) * blend_light + img_hsl[2] * (1. - blend_light)
        return img_hsl

    N, _, H, W = img.shape
    hsl = utils.color.rgb2hsl(img)  # [N, 3(HSL), H, W]
    return utils.color.hsl2rgb(torch.stack([per_image(i, m[0]) for i, m in zip(hsl, mask)], dim=0))


def iso_noise(img: torch.Tensor, color_shift: float = .05, intensity: float = .5) -> torch.Tensor:
    """
    Apply poisson noise to image to simulate camera sensor noise.
    https://github.com/albumentations-team/albumentations/blob/8e958a324cb35d3adf13c03b180b3dc066ef21d5/albumentations/augmentations/transforms.py#L1406
    Args:
        img: [N, 3(RGB), H, W] \in [0, 1]
        color_shift (float):
        intensity (float): Multiplication factor for noise values. Values of ~0.5 are produce noticeable,
                   yet acceptable level of noise.
    Returns:
        [N, 3(RGB), H, W] \in [0, 1]
    """
    N, _, H, W = img.shape
    hsl = utils.color.rgb2hsl(img)
    result = hsl.clone()  # to avoid 'in-place operation' error during back propagation
    stddev = hsl.std(dim=[-2, -1], unbiased=False)[:, 2, None, None]  # [N, 1, 1]
    luminance_noise = torch.poisson((stddev * intensity * 255.).expand(-1, H, W)) / 255.  # [N, H, W]
    result[:, 2] = (hsl[:, 2] + luminance_noise * (1. - hsl[:, 2])).clamp(min=0., max=1.)
    hue_color_noise = torch.randn_like(hsl[:, 0]) * color_shift * intensity  # [N, H, W]
    result[:, 0] = (hsl[:, 0] + hue_color_noise) % 1.
    return utils.color.hsl2rgb(result)


# if __name__ == '__main__':
#     import utils.io
#     from utils.image_2d import visualize
#
#     im0 = utils.io.read_img_file('/data/coco/train2017/000000000009.jpg')
#     # im0 = utils.io.read_img_file('/data/lm/test/000001/rgb/000200.png')
#     im1 = utils.io.read_img_file('/data/lm/train/000001/rgb/000200.png')
#     mask = utils.io.read_img_file('/data/lm/train/000001/mask_visib/000200_000000.png')[:, :1].bool()
#     im = im0 * ~mask + im1 * mask
#     # x = match_background_mean(im, mask, 1., 0.)
#     x = match_background_histogram(im.cuda(), mask.cuda(), 1., 1., 1.)
#     visualize(im0)
#     visualize(x)
