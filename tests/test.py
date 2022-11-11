import math
import torch
import torch.nn.functional as F
from torchvision.models import resnet18


def bnn_test0(pred: torch.Tensor, gt: torch.Tensor):
    # pred [uX, uY, uZ, vX, vY, vZ, pXY, pXZ, pYZ]
    N, _, H, W = pred.shape
    pred = pred.permute(0, 2, 3, 1)  # [N, H, W, 9]
    gt = gt.permute(0, 2, 3, 1)  # [N, H, W, 3]
    mu, log_var_3d, tan_corr_3d = pred.split(split_size=3, dim=-1)  # [N, H, W, 3]
    sigma = ((log_var_3d[..., None, :] + log_var_3d[..., None]) * .5).exp()  # [N, H, W, 3, 3]
    corr_xy, corr_xz, corr_yz = tan_corr_3d.tanh().unbind(dim=-1)  # [N, H, W]
    corr = torch.ones_like(sigma)  # [N, H, W, 3, 3]
    corr[..., 1, 0] = corr[..., 0, 1] = corr_xy
    corr[..., 2, 0] = corr[..., 0, 2] = corr_xz
    corr[..., 2, 1] = corr[..., 1, 2] = corr_yz
    sigma *= corr
    diff = gt - mu  # [N, H, W, 3]
    sigma_det = torch.linalg.det(sigma)
    neg_log_p = ((diff[..., None, :] @ torch.linalg.inv(sigma) @ diff[..., None])[..., 0, 0] + sigma_det.log())[:, None]
    # [N, 1, H, W]
    mode = (torch.pi * 2.) ** -1.5 * sigma_det ** -.5
    return neg_log_p, mode


def bnn_test(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor = None, l: int = 2):
    # pred [uX, uY, uZ, vX, vY, vZ]
    mu, log_var = pred.split(split_size=3, dim=-3)  # [N, 3, H, W]
    neg_log_p = (gt - mu).abs() ** l * (-log_var).exp() + log_var  # [N, 3, H, W]
    log_mode = -log_var.sum(dim=-3, keepdim=True) / l  # [N, 1, H, W]
    if mask is not None:
        loss = ((neg_log_p * mask).sum(dim=[-3, -2, -1]) / mask.sum(dim=[-3, -2, -1])).mean()
    else:
        loss = neg_log_p.mean() * 3.
    weight_map = F.softmax(log_mode.reshape(len(log_mode), -1), dim=-1).reshape(log_mode.shape)  # [N, 1, H, W]
    return loss, weight_map

bnn_test(torch.randn(5, 6, 13, 17), torch.randn(5, 3, 13, 17))
bnn_test(torch.tensor([0., 1, 2, 3, 4, 5, .6, .7, .8]).reshape(1, -1, 1, 1), torch.tensor([-1., -2, -3]).reshape(1, -1, 1, 1))



model = resnet18(pretrained=True)	# 加载模型
optimizer = torch.optim.SGD(params=[	# 初始化优化器，并设置两个param_groups
    {'params': model.layer2.parameters()},
    {'params': model.layer3.parameters(), 'lr':0.2},
], lr=0.1)	# base_lr = 0.1

# 设置warm up的轮次为100次
warm_up_iter = 10
T_max = 50	# 周期
r_max = 10	# 最大值
r_min = 1	# 最小值
anneal_iter = 20


def f(cur_iter):
    if cur_iter < warm_up_iter:
        rate = cur_iter / warm_up_iter
    elif cur_iter >= anneal_iter:
        rate = .5 + .5 * math.cos((cur_iter - anneal_iter) / (T_max - anneal_iter) * math.pi)
    else:
        rate = 1.
    return rate


# # 为param_groups[0] (即model.layer2) 设置学习率调整规则 - Warm up + Cosine Anneal
# lambda0 = lambda cur_iter: cur_iter / warm_up_iter if  cur_iter < warm_up_iter else \
#         (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))/0.1

#  param_groups[1] 不进行调整
lambda1 = lambda cur_iter: 1

# LambdaLR
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[f, lambda1])

x = list(range(50))
y0 = []
y1 = []
for epoch in x:
    y0i, y1i = optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr']
    y0.append(y0i)
    y1.append(y1i)
    optimizer.step()
    scheduler.step()

from matplotlib import pyplot as plt

plt.plot(x, y0)
plt.plot(x, y1)
plt.show()
