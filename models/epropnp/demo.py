import torch
import torch.nn as nn

from models.epropnp.epropnp import EProPnP6DoF
from models.epropnp.levenberg_marquardt import LMSolver, RSLMSolver
from models.epropnp.camera import PerspectiveCamera
from models.epropnp.cost_fun import AdaptiveHuberPnPCost


class MonteCarloPoseLoss(nn.Module):
    def __init__(self, init_norm_factor=1.0, momentum=0.1):
        super(MonteCarloPoseLoss, self).__init__()
        self.register_buffer('norm_factor', torch.tensor(init_norm_factor, dtype=torch.float))
        self.momentum = momentum

    def forward(self, pose_sample_logweights, cost_target, norm_factor):
        """
        Args:
            pose_sample_logweights: Shape (mc_samples, num_obj)
            cost_target: Shape (num_obj, )
            norm_factor: Shape ()
        """
        if self.training:
            with torch.no_grad():
                self.norm_factor.mul_(
                    1 - self.momentum).add_(self.momentum * norm_factor)

        loss_tgt = cost_target
        loss_pred = torch.logsumexp(pose_sample_logweights, dim=0)  # (num_obj, )

        loss_pose = loss_tgt + loss_pred  # (num_obj, )
        loss_pose[torch.isnan(loss_pose)] = 0
        loss_pose = loss_pose.mean() / self.norm_factor

        return loss_pose.mean()


class EProPnPDemo(nn.Module):
    def __init__(
            self,
            epropnp=EProPnP6DoF(
                mc_samples=512,
                num_iter=4,
                solver=LMSolver(
                    dof=6,
                    num_iter=10,
                    init_solver=RSLMSolver(
                        dof=6,
                        num_points=8,
                        num_proposals=128,
                        num_iter=5))),
            camera=PerspectiveCamera(),
            cost_fun=AdaptiveHuberPnPCost(
                relative_delta=0.5)):
        super().__init__()
        # Here we use static weight_scale because the data noise is homoscedastic
        self.epropnp = epropnp
        self.camera = camera
        self.cost_fun = cost_fun
        self.mc_loss_fun = MonteCarloPoseLoss()
        # self.log_weight_scale = nn.Parameter(torch.zeros(2))

    @staticmethod
    def forward_w2d(w2d, log_weight_scale):
        """

        :param w2d: [N, V, 2(XY)]
        :param log_weight_scale: [N, 2(XY)]
        :return: [N, V, 2(XY)]
        """
        w2d = (w2d.log_softmax(dim=-2) + log_weight_scale[:, None]).exp()
        # equivalant to:
        #     w2d = w2d.softmax(dim=-2) * log_weight_scale.exp()
        return w2d

    def _compute_loss(self, pose_opt_plus, pose_sample_logweights, cost_tgt, norm_factor, out_pose):
        # monte carlo pose loss
        loss_mc = self.mc_loss_fun(pose_sample_logweights, cost_tgt, norm_factor)

        # derivative regularization
        dist_t = (pose_opt_plus[:, :3] - out_pose[:, :3]).norm(dim=-1)
        beta = 1.0
        loss_t = torch.where(dist_t < beta, 0.5 * dist_t.square() / beta, dist_t - 0.5 * beta)
        loss_t = loss_t.mean()

        dot_quat = (pose_opt_plus[:, None, 3:] @ out_pose[:, 3:, None]).squeeze(-1).squeeze(-1)
        loss_r = (1 - dot_quat.square()) * 2
        loss_r = loss_r.mean()

        loss = loss_mc + 0.1 * loss_t + 0.1 * loss_r
        return loss

    def forward_train(self, x3d, x2d, w2d, log_weight_scale, out_pose):
        N, V, _ = x3d.shape
        w2d = self.forward_w2d(w2d, log_weight_scale)
        self.camera.set_param(torch.eye(3, device=x3d.device).expand(N, -1, -1))
        self.cost_fun.set_param(x2d.detach(), w2d)  # compute dynamic delta
        pose_opt, cost, pose_opt_plus, pose_samples, pose_sample_logweights, cost_tgt = self.epropnp.monte_carlo_forward(
            x3d,
            x2d,
            w2d,
            self.camera,
            self.cost_fun,
            pose_init=out_pose,
            force_init_solve=True,
            with_pose_opt_plus=True)  # True for derivative regularization loss
        norm_factor = log_weight_scale.detach().exp().mean()

        loss = self._compute_loss(pose_opt_plus, pose_sample_logweights, cost_tgt, norm_factor, out_pose)
        return pose_opt, cost, pose_opt_plus, pose_samples, pose_sample_logweights, cost_tgt, norm_factor, loss

    def forward_test(self, x3d, x2d, w2d, log_weight_scale, fast_mode=False):
        N, V, _ = x3d.shape
        w2d = self.forward_w2d(w2d, log_weight_scale)
        self.camera.set_param(torch.eye(3, device=x3d.device).expand(N, -1, -1))
        self.cost_fun.set_param(x2d.detach(), w2d)
        # returns a mode of the distribution
        pose_opt, _, _, _ = self.epropnp(
            x3d, x2d, w2d, self.camera, self.cost_fun,
            fast_mode=fast_mode)  # fast_mode=True activates Gauss-Newton solver (no trust region)
        return pose_opt
        # or returns weighted samples drawn from the distribution (slower):
        #     _, _, _, pose_samples, pose_sample_logweights, _ = self.epropnp.monte_carlo_forward(
        #         x3d, x2d, w2d, self.camera, self.cost_fun, fast_mode=fast_mode)
        #     pose_sample_weights = pose_sample_logweights.softmax(dim=0)
        #     return pose_samples, pose_sample_weights


# # start training
# for epoch_id in range(n_epoch):
#     for iter_id, (batch_in_pose, batch_out_pose) in enumerate(loader):  # for each training step
#         _, _, pose_opt_plus, _, pose_sample_logweights, cost_tgt, norm_factor, loss = model.forward_train(
#             batch_in_pose,
#             batch_out_pose)
#         optimizer.zero_grad()
#         loss.backward()
#
#         grad_norm = []
#         for p in model.parameters():
#             if (p.grad is None) or (not p.requires_grad):
#                 continue
#             else:
#                 grad_norm.append(torch.norm(p.grad.detach()))
#         grad_norm = torch.norm(torch.stack(grad_norm))
#
#         optimizer.step()
#
#         print(
#             'Epoch {}: {}/{} - loss_mc={:.4f}, loss_t={:.4f}, loss_r={:.4f}, loss={:.4f}, norm_factor={:.4f}, grad_norm={:.4f}'.format(
#                 epoch_id + 1, iter_id + 1, len(loader), loss_mc, loss_t, loss_r, loss, norm_factor, grad_norm))
#
# # batch inference
# with torch.no_grad():
#     pose_opt = model.forward_test(batch_in_pose_test, batch_cam_mats)
#
# # evaluation
# dist_t = (pose_opt[:, :3] - batch_in_pose_test[:, :3]).norm(dim=-1)
# dot_quat = (pose_opt[:, None, 3:] @ batch_in_pose_test[:, 3:, None]).squeeze(-1).squeeze(-1)
# dist_theta = 2 * torch.acos(dot_quat.abs())
# print('Mean Translation Error: {:4f}'.format(dist_t.mean()))
# print('Mean Orientation Error: {:4f}'.format(dist_theta.mean()))
