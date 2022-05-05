import torch
from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRasterizer, \
    MeshRenderer, SoftPhongShader, BlendParams, DirectionalLights, Materials, HardPhongShader
from pytorch3d.structures import join_meshes_as_scene, join_meshes_as_batch

from utils.SimpleShader import SimpleShader
from utils.const import debug_mode
from utils.io import parse_device


class Scene:
    def __init__(self, objects, cam_K, obj_id, cam_R_m2c, cam_t_m2c, width=640, height=480, device=None, **kwargs):
        self.kwargs = kwargs
        self.objects = objects  # dict
        self.device = parse_device(device)
        self.cam_K = cam_K / cam_K[-1, -1]  # [3, 3]
        self.gt_obj_id = obj_id  # [N]
        self.gt_cam_R_m2c = cam_R_m2c  # [N, 3, 3]
        self.gt_cam_t_m2c = cam_t_m2c  # [N, 3]
        self.width, self.height = width, height

        R = torch.eye(3)[None]
        R[0, 0, 0] = R[0, 1, 1] = -1.
        K = torch.zeros((1, 4, 4))
        K[0, :2, :3] = self.cam_K[:2]
        K[0, 2, 3] = K[0, 3, 2] = 1.
        self.cameras = PerspectiveCameras(R=R, K=K, image_size=((self.height, self.width),),
                                          device=self.device, in_ndc=False)

        self.coor2d = torch.stack(torch.meshgrid(torch.arange(float(self.width)), torch.arange(float(self.height)),
                                                 indexing='xy'), dim=0).to(self.device)  # [2(XY), H, W]
        if debug_mode:
            self.coor2d /= torch.tensor([self.width, self.height]).to(self.device)[..., None, None]
        else:
            coor2d = self.coor2d.permute(1, 2, 0)  # [H, W, 2(XY)]
            coor2d = torch.cat([coor2d, torch.ones_like(coor2d[..., :1])], dim=-1)[..., None]  # [H, W, 3(XY1), 1]
            coor2d = torch.linalg.solve(self.cam_K[None, None], coor2d)  # solve(A, B) == A.inv() @ B
            self.coor2d = coor2d[..., :2, 0].permute(2, 0, 1)  # [2(XY), H, W]

        self.transformed_meshes = [self.objects[int(self.gt_obj_id[i])]
                                       .get_transformed_mesh(self.gt_cam_R_m2c[i], self.gt_cam_t_m2c[i])
                                   for i in range(len(self.gt_obj_id))]
        for i in range(len(self.transformed_meshes)):
            self.transformed_meshes[i].textures = self.objects[int(self.gt_obj_id[i])].get_gt_texture()
        scene_mesh = join_meshes_as_scene(self.transformed_meshes, include_textures=True)

        gt_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cameras,
                                      raster_settings=RasterizationSettings(image_size=(self.height, self.width))),
            shader=SimpleShader(background=0.),
        )

        images = gt_renderer(join_meshes_as_batch([scene_mesh, *self.transformed_meshes], include_textures=True))\
            .permute(0, 3, 1, 2)  # [1+N, H, W, 4(IXYZ)] -> [1+N, 4(IXYZ), H, W]
        coor3d = images[:, 1:4]  # [1+N, 3(XYZ), H, W]
        self.gt_coor3d_vis = coor3d[0]  # [1, 3(XYZ), H, W]
        self.gt_coor3d_obj = coor3d[1:]  # [N, 3(XYZ), H, W]

        masks = images[:, 0].round().to(dtype=torch.uint8)  # [1+N, H, W]
        self.gt_mask = masks[0].bool()[None, None]  # [1, 1, H, W]
        self.gt_mask_vis = torch.stack([masks[0] == obj_id for obj_id in self.gt_obj_id], dim=0)[:, None]
        # [N, 1, H, W]
        self.gt_mask_obj = masks[1:].bool()[:, None]  # [N, 1, H, W]
        self.gt_vis_ratio = self.gt_mask_vis.sum(dim=(1, 2, 3)) / self.gt_mask_obj.sum(dim=(1, 2, 3))  # [N]

        def get_bbox(m):
            # mask: [..., H, W]
            w_mask = m.any(dim=-2)  # [..., W]
            h_mask = m.any(dim=-1)  # [..., H]
            x0 = w_mask.to(dtype=torch.uint8).argmax(dim=-1)  # [...]
            y0 = h_mask.to(dtype=torch.uint8).argmax(dim=-1)  # [...]
            w = w_mask.sum(dim=-1)  # [...]
            h = h_mask.sum(dim=-1)  # [...]
            return torch.stack([x0 + w * .5, y0 + h * .5, w, h], dim=-1)  # [..., 4(XYWH)]

        self.gt_bbox_vis = get_bbox(self.gt_mask_vis)[:, 0]  # [N, 4(XYWH)]
        self.gt_bbox_obj = get_bbox(self.gt_mask_obj)[:, 0]  # [N, 4(XYWH)]

    def render_scene_mesh(self, f=None, ambient=((1.,) * 3,), diffuse=((0.,) * 3,), specular=((0.,) * 3,),
                          direction=((0., 0., -1.),), shininess=64, bg=None):
        for i in range(len(self.transformed_meshes)):
            self.transformed_meshes[i].textures = self.objects[int(self.gt_obj_id[i])].get_texture(f)
        scene_mesh = join_meshes_as_scene(self.transformed_meshes, include_textures=True)

        lights = DirectionalLights(ambient_color=ambient, diffuse_color=diffuse, specular_color=specular,
                                   direction=direction, device=self.device)
        materials = Materials(shininess=shininess, device=self.device)
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.,) * 3)

        raster_settings = RasterizationSettings(
            image_size=(self.height, self.width),
            blur_radius=0.,
            faces_per_pixel=1,
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings,
            ),
            shader=HardPhongShader(
                device=self.device,
                cameras=self.cameras,
                lights=lights,
                materials=materials,
                blend_params=blend_params,
            )
        )

        image = renderer(scene_mesh, include_textures=True).permute(0, 3, 1, 2)[:, :3]  # [1(B), 3(RGB), H, W]
        # image = image.clamp(max=1.)

        if bg is not None:
            # bg [1(B), 3(RGB), H, W] \in [0, 1]
            image = image * self.gt_mask + bg * ~self.gt_mask
        return image
