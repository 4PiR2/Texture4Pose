import torch
import torch.nn.functional as F
from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRasterizer, \
    MeshRenderer, SoftPhongShader, AmbientLights, BlendParams
from pytorch3d.structures import join_meshes_as_scene, join_meshes_as_batch

from utils.SimpleShader import SimpleShader
from utils.const import debug_mode


class Scene:
    def __init__(self, obj_set, cam_K, obj_id, cam_R_m2c, cam_t_m2c, width, height, **kwargs):
        self.kwargs = kwargs
        self.obj_set = obj_set
        self.device = obj_set.device
        self.gt_cam_K = cam_K  # [3, 3]
        self.gt_obj_id = obj_id  # [N]
        self.gt_cam_R_m2c = cam_R_m2c  # [N, 3, 3]
        self.gt_cam_t_m2c = cam_t_m2c  # [N, 3]
        self.width, self.height = width, height

        R = torch.eye(3)[None]
        R[0, 0, 0] = R[0, 1, 1] = -1.
        K = torch.zeros((1, 4, 4))
        K[0, :2, :3] = self.gt_cam_K[:2]
        K[0, 2, 3] = K[0, 3, 2] = 1.
        self.cameras = PerspectiveCameras(R=R, K=K, image_size=((self.height, self.width),),
                                          device=self.device, in_ndc=False)


        self.coor2d = torch.stack(torch.meshgrid(torch.arange(float(self.width)), torch.arange(float(self.height)),
                                                 indexing='xy'), dim=0).to(self.device)
        if debug_mode:
            self.coor2d /= torch.tensor([self.width, self.height]).to(self.device)[..., None, None]

        self.transformed_meshes = [self.obj_set.objects[int(self.gt_obj_id[i])]
                                       .get_transformed_mesh(self.gt_cam_R_m2c[i], self.gt_cam_t_m2c[i])
                                   for i in range(len(self.gt_obj_id))]
        for i in range(len(self.transformed_meshes)):
            self.transformed_meshes[i].textures = self.obj_set.objects[int(self.gt_obj_id[i])].get_gt_texture()
        scene_mesh = join_meshes_as_scene(self.transformed_meshes, include_textures=True)

        gt_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cameras,
                                      raster_settings=RasterizationSettings(image_size=(self.height, self.width))),
            shader=SimpleShader(background=0.),
        )

        images = gt_renderer(join_meshes_as_batch([scene_mesh, *self.transformed_meshes], include_textures=True))\
            .permute(0, 3, 1, 2)  # [1+N, H, W, 4(IXYZ)] -> [1+N, 4(IXYZ), H, W]
        masks = images[:, 0].round().to(dtype=torch.uint8)  # [1+N, H, W]
        coor3d = images[:, 1:4]  # [1+N, 3(XYZ), H, W]
        self.gt_coor3d_vis = coor3d[0]  # [1, 3(XYZ), H, W]
        self.gt_coor3d_obj = coor3d[1:]  # [N, 3(XYZ), H, W]

        def get_bbox(m):
            # mask: [..., H, W]
            w_mask = m.any(dim=-2)  # [..., W]
            h_mask = m.any(dim=-1)  # [..., H]
            x0 = w_mask.to(dtype=torch.uint8).argmax(dim=-1)  # [...]
            y0 = h_mask.to(dtype=torch.uint8).argmax(dim=-1)  # [...]
            w = w_mask.sum(dim=-1)  # [...]
            h = h_mask.sum(dim=-1)  # [...]
            return torch.stack([x0 + w * .5, y0 + h * .5, w, h], dim=-1)  # [..., 4(XYWH)]

        self.gt_mask = masks[0].bool()[None, None]  # [1, 1, H, W]
        self.gt_mask_vis = torch.stack([masks[0] == obj_id for obj_id in self.gt_obj_id], dim=0)[:, None]
        # [N, 1, H, W]
        self.gt_bbox_vis = get_bbox(self.gt_mask_vis)[:, 0]  # [N, 4(XYWH)]
        self.gt_mask_obj = masks[1:].bool()[:, None]  # [N, 1, H, W]
        self.gt_bbox_obj = get_bbox(self.gt_mask_obj)[:, 0]  # [N, 4(XYWH)]

        self.images = None

    def render_scene_mesh(self, f=None, bg=None, num_images=1):
        for i in range(len(self.transformed_meshes)):
            self.transformed_meshes[i].textures = self.obj_set.objects[int(self.gt_obj_id[i])].get_texture(f)
        scene_mesh = join_meshes_as_scene(self.transformed_meshes, include_textures=True)

        lights = AmbientLights(device=self.device)
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0., 0., 0.))

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
            shader=SoftPhongShader(
                device=self.device,
                cameras=self.cameras,
                lights=lights,
                blend_params=blend_params,
            )
        )

        self.images = renderer(scene_mesh, include_textures=True).permute(0, 3, 1, 2)[:, :3]  # [B, 3(RGB), H, W]

        if bg is not None:
            # bg [B or 1, 3(RGB), H, W] \in [0, 1]
            self.images = self.images * self.gt_mask + bg * ~self.gt_mask
        return self.images

    def get_data(self, bbox, output_size):
        # bbox [N, 4(XYWH)]
        # output_size int, output is [N, C, output_size x output_size] image
        # imgs: list of [N, C, H, W] or [C, H, W] images
        crop_size, _ = bbox[:, 2:].max(dim=-1)
        max_crop_size = int(crop_size.max())
        pad_size = (max_crop_size + 1) // 2
        x0, y0 = (bbox[:, :2].T - crop_size * .5).int() + pad_size
        crop_size = crop_size.int()
        x1, y1 = x0 + crop_size, y0 + crop_size

        def crop(img):
            padded_img = torch.zeros(
                [*img.shape[:-2], img.shape[-2] + max_crop_size, img.shape[-1] + max_crop_size],
                dtype=img.dtype, device=img.device)
            padded_img[..., pad_size:pad_size + img.shape[-2], pad_size:pad_size + img.shape[-1]] = img
            c_imgs = []
            for i in range(len(bbox)):
                if img.dim() > 3:  # img [N, C, H, W]
                    c_img = padded_img[i:i + 1, ..., y0[i]:y1[i], x0[i]:x1[i]]
                else:  # img [C, H, W]
                    c_img = padded_img[None, ..., y0[i]:y1[i], x0[i]:x1[i]]
                if padded_img.dtype in [torch.uint8]:
                    c_img = F.interpolate(c_img, size=output_size, mode='nearest')
                else:
                    c_img = F.interpolate(c_img, size=output_size, mode='bilinear', align_corners=True)
                c_imgs.append(c_img)  # c_img [1, C, H, W]
            return torch.cat(c_imgs, dim=0)  # [N, C, H, W]

        # F.interpolate doesn't support bool
        result = {'gt_obj_id': self.gt_obj_id, 'gt_cam_K': self.gt_cam_K,
                  'gt_cam_R_m2c': self.gt_cam_R_m2c, 'gt_cam_t_m2c': self.gt_cam_t_m2c,
                  'gt_coor2d': crop(self.coor2d), 'gt_coor3d': crop(self.gt_coor3d_obj),
                  'gt_mask_vis': crop(self.gt_mask_vis.to(dtype=torch.uint8)).bool(),
                  'gt_mask_obj': crop(self.gt_mask_obj.to(dtype=torch.uint8)).bool(),
                  'imgs': [crop(img) for img in self.images], 'dbg_img': self.images, 'dbg_bbox': bbox}
        return result
