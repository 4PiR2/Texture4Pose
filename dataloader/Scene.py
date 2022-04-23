import random

import torch
import torch.nn.functional as F
from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRasterizer, \
    MeshRenderer, SoftPhongShader, AmbientLights, BlendParams
from pytorch3d.structures import join_meshes_as_scene, join_meshes_as_batch
from pytorch3d.transforms import euler_angles_to_matrix

from utils.SimpleShader import SimpleShader


class Scene:
    width = 640
    height = 480

    def __init__(self, obj_set, cam_K, gt_poses, **kwargs):
        self.kwargs = kwargs
        self.obj_set = obj_set
        self.device = obj_set.device
        self.cam_K = torch.tensor(cam_K, device=self.device).reshape(3, 3)
        self.gt_poses = gt_poses
        image_size = (Scene.height, Scene.width)
        R = torch.eye(3)[None]
        R[0, 0, 0] = R[0, 1, 1] = -1.
        K = torch.zeros((1, 4, 4))
        K[0, :2, :3] = self.cam_K[:2]
        K[0, 2, 3] = K[0, 3, 2] = 1.
        self.cameras = PerspectiveCameras(R=R, K=K, image_size=(image_size,), device=self.device, in_ndc=False)

        self.raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.,
            faces_per_pixel=1,
            # max_faces_per_bin=mesh.num_faces_per_mesh()[0],
        )
        self.gt_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings),
            shader=SimpleShader(background=0.),
        )

    def get_transformed_meshes(self):
        meshes = []
        for i in range(len(self.gt_poses['obj_id'])):
            mesh = self.obj_set.objects[int(self.gt_poses['obj_id'][i])]\
                .get_transformed_mesh(self.gt_poses['cam_R_m2c'][i], self.gt_poses['cam_t_m2c'][i])
            meshes.append(mesh)
        return meshes

    def render_scene_mesh_gt(self, transformed_meshes=None):
        if transformed_meshes is None:
            transformed_meshes = self.get_transformed_meshes()
        total_meshes = len(transformed_meshes)
        for i in range(total_meshes):
            transformed_meshes[i].textures = self.obj_set.objects[int(self.gt_poses['obj_id'][i])].get_gt_texture()
        scene_mesh = join_meshes_as_scene(transformed_meshes, include_textures=True)
        images = self.gt_renderer(join_meshes_as_batch([scene_mesh, *transformed_meshes], include_textures=True))
        # [1+N, H, W, 4(IXYZ)]
        mask = images[..., 0].round().to(dtype=torch.uint8)  # [1+N, H, W]
        coor3d = images[..., 1:4]  # [1+N, H, W, 3(XYZ)]

        def get_bbox(m):
            # mask: [..., H, W]
            w_mask = m.any(dim=-2)  # [..., W]
            h_mask = m.any(dim=-1)  # [..., H]
            x0 = w_mask.to(dtype=torch.uint8).argmax(dim=-1)  # [...]
            y0 = h_mask.to(dtype=torch.uint8).argmax(dim=-1)  # [...]
            w = w_mask.sum(dim=-1)  # [...]
            h = h_mask.sum(dim=-1)  # [...]
            return torch.stack([x0 + w * .5, y0 + h * .5, w, h], dim=-1)  # [..., 4(XYWH)]

        mask_vis = torch.stack([mask[0] == obj_id for obj_id in self.gt_poses['obj_id']], dim=0)
        # [N, H, W]
        bbox_vis = get_bbox(mask_vis)  # [N, 4(XYWH)]

        mask_obj = mask[1:].bool()  # [N, H, W]
        bbox_obj = get_bbox(mask_obj)  # [N, 4(XYWH)]

        return mask[0].bool(), (coor3d[0], mask_vis, bbox_vis), (coor3d[1:], mask_obj, bbox_obj)

    def render_scene_mesh(self, f=None, num_images=1, transformed_meshes=None):
        if transformed_meshes is None:
            transformed_meshes = self.get_transformed_meshes()
        total_meshes = len(transformed_meshes)
        for i in range(total_meshes):
            transformed_meshes[i].textures = self.obj_set.objects[int(self.gt_poses['obj_id'][i])].get_texture(f)
        scene_mesh = join_meshes_as_scene(transformed_meshes, include_textures=True)

        lights = AmbientLights(device=self.device)
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0., 0., 0.))
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self.raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device,
                cameras=self.cameras,
                lights=lights
            )
        )

        images = renderer(scene_mesh, include_textures=True)[..., :3]
        return images


def poses_from_random(dataset, num_obj):
    device = dataset.device
    euler_angles = (2. * torch.pi) * torch.rand((num_obj, 3), device=device)
    Rs = euler_angles_to_matrix(euler_angles, 'ZYX')

    objects = dataset.objects
    selected_obj = [obj_id for obj_id in objects]
    random.shuffle(selected_obj)
    selected_obj = selected_obj[:num_obj]
    selected_obj.sort()
    radii = torch.tensor([objects[obj_id].radius for obj_id in selected_obj], device=device)
    centers = torch.stack([objects[obj_id].center for obj_id in selected_obj], dim=0)
    triu_indices = torch.triu_indices(num_obj, num_obj, 1)
    mdist = (radii + radii[..., None])[triu_indices[0], triu_indices[1]]

    flag = False
    while not flag:
        positions = torch.rand((num_obj, 3), device=device)\
                    * torch.tensor((.5, .5, .5), device=device) + torch.tensor((-.25, -.25, 1.), device=device)
        flag = (F.pdist(positions) >= mdist).all()
    positions -= centers

    poses = []
    for i in range(num_obj):
        poses.append({'obj_id': selected_obj[i], 'cam_R_m2c': Rs[i], 'cam_t_m2c': positions[i]})
    return poses
