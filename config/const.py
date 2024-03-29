import torch

dtype = torch.float32
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

debug_mode = False

plot_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

lm_cam_K = torch.tensor([[[572.4114, 0., 325.2611], [0., 573.57043, 242.04899], [0., 0., 1.]]], dtype=dtype)
i12P_cam_K = torch.tensor([[[3076.2150484237645, 0., 2046.65792208993],
                            [0., 3076.2150484237645, 1492.683079664184],
                            [0., 0., 1.]]], dtype=dtype)
video_cam_K = torch.tensor([[[1.046, 0., -(4032. - 3840. / 1.046) * .5],
                             [0., 1.046, -(3024. - 2160. / 1.046) * .5],
                             [0., 0., 1.]]], dtype=dtype) @ i12P_cam_K
# video_cam_K = torch.tensor([[[.5, 0., -(4032. - 1920. / .5) * .5],
#                              [0., .5, -490.],
#                              [0., 0., 1.]]], dtype=dtype) @ i12P_cam_K
# video_cam_K = torch.tensor([[[3147.928263704622, 0., 1968.496833871099],
#                              [0., 3147.928263704622, 1040.2614106295287],
#                              [0., 0., 1.]]], dtype=dtype)
fov70_cam_K = torch.tensor([[[256. / .7, 0., 256.], [0., 256. / .7, 256.], [0., 0., 1.]]], dtype=dtype)
real_cam_K = i12P_cam_K

lm_objects = {1: 'ape', 2: 'benchvise', 3: 'bowl', 4: 'camera', 5: 'can', 6: 'cat', 7: 'cup', 8: 'driller', 9: 'duck',
              10: 'eggbox', 11: 'glue', 12: 'holepuncher', 13: 'iron', 14: 'lamp', 15: 'phone'}
lm13_objects = {i: lm_objects[i] for i in [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]}
lmo_objects = {i: lm_objects[i] for i in [1, 5, 6, 8, 9, 10, 11, 12]}

regular_objects = {101: 'sphere', 102: 'cube', 103: 'tetrahedron', 104: 'cylinderstrip', 105: 'sphericon'}

gdr_mode = True
