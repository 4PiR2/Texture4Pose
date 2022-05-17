import torch

debug_mode = False
gdr_mode = True

plot_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

lm_objects = {1: 'ape', 2: 'benchvise', 3: 'bowl', 4: 'camera', 5: 'can', 6: 'cat', 7: 'cup', 8: 'driller', 9: 'duck',
              10: 'eggbox', 11: 'glue', 12: 'holepuncher', 13: 'iron', 14: 'lamp', 15: 'phone'}
lm13_objects = {1: 'ape', 2: 'benchvise', 4: 'camera', 5: 'can', 6: 'cat', 8: 'driller', 9: 'duck', 10: 'eggbox',
                11: 'glue', 12: 'holepuncher', 13: 'iron', 14: 'lamp', 15: 'phone'}
lmo_objects = {1: 'ape', 5: 'can', 6: 'cat', 8: 'driller', 9: 'duck', 10: 'eggbox', 11: 'glue', 12: 'holepuncher'}

regular_objects = {1: 'sphere', 2: 'cube'}

dtype = torch.float
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

img_render_size = 512
img_input_size = 256
pnp_input_size = 64

vis_ratio_threshold = .5
t_depth_min = .5
t_depth_max = 1.2

lm_cam_K = torch.tensor([[572.4114, 0., 325.2611], [0., 573.57043, 242.04899], [0., 0., 1.]])
