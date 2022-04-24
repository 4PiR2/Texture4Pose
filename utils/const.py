import torch

debug_mode = False

plot_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

lm_objects = {1: 'ape', 2: 'benchvise', 3: 'bowl', 4: 'camera', 5: 'can', 6: 'cat', 7: 'cup', 8: 'driller', 9: 'duck',
              10: 'eggbox', 11: 'glue', 12: 'holepuncher', 13: 'iron', 14: 'lamp', 15: 'phone'}

lm13_objects = {1: 'ape', 2: 'benchvise', 4: 'camera', 5: 'can', 6: 'cat', 8: 'driller', 9: 'duck', 10: 'eggbox',
                11: 'glue', 12: 'holepuncher', 13: 'iron', 14: 'lamp', 15: 'phone'}

lmo_objects = {1: 'ape', 5: 'can', 6: 'cat', 8: 'driller', 9: 'duck', 10: 'eggbox', 11: 'glue', 12: 'holepuncher'}

device = torch.device('cuda:0')