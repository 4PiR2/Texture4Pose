import os
import pickle

import numpy as np
import torch

# synt
# versions = [330, 332, 331, 280, 343, 345, 292, 337, 334]
# real
# versions = [281, 344, 346, 293, 338, 335]

# synt occ
# versions = [363, 364, 365, 366, 367, 368, 369, 370, 371]
# real occ
# versions = [357, 355, 354, 362, 359, 358]

# abl init synt
# versions = [345, 374]
# abl init real
# versions = [346, 375]

# abl aug synt
versions = [345, 372]
# abl aug real
# versions = [346, 373]

root_dirs = [f'outputs/lightning_logs', f'/data/lightning_logs_old']

values = []

for version in versions:
    list_dir = []
    for root_dir in root_dirs:
        version_metric_dir = os.path.join(root_dir, f'version_{version}', 'metrics')
        try:
            list_dir = os.listdir(version_metric_dir)
        except Exception:
            continue
        break
    assert len(list_dir) == 1
    v = torch.load(os.path.join(version_metric_dir, list_dir[0]))
    values.append(v)

keys_global = None
metrics_list = []
quantiles_list = []
for outputs in values:
    keys = list(outputs[0].keys())
    assert keys_global is None or keys_global == keys
    keys_global = keys
    outputs = {key: torch.cat([output[key] for output in outputs], dim=0) for key in keys}
    metrics = torch.stack([outputs[key] for key in keys], dim=0)
    metrics[metrics.isnan()] = torch.inf
    metrics_list.append(metrics)
    q = torch.linspace(0., 1., 20+1, dtype=metrics.dtype, device=metrics.device)[1:-1]
    quantiles = metrics.quantile(q, dim=1).T
    quantiles_list.append(quantiles)

all_quantiles = torch.stack(quantiles_list, dim=0)  # [V, 5(M), Q]
q_idx = -1
# q_idx = -5
# q_idx = -10
print('quantile:', float(q[q_idx]) * 100.)
qq = all_quantiles[..., q_idx].T  # [5(M), V]

for i in range(len(keys_global)):
    key = keys_global[i]
    print(key)
    if key.startswith('ad'):
        print(' & '.join([f'{float(x * 100.):.3f}' for x in qq[i]]) + '\\\\')
    else:
        print(' & '.join([f'{float(x):.3f}' for x in qq[i]]) + '\\\\')

a = 0

