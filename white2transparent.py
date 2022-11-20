from matplotlib import pyplot as plt
import numpy as np

import utils.io

# import numpy as np
# from matplotlib import pyplot as plt
# N = 1000000000
# n_bins = 180
# angles = np.arccos(np.cos(np.random.rand(N) * (np.pi * .5)) * np.sqrt(np.random.rand(N))) * 2.
# angles_deg = angles * (180. / np.pi)
# plt.rcParams.update({'font.size': 14})
# plt.hist(angles_deg, bins=n_bins, density=True)
# plt.xticks(np.arange(0, 181, 30))
# plt.xlim(0, 180)
# plt.ylim(bottom=0)
# plt.xlabel('Rotation Angle [°]')
# plt.ylabel('Probability Density')
# plt.tight_layout()
# plt.savefig('rotation_angle_distribution.pdf', bbox_inches='tight', pad_inches = 0, transparent=True)
# plt.savefig('rotation_angle_distribution.svg', bbox_inches='tight', pad_inches = 0, transparent=True)
# plt.show()


rgb = utils.io.imread('/home/user/Desktop/RUNE-Tag.png', opencv_bgr=False)
a = 255 - rgb[..., :1]
rgba = np.concatenate([rgb, a], axis=-1)
plt.imsave('/home/user/Desktop/Picture2.png', rgba, vmin=0, vmax=255)
