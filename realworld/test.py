import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch

import realworld.charuco_board
import realworld.print_unroll
import utils.io
import utils.print_paper


def show_ndarray(img):
    plt.figure(figsize=(16, 12))
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    # from utils.image_2d import visualize
    # img = realworld.print_unroll.unroll_sphericon(.05, dpi=300)
    # visualize(img)

    from renderer.cube_mesh import sphericon
    from pytorch3d.renderer import TexturesVertex
    import pytorch3d.vis.plotly_vis
    mesh = sphericon(5)
    mesh.textures = TexturesVertex(mesh.verts_packed()[None] * .49 + .5)
    pytorch3d.vis.plotly_vis.plot_scene({'subplot1': {'sphericon_mesh': mesh}}).show()

    chboard: realworld.charuco_board.ChArUcoBoard = realworld.charuco_board.ChArUcoBoard(7, 10, .04)
    print('Starting camera calibration ...')
    camera_matrix = chboard.calibrate_camera(utils.io.list_img_from_dir('/data/real_exp/i12P_26mm/calib', ext='heic'))[0]
    print('Finished camera calibration')

    frame = utils.io.imread('/data/real_exp/i12P_26mm/000104/IMG_8170.HEIC')
    # frame = utils.io.imread('/data/coco/train2017/000000000009.jpg')
    # frame = cv2.undistort(frame, camera_matrix, distortion)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # cap = cv2.VideoCapture('/data/calib/4e9eaa7bdc/rgb.mp4')
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     corners, ids = chboard.detect_markers(gray)
    #     frame_markers = cv2.aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    #     show_ndarray(frame_markers)
    #     a = 0

    ret, p_rmat, p_tvec, p_rvec = chboard.estimate_pose(gray, camera_matrix)
    if ret:
        frame_pose = cv2.aruco.drawAxis(frame, camera_matrix, None, p_rvec, p_tvec, .1)
        show_ndarray(frame_pose[..., ::-1])

    a = 0
