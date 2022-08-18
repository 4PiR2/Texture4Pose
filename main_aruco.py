import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch

import aruco.charuco_board as ac
import aruco.print_cylinder as ap
import utils.io
import utils.print_paper


def show_ndarray(img):
    plt.figure(figsize=(16, 12))
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    # ac.ChArUcoBoard(7, 10, .04).to_paper_pdf('/home/user/Desktop/1.pdf', paper_size='a3')
    dpi = 300
    img_84 = ap.unroll_cylinder_side(r=.042, dpi=dpi)
    img_82 = ap.unroll_cylinder_side(r=.041, dpi=dpi)
    img = utils.print_paper.make_grid(img_84, (1, 2), margin=.05)
    img[..., :img_84.shape[-1]] = 1.
    img[..., :img_82.shape[-2], :img_82.shape[-1]] = img_82
    utils.print_paper.print_tensor_to_paper_pdf(img, '/home/user/Desktop/3.pdf', dpi=dpi)

    chboard: ac.ChArUcoBoard = ac.ChArUcoBoard(7, 10, .04)
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
