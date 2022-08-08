from typing import Union

import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch


def get_a4_size(unit_or_dpi: Union[str, int]):
    w, h = 210, 297
    if unit_or_dpi == 'mm':
        return w, h
    elif unit_or_dpi == 'inch':
        return w / 25.4, h / 25.4
    elif isinstance(unit_or_dpi, int):
        w_inch, h_inch = get_a4_size('inch')
        return round(w_inch * unit_or_dpi), round(h_inch * unit_or_dpi)


def get_ax_a4(dpi: int = 72):
    fig = plt.figure(figsize=get_a4_size('inch'), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.add_axes(ax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_axis_off()
    return ax


def plt_save_pdf(path: str):
    plt.savefig(path, format='pdf')


def show_tensor(img: Union[torch.Tensor, np.ndarray], ax: plt.Axes = None, zorder: int = 10):
    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0).detach().cpu().numpy()
    elif isinstance(img, np.ndarray):
        img = img[..., ::-1]  # BGR2RGB
    if img.dtype != np.uint8:
        img = (img * 255.).astype('uint8')
    if ax is not None:
        ax.imshow(img, interpolation='nearest', zorder=zorder)
    else:
        plt.figure(figsize=(16, 12))
        plt.imshow(img)
        plt.show()


class ChArUcoBoard:
    def __init__(self, w_square: int = 10, h_square: int = 14, square_length: float = .02):
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        # img = cv2.aruco.drawMarker(aruco_dict, 3, 700, borderBits=1)
        self.board = cv2.aruco.CharucoBoard_create(
            w_square, h_square, square_length, .8 * square_length, self.aruco_dict)  # square length is in meter

    def to_a4_pdf(self, path: str, dpi: int = 300):
        w_square, h_square = self.board.getChessboardSize()
        square_length = self.board.getSquareLength()
        w, h = int(w_square * square_length * dpi / .0254), int(h_square * square_length * dpi / .0254)
        W, H = get_a4_size(dpi)
        imboard = np.full([H, W], 255, dtype=np.uint8)
        self.board.draw((w, h), imboard[(H - h) // 2:(H + h) // 2, (W - w) // 2:(W + w) // 2])
        ax = get_ax_a4(dpi)
        show_tensor(imboard[..., None].repeat(3, axis=-1), ax)
        # Matplotlib's linewidth is in points, 1 inch = 72 point, default dpi = 72, 1 point = 0.352778 mm
        # https://stackoverflow.com/questions/57657419/how-to-draw-a-figure-in-specific-pixel-size-with-matplotlib
        plt_save_pdf(path)

    def detect_markers(self, img_gray: np.ndarray):
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img_gray, self.aruco_dict, parameters=parameters)
        # SUB PIXEL DETECTION
        for corner in corners:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
            cv2.cornerSubPix(img_gray, corner, winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria)
        return corners, ids

    def detect_checkerboard_corners(self, img_gray: np.ndarray, camera_matrix: np.ndarray = None,
                                    distortion: np.ndarray = None):
        corners, ids = self.detect_markers(img_gray)
        if len(corners) == 0:
            return 0, None, None
        num, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(
            corners, ids, img_gray, self.board, cameraMatrix=camera_matrix, distCoeffs=distortion, minMarkers=2)
        return num, charucoCorners, charucoIds

    def calibrate_camera(self, images):
        allCorners = []
        allIds = []
        for im in images:
            img_gray = cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2GRAY)
            num, charucoCorners, charucoIds = self.detect_checkerboard_corners(img_gray)
            if num <= 3:
                continue
            allCorners.append(charucoCorners)
            allIds.append(charucoIds)
            imsize = img_gray.shape

        cameraMatrixInit = np.array([[1000., 0., imsize[0] / 2.],
                                     [0., 1000., imsize[1] / 2.],
                                     [0., 0., 1.]])
        distCoeffsInit = np.zeros((5, 1))
        # calib flags: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
        # flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO
        flags = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | \
         cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 | cv2.CALIB_FIX_K6
        reproj_err, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors,\
        stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors = cv2.aruco.calibrateCameraCharucoExtended(
            charucoCorners=allCorners, charucoIds=allIds, board=self.board, imageSize=imsize,
            cameraMatrix=cameraMatrixInit, distCoeffs=distCoeffsInit, flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

        return camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors

    def estimate_pose(self, img_gray: np.ndarray, camera_matrix: np.ndarray, distortion: np.ndarray = None):
        num, c_corners, c_ids = self.detect_checkerboard_corners(img_gray, camera_matrix, distortion)
        ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            c_corners, c_ids, self.board, camera_matrix, distortion, None, None, useExtrinsicGuess=False)
        if ret:
            rmat, _ = cv2.Rodrigues(rvec)
        else:
            rmat = None
        return ret, rmat, tvec, rvec
