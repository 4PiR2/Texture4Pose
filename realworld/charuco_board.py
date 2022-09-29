# https://mecaruco2.readthedocs.io/en/latest/notebooks_rst/Aruco/sandbox/ludovic/aruco_calibration_rotation.html
# https://docs.opencv.org/4.x/d9/d6a/group__aruco.html

import cv2
import numpy as np

import utils.io
import utils.print_paper


class ChArUcoBoard:
    def __init__(self, w_square: int = 10, h_square: int = 14, square_length: float = .02):
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        # img = cv2.aruco.drawMarker(aruco_dict, 3, 700, borderBits=1)
        self.board: cv2.aruco_CharucoBoard = cv2.aruco.CharucoBoard_create(
            w_square, h_square, square_length, .8 * square_length, self.aruco_dict)  # square length is in meter
        # origin point is at bottom left corner, x+: right, y+: up

    def to_paper_pdf(self, path: str, dpi: int = 300, paper_size: str = 'a4'):
        w_square, h_square = self.board.getChessboardSize()
        square_length = self.board.getSquareLength()
        w = utils.print_paper.meter2px(w_square * square_length, dpi)
        h = utils.print_paper.meter2px(h_square * square_length, dpi)
        img = self.board.draw((w, h))
        utils.print_paper.print_tensor_to_paper_pdf(img, path, 'nearest', dpi, paper_size)

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
            img_gray = cv2.cvtColor(utils.io.imread(im), cv2.COLOR_BGR2GRAY)
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
