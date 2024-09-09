import cv2
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
import copy
import sys
from utils_io import save_calib
from typing import List, Optional


def calibrate_camera(
        args: argparse.Namespace,
        images: List[np.ndarray],
        rows: int, columns:
        int, size: float,
        cameraMatrix: Optional = None,
        distCoeffs: Optional = None) -> dict:
    # criteria used by checkerboard pattern detector.
    # Change this if the code can't find the checkerboard
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = size * objp

    # Pixel coordinates of checkerboards
    imgpoints = []  # 2d points in image plane.

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space

    image_indices = []

    window_name = "calibration images"

    if args.show_calibration_checkerboards:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    width = images[0].shape[1]
    height = images[0].shape[0]
    for i, frame in tqdm(enumerate(images), desc='Finding corners...'):
        frame_copy = copy.deepcopy(frame)
        gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
        # find the checkerboard
        ret, corners = cv2.findChessboardCorners(gray, (rows, columns), None)

        if ret == True:
            # Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)
            # opencv can attempt to improve the checkerboard coordinates
            corners = cv2.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            if args.show_calibration_checkerboards:
                cv2.drawChessboardCorners(frame_copy, (rows, columns), corners, ret)
                cv2.imshow(window_name, frame_copy)
                k = cv2.waitKey(500)

            objpoints.append(objp)
            imgpoints.append(corners)
            image_indices.append(i)
    if args.show_calibration_checkerboards:
        cv2.destroyAllWindows()
    rmse, mtx, dist, rvecs, tvecs, std_intrinsics, std_extrinsics, per_view_errors = cv2.calibrateCameraExtended(
        objpoints, imgpoints, (width, height), cameraMatrix, distCoeffs)
    return {"rmse": rmse,
            "K": mtx,
            "D": dist,
            "rvecs": rvecs,
            "tvecs": tvecs,
            "imgpoints": imgpoints,
            "objpoints": objpoints,
            "std_intrinsics": std_intrinsics,
            "std_extrinsics": std_extrinsics,
            "per_view_errors": per_view_errors,
            "image_indices": image_indices}


def refine_calibration(calibration_dict: dict, width: int, height: int, threshold: float = 0.5) -> dict:
    # extract the goodies
    cameraMatrix = calibration_dict["K"]
    distCoeffs = calibration_dict["D"]
    image_pts = calibration_dict["imgpoints"]
    object_pts = calibration_dict["objpoints"]
    per_view_errors = calibration_dict["per_view_errors"]
    image_indices = calibration_dict["image_indices"]

    # remove the badies
    mask = per_view_errors.reshape(-1) < threshold
    image_indices = list(np.array(image_indices)[mask])
    good_per_view_errors = per_view_errors[mask]
    good_image_pts = list(np.array(image_pts)[mask])
    good_object_pts = list(np.array(object_pts)[mask])

    # re-calibrate
    rmse, mtx, dist, rvecs, tvecs, std_intrinsics, std_extrinsics, per_view_errors = cv2.calibrateCameraExtended(
        good_object_pts, good_image_pts, (width, height), cameraMatrix, distCoeffs)
    return {"rmse": rmse,
            "K": mtx,
            "D": dist,
            "imgpoints": good_image_pts,
            "objpoints": good_object_pts,
            "rvecs": rvecs,
            "tvecs": tvecs,
            "std_intrinsics": std_intrinsics,
            "std_extrinsics": std_extrinsics,
            "per_view_errors": good_per_view_errors,
            "image_indices": image_indices}


def check_calibration_reprojection(calibration_dict: dict, images: List[np.ndarray],
                                   save_path: Path) -> None:
    cameraMatrix = calibration_dict["K"]
    distCoeffs = calibration_dict["D"]
    image_pts = calibration_dict["imgpoints"]
    object_pts = calibration_dict["objpoints"]
    rvecs = calibration_dict["rvecs"]
    tvecs = calibration_dict["tvecs"]
    image_indices = calibration_dict["image_indices"]
    per_view_errors = calibration_dict["per_view_errors"]

    save_path.mkdir(exist_ok=True, parents=True)
    images = list(np.array(images)[image_indices])
    h, w, _ = images[0].shape
    window_name = f"Calibration reprojection images"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    view = True
    print("Instructions for viewing pleasure: Esc to quit viewing, q to quit.")
    for idx, img, imgpts, objpts, viewerr, rvec, tvec in zip(image_indices, images, image_pts, object_pts,
                                                             per_view_errors, rvecs, tvecs):
        img = copy.copy(img)
        imgpts = copy.copy(imgpts).reshape(-1, 2)
        cv2.putText(img, f"err: {viewerr.item():.2f}", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3,
                    cv2.LINE_AA)

        for uv, obj in zip(imgpts, objpts):
            projpts, _ = cv2.projectPoints(obj, rvec, tvec, cameraMatrix, distCoeffs)
            cv2.circle(img, tuple(uv.astype(int)), 4, (255, 0, 0), -1)  # blue for detected
            cv2.circle(img, projpts.reshape(2).astype(int), 3, (0, 0, 255), -1)  # red for reprojection
        if view is True:
            cv2.imshow(window_name, img)
        cv2.imwrite(str(save_path / f"reprojection_{str(idx).zfill(4)}.png"), img)
        k = cv2.waitKey(0)
        if k == 27:
            print("Done lookin'")
            view = False
            cv2.destroyAllWindows()
        elif k == ord('q'):
            print('Quitting!')
            sys.exit(0)


def plot_and_save_corners(figs_path: Path, width: int, height: int,
                          imgpoints1: np.ndarray, imgpoints2: np.ndarray,
                          per_view_errors1: np.ndarray, per_view_errors2: np.ndarray,
                          filename: str = "detected_corners",
                          title: str = "Detected corners for each camera coloured by RMSE") -> None:
    fig, ax = plt.subplots(1, 2)
    plt.title("Corners detected during camera calibration.")
    u1, v1, c1 = [], [], []
    for view_pts, view_error in zip(imgpoints1, per_view_errors1.reshape(-1)):
        plt_pts = view_pts.reshape(-1, 2)
        cols = np.ones_like(imgpoints1[0].reshape(-1, 2)[:, 0]) * view_error
        u1.append(plt_pts[:, 0])
        v1.append(-plt_pts[:, 1])
        c1.append(cols)
    corner1_plot = ax[0].scatter(u1, v1, alpha=0.5, c=c1, cmap="inferno")
    ax[0].set_title('left')
    ax[0].set_ylabel('v')
    ax[0].set_ylim(-height, 0)
    ax[0].set_xlim(0, width)
    ax[0].set_xlabel('u')
    divider1 = make_axes_locatable(ax[0])
    cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(corner1_plot, cax=cax1, orientation='vertical')

    u2, v2, c2 = [], [], []
    for view_pts, view_error in zip(imgpoints2, per_view_errors2.reshape(-1)):
        plt_pts = view_pts.reshape(-1, 2)
        cols = np.ones_like(imgpoints2[0].reshape(-1, 2)[:, 0]) * view_error
        u2.append(plt_pts[:, 0])
        v2.append(-plt_pts[:, 1])
        c2.append(cols)
    corner2_plot = ax[1].scatter(u2, v2, alpha=0.5, c=c2, cmap="inferno")
    divider2 = make_axes_locatable(ax[1])
    cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(corner2_plot, cax=cax2, orientation='vertical')
    ax[1].set_ylabel('v')
    ax[1].set_xlabel('u')
    ax[1].set_ylim(-height, 0)
    ax[1].set_yticks([])
    ax[1].set_xlim(0, width)
    ax[1].set_title('right')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(str((figs_path / filename).with_suffix('.pdf')))


def save_camera_calibration(calibration_dict: dict, image_paths: List[Path], filename: Path) -> None:
    np.savez_compressed(filename,
                        image_paths=image_paths,
                        rmse=calibration_dict["rmse"],
                        K=calibration_dict["K"],
                        D=calibration_dict["D"],
                        rvecs=calibration_dict["rvecs"],
                        tvecs=calibration_dict["tvecs"],
                        imgpoints=calibration_dict["imgpoints"],
                        objpoints=calibration_dict["objpoints"],
                        std_intrinsics=calibration_dict["std_intrinsics"],
                        std_extrinsics=calibration_dict["std_extrinsics"],
                        per_view_errors=calibration_dict["per_view_errors"],
                        image_indices=calibration_dict["image_indices"])


def print_calibration_results(calibration_dict: dict, refined_calibration_dict: dict) -> None:
    print(f"Initial results:\n")
    print(f"Initial RSME: {calibration_dict['rmse']:.2f}")
    for k in list(calibration_dict.keys()):
        if k == "K" or k == "D":
            print(f"{k}:\n{calibration_dict[k]}")

    print(f"\nRefined results:\n")
    print(f"Refined RSME: {refined_calibration_dict['rmse']:.2f}")
    for k in list(refined_calibration_dict.keys()):
        if k == "K" or k == "D":
            print(f"{k}:\n{refined_calibration_dict[k]}")


def plot_undistorted_images_and_epipoles(save_path: Path,
                                         left_images: List[np.ndarray], right_images: List[np.ndarray],
                                         rectify_dict: dict) -> None:
    save_path.mkdir(parents=True, exist_ok=True)
    left_image = left_images[0]
    right_image = right_images[0]

    rmse = rectify_dict["rmse"]
    w = rectify_dict["width"]
    h = rectify_dict["height"]
    K1 = rectify_dict["K1"]
    D1 = rectify_dict["D1"]
    R1 = rectify_dict["R1"]
    P1 = rectify_dict["P1"]
    K2 = rectify_dict["K2"]
    D2 = rectify_dict["D2"]
    R2 = rectify_dict["R2"]
    P2 = rectify_dict["P2"]

    map_l1, map_l2 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_32FC1)
    map_r1, map_r2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w, h), cv2.CV_32FC1)

    rect1 = cv2.remap(left_image, map_l1, map_l2, cv2.INTER_LINEAR)
    rect2 = cv2.remap(right_image, map_r1, map_r2, cv2.INTER_LINEAR)

    sift = cv2.SIFT_create()
    rect1_gray = cv2.cvtColor(rect1, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    rect2_gray = cv2.cvtColor(rect2, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    kp1_rect, des1_rect = sift.detectAndCompute(rect1_gray, None)
    kp2_rect, des2_rect = sift.detectAndCompute(rect2_gray, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches_rect = flann.knnMatch(des1_rect, des2_rect, k=2)
    pts1_rect = []
    pts2_rect = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches_rect):
        if m.distance < 0.8 * n.distance:
            pts2_rect.append(kp2_rect[m.trainIdx].pt)
            pts1_rect.append(kp1_rect[m.queryIdx].pt)
    pts1_rect = np.int32(pts1_rect)
    pts2_rect = np.int32(pts2_rect)
    F_rect, mask_rect = cv2.findFundamentalMat(pts1_rect, pts2_rect, cv2.FM_LMEDS)
    # We select only inlier points
    pts1_rect = pts1_rect[mask_rect.ravel() == 1][::25, :]
    pts2_rect = pts2_rect[mask_rect.ravel() == 1][::25, :]
    lines1_rect = cv2.computeCorrespondEpilines(pts2_rect.reshape(-1, 1, 2), 2, F_rect)
    lines1_rect = lines1_rect.reshape(-1, 3)
    rect1_lines = draw_epilines_lines(rect1_gray, lines1_rect, pts1_rect)
    lines2_rect = cv2.computeCorrespondEpilines(pts1_rect.reshape(-1, 1, 2), 1, F_rect)
    lines2_rect = lines2_rect.reshape(-1, 3)
    rect2_lines = draw_epilines_lines(rect2_gray, lines2_rect, pts2_rect)

    fig = np.zeros((h, 2 * w, 3), dtype=np.uint8)
    fig[:, :w, :] = rect1_lines
    fig[:, w:, :] = rect2_lines
    cv2.putText(fig, f"err: {rmse:.2f}", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3, cv2.LINE_AA)

    cv2.imwrite(str(save_path / "final_undistorted_w_epilines_image.png"), fig)


def draw_epilines_lines(img: np.ndarray, lines: np.ndarray, pts: np.ndarray) -> np.ndarray:
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img.shape
    img1 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for r, pt1 in zip(lines, pts):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 5)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
    return img1


def save_kitti_calibs(filepath: Path, rectify_dict: dict) -> None:
    """
    This just converts the rectified calibs into kitti style .txt ones.
    (from kitti docs) Note: All matrices are stored row-major, i.e., the first values correspond
    to the first row. R0_rect contains a 3x3 matrix which you need to extend to
    a 4x4 matrix by adding a 1 as the bottom-right element and 0's elsewhere.
    Tr_xxx is a 3x4 matrix (R|t), which you need to extend to a 4x4 matrix
    in the same way!

    The kitti style has a namingg convention which is duplicated here
    P0 - left mono cam (not used) i.e. R=eye, T=0
    P1 - right mono cam (not used) i.e. R=eye, T=0
    P2 - left color cam
    P3 - right color cam
    R_rect - rotation to left rectified  (R0_rect)
    Tr_velo_cam - rigid transform from velo (lidar used in kitti) to cam (not calculated here) i.e. R=eye, T=0
    Tr_imu_velo - rigid transform from imu to velo (lidar used in kitti) (not calculated here) i.e. R=eye, T=0
    """
    calib = {}
    calib["P0"] = np.block([np.eye(3), np.zeros((3, 1))])
    calib["P1"] = np.block([np.eye(3), np.zeros((3, 1))])
    calib["P2"] = rectify_dict["P1"]
    calib["P3"] = rectify_dict["P2"]
    calib["R_rect"] = rectify_dict["R1"]
    calib["Tr_velo_cam"] = np.block([np.eye(3), np.zeros((3, 1))])
    calib["Tr_imu_velo"] = np.block([np.eye(3), np.zeros((3, 1))])
    save_calib(calib_path=filepath, calib=calib)


def run(args: argparse.Namespace) -> None:
    root = Path(args.root)
    outputs_root = root / "outputs"
    outputs_root.mkdir(exist_ok=True, parents=True)
    figs_path = outputs_root / "figs"
    figs_path.mkdir(exist_ok=True, parents=True)
    data_path = outputs_root / "data"
    data_path.mkdir(exist_ok=True, parents=True)
    camera_images = sorted(root.glob('*.png'))

    print(f"Found: {len(camera_images)} images")
    camera_images_list = []
    for image_path in tqdm(camera_images, desc='Loading images...'):
        im = cv2.imread(str(image_path), 1)
        camera_images_list.append(im)
    height, width, _ = camera_images_list[0].shape

    if not args.precomputed:
        calibration_dict1 = calibrate_camera(args, camera_images_list, args.rows, args.columns, args.size)
        save_camera_calibration(calibration_dict=calibration_dict1,
                                image_paths=camera_images,
                                filename=data_path / "calibration.npz")
    else:
        calibration_dict1 = np.load(data_path / "calibration.npz")

    refined_calibration_dict1 = refine_calibration(calibration_dict1, width, height, threshold=args.camera_threshold)
    if args.check:
        check_calibration_reprojection(refined_calibration_dict1, camera_images_list, figs_path)
    save_camera_calibration(calibration_dict=refined_calibration_dict1,
                            image_paths=camera_images,
                            filename=data_path / "refined_calibration.npz")

    print_calibration_results(calibration_dict1, refined_calibration_dict1)


def make_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Calibrate from a set of synchronised left and right images.")
    parser.add_argument('--root', type=str,
                        help='Root to images')
    parser.add_argument('--precomputed', action="store_true", default=False,
                        help='Use precomputed calibrations matricies stored in outputs\/data/. Default=False')
    parser.add_argument('--check', action="store_true", default=False,
                        help='Check the re-projections of each set of left corners using the refined results. Default=False')
    parser.add_argument('--columns', type=int, default=7,
                        help='Number of columns on the checkerboard. Default=7')
    parser.add_argument('--rows', type=int, default=5,
                        help='Number of rows on the checkerboard. Default=5')
    parser.add_argument('--size', type=float, default=0.139,
                        help='Realworld measurement of checkerboard squares. Default=0.140 (m)')
    parser.add_argument('--camera_threshold', type=float, default=0.5,
                        help='RMSE threshold for filtering image and object points during camera calibration '
                             'refinement.')
    parser.add_argument('--show_calibration_checkerboards', action='store_true', default=False,
                        help='Show checkerboards with their detected corners overlaid. Default=False')
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    run(args)
