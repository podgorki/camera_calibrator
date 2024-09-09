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


def plot_undistorted_images(
        save_path: Path,
        images: List[Path],
        rectify_dict: dict) -> None:
    rectified_save_path = (save_path / 'rectified')
    rectified_save_path.mkdir(parents=True, exist_ok=True)

    rmse = rectify_dict["rmse"]
    K1 = rectify_dict["K"]
    D1 = rectify_dict["D"]

    for i, image in enumerate(images):
        h, w, _ = image.shape
        new_K, roi = cv2.getOptimalNewCameraMatrix(K1, D1, (w, h), 1, (w, h))
        map_l1, map_l2 = cv2.initUndistortRectifyMap(K1, D1, None, new_K, (w, h), cv2.CV_32FC1)
        rectified_image = cv2.remap(image, map_l1, map_l2, cv2.INTER_LINEAR)
        fig = rectified_image
        cv2.putText(
            fig,
            f"err: {rmse:.2f}", (0, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (255, 0, 0),
            3,
            cv2.LINE_AA
        )

        cv2.imwrite(str(rectified_save_path / f"undistorted_{str(i).zfill(4)}.png"), fig)


def run(args: argparse.Namespace) -> None:
    root = Path(args.root)
    outputs_root = root / "outputs"
    outputs_root.mkdir(exist_ok=True, parents=True)
    data_path = outputs_root / "data"
    data_path.mkdir(exist_ok=True, parents=True)
    camera_images_paths = sorted(root.glob('*.png'))

    print(f"Found: {len(camera_images_paths)} images")
    camera_images_list = []
    for image_path in tqdm(camera_images_paths, desc='Loading images...'):
        im = cv2.imread(str(image_path), 1)
        camera_images_list.append(im)
    height, width, _ = camera_images_list[0].shape

    if not args.precomputed:
        calibration_dict = calibrate_camera(args, camera_images_list, args.rows, args.columns, args.size)
        save_camera_calibration(calibration_dict=calibration_dict,
                                image_paths=camera_images_paths,
                                filename=data_path / "calibration.npz")
    else:
        calibration_dict = np.load(data_path / "calibration.npz")

    refined_calibration_dict = refine_calibration(calibration_dict, width, height, threshold=args.camera_threshold)
    if args.check:
        check_calibration_reprojection(refined_calibration_dict, camera_images_list, figs_path)
    save_camera_calibration(calibration_dict=refined_calibration_dict,
                            image_paths=camera_images_paths,
                            filename=data_path / "refined_calibration.npz")

    print_calibration_results(calibration_dict, refined_calibration_dict)

    plot_undistorted_images(root, camera_images_list, calibration_dict)


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
