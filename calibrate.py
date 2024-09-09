import cv2
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
import copy
import sys
from utils.io_utils import save_calib
from typing import List


def calibrate_camera(args: argparse.Namespace, images: List[np.ndarray], rows: int, columns: int, size: float,
                     cameraMatrix=None, distCoeffs=None) -> dict:
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


def check_calibration_reprojection(calibration_dict: dict, images: List[np.ndarray], camera: str,
                                   save_path: Path) -> None:
    cameraMatrix = calibration_dict["K"]
    distCoeffs = calibration_dict["D"]
    image_pts = calibration_dict["imgpoints"]
    object_pts = calibration_dict["objpoints"]
    rvecs = calibration_dict["rvecs"]
    tvecs = calibration_dict["tvecs"]
    image_indices = calibration_dict["image_indices"]
    per_view_errors = calibration_dict["per_view_errors"]

    camera_save_path = save_path / camera
    camera_save_path.mkdir(exist_ok=True, parents=True)
    images = list(np.array(images)[image_indices])
    h, w, _ = images[0].shape
    window_name = f"{camera} calibration reprojection images"
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
        if view == True:
            cv2.imshow(window_name, img)
        cv2.imwrite(str(camera_save_path / f"{camera}_reprojection_{str(idx).zfill(4)}.png"), img)
        k = cv2.waitKey(0)
        if k == 27:
            print("Done lookin'")
            view = False
            cv2.destroyAllWindows()
        elif k == ord('q'):
            print('Quitting!')
            sys.exit(0)


def stereo_calibrate(calibration_dict_left: dict, calibration_dict_right: dict, width: int, height: int) -> dict:
    """
    Returned R, T are take points from the right to the left frame
    """
    mtx_left, dist_left = calibration_dict_left["K"], calibration_dict_left["D"]
    mtx_right, dist_right = calibration_dict_right["K"], calibration_dict_right["D"]

    image_indices_left = calibration_dict_left["image_indices"]
    image_indices_right = calibration_dict_right["image_indices"]

    mask_left = np.isin(image_indices_left, image_indices_right)
    mask_right = np.isin(image_indices_right, image_indices_left)

    image_indices_left = np.array(image_indices_left)[mask_left]
    image_indices_right = np.array(image_indices_right)[mask_right]

    assert (image_indices_left == image_indices_right).all()  # these should be the same

    objpoints_left = np.array(calibration_dict_left["objpoints"])[mask_left]
    objpoints_right = np.array(calibration_dict_right["objpoints"])[mask_right]

    assert (objpoints_left == objpoints_right).all()  # these should be the same

    image_indices_left = list(image_indices_left)

    rvecs_left = list(np.array(calibration_dict_left["rvecs"])[mask_left])
    tvecs_left = list(np.array(calibration_dict_left["tvecs"])[mask_left])

    rvecs_right = list(np.array(calibration_dict_right["rvecs"])[mask_right])
    tvecs_right = list(np.array(calibration_dict_right["tvecs"])[mask_right])

    imgpoints_left = list(np.array(calibration_dict_left["imgpoints"])[mask_left])
    imgpoints_right = list(np.array(calibration_dict_right["imgpoints"])[mask_right])

    # change this if stereo calibration not good.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    stereocalibration_flags = cv2.CALIB_USE_EXTRINSIC_GUESS + cv2.CALIB_FIX_INTRINSIC
    ret, CM_left, D_left, CM_right, D_right, R, T, E, F, per_view_errors = cv2.stereoCalibrateExtended(
        objectPoints=objpoints_left,
        imagePoints1=imgpoints_left,
        imagePoints2=imgpoints_right,
        cameraMatrix1=mtx_left,
        distCoeffs1=dist_left,
        cameraMatrix2=mtx_right,
        distCoeffs2=dist_right,
        imageSize=(width, height),
        R=np.eye(3),
        T=np.array(
            [[-1.1], [0.], [0.]]),
        criteria=criteria,
        flags=stereocalibration_flags)

    return {"rmse": ret,
            "K1": CM_left, "D1": D_left,
            "K2": CM_right, "D2": D_right,
            "R": R, "T": T, "E": E, "F": F,  # R, T extrinsics from right to left
            "image_points_left": imgpoints_left,
            "image_points_right": imgpoints_right,
            "object_points": objpoints_left,  # left and right should now be the same
            "rvecs_left": rvecs_left,  # rotate objpts to left frame
            "tvecs_left": tvecs_left,  # translate objpts to left frame
            "rvecs_right": rvecs_right,  # rotate objpts to right frame
            "tvecs_right": tvecs_right,  # translate objpts to right frame
            "per_view_errors": per_view_errors,  # per view errors for both left and right
            "image_indices": image_indices_left}


def refine_stereo_calibration(stereo_calibration_dict: dict, threshold: float, width: int, height: int) -> dict:
    # get the goodies
    mtx_left = stereo_calibration_dict["K1"]
    dist_left = stereo_calibration_dict["D1"]
    mtx_right = stereo_calibration_dict["K2"]
    dist_right = stereo_calibration_dict["D2"]
    R = stereo_calibration_dict["R"]
    T = stereo_calibration_dict["T"]
    per_view_errors = stereo_calibration_dict["per_view_errors"]
    image_points_left = stereo_calibration_dict["image_points_left"]
    image_points_right = stereo_calibration_dict["image_points_right"]
    rvectors_left = stereo_calibration_dict["rvecs_left"]
    tvectors_left = stereo_calibration_dict["tvecs_left"]
    rvectors_right = stereo_calibration_dict["rvecs_right"]
    tvectors_right = stereo_calibration_dict["tvecs_right"]
    object_points = stereo_calibration_dict["object_points"]
    image_indices = stereo_calibration_dict["image_indices"]

    # remove the badies
    mask1 = per_view_errors[:, 0] < threshold
    mask2 = per_view_errors[:, 1] < threshold
    indices = np.arange(per_view_errors[:, 0].shape[0])[mask1 & mask2]
    if len(indices) < 2:
        raise ValueError(f"Number of low rsme image sets too low: {len(indices)}!")
    image_indices = list(np.array(image_indices)[indices])
    good_image_points_left = list(np.array(image_points_left)[indices])
    good_image_points_right = list(np.array(image_points_right)[indices])
    good_object_points = list(np.array(object_points)[indices])
    good_rvectors_left = list(np.array(rvectors_left)[indices])
    good_tvectors_left = list(np.array(tvectors_left)[indices])
    good_rvectors_right = list(np.array(rvectors_right)[indices])
    good_tvectors_right = list(np.array(tvectors_right)[indices])

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000000, 0.00000001)
    stereocalibration_flags = cv2.CALIB_USE_EXTRINSIC_GUESS + cv2.CALIB_FIX_INTRINSIC

    ret, CM_left, D_left, CM_right, D_right, new_R, new_T, E, F, good_per_view_errors = cv2.stereoCalibrateExtended(
        objectPoints=good_object_points,
        imagePoints1=good_image_points_left,
        imagePoints2=good_image_points_right,
        cameraMatrix1=mtx_left,
        distCoeffs1=dist_left,
        cameraMatrix2=mtx_right,
        distCoeffs2=dist_right,
        imageSize=(width, height),
        R=R,
        T=T,
        criteria=criteria,
        flags=stereocalibration_flags)

    return {"rmse": ret,
            "K1": CM_left, "D1": D_left,
            "K2": CM_right, "D2": D_right,
            "R": new_R, "T": new_T, "E": E, "F": F,  # R, T extrinsics from right to left
            "image_points_left": good_image_points_left,
            "image_points_right": good_image_points_right,
            "object_points": good_object_points,
            "rvecs_left": good_rvectors_left,
            "tvecs_left": good_tvectors_left,
            "rvecs_right": good_rvectors_right,
            "tvecs_right": good_tvectors_right,
            "per_view_errors": good_per_view_errors,
            "image_indices": image_indices}


def check_stereo_reprojection(stereo_dict: dict, images_left: List[np.ndarray], images_right: List[np.ndarray],
                              save_path: Path) -> None:
    camera_save_path = save_path / 'stereo'
    camera_save_path.mkdir(exist_ok=True, parents=True)

    K1, D1 = stereo_dict["K1"], stereo_dict["D1"]
    K2, D2 = stereo_dict["K2"], stereo_dict["D2"]

    R = stereo_dict["R"]  # coordinates 2 -> 1
    T = stereo_dict["T"]  # coordinates 2 -> 1

    image_indices = stereo_dict["image_indices"]
    images_left = list(np.array(images_left)[image_indices])
    images_right = list(np.array(images_right)[image_indices])
    assert len(images_left) == len(images_right)

    rvecs_left = stereo_dict['rvecs_left']
    tvecs_left = stereo_dict['tvecs_left']

    tvecs_right = stereo_dict['tvecs_right']
    rvecs_right = stereo_dict['rvecs_right']

    image_points_left = stereo_dict["image_points_left"]
    image_points_right = stereo_dict["image_points_right"]

    object_points_left = stereo_dict["object_points"]
    object_points_right = stereo_dict["object_points"]

    h, w, c = images_left[0].shape
    window_name = f"stereo calibration reprojection images"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    view = True
    print("Instructions for viewing pleasure: Esc to quit viewing, q to quit.")
    for idx, img_left, img_right, imgpts_left, imgpts_right, objptsl, objptsr, rvecl, rvecr, tvecl, tvecr in zip(
            image_indices,
            images_left,
            images_right,
            image_points_left,
            image_points_right,
            object_points_left,
            object_points_right,
            rvecs_left,
            rvecs_right,
            tvecs_left,
            tvecs_right):

        img_display = np.zeros((h, 2 * w, c), dtype=np.uint8)
        img_left = copy.copy(img_left)
        img_right = copy.copy(img_right)
        imgpts_left = copy.copy(imgpts_left).reshape(-1, 2)
        imgpts_right = copy.copy(imgpts_right).reshape(-1, 2)
        # project the right points to the left imaqge
        for uv, obj in zip(imgpts_left, objptsr):
            R_to_camera = cv2.Rodrigues(rvecr)
            proj_mtx = np.block([[np.linalg.inv(R), -T], [0, 0, 0, 1]]) @ \
                       np.block([[R_to_camera[0], tvecr], [0, 0, 0, 1]])
            projpts_left, _ = cv2.projectPoints(obj, cv2.Rodrigues(proj_mtx[:3, :3])[0], proj_mtx[:3, 3], K1, D1)
            cv2.circle(img_left, tuple(uv.astype(int)), 4, (255, 0, 0), -1)  # blue for detected on left
            cv2.circle(img_left, projpts_left.reshape(2).astype(int), 3, (0, 0, 255), -1)  # red for reprojection

            projpts_right, _ = cv2.projectPoints(obj, rvecr, tvecr, K2, D2)
            cv2.circle(img_right, projpts_right.reshape(2).astype(int), 2, (0, 255, 0), -1)
            # green for original detected on right
            # green and blue should be the same

        # # project the left points to the right image
        for uv, obj in zip(imgpts_right, objptsl):
            R_to_camera = cv2.Rodrigues(rvecl)
            # move obj points to the left camera frame
            proj_mtx = np.block([[R, T], [0, 0, 0, 1]]) @ \
                       np.block([[R_to_camera[0], tvecl], [0, 0, 0, 1]])

            projpts_right, _ = cv2.projectPoints(obj, cv2.Rodrigues(proj_mtx[:3, :3])[0], proj_mtx[:3, 3], K2, D2)

            cv2.circle(img_right, tuple(uv.astype(int)), 4, (255, 0, 0), -1)  # blue for detected on right
            cv2.circle(img_right, projpts_right.reshape(2).astype(int), 3, (0, 0, 255), -1)  # red for reprojection

            projpts_left, _ = cv2.projectPoints(obj, rvecl, tvecl, K1, D1)
            cv2.circle(img_left, projpts_left.reshape(2).astype(int), 2, (0, 255, 0), -1)
            # green for original detected on left
            # green and blue should be the same

        img_display[:, :w, :] = img_left
        img_display[:, w:, :] = img_right
        if view == True:
            cv2.imshow(window_name, img_display)
        cv2.imwrite(str(camera_save_path / f"stereo_reprojection_{str(idx).zfill(4)}.png"), img_display)
        k = cv2.waitKey(0)
        if k == 27:
            print("Done lookin'")
            view = False
            cv2.destroyAllWindows()
        elif k == ord('q'):
            print('Quitting!')
            sys.exit(0)


def stereo_rectify(refined_stereo_calibration_dict: dict, width: int, height: int) -> dict:
    rmse = refined_stereo_calibration_dict["rmse"]
    K1 = refined_stereo_calibration_dict["K1"]
    D1 = refined_stereo_calibration_dict["D1"]
    K2 = refined_stereo_calibration_dict["K2"]
    D2 = refined_stereo_calibration_dict["D2"]
    R = refined_stereo_calibration_dict["R"]
    T = refined_stereo_calibration_dict["T"]
    F = refined_stereo_calibration_dict["F"]
    E = refined_stereo_calibration_dict["E"]

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1=K1, distCoeffs1=D1,
                                                      cameraMatrix2=K2, distCoeffs2=D2,
                                                      imageSize=(width, height),
                                                      R=R, T=T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)
    return {"rmse": rmse, "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q, "roi1": roi1, "roi2": roi2,
            "width": width, "height": height, "F": F, "E": E, "R": R, "T": T,
            "K1": K1, "K2": K2, "D1": D1, "D2": D2}


def save_stereo_rectify(filename, rectify_dict: dict) -> None:
    np.savez_compressed(filename,
                        rmse=rectify_dict["rmse"],
                        width=rectify_dict["width"],
                        heigth=rectify_dict["height"],
                        E=rectify_dict["E"],
                        F=rectify_dict["F"],
                        Q=rectify_dict["Q"],
                        R=rectify_dict["R"],
                        T=rectify_dict["T"],
                        R1=rectify_dict["R1"],
                        P1=rectify_dict["P1"],
                        K1=rectify_dict["K1"],
                        D1=rectify_dict["D1"],
                        roi1=rectify_dict["roi1"],
                        R2=rectify_dict["R2"],
                        P2=rectify_dict["P2"],
                        K2=rectify_dict["K2"],
                        D2=rectify_dict["D2"],
                        roi2=rectify_dict["roi2"])


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


def save_stereo_calibration(filename: Path, image_paths: List[Path], stereo_calibration_dict: dict) -> None:
    np.savez_compressed(filename,
                        image_paths=image_paths,
                        rmse=stereo_calibration_dict["rmse"],
                        K1=stereo_calibration_dict["K1"],
                        D1=stereo_calibration_dict["D1"],
                        K2=stereo_calibration_dict["K2"],
                        D2=stereo_calibration_dict["D2"],
                        R=stereo_calibration_dict["R"],
                        T=stereo_calibration_dict["T"],
                        E=stereo_calibration_dict["E"],
                        F=stereo_calibration_dict["F"],
                        image_points_left=stereo_calibration_dict["image_points_left"],
                        image_points_right=stereo_calibration_dict["image_points_right"],
                        object_points=stereo_calibration_dict["object_points"],
                        rvecs_left=stereo_calibration_dict["rvecs_left"],
                        tvecs_left=stereo_calibration_dict["tvecs_left"],
                        rvecs_right=stereo_calibration_dict["rvecs_right"],
                        tvecs_right=stereo_calibration_dict["tvecs_right"],
                        per_view_errors=stereo_calibration_dict["per_view_errors"],
                        image_indices=stereo_calibration_dict["image_indices"])


def print_calibration_results(calibration_dict: dict, refined_calibration_dict: dict, camera: str) -> None:
    print(f"\n{camera} results:\n")
    print(f"Initial {camera} results:\n")
    print(f"Initial RSME: {calibration_dict['rmse']:.2f}")
    for k in list(calibration_dict.keys()):
        if camera == "stereo":
            if k == "K1" or k == "K2" or k == "D1" or k == "D2" or k == "R" or k == "T" or k == "E" or k == "E" or k == "F":
                print(f"{k}:\n{calibration_dict[k]}")
        else:
            if k == "K" or k == "D":
                print(f"{k}:\n{calibration_dict[k]}")

    print(f"\nRefined {camera} results:\n")
    print(f"Refined RSME: {refined_calibration_dict['rmse']:.2f}")
    for k in list(refined_calibration_dict.keys()):
        if camera == "stereo":
            if k == "K1" or k == "K2" or k == "D1" or k == "D2" or k == "R" or k == "T" or k == "E" or k == "E" or k == "F":
                print(f"{k}:\n{refined_calibration_dict[k]}")
        else:
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

    left_images = sorted((root / args.left_folder).glob('*.png'))
    right_images = sorted((root / args.right_folder).glob('*.png'))
    print(f"Found: {len(left_images)} left images and {len(right_images)} right images...")
    left_images_list = []
    right_images_list = []
    for im1, im2 in tqdm(zip(left_images, right_images), desc='Loading images...'):
        _im = cv2.imread(str(im1), 1)
        left_images_list.append(_im)

        _im = cv2.imread(str(im2), 1)
        right_images_list.append(_im)
    height, width, _ = left_images_list[0].shape

    if not args.precomputed_left:
        calibration_dict1 = calibrate_camera(args, left_images_list, args.rows, args.columns, args.size)
        save_camera_calibration(calibration_dict=calibration_dict1,
                                image_paths=left_images,
                                filename=data_path / "left_calibration.npz")
    else:
        calibration_dict1 = np.load(data_path / "left_calibration.npz")

    refined_calibration_dict1 = refine_calibration(calibration_dict1, width, height, threshold=args.camera_threshold)
    if args.check_left:
        check_calibration_reprojection(refined_calibration_dict1, left_images_list, 'left', figs_path)
    save_camera_calibration(calibration_dict=refined_calibration_dict1,
                            image_paths=left_images,
                            filename=data_path / "left_refined_calibration.npz")

    print_calibration_results(calibration_dict1, refined_calibration_dict1, 'left')

    if not args.precomputed_right:
        calibration_dict2 = calibrate_camera(args,
                                             right_images_list,
                                             args.rows,
                                             args.columns,
                                             args.size)

        save_camera_calibration(calibration_dict=calibration_dict2,
                                image_paths=right_images,
                                filename=data_path / "right_calibration.npz")
    else:
        calibration_dict2 = np.load(data_path / "right_calibration.npz")

    # refine cals
    refined_calibration_dict2 = refine_calibration(calibration_dict2, width, height, threshold=args.camera_threshold)

    if args.check_right:
        check_calibration_reprojection(refined_calibration_dict2, right_images_list, 'right', figs_path)

    save_camera_calibration(calibration_dict=refined_calibration_dict2,
                            image_paths=right_images,
                            filename=data_path / "right_refined_calibration.npz")

    print_calibration_results(calibration_dict2, refined_calibration_dict2, 'right')

    plot_and_save_corners(figs_path,
                          width, height,
                          calibration_dict1["imgpoints"],
                          calibration_dict2["imgpoints"],
                          calibration_dict1["per_view_errors"],
                          calibration_dict2["per_view_errors"])

    plot_and_save_corners(figs_path,
                          width, height,
                          refined_calibration_dict1["imgpoints"],
                          refined_calibration_dict2["imgpoints"],
                          refined_calibration_dict1["per_view_errors"],
                          refined_calibration_dict2["per_view_errors"],
                          filename="refined_detected_corners",
                          title="Refined detected corners for each camera coloured by RMSE")

    if not args.precomputed_stereo:
        stereo_calibration_dict = stereo_calibrate(refined_calibration_dict1, refined_calibration_dict2, width, height)
        save_stereo_calibration(data_path / "stereo_calibration.npz", left_images, stereo_calibration_dict)

        plot_and_save_corners(figs_path,
                              width, height,
                              stereo_calibration_dict["image_points_left"],
                              stereo_calibration_dict["image_points_right"],
                              stereo_calibration_dict["per_view_errors"][:, 0],
                              stereo_calibration_dict["per_view_errors"][:, 1],
                              filename="detected_stereo_corners",
                              title="Detected stereo corners for each camera coloured by RMSE")

    else:
        stereo_calibration_dict = np.load(data_path / "stereo_calibration.npz")

    refined_stereo_calibration_dict = refine_stereo_calibration(stereo_calibration_dict, args.stereo_threshold,
                                                                width, height)

    print_calibration_results(stereo_calibration_dict, refined_stereo_calibration_dict, "stereo")

    if args.check_stereo:
        check_stereo_reprojection(refined_stereo_calibration_dict, left_images_list, right_images_list, figs_path)

    save_stereo_calibration(data_path / "refined_stereo_calibration.npz",
                            left_images,
                            refined_stereo_calibration_dict)

    rectify_dict = stereo_rectify(refined_stereo_calibration_dict, width, height)
    save_stereo_rectify(data_path / "rectified_stereo_calibration.npz", rectify_dict)

    plot_and_save_corners(figs_path,
                          width, height,
                          refined_stereo_calibration_dict["image_points_left"],
                          refined_stereo_calibration_dict["image_points_right"],
                          refined_stereo_calibration_dict["per_view_errors"][:, 0],
                          refined_stereo_calibration_dict["per_view_errors"][:, 1],
                          filename="refined_detected_stereo_corners",
                          title="Refined detected stereo corners for each camera coloured by RMSE")

    plot_undistorted_images_and_epipoles(figs_path / "undistorted", left_images_list, right_images_list, rectify_dict)
    if not args.no_output_kitti:
        save_kitti_calibs(data_path / 'calibration.txt', rectify_dict)


def make_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Calibrate from a set of synchronised left and right images.")
    parser.add_argument('--root', type=str,
                        help='Root to left and right images')
    parser.add_argument('--left_folder', type=str, default='left',
                        help='Folder containing the left images. Default=\'left\'.')
    parser.add_argument('--right_folder', type=str, default='right',
                        help='Folder containing the right images. Default=\'right\'.')
    parser.add_argument('--precomputed_left', action="store_true", default=False,
                        help='Use precomputed left calibrations matricies stored in outputs\/data/. Default=False')
    parser.add_argument('--precomputed_right', action="store_true", default=False,
                        help='Use precomputed right calibrations matricies stored in outputs\/data/. Default=False')
    parser.add_argument('--precomputed_stereo', action="store_true", default=False,
                        help='Use precomputed stereo extrinsic matricies stored in outputs\/data/. Default=False')
    parser.add_argument('--check_left', action="store_true", default=False,
                        help='Check the re-projections of each set of left corners using the refined results. Default=False')
    parser.add_argument('--check_right', action="store_true", default=False,
                        help='Check the re-projections of each set of right corners using the refined results. Default=False')
    parser.add_argument('--check_stereo', action="store_true", default=False,
                        help='Check the re-projections of the refined stereo calibrations. Default=False')
    parser.add_argument('--columns', type=int, default=7,
                        help='Number of columns on the checkerboard. Default=7')
    parser.add_argument('--rows', type=int, default=5,
                        help='Number of rows on the checkerboard. Default=5')
    parser.add_argument('--size', type=float, default=0.139,
                        help='Realworld measurement of checkerboard squares. Default=0.140 (m)')
    parser.add_argument('--camera_threshold', type=float, default=0.5,
                        help='RMSE threshold for filtering image and object points during camera calibration '
                             'refinement.')
    parser.add_argument('--stereo_threshold', type=float, default=10,
                        help='RMSE threshold for filtering image and object points during stereo calibration '
                             'refinement.')
    parser.add_argument('--show_calibration_checkerboards', action='store_true', default=False,
                        help='Show checkerboards with their detected corners overlaid. Default=False')
    parser.add_argument('--show_stereo_checkerboards', action='store_true', default=False,
                        help='Show stereo checkerboards with their detected corners overlaid. Default=False')
    parser.add_argument('--no_output_kitti', action='store_true', default=False,
                        help='Do not output kitti calib.txt. Default=False')
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    run(args)
