import numpy as np
from pathlib import Path


def convert_2_kitti_str(calib: np.array) -> str:
    kitti_calib = ""
    calib = calib.reshape(-1)
    for index in range(len(calib)):
        if index < len(calib) - 1:
            kitti_calib += str(calib[index]) + " "
        else:
            kitti_calib += str(calib[index])
    return kitti_calib


def save_calib(calib_path: Path, calib: dict) -> None:
    P0 = convert_2_kitti_str(calib['P0'])
    P1 = convert_2_kitti_str(calib['P1'])
    P2 = convert_2_kitti_str(calib['P2'])
    P3 = convert_2_kitti_str(calib['P3'])
    R_rect = convert_2_kitti_str(calib['R_rect'][:3, :3])
    Tr_velo_cam = convert_2_kitti_str(calib['Tr_velo_cam'][:3, :4])
    Tr_imu_velo = convert_2_kitti_str(calib['Tr_imu_velo'][:3, :4])
    with open(str(calib_path), 'w') as f:
        f.write("P0: " + P0 +
                "\nP1: " + P1 +
                "\nP2: " + P2 +
                "\nP3: " + P3 +
                "\nR_rect: " + R_rect +
                "\nTr_velo_cam: " + Tr_velo_cam +
                "\nTr_imu_velo: " + Tr_imu_velo)
