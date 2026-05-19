from typing import Tuple

import numpy as np

from fishsense_imwut.constants import (
    FOCAL_LENGTH_PX,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)


def make_camera_intrinsics(
    image_width: int = IMAGE_WIDTH,
    image_height: int = IMAGE_HEIGHT,
    focal_length_px: float = FOCAL_LENGTH_PX,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (K, K^-1) for a centered-principal-point pinhole camera.

    Defaults match the reference FishCamera hardware (Section 3.3).
    """
    K = np.array(
        [
            [focal_length_px, 0, image_width / 2],
            [0, focal_length_px, image_height / 2],
            [0, 0, 1],
        ]
    )
    return K, np.linalg.inv(K)


def point_to_plane(point: np.ndarray, std: float = 0.1) -> np.ndarray:
    random_normal = np.random.normal(0, std, (3,))
    random_normal /= np.linalg.norm(random_normal)

    if random_normal[2] < 0:
        random_normal = -random_normal

    return point, random_normal


def calculate_normals(points: np.ndarray) -> np.ndarray:
    np.random.seed(0)

    normals = []
    for point in points.T:
        _, normal = point_to_plane(point)

        normals.append(normal)

    normals = np.array(normals)
    return normals


def reconstruct_points(
    image_points: np.ndarray,
    inverted_camera_intrinsics: np.ndarray,
    laser_origin: np.ndarray,
    laser_axis: np.ndarray,
) -> Tuple[np.ndarray, float]:
    projected_points = inverted_camera_intrinsics @ image_points
    norms = np.linalg.norm(projected_points, axis=0)
    final_laser_axis = -projected_points / norms

    point_constants_noisy = (
        (final_laser_axis.T @ laser_origin)
        - (laser_axis.T @ laser_origin) * (laser_axis.T @ final_laser_axis)
    ) / (1 - (laser_axis.T @ final_laser_axis) ** 2)
    world_points = np.tile(point_constants_noisy, (3, 1)) * final_laser_axis

    L = laser_axis / np.linalg.norm(laser_axis)
    dot = (L.reshape(1, 3) @ final_laser_axis).ravel()
    denom = 1 - dot**2

    return world_points, denom
