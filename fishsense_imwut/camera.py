from typing import Tuple

import numpy as np


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
