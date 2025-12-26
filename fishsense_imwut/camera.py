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
