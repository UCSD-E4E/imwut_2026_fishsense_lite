from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_position_percent_error(
    calibrations_dict: Dict[str, Tuple[float, str, List[np.ndarray]]],
    laser_params: np.ndarray,
    step_count: int,
    type: str,
    start: int = None,
    end: int = None,
) -> None:
    if start is None:
        start = 2
    if end is None:
        end = step_count

    fig, ax = plt.subplots()

    for key, calibrations_by_std in calibrations_dict.items():
        name, linestyle = key.split("/")

        for std, arrays in calibrations_by_std:
            percent_error = np.abs(
                (np.array(arrays) - np.tile(laser_params, (len(arrays), 1)))
                / np.tile(laser_params, (len(arrays), 1))
                * 100
            )

            direction_percent_error = percent_error[:,][:, 0:3]
            position_percent_error = np.mean(percent_error[:,][:, 3:5], axis=1)

            ax.plot(
                np.arange(start, end),
                position_percent_error[start - 2 : end - 2],
                label=f"Std {std} Position Percent Error ({name})",
                linestyle=linestyle,
            )

    ax.set_title(
        f"Position Percent Error vs Number of Points Used for Calibration ({type})"
    )
    ax.set_xlabel("Number of Points Used for Calibration")
    ax.set_ylabel("Percent Error (%)")
    ax.legend()

    return fig


def plot_mean_reconstruction_error(
    calibrations_dict: Dict[str, Tuple[float, str, List[np.ndarray]]],
    inverted_camera_intrinsics: np.ndarray,
    image_points: np.ndarray,
    world_points: np.ndarray,
    step_count: int,
    type: str,
    start: int = None,
    end: int = None,
) -> None:
    if start is None:
        start = 2
    if end is None:
        end = step_count

    fig, ax = plt.subplots()

    for key, calibrations_by_std in calibrations_dict.items():
        name, linestyle = key.split("/")

        for std, arrays in calibrations_by_std:
            mean_errors_noisy = []

            for calibration_noisy in arrays:
                laser_axis_noisy = calibration_noisy[0:3]

                laser_origin_noisy = np.zeros((3,))
                laser_origin_noisy[0:2] = calibration_noisy[3:5]
                laser_origin_noisy[2] = 0.0

                projected_points = inverted_camera_intrinsics @ image_points
                norms = np.linalg.norm(projected_points, axis=0)
                final_laser_axis = -projected_points / norms

                point_constants_noisy = (
                    (final_laser_axis.T @ laser_origin_noisy)
                    - (laser_axis_noisy.T @ laser_origin_noisy)
                    * (laser_axis_noisy.T @ final_laser_axis)
                ) / (1 - (laser_axis_noisy.T @ final_laser_axis) ** 2)
                world_points_noisy = (
                    np.tile(point_constants_noisy, (3, 1)) * final_laser_axis
                )
                mean_error_noisy = np.mean(
                    np.sqrt(np.sum((world_points_noisy - world_points) ** 2, axis=1))
                )

                mean_errors_noisy.append(mean_error_noisy)

            ax.plot(
                np.arange(start, end),
                mean_errors_noisy[start - 2 : end - 2],
                label=f"Std {std} Mean Reconstruction Error ({name})",
                linestyle=linestyle,
            )

    ax.set_title(
        f"Mean Reconstruction Error vs Number of Points Used for Calibration ({type})"
    )
    ax.set_xlabel("Number of Points Used for Calibration")
    ax.set_ylabel("Mean Reconstruction Error (m)")
    ax.legend()
