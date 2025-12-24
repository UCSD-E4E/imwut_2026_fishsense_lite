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


# for std, arrays, arrays_reconstructed, _, _ in calibrations_by_stds:
#     percent_error = np.abs(
#         (np.array(arrays) - np.tile(laser_params, (len(arrays), 1)))
#         / np.tile(laser_params, (len(arrays), 1))
#         * 100
#     )
#     percent_error_reconstructed = np.abs(
#         (
#             np.array(arrays_reconstructed)
#             - np.tile(laser_params, (len(arrays_reconstructed), 1))
#         )
#         / np.tile(laser_params, (len(arrays_reconstructed), 1))
#         * 100
#     )

#     direction_percent_error = percent_error[:,][:, 0:3]
#     position_percent_error = np.mean(percent_error[:,][:, 3:5], axis=1)

#     direction_percent_error_reconstructed = percent_error_reconstructed[:,][:, 0:3]
#     position_percent_error_reconstructed = np.mean(
#         percent_error_reconstructed[:,][:, 3:5], axis=1
#     )

#     # plt.plot(np.arange(2, STEP_COUNT), position_percent_error, label=f'Std {std} Position Percent Error')
#     plt.plot(
#         np.arange(2, STEP_COUNT)[100:],
#         position_percent_error_reconstructed[100:],
#         linestyle="dashed",
#         label=f"Std {std} Position Percent Error (Reconstructed)",
#     )

# plt.title(
#     "Position Percent Error vs Number of Points Used for Calibration (Continuous)"
# )
# plt.xlabel("Number of Points Used for Calibration")
# plt.ylabel("Percent Error")
# plt.legend()
