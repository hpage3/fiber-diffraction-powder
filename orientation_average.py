import numpy as np

from scripts import Rx, Ry, Rz, generate_fiber_diffraction


def parse_angle_list(text):
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def rotate_coordinates(coords, rx=0.0, ry=0.0, rz=0.0):
    rotated = np.dot(Rx(rx), coords.T).T
    rotated = np.dot(Ry(ry), rotated.T).T
    rotated = np.dot(Rz(rz), rotated.T).T
    return rotated


def fiber_orientations(tilts, rotations):
    for tilt in tilts:
        for rotation in rotations:
            yield {"tilt": tilt, "rotation": rotation}


def average_fiber_diffraction(
    atomic_numbers,
    coords,
    wavelength,
    distance_to_detector,
    z_grid_limits,
    x_grid_limits,
    z_grid_size,
    x_grid_size,
    tilts,
    rotations,
    progress_callback=None,
):
    diffraction_data = np.zeros((z_grid_size, x_grid_size))

    for orientation in fiber_orientations(tilts, rotations):
        tilt = orientation["tilt"]
        rotation = orientation["rotation"]
        if progress_callback is not None:
            progress_callback(tilt, rotation)

        rotated_coords = rotate_coordinates(coords, ry=tilt, rz=rotation)
        diffraction_data += generate_fiber_diffraction(
            atomic_numbers,
            rotated_coords,
            wavelength,
            distance_to_detector,
            z_grid_limits,
            x_grid_limits,
            z_grid_size,
            x_grid_size,
        )

    return diffraction_data
