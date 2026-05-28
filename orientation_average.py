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


def powder_orientations(theta_count, phi_count, psi_count, theta_max=180.0):
    if theta_count < 1 or phi_count < 1 or psi_count < 1:
        raise ValueError("theta_count, phi_count, and psi_count must all be positive")

    cos_min = np.cos(np.deg2rad(theta_max))
    cos_max = 1.0
    bin_width = (cos_max - cos_min) / theta_count
    cos_theta_values = cos_max - (np.arange(theta_count) + 0.5) * bin_width
    theta_values = np.rad2deg(np.arccos(np.clip(cos_theta_values, -1.0, 1.0)))

    phi_values = np.linspace(0.0, 360.0, phi_count, endpoint=False)
    psi_values = np.linspace(0.0, 360.0, psi_count, endpoint=False)
    total = theta_count * phi_count * psi_count
    index = 0

    for theta in theta_values:
        for phi in phi_values:
            for psi in psi_values:
                index += 1
                yield {
                    "theta": float(theta),
                    "phi": float(phi),
                    "psi": float(psi),
                    "index": index,
                    "total": total,
                }


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


def average_powder_diffraction(
    atomic_numbers,
    coords,
    wavelength,
    distance_to_detector,
    z_grid_limits,
    x_grid_limits,
    z_grid_size,
    x_grid_size,
    theta_count,
    phi_count,
    psi_count,
    theta_max=180.0,
):
    diffraction_data = np.zeros((z_grid_size, x_grid_size))

    for orientation in powder_orientations(
        theta_count,
        phi_count,
        psi_count,
        theta_max=theta_max,
    ):
        theta = orientation["theta"]
        phi = orientation["phi"]
        psi = orientation["psi"]
        print(
            "Calculating powder orientation "
            f"{orientation['index']}/{orientation['total']}: "
            f"theta={theta:g}, phi={phi:g}, psi={psi:g}"
        )

        rotation = np.dot(Rz(phi), np.dot(Ry(theta), Rz(psi)))
        rotated_coords = np.dot(rotation, coords.T).T
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
