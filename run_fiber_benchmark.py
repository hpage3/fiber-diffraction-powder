import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts import (
    atomic_number,
    helix_maker,
)
from orientation_average import average_fiber_diffraction, parse_angle_list


def load_xyz(path):
    coords = np.loadtxt(path, usecols=(1, 2, 3), skiprows=2)
    atoms = np.loadtxt(path, usecols=(0,), skiprows=2, dtype="str")
    return atoms, coords


def atomic_numbers_for(atoms):
    unsupported = sorted({atom for atom in atoms if atom not in atomic_number})
    if unsupported:
        supported = ", ".join(sorted(atomic_number))
        unsupported_text = ", ".join(unsupported)
        raise ValueError(
            f"Unsupported atom symbol(s): {unsupported_text}. "
            f"Supported atom symbols: {supported}"
        )
    return np.array([atomic_number[atom] for atom in atoms])


def main():
    parser = argparse.ArgumentParser(
        description="Generate a modest fiber diffraction benchmark image."
    )
    parser.add_argument(
        "--coordinate-file",
        default="geometries/TAP_4MCyCo6.xyz",
        help="Input XYZ coordinate file.",
    )
    parser.add_argument("--rise", type=float, default=3.4)
    parser.add_argument("--twist", type=float, default=-26.67)
    parser.add_argument("--number-of-hexads", type=int, default=10)
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--grid-limit", type=float, default=100.0)
    parser.add_argument("--tilts", default="2,6,10,14", help="Comma-separated tilt angles.")
    parser.add_argument(
        "--rotations",
        default="0,30,60,90,120,150,180,210,240,270,300,330",
        help="Comma-separated rotation angles.",
    )
    parser.add_argument("--output-prefix", default="benchmark_fiber")
    args = parser.parse_args()

    tilts = parse_angle_list(args.tilts)
    rotations = parse_angle_list(args.rotations)
    orientation_count = len(tilts) * len(rotations)

    wavelength = 0.7749e-7
    distance_to_detector = 338.4
    z_grid_limits = [-args.grid_limit, args.grid_limit]
    x_grid_limits = [-args.grid_limit, args.grid_limit]

    print(f"Input file: {args.coordinate_file}")
    atoms, coords = load_xyz(args.coordinate_file)
    print(f"Atoms before helix_maker: {len(atoms)}")

    atoms, coords = helix_maker(
        atoms,
        coords,
        args.rise,
        args.twist,
        args.number_of_hexads,
    )
    print(f"Atoms after helix_maker: {len(atoms)}")
    print(f"Grid size: {args.grid_size} x {args.grid_size}")
    print(f"Number of orientations: {orientation_count}")

    coords *= 1e-7  # Angstroms to mm
    atomic_numbers = atomic_numbers_for(atoms)

    diffraction_data = average_fiber_diffraction(
        atomic_numbers,
        coords,
        wavelength,
        distance_to_detector,
        z_grid_limits,
        x_grid_limits,
        args.grid_size,
        args.grid_size,
        tilts,
        rotations,
        progress_callback=lambda tilt, rotation: print(
            f"Calculating tilt={tilt:g}, rotation={rotation:g}"
        ),
    )

    npy_file = f"{args.output_prefix}.npy"
    png_file = f"{args.output_prefix}.png"
    np.save(npy_file, diffraction_data)

    max_intensity = np.max(diffraction_data)
    vmax = max_intensity * 0.01 if max_intensity != 0 else None

    plt.figure(figsize=(6, 6))
    plt.imshow(
        diffraction_data,
        extent=(
            x_grid_limits[0],
            x_grid_limits[1],
            z_grid_limits[0],
            z_grid_limits[1],
        ),
        cmap="gray_r",
        vmax=vmax,
    )
    plt.xlabel("x detector position, mm")
    plt.ylabel("z detector position, mm")
    plt.title("Fiber diffraction benchmark")
    plt.tight_layout()
    plt.savefig(png_file, dpi=150)

    print(f"Saved {npy_file}")
    print(f"Saved {png_file}")
    print(f"Min intensity: {diffraction_data.min()}")
    print(f"Max intensity: {max_intensity}")
    print(f"Mean intensity: {diffraction_data.mean()}")


if __name__ == "__main__":
    main()
