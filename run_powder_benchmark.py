import argparse
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from orientation_average import average_powder_diffraction
from scripts import atomic_number


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
        description="Generate a small powder-style orientation averaging benchmark."
    )
    parser.add_argument(
        "--coordinate-file",
        default="inputs/Hexaplex_AntiParallel_30deg_Ideal_no_h_deduped.xyz",
        help="Input XYZ coordinate file.",
    )
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--grid-limit", type=float, default=100.0)
    parser.add_argument("--theta-count", type=int, default=4)
    parser.add_argument("--phi-count", type=int, default=8)
    parser.add_argument("--psi-count", type=int, default=3)
    parser.add_argument("--theta-max", type=float, default=180.0)
    parser.add_argument("--output-prefix", default="outputs/hexaplex_powder_test")
    args = parser.parse_args()
    start_time = time.perf_counter()

    orientation_count = args.theta_count * args.phi_count * args.psi_count
    detector_pixels = args.grid_size * args.grid_size
    wavelength = 0.7749e-7
    distance_to_detector = 338.4
    z_grid_limits = [-args.grid_limit, args.grid_limit]
    x_grid_limits = [-args.grid_limit, args.grid_limit]

    print(f"Input file: {args.coordinate_file}")
    atoms, coords = load_xyz(args.coordinate_file)
    print(f"Atoms: {len(atoms)}")
    print(f"Grid size: {args.grid_size} x {args.grid_size}")
    print(f"Detector pixels: {detector_pixels}")
    print(f"Number of atoms: {len(atoms)}")
    print(f"Number of orientations: {orientation_count}")
    print(
        "Approximate atom-pixel-orientation operations: "
        f"{len(atoms) * detector_pixels * orientation_count}"
    )

    coords *= 1e-7  # Angstroms to mm
    atomic_numbers = atomic_numbers_for(atoms)

    diffraction_data = average_powder_diffraction(
        atomic_numbers,
        coords,
        wavelength,
        distance_to_detector,
        z_grid_limits,
        x_grid_limits,
        args.grid_size,
        args.grid_size,
        args.theta_count,
        args.phi_count,
        args.psi_count,
        theta_max=args.theta_max,
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
    plt.title("Powder-style orientation benchmark")
    plt.tight_layout()
    plt.savefig(png_file, dpi=150)

    print(f"Saved {npy_file}")
    print(f"Saved {png_file}")
    print(f"Min intensity: {diffraction_data.min()}")
    print(f"Max intensity: {max_intensity}")
    print(f"Mean intensity: {diffraction_data.mean()}")
    elapsed_seconds = time.perf_counter() - start_time
    seconds_per_orientation = elapsed_seconds / orientation_count
    print(f"Total wall-clock runtime, seconds: {elapsed_seconds:.3f}")
    print(f"Average seconds per orientation: {seconds_per_orientation:.3f}")


if __name__ == "__main__":
    main()
