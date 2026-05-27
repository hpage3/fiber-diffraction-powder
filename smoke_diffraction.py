import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts import (
    atomic_number,
    helix_maker,
    generate_fiber_diffraction,
    Rx,
    Ry,
    Rz,
)

coordinates_file = "geometries/TAP_4MCyCo6.xyz"

# Tiny test values so this runs quickly
rise = 3.4
twist = -26.67
number_of_hexads = 3

tilts = [6]
rotations = [0]

wavelength = 0.7749e-7
distance_to_detector = 338.4

z_grid_limits = [-100.0, 100.0]
x_grid_limits = [-100.0, 100.0]
z_grid_size = 31
x_grid_size = 31

coords = np.loadtxt(coordinates_file, usecols=(1, 2, 3), skiprows=2)
atoms = np.loadtxt(coordinates_file, usecols=(0,), skiprows=2, dtype="str")

atoms, coords = helix_maker(atoms, coords, rise, twist, number_of_hexads)

coords *= 1e-7  # Angstroms to mm
atomic_numbers = np.array([atomic_number[a] for a in atoms])

diffraction_data = np.zeros((z_grid_size, x_grid_size))

for tilt in tilts:
    tilted_coords = np.dot(Ry(tilt), coords.T).T

    for rotation in rotations:
        print(f"Tilt: {tilt}, Rotation: {rotation}")
        rotated_coords = np.dot(Rz(rotation), tilted_coords.T).T

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

np.save("smoke_diffraction.npy", diffraction_data)

plt.figure(figsize=(6, 6))
plt.imshow(
    diffraction_data,
    extent=(x_grid_limits[0], x_grid_limits[1], z_grid_limits[0], z_grid_limits[1]),
    cmap="gray_r",
    vmax=np.max(diffraction_data) * 0.01,
)
plt.xlabel("x detector position, mm")
plt.ylabel("z detector position, mm")
plt.title("Smoke-test fiber diffraction")
plt.tight_layout()
plt.savefig("smoke_diffraction.png", dpi=150)

print("Saved smoke_diffraction.npy")
print("Saved smoke_diffraction.png")
print("Shape:", diffraction_data.shape)
print("Min:", diffraction_data.min())
print("Max:", diffraction_data.max())
print("Mean:", diffraction_data.mean())
