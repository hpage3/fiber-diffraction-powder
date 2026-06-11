import argparse
import math
from collections import defaultdict
from pathlib import Path


BASELINE_TWIST_DEG = 30.0


class PdbAtom:
    def __init__(self, line, index):
        self.line = line.rstrip("\n")
        self.index = index
        self.record = self.line[0:6]
        self.atom_serial = self.line[6:11]
        self.atom_name = self.line[12:16]
        self.residue_name = self.line[17:20]
        self.chain_id = self.line[21:22]
        self.residue_number = int(self.line[22:26])
        self.x = float(self.line[30:38])
        self.y = float(self.line[38:46])
        self.z = float(self.line[46:54])
        self.element = self.line[76:78] if len(self.line) >= 78 else ""

    def with_xy(self, x, y):
        padded = self.line
        if len(padded) < 80:
            padded = padded.ljust(80)
        return f"{padded[:30]}{x:8.3f}{y:8.3f}{self.z:8.3f}{padded[54:]}"


def parse_pdb(path):
    atoms = []
    lines = []
    with open(path, "r", encoding="utf-8") as pdb_file:
        for line in pdb_file:
            lines.append(line.rstrip("\n"))
            if line.startswith(("ATOM  ", "HETATM")):
                atoms.append(PdbAtom(line, len(lines) - 1))
    return lines, atoms


def centroid(points):
    count = len(points)
    return (
        sum(point[0] for point in points) / count,
        sum(point[1] for point in points) / count,
        sum(point[2] for point in points) / count,
    )


def group_layers(atoms, strand_length, num_strands):
    layers = defaultdict(list)
    strands = defaultdict(list)
    max_residue = strand_length * num_strands
    for atom in atoms:
        if atom.residue_number < 1 or atom.residue_number > max_residue:
            raise ValueError(
                f"Residue number {atom.residue_number} is outside expected 1..{max_residue}"
            )
        layer_index = (atom.residue_number - 1) % strand_length
        strand_index = (atom.residue_number - 1) // strand_length
        layers[layer_index].append(atom)
        strands[strand_index].append(atom)

    missing_layers = [index for index in range(strand_length) if index not in layers]
    missing_strands = [index for index in range(num_strands) if index not in strands]
    if missing_layers:
        raise ValueError(f"Missing layer indices: {missing_layers}")
    if missing_strands:
        raise ValueError(f"Missing strand indices: {missing_strands}")

    return dict(layers), dict(strands)


def layer_centroids(layers):
    centers = {}
    for layer_index, atoms in layers.items():
        centers[layer_index] = centroid([(atom.x, atom.y, atom.z) for atom in atoms])
    return centers


def rotate_xy(x, y, center_x, center_y, angle_deg):
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    dx = x - center_x
    dy = y - center_y
    return (
        center_x + dx * cos_a - dy * sin_a,
        center_y + dx * sin_a + dy * cos_a,
    )


def write_variant(lines, layers, output_path, center_x, center_y, target_twist):
    output_lines = list(lines)
    reference_layer = (len(layers) - 1) / 2.0
    for layer_index, atoms in layers.items():
        delta_angle = (target_twist - BASELINE_TWIST_DEG) * (layer_index - reference_layer)
        for atom in atoms:
            x, y = rotate_xy(atom.x, atom.y, center_x, center_y, delta_angle)
            output_lines[atom.index] = atom.with_xy(x, y)

    with open(output_path, "w", encoding="ascii", newline="\n") as pdb_file:
        for line in output_lines:
            pdb_file.write(line + "\n")


def atom_coordinates(atoms):
    return [(atom.x, atom.y, atom.z) for atom in atoms]


def parse_output_atoms(path):
    _, atoms = parse_pdb(path)
    return atoms


def rmsd(coords_a, coords_b):
    if len(coords_a) != len(coords_b):
        raise ValueError("Cannot compute RMSD with different atom counts")
    total = 0.0
    for a, b in zip(coords_a, coords_b):
        total += (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    return math.sqrt(total / len(coords_a))


def radius_of_gyration_xy(atoms, center_x, center_y):
    total = 0.0
    for atom in atoms:
        total += (atom.x - center_x) ** 2 + (atom.y - center_y) ** 2
    return math.sqrt(total / len(atoms))


def unwrap_angles(angles):
    if not angles:
        return []
    unwrapped = [angles[0]]
    for angle in angles[1:]:
        candidate = angle
        while candidate - unwrapped[-1] > 180.0:
            candidate -= 360.0
        while candidate - unwrapped[-1] < -180.0:
            candidate += 360.0
        unwrapped.append(candidate)
    return unwrapped


def measure_layer_angles(layer_centers, center_x, center_y):
    angles = []
    for index in sorted(layer_centers):
        x, y, _ = layer_centers[index]
        angles.append(math.degrees(math.atan2(y - center_y, x - center_x)))
    return unwrap_angles(angles)


def mean_adjacent_twist(layer_centers, center_x, center_y):
    angles = measure_layer_angles(layer_centers, center_x, center_y)
    deltas = [angles[index + 1] - angles[index] for index in range(len(angles) - 1)]
    if not deltas:
        return 0.0
    return sum(deltas) / len(deltas)


def residue_orientation_centers(atoms, strand_length):
    residues = defaultdict(list)
    for atom in atoms:
        layer_index = (atom.residue_number - 1) % strand_length
        strand_index = (atom.residue_number - 1) // strand_length
        if strand_index == 0:
            residues[layer_index].append(atom)

    centers = {}
    for layer_index, residue_atoms in residues.items():
        centers[layer_index] = centroid(
            [(atom.x, atom.y, atom.z) for atom in residue_atoms]
        )
    return centers


def validate_variant(
    path,
    input_atoms,
    input_layer_z,
    strand_length,
    num_strands,
    center_x,
    center_y,
):
    atoms = parse_output_atoms(path)
    layers, _ = group_layers(atoms, strand_length, num_strands)
    centers = layer_centroids(layers)
    orientation_centers = residue_orientation_centers(atoms, strand_length)
    z_max_delta = max(
        abs(centers[index][2] - input_layer_z[index]) for index in range(strand_length)
    )
    return {
        "atom_count": len(atoms),
        "rmsd": rmsd(atom_coordinates(input_atoms), atom_coordinates(atoms)),
        "max_layer_z_delta": z_max_delta,
        "radius_gyration_xy": radius_of_gyration_xy(atoms, center_x, center_y),
        "measured_adjacent_twist": mean_adjacent_twist(
            orientation_centers, center_x, center_y
        ),
    }


def write_report(
    report_path,
    input_path,
    output_rows,
    atom_count,
    strand_length,
    num_strands,
    center_x,
    center_y,
):
    with open(report_path, "w", encoding="utf-8", newline="\n") as report:
        report.write("# Compact Hexaplex Twist Variant Report\n\n")
        report.write(f"- Input file: `{input_path}`\n")
        report.write(
            f"- Grouping: {num_strands} sequential strands x {strand_length} residues; "
            "`layer_index = (residue_number - 1) % strand_length`\n"
        )
        report.write(
            "- Helix axis assumption: z-axis; rotation center is the all-atom x/y centroid "
            f"({center_x:.6f}, {center_y:.6f}); original atom z coordinates are preserved.\n"
        )
        report.write(f"- Input atom count: {atom_count}\n\n")
        report.write("## Validation Table\n\n")
        report.write(
            "| Target twist | Output | Atom count | RMSD vs input (A) | "
            "Max layer z delta (A) | XY radius of gyration (A) | "
            "Raw marker twist | Baseline-corrected measured twist |\n"
        )
        report.write("|---:|---|---:|---:|---:|---:|---:|---:|\n")
        for row in output_rows:
            report.write(
                f"| {row['target_twist']:.1f} | `{row['output']}` | {row['atom_count']} | "
                f"{row['rmsd']:.6f} | {row['max_layer_z_delta']:.6e} | "
                f"{row['radius_gyration_xy']:.6f} | {row['measured_adjacent_twist']:.6f} | "
                f"{row['baseline_corrected_twist']:.6f} |\n"
            )
        report.write("\n## Warnings And Limitations\n\n")
        report.write(
            "- These variants are controlled rigid-layer transforms of the compact ideal "
            "30 degree model, not energy-relaxed pNAB conformers.\n"
        )
        report.write(
            "- Raw marker twist is computed from the strand-1 residue centroid in each "
            "layer around the chosen z-axis center. Because the antiparallel compact model "
            "has an intrinsic marker-angle offset, the baseline-corrected measured twist "
            "adds the measured change relative to the regenerated 30 degree model back to "
            "the 30 degree baseline.\n"
        )
        report.write(
            "- PDB coordinate fields are rewritten with standard 8.3 precision; atom, "
            "residue, serial, chain, occupancy, temperature, and element fields are retained "
            "from the input line layout.\n"
        )
        report.write("- Diffraction was not run in this Track B generation pass.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate compact rigid-layer hexaplex twist variants from an ideal PDB."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--twists", nargs="+", type=float, required=True)
    parser.add_argument("--strand-length", type=int, required=True)
    parser.add_argument("--num-strands", type=int, required=True)
    parser.add_argument(
        "--report",
        default="outputs/reports/compact_twist_variant_report.md",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    report_path = Path(args.report)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines, atoms = parse_pdb(input_path)
    expected_residues = args.strand_length * args.num_strands
    observed_residues = sorted({atom.residue_number for atom in atoms})
    if len(atoms) == 0:
        raise ValueError("No ATOM/HETATM records found")
    if len(observed_residues) != expected_residues:
        raise ValueError(
            f"Expected {expected_residues} residues, found {len(observed_residues)}"
        )

    layers, _ = group_layers(atoms, args.strand_length, args.num_strands)
    centers = layer_centroids(layers)
    input_layer_z = {index: centers[index][2] for index in centers}
    center_x, center_y, _ = centroid(atom_coordinates(atoms))

    output_rows = []
    for twist in args.twists:
        twist_label = f"{twist:g}".replace(".", "p")
        output_path = out_dir / f"compact_hexaplex_twist_{twist_label}.pdb"
        write_variant(lines, layers, output_path, center_x, center_y, twist)
        validation = validate_variant(
            output_path,
            atoms,
            input_layer_z,
            args.strand_length,
            args.num_strands,
            center_x,
            center_y,
        )
        output_rows.append(
            {
                "target_twist": twist,
                "output": output_path.as_posix(),
                **validation,
            }
        )

    baseline_rows = [
        row for row in output_rows if abs(row["target_twist"] - BASELINE_TWIST_DEG) < 1e-9
    ]
    if not baseline_rows:
        raise ValueError("The twist list must include the 30 degree baseline for validation")
    baseline_marker_twist = baseline_rows[0]["measured_adjacent_twist"]
    for row in output_rows:
        row["baseline_corrected_twist"] = (
            BASELINE_TWIST_DEG
            + row["measured_adjacent_twist"]
            - baseline_marker_twist
        )

    write_report(
        report_path,
        input_path.as_posix(),
        output_rows,
        len(atoms),
        args.strand_length,
        args.num_strands,
        center_x,
        center_y,
    )

    for row in output_rows:
        print(
            f"{row['target_twist']:g} deg -> {row['output']} "
            f"atoms={row['atom_count']} rmsd={row['rmsd']:.6f} "
            f"z_delta={row['max_layer_z_delta']:.3e} "
            f"rg_xy={row['radius_gyration_xy']:.6f} "
            f"measured_twist={row['baseline_corrected_twist']:.6f}"
        )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
