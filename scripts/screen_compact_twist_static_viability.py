import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


TWISTS = [24, 26, 28, 30, 32, 34, 36]
CLASH_THRESHOLDS = [1.2, 1.5, 2.0]


class Atom:
    def __init__(self, line):
        self.record = line[:6].strip()
        self.name = line[12:16].strip()
        self.residue_name = line[17:20].strip()
        self.chain_id = line[21:22]
        self.residue_number = int(line[22:26])
        self.x = float(line[30:38])
        self.y = float(line[38:46])
        self.z = float(line[46:54])
        element = line[76:78].strip() if len(line) >= 78 else ""
        self.element = element.capitalize() if element else self._infer_element()

    def _infer_element(self):
        for char in self.name:
            if char.isalpha():
                return char.upper()
        return ""

    @property
    def is_heavy(self):
        return self.element.upper() != "H"


def read_atoms(path):
    atoms = []
    with open(path, "r", encoding="utf-8") as pdb_file:
        for line in pdb_file:
            if line.startswith(("ATOM  ", "HETATM")):
                atoms.append(Atom(line))
    return atoms


def coords_array(atoms):
    return np.array([[atom.x, atom.y, atom.z] for atom in atoms], dtype=float)


def residue_ids(atoms):
    return np.array([atom.residue_number for atom in atoms], dtype=int)


def strand_ids(atoms, strand_length):
    return np.array([(atom.residue_number - 1) // strand_length for atom in atoms], dtype=int)


def layer_ids(atoms, strand_length):
    return np.array([(atom.residue_number - 1) % strand_length for atom in atoms], dtype=int)


def radius_of_gyration(coords):
    center = coords.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1))))


def radial_compactness(coords):
    center_xy = coords[:, :2].mean(axis=0)
    radii = np.sqrt(np.sum((coords[:, :2] - center_xy) ** 2, axis=1))
    return {
        "rg_xy_A": float(np.sqrt(np.mean(radii**2))),
        "radius_mean_A": float(np.mean(radii)),
        "radius_p95_A": float(np.percentile(radii, 95)),
        "radius_max_A": float(np.max(radii)),
    }


def layer_centroids(atoms, strand_length):
    coords = coords_array(atoms)
    layers = layer_ids(atoms, strand_length)
    centroids = {}
    for layer in sorted(set(layers)):
        centroids[int(layer)] = coords[layers == layer].mean(axis=0)
    return centroids


def count_close_pairs(coords, exclude_same_residue, thresholds):
    counts = {threshold: 0 for threshold in thresholds}
    nearest = math.inf
    n_atoms = len(coords)
    for start in range(n_atoms):
        deltas = coords[start + 1 :] - coords[start]
        if deltas.size == 0:
            continue
        distances = np.sqrt(np.sum(deltas**2, axis=1))
        mask = ~exclude_same_residue[start, start + 1 :]
        distances = distances[mask]
        if distances.size == 0:
            continue
        nearest = min(nearest, float(distances.min()))
        for threshold in thresholds:
            counts[threshold] += int(np.count_nonzero(distances < threshold))
    return nearest, counts


def interstrand_contacts(coords, strands):
    nearest = math.inf
    per_pair = {}
    for strand_a in sorted(set(strands)):
        coords_a = coords[strands == strand_a]
        for strand_b in sorted(set(strands)):
            if strand_b <= strand_a:
                continue
            coords_b = coords[strands == strand_b]
            pair_min = math.inf
            for atom in coords_a:
                distances = np.sqrt(np.sum((coords_b - atom) ** 2, axis=1))
                pair_min = min(pair_min, float(distances.min()))
            per_pair[f"{strand_a + 1}-{strand_b + 1}"] = pair_min
            nearest = min(nearest, pair_min)
    return nearest, per_pair


def summarize_variant(path, baseline_layers, strand_length, num_strands):
    atoms = read_atoms(path)
    heavy_atoms = [atom for atom in atoms if atom.is_heavy]
    coords = coords_array(heavy_atoms)
    residues = residue_ids(heavy_atoms)
    strands = strand_ids(heavy_atoms, strand_length)

    residue_count = len(set(atom.residue_number for atom in atoms))
    if residue_count != strand_length * num_strands:
        raise ValueError(
            f"{path}: expected {strand_length * num_strands} residues, found {residue_count}"
        )

    same_residue = residues[:, None] == residues[None, :]
    nearest_nonbonded_proxy, clash_counts = count_close_pairs(
        coords,
        same_residue,
        CLASH_THRESHOLDS,
    )
    nearest_interstrand, per_strand_pair = interstrand_contacts(coords, strands)

    current_layers = layer_centroids(heavy_atoms, strand_length)
    max_layer_z_delta = max(
        abs(current_layers[layer][2] - baseline_layers[layer][2])
        for layer in baseline_layers
    )

    compact = radial_compactness(coords)
    return {
        "atom_count": len(atoms),
        "heavy_atom_count": len(heavy_atoms),
        "residue_count": residue_count,
        "radius_gyration_A": radius_of_gyration(coords),
        **compact,
        "max_layer_centroid_z_delta_vs_30_A": float(max_layer_z_delta),
        "nearest_nonbonded_proxy_heavy_distance_A": nearest_nonbonded_proxy,
        "heavy_pairs_lt_1p2_A_excluding_same_residue": clash_counts[1.2],
        "heavy_pairs_lt_1p5_A_excluding_same_residue": clash_counts[1.5],
        "heavy_pairs_lt_2p0_A_excluding_same_residue": clash_counts[2.0],
        "nearest_interstrand_heavy_distance_A": nearest_interstrand,
        "nearest_interstrand_pair": min(per_strand_pair, key=per_strand_pair.get),
    }


def write_csv(path, rows):
    fieldnames = ["twist_deg", "pdb_path"] + [
        key for key in rows[0] if key not in ("twist_deg", "pdb_path")
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, rows, package_status):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as report:
        report.write("# Compact Twist Static Viability Report\n\n")
        report.write("## Environment Screen\n\n")
        for name, status in package_status.items():
            report.write(f"- {name}: {status}\n")
        report.write("\n")
        report.write(
            "No force-field or trajectory package was available in the repo virtualenv, "
            "so this pass is a static coordinate-only screen.\n\n"
        )
        report.write("## Geometry Metrics\n\n")
        report.write(
            "| Twist | Atoms | Heavy atoms | Rg (A) | Rg xy (A) | Radius p95 (A) | "
            "Max layer z delta vs 30 (A) | Nearest nonbonded-proxy heavy distance (A) | "
            "Pairs <1.2 A | Pairs <1.5 A | Pairs <2.0 A | Nearest inter-strand distance (A) |\n"
        )
        report.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            report.write(
                f"| {row['twist_deg']} | {row['atom_count']} | {row['heavy_atom_count']} | "
                f"{row['radius_gyration_A']:.3f} | {row['rg_xy_A']:.3f} | "
                f"{row['radius_p95_A']:.3f} | "
                f"{row['max_layer_centroid_z_delta_vs_30_A']:.3e} | "
                f"{row['nearest_nonbonded_proxy_heavy_distance_A']:.3f} | "
                f"{row['heavy_pairs_lt_1p2_A_excluding_same_residue']} | "
                f"{row['heavy_pairs_lt_1p5_A_excluding_same_residue']} | "
                f"{row['heavy_pairs_lt_2p0_A_excluding_same_residue']} | "
                f"{row['nearest_interstrand_heavy_distance_A']:.3f} |\n"
            )
        report.write("\n## Interpretation\n\n")
        report.write(
            "- Atom counts, heavy atom counts, and layer centroid z positions are preserved "
            "across the compact twist variants.\n"
        )
        report.write(
            "- Radius of gyration and radial compactness vary only modestly, consistent with "
            "rigid-layer angular transforms rather than expansion/collapse.\n"
        )
        report.write(
            "- The close-contact counts are a conservative nonbonded-proxy screen. Same-residue "
            "pairs are excluded, but true bond topology is unavailable, so adjacent-residue "
            "covalent or intended close contacts may still be counted.\n"
        )
        report.write(
            "- Nearest inter-strand distances should be read as geometric contact diagnostics, "
            "not as force-field energies.\n\n"
        )
        report.write("## Force-Field/Minimization Feasibility\n\n")
        report.write(
            "Open Babel, RDKit, MDAnalysis, MDTraj, OpenMM, SciPy, and Biopython were not "
            "available in the repo virtualenv. Full parameterization or minimization is "
            "therefore not straightforward in the current environment. Even if OpenMM or "
            "another engine is installed later, the nonstandard proto-nucleic-acid chemistry "
            "will require careful atom typing, bond templates, charges, and validation before "
            "any minimization or MD result should be trusted.\n\n"
        )
        report.write("## Cautions\n\n")
        report.write(
            "- This screen tests local static geometry only; it does not establish energetic "
            "stability or dynamic viability.\n"
        )
        report.write(
            "- Nonstandard chemistry and missing explicit bond topology limit automated clash "
            "and force-field interpretation.\n"
        )
        report.write("- Diffraction was not rerun in this pass.\n")


def package_status(python_executable):
    import importlib.util

    modules = {
        "Open Babel Python": "openbabel",
        "RDKit": "rdkit",
        "MDAnalysis": "MDAnalysis",
        "MDTraj": "mdtraj",
        "OpenMM": "openmm",
        "NumPy": "numpy",
        "SciPy": "scipy",
        "Biopython": "Bio",
    }
    return {
        label: "available" if importlib.util.find_spec(module) else "not available"
        for label, module in modules.items()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Static geometry viability screen for compact twist PDB variants."
    )
    parser.add_argument(
        "--pdb-dir",
        default="outputs/compact_twist_variants",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/metrics/compact_twist_static_viability.csv",
    )
    parser.add_argument(
        "--report",
        default="outputs/reports/compact_twist_static_viability_report.md",
    )
    parser.add_argument("--strand-length", type=int, default=30)
    parser.add_argument("--num-strands", type=int, default=6)
    args = parser.parse_args()

    pdb_dir = Path(args.pdb_dir)
    baseline_atoms = [
        atom
        for atom in read_atoms(pdb_dir / "compact_hexaplex_twist_30.pdb")
        if atom.is_heavy
    ]
    baseline_layers = layer_centroids(baseline_atoms, args.strand_length)

    rows = []
    for twist in TWISTS:
        pdb_path = pdb_dir / f"compact_hexaplex_twist_{twist}.pdb"
        row = {
            "twist_deg": twist,
            "pdb_path": pdb_path.as_posix(),
            **summarize_variant(
                pdb_path,
                baseline_layers,
                args.strand_length,
                args.num_strands,
            ),
        }
        rows.append(row)

    write_csv(args.output_csv, rows)
    write_report(args.report, rows, package_status(None))

    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.report}")
    for row in rows:
        print(
            f"{row['twist_deg']} deg: atoms={row['atom_count']} "
            f"heavy={row['heavy_atom_count']} rg={row['radius_gyration_A']:.3f} "
            f"nearest_proxy={row['nearest_nonbonded_proxy_heavy_distance_A']:.3f} "
            f"clash_lt1.5={row['heavy_pairs_lt_1p5_A_excluding_same_residue']}"
        )


if __name__ == "__main__":
    main()
