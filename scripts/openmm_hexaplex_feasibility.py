import argparse
import csv
import importlib.util
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


RELATED_PACKAGES = [
    ("openmm", "openmm"),
    ("pdbfixer", "pdbfixer"),
    ("openff", "openff"),
    ("openff-toolkit", "openff.toolkit"),
    ("openmmforcefields", "openmmforcefields"),
    ("rdkit", "rdkit"),
    ("openbabel", "openbabel"),
    ("parmed", "parmed"),
    ("mdtraj", "mdtraj"),
    ("MDAnalysis", "MDAnalysis"),
]


STANDARD_FORCEFIELDS = [
    ("amber14-all.xml + amber14/tip3pfb.xml", ["amber14-all.xml", "amber14/tip3pfb.xml"]),
    ("amber14/protein.ff14SB.xml + amber14/tip3pfb.xml", ["amber14/protein.ff14SB.xml", "amber14/tip3pfb.xml"]),
]


class Atom:
    def __init__(self, line):
        self.name = line[12:16].strip()
        self.residue_name = line[17:20].strip()
        self.chain_id = line[21:22].strip()
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


def package_available(module_name):
    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def environment_status():
    return {
        label: "available" if package_available(module) else "not available"
        for label, module in RELATED_PACKAGES
    }


def read_atoms(path):
    atoms = []
    with open(path, "r", encoding="utf-8") as pdb_file:
        for line in pdb_file:
            if line.startswith(("ATOM  ", "HETATM")):
                atoms.append(Atom(line))
    return atoms


def coords_array(atoms):
    return np.array([[atom.x, atom.y, atom.z] for atom in atoms], dtype=float)


def radius_of_gyration(atoms):
    coords = coords_array([atom for atom in atoms if atom.is_heavy])
    center = coords.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((coords - center) ** 2, axis=1))))


def count_close_heavy_pairs(atoms, threshold=1.5):
    heavy_atoms = [atom for atom in atoms if atom.is_heavy]
    coords = coords_array(heavy_atoms)
    residues = np.array([atom.residue_number for atom in heavy_atoms], dtype=int)
    count = 0
    nearest = math.inf
    for start in range(len(coords)):
        deltas = coords[start + 1 :] - coords[start]
        if deltas.size == 0:
            continue
        distances = np.sqrt(np.sum(deltas**2, axis=1))
        mask = residues[start + 1 :] != residues[start]
        distances = distances[mask]
        if distances.size == 0:
            continue
        nearest = min(nearest, float(distances.min()))
        count += int(np.count_nonzero(distances < threshold))
    return nearest, count


def pdb_static_summary(path):
    atoms = read_atoms(path)
    residues = Counter(atom.residue_name for atom in atoms)
    chain_ids = {atom.chain_id for atom in atoms}
    chain_count = len(chain_ids) if chain_ids != {""} else 0
    nearest, close_count = count_close_heavy_pairs(atoms, threshold=1.5)
    return {
        "atom_count_text_parse": len(atoms),
        "heavy_atom_count_text_parse": sum(1 for atom in atoms if atom.is_heavy),
        "residue_name_counts": dict(sorted(residues.items())),
        "chain_count_text_parse": chain_count,
        "radius_gyration_heavy_A": radius_of_gyration(atoms),
        "nearest_nonbonded_proxy_heavy_distance_A": nearest,
        "heavy_pairs_lt_1p5_A_excluding_same_residue": close_count,
    }


def try_openmm_diagnostics(path, forcefield_mode):
    result = {
        "openmm_available": package_available("openmm"),
        "pdbfile_loads": False,
        "topology_created": False,
        "openmm_atom_count": "",
        "openmm_residue_count": "",
        "openmm_chain_count": "",
        "openmm_bond_count": "",
        "system_created": False,
        "forcefield_attempt": "",
        "system_error": "",
        "load_error": "",
    }
    if not result["openmm_available"]:
        result["load_error"] = "OpenMM is not installed in the repo virtualenv."
        return result

    try:
        from openmm.app import ForceField, PDBFile
    except Exception as exc:
        result["load_error"] = f"Could not import OpenMM app API: {exc}"
        return result

    try:
        pdb = PDBFile(str(path))
        result["pdbfile_loads"] = True
        topology = pdb.topology
        result["topology_created"] = True
        result["openmm_atom_count"] = sum(1 for _ in topology.atoms())
        result["openmm_residue_count"] = sum(1 for _ in topology.residues())
        result["openmm_chain_count"] = sum(1 for _ in topology.chains())
        result["openmm_bond_count"] = sum(1 for _ in topology.bonds())
    except Exception as exc:
        result["load_error"] = str(exc)
        return result

    forcefields = STANDARD_FORCEFIELDS
    if forcefield_mode != "standard":
        result["system_error"] = f"Unsupported forcefield mode: {forcefield_mode}"
        return result

    errors = []
    for label, xml_files in forcefields:
        result["forcefield_attempt"] = label
        try:
            forcefield = ForceField(*xml_files)
            forcefield.createSystem(topology)
            result["system_created"] = True
            result["system_error"] = ""
            return result
        except Exception as exc:
            errors.append(f"{label}: {exc}")

    result["system_error"] = " | ".join(errors)
    return result


def write_csv(path, rows):
    fieldnames = [
        "pdb",
        "atom_count_text_parse",
        "heavy_atom_count_text_parse",
        "residue_name_counts",
        "chain_count_text_parse",
        "radius_gyration_heavy_A",
        "nearest_nonbonded_proxy_heavy_distance_A",
        "heavy_pairs_lt_1p5_A_excluding_same_residue",
        "openmm_available",
        "pdbfile_loads",
        "topology_created",
        "openmm_atom_count",
        "openmm_residue_count",
        "openmm_chain_count",
        "openmm_bond_count",
        "system_created",
        "forcefield_attempt",
        "load_error",
        "system_error",
    ]
    with open(path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output_row = dict(row)
            residue_counts = output_row.get("residue_name_counts", {})
            output_row["residue_name_counts"] = ";".join(
                f"{name}:{count}" for name, count in residue_counts.items()
            )
            writer.writerow(output_row)


def write_report(path, env, rows, static_summaries):
    with open(path, "w", encoding="utf-8", newline="\n") as report:
        report.write("# OpenMM Hexaplex Feasibility Report\n\n")
        report.write("## Environment And Tool Availability\n\n")
        for label, status in env.items():
            report.write(f"- {label}: {status}\n")
        report.write("\n")
        report.write("## PDB Diagnostics\n\n")
        report.write(
            "| PDB | Text atoms | Heavy atoms | Residue names | Text chain count | "
            "OpenMM loads | Topology created | OpenMM atoms | OpenMM residues | "
            "OpenMM chains | OpenMM bonds | Force-field system created |\n"
        )
        report.write("|---|---:|---:|---|---:|---|---|---:|---:|---:|---:|---|\n")
        for row in rows:
            static = static_summaries[row["pdb"]]
            residue_counts = ", ".join(
                f"{name}:{count}" for name, count in static["residue_name_counts"].items()
            )
            report.write(
                f"| `{row['pdb']}` | {row['atom_count_text_parse']} | "
                f"{row['heavy_atom_count_text_parse']} | {residue_counts} | "
                f"{row['chain_count_text_parse']} | {row['pdbfile_loads']} | "
                f"{row['topology_created']} | {row['openmm_atom_count']} | "
                f"{row['openmm_residue_count']} | {row['openmm_chain_count']} | "
                f"{row['openmm_bond_count']} | {row['system_created']} |\n"
            )
        report.write("\n## Parameterization Result\n\n")
        if all(row["openmm_available"] is False for row in rows):
            report.write(
                "OpenMM is not installed in the repo virtualenv, so OpenMM PDBFile loading, "
                "Topology inspection, and ForceField system creation cannot be attempted in "
                "the current environment.\n\n"
            )
        else:
            for row in rows:
                report.write(f"- `{row['pdb']}`\n")
                if row["load_error"]:
                    report.write(f"  - Load/topology error: {row['load_error']}\n")
                if row["system_error"]:
                    report.write(f"  - System creation error: {row['system_error']}\n")
                if row["system_created"]:
                    report.write("  - A ForceField System was created successfully.\n")
        report.write("## Static Contact Context\n\n")
        for row in rows:
            report.write(
                f"- `{row['pdb']}`: heavy-atom Rg {row['radius_gyration_heavy_A']:.3f} A; "
                f"nearest nonbonded-proxy heavy distance "
                f"{row['nearest_nonbonded_proxy_heavy_distance_A']:.3f} A; "
                f"same-residue-excluded heavy pairs <1.5 A: "
                f"{row['heavy_pairs_lt_1p5_A_excluding_same_residue']}.\n"
            )
        report.write("\n## Feasibility Assessment\n\n")
        report.write(
            "Restrained minimization is not currently feasible in this repo environment. "
            "The required OpenMM package is absent, and no supporting parameterization "
            "toolkit such as pdbfixer, OpenFF Toolkit, openmmforcefields, RDKit, Open Babel, "
            "ParmEd, MDTraj, or MDAnalysis is available in the repo virtualenv.\n\n"
        )
        report.write(
            "Even after OpenMM is installed, these compact hexaplexes use nonstandard "
            "proto-nucleic-acid chemistry. Standard DNA/RNA templates should not be assumed "
            "to apply. A safe minimization workflow would need explicit residue templates, "
            "bond definitions, atom types, charges, inter-residue connectivity, and terminal "
            "definitions. Only after those are validated should a restrained local "
            "minimization smoke test be run on copied output files.\n\n"
        )
        report.write("## Recommended Next Steps\n\n")
        report.write(
            "1. Install OpenMM and a suitable parameterization stack in a dedicated environment.\n"
        )
        report.write(
            "2. Build or obtain templates/topology for the nonstandard hexaplex residues.\n"
        )
        report.write(
            "3. Validate topology and charges first on the compact 30 degree control.\n"
        )
        report.write(
            "4. If the 30 degree control parameterizes cleanly, test a heavily restrained "
            "minimization on a copy, then compare RMSD, close contacts, and Rg before/after.\n"
        )
        report.write(
            "5. Only then attempt the strained 24 degree variant with the same restraints "
            "and diagnostics.\n\n"
        )
        report.write("No MD, minimization, chemistry repair, or diffraction rerun was performed.\n")


def main():
    parser = argparse.ArgumentParser(
        description="OpenMM feasibility diagnostics for compact hexaplex PDBs."
    )
    parser.add_argument("--pdb", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--forcefield-mode", default="standard")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path("outputs/reports/openmm_hexaplex_feasibility_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "openmm_hexaplex_feasibility.csv"

    env = environment_status()
    rows = []
    static_summaries = {}
    for pdb_text in args.pdb:
        pdb_path = Path(pdb_text)
        static = pdb_static_summary(pdb_path)
        static_summaries[pdb_text] = static
        openmm_result = try_openmm_diagnostics(pdb_path, args.forcefield_mode)
        rows.append(
            {
                "pdb": pdb_text,
                **static,
                **openmm_result,
            }
        )

    write_csv(csv_path, rows)
    write_report(report_path, env, rows, static_summaries)
    print(f"Wrote {csv_path}")
    print(f"Wrote {report_path}")
    for row in rows:
        print(
            f"{row['pdb']}: openmm={row['openmm_available']} "
            f"loads={row['pdbfile_loads']} system={row['system_created']} "
            f"atoms={row['atom_count_text_parse']} heavy={row['heavy_atom_count_text_parse']}"
        )


if __name__ == "__main__":
    main()
