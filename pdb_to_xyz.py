import argparse
import os
import sys
from collections import Counter


WATER_RESIDUES = {"HOH", "WAT", "H2O"}


def infer_element(atom_name):
    stripped = atom_name.strip()
    if not stripped:
        return ""

    for char in stripped:
        if char.isalpha():
            return char.upper()
    return ""


def parse_pdb_atom_line(line, line_number):
    try:
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
    except ValueError as exc:
        raise ValueError(f"Malformed coordinate record on line {line_number}") from exc

    atom_name = line[12:16]
    element = line[76:78].strip() if len(line) >= 78 else ""
    if not element:
        element = infer_element(atom_name)
    else:
        element = element.capitalize()

    return {
        "record": line[0:6].strip(),
        "atom_name": atom_name,
        "altloc": line[16:17],
        "residue_name": line[17:20].strip(),
        "x": x,
        "y": y,
        "z": z,
        "element": element,
    }


def convert_pdb_to_xyz(args):
    if not os.path.isfile(args.input_pdb):
        raise FileNotFoundError(f"Input PDB file not found: {args.input_pdb}")

    atoms = []
    atoms_read = 0
    skipped_hydrogens = 0
    skipped_waters = 0
    skipped_hetatm = 0
    skipped_altloc = 0
    skipped_exact_duplicates = 0

    with open(args.input_pdb, "r", encoding="utf-8") as pdb_file:
        for line_number, line in enumerate(pdb_file, start=1):
            record = line[0:6].strip()
            if record == "HETATM" and not args.include_hetatm:
                skipped_hetatm += 1
                continue
            if record != "ATOM" and not (args.include_hetatm and record == "HETATM"):
                continue

            atom = parse_pdb_atom_line(line, line_number)
            atoms_read += 1

            if atom["altloc"] not in (" ", "A"):
                skipped_altloc += 1
                continue

            if not args.include_water and atom["residue_name"] in WATER_RESIDUES:
                skipped_waters += 1
                continue

            if not args.keep_hydrogens and atom["element"].upper() == "H":
                skipped_hydrogens += 1
                continue

            atoms.append(atom)

    if args.dedupe_exact:
        deduped_atoms = []
        seen = set()
        for atom in atoms:
            key = (atom["element"], atom["x"], atom["y"], atom["z"])
            if key in seen:
                skipped_exact_duplicates += 1
                continue
            seen.add(key)
            deduped_atoms.append(atom)
        atoms = deduped_atoms

    if not atoms:
        raise ValueError("No atoms written after applying filters")

    filters = [
        f"include_hetatm={args.include_hetatm}",
        f"keep_hydrogens={args.keep_hydrogens}",
        f"include_water={args.include_water}",
        f"dedupe_exact={args.dedupe_exact}",
    ]
    comment = f"Converted from {args.input_pdb}; " + ", ".join(filters)

    with open(args.output_xyz, "w", encoding="utf-8") as xyz_file:
        xyz_file.write(f"{len(atoms)}\n")
        xyz_file.write(f"{comment}\n")
        for atom in atoms:
            xyz_file.write(
                f"{atom['element']:<2s} "
                f"{atom['x']:12.6f} {atom['y']:12.6f} {atom['z']:12.6f}\n"
            )

    return {
        "atoms_read": atoms_read,
        "atoms_written": len(atoms),
        "elements": Counter(atom["element"] for atom in atoms),
        "skipped_hydrogens": skipped_hydrogens,
        "skipped_waters": skipped_waters,
        "skipped_hetatm": skipped_hetatm,
        "skipped_altloc": skipped_altloc,
        "skipped_exact_duplicates": skipped_exact_duplicates,
    }


def print_summary(args, summary):
    print(f"Input file: {args.input_pdb}")
    print(f"Output file: {args.output_xyz}")
    print(f"Atoms read: {summary['atoms_read']}")
    print(f"Atoms written: {summary['atoms_written']}")
    print("Elements written:")
    for element, count in sorted(summary["elements"].items()):
        print(f"  {element}: {count}")
    print(f"Skipped hydrogens: {summary['skipped_hydrogens']}")
    print(f"Skipped waters: {summary['skipped_waters']}")
    print(f"Skipped HETATM records: {summary['skipped_hetatm']}")
    print(f"Skipped alternate locations: {summary['skipped_altloc']}")
    print(f"Skipped exact duplicates: {summary['skipped_exact_duplicates']}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert PDB ATOM/HETATM coordinates to XYZ format."
    )
    parser.add_argument("--input-pdb", required=True, help="Input PDB file path.")
    parser.add_argument("--output-xyz", required=True, help="Output XYZ file path.")
    parser.add_argument(
        "--include-hetatm",
        action="store_true",
        help="Include HETATM records in addition to ATOM records.",
    )
    parser.add_argument(
        "--keep-hydrogens",
        action="store_true",
        help="Keep hydrogen atoms instead of removing them.",
    )
    parser.add_argument(
        "--include-water",
        action="store_true",
        help="Keep water residues such as HOH, WAT, and H2O.",
    )
    parser.add_argument(
        "--dedupe-exact",
        action="store_true",
        help="Remove exact duplicate atoms after filtering, keyed by element and coordinates.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        summary = convert_pdb_to_xyz(args)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print_summary(args, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
