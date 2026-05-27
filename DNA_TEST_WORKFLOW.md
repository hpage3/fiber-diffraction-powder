# DNA Test Workflow

## Purpose

This is a sanity-check workflow for running the existing fiber-diffraction
calculation on DNA structures without hydrogens. It does not add powder
diffraction behavior or change the diffraction algorithm.

## What the Workflow Does

- Starts from a DNA PDB file.
- Removes hydrogens by default.
- Removes HETATM records by default.
- Converts ATOM records to XYZ.
- Runs `run_fiber_benchmark.py` using the generated XYZ file.

## Important Note

DNA without hydrogens still includes phosphorus atoms in the phosphate backbone.
That is why `scripts.atomic_number` must support `"P": 15`.

## Example Directory Setup

```bash
mkdir -p inputs outputs
```

## Convert DNA PDB to XYZ

```bash
python pdb_to_xyz.py \
  --input-pdb inputs/dna_test.pdb \
  --output-xyz inputs/dna_test_no_h.xyz
```

## Run the Fiber Benchmark

```bash
python run_fiber_benchmark.py \
  --coordinate-file inputs/dna_test_no_h.xyz \
  --number-of-hexads 1 \
  --grid-size 101 \
  --tilts 2,6,10,14 \
  --rotations 0,30,60,90,120,150,180,210,240,270,300,330 \
  --output-prefix outputs/dna_test_fiber
```

Use `--number-of-hexads 1` as the safe default for an already-complete DNA PDB.
`helix_maker` duplicates the input coordinates into a helical stack, so if the
PDB already contains a full DNA fragment, blindly stacking it many times may
produce an artificial repeated superstructure.

## Expected Outputs

- `outputs/dna_test_fiber.npy`
- `outputs/dna_test_fiber.png`

## Troubleshooting

- Unsupported atom symbol: add element support or clean the PDB.
- Empty XYZ output: check ATOM records and filters.
- Very slow calculation: reduce grid size, orientations, or atom count.
- Matplotlib cache warning on headless server: usually harmless if PNG is
  produced.
