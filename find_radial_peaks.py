import argparse
import csv


PEAK_FIELDS = [
    "q_center_inv_angstrom",
    "d_center_angstrom",
    "mean_intensity",
    "normalized_intensity",
    "pixel_count",
]


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def read_retained_points(path, q_min):
    points = []
    with open(path, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            q = parse_float(row.get("q_center_inv_angstrom"))
            d_spacing = parse_float(row.get("d_center_angstrom"))
            mean_intensity = parse_float(row.get("mean_intensity"))
            pixel_count = parse_int(row.get("pixel_count"))

            if q is None or d_spacing is None or mean_intensity is None:
                continue
            if pixel_count is None or pixel_count <= 0:
                continue
            if q < q_min:
                continue

            points.append(
                {
                    "q_center_inv_angstrom": q,
                    "d_center_angstrom": d_spacing,
                    "mean_intensity": mean_intensity,
                    "pixel_count": pixel_count,
                }
            )

    return points


def find_local_maxima(points):
    peaks = []
    for previous_point, point, next_point in zip(points, points[1:], points[2:]):
        if (
            point["mean_intensity"] > previous_point["mean_intensity"]
            and point["mean_intensity"] > next_point["mean_intensity"]
        ):
            peaks.append(dict(point))

    max_intensity = max((peak["mean_intensity"] for peak in peaks), default=0.0)
    for peak in peaks:
        peak["normalized_intensity"] = (
            peak["mean_intensity"] / max_intensity if max_intensity else 0.0
        )

    return peaks


def print_peak_table(peaks):
    print(
        "q_center_inv_angstrom,d_center_angstrom,mean_intensity,"
        "normalized_intensity,pixel_count"
    )
    for peak in peaks:
        print(
            f"{peak['q_center_inv_angstrom']:.6f},"
            f"{peak['d_center_angstrom']:.6f},"
            f"{peak['mean_intensity']:.6g},"
            f"{peak['normalized_intensity']:.6f},"
            f"{peak['pixel_count']}"
        )


def write_peaks_csv(path, peaks):
    with open(path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PEAK_FIELDS)
        writer.writeheader()
        writer.writerows(peaks)


def main():
    parser = argparse.ArgumentParser(
        description="Find simple local maxima in a radial diffraction profile CSV."
    )
    parser.add_argument("--input-csv", required=True, help="Input radial profile CSV.")
    parser.add_argument("--q-min", type=float, default=0.4)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--output-csv", help="Optional CSV for all peaks sorted by q.")
    args = parser.parse_args()

    if args.top < 1:
        raise ValueError("--top must be positive")

    points = read_retained_points(args.input_csv, args.q_min)
    peaks = find_local_maxima(points)
    peaks_by_q = sorted(peaks, key=lambda peak: peak["q_center_inv_angstrom"])
    top_peaks = sorted(
        peaks,
        key=lambda peak: peak["normalized_intensity"],
        reverse=True,
    )[: args.top]

    print(f"Input file: {args.input_csv}")
    print(f"q-min: {args.q_min}")
    print(f"Retained radial points: {len(points)}")
    print(f"Local maxima found: {len(peaks)}")
    print()
    print("All peaks sorted by q:")
    print_peak_table(peaks_by_q)
    print()
    print(f"Top {args.top} peaks by normalized intensity:")
    print_peak_table(top_peaks)

    if args.output_csv:
        write_peaks_csv(args.output_csv, peaks_by_q)
        print()
        print(f"Wrote peaks CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
