import argparse
import csv
from pathlib import Path


FEATURES = [
    ("3.4 A", "d_3p4_max_norm_intensity"),
    ("3.0 A", "d_3p0_max_norm_intensity"),
    ("4.5-5.0 A", "d_4p5_5p0_max_norm_intensity"),
    ("4.1 A", "d_4p1_max_norm_intensity"),
    ("5.5 A", "d_5p5_max_norm_intensity"),
    ("7.0 A", "d_7p0_max_norm_intensity"),
    ("8.4 A", "d_8p4_max_norm_intensity"),
]


def read_summary(path):
    with open(path, "r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def build_rankings(summary_rows):
    rankings = []
    for feature_window, field in FEATURES:
        values = []
        for row in summary_rows:
            value = row.get(field, "")
            if value == "":
                continue
            values.append(
                {
                    "feature_window": feature_window,
                    "twist_deg": float(row["twist"]),
                    "normalized_intensity": float(value),
                }
            )
        values.sort(key=lambda row: row["normalized_intensity"], reverse=True)
        for rank, row in enumerate(values, start=1):
            rankings.append(
                {
                    "feature_window": row["feature_window"],
                    "rank": rank,
                    "twist_deg": f"{row['twist_deg']:g}",
                    "normalized_intensity": f"{row['normalized_intensity']:.12g}",
                }
            )
    return rankings


def write_rankings(path, rankings):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "feature_window",
                "rank",
                "twist_deg",
                "normalized_intensity",
            ],
        )
        writer.writeheader()
        writer.writerows(rankings)


def main():
    parser = argparse.ArgumentParser(
        description="Rank compact twist feature responses from strongest to weakest."
    )
    parser.add_argument(
        "--input-csv",
        default="outputs/metrics/compact_twist_feature_summary.csv",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/metrics/compact_twist_feature_rankings.csv",
    )
    args = parser.parse_args()

    rows = read_summary(args.input_csv)
    rankings = build_rankings(rows)
    write_rankings(args.output_csv, rankings)

    print(f"Wrote {args.output_csv}")
    for row in rankings:
        if row["rank"] == 1:
            print(
                f"{row['feature_window']}: "
                f"{row['twist_deg']} deg ({row['normalized_intensity']})"
            )


if __name__ == "__main__":
    main()
