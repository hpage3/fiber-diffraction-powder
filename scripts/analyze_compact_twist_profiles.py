import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FEATURE_WINDOWS = [
    ("d_3p4", 3.3, 3.5),
    ("d_3p0", 2.9, 3.1),
    ("d_4p5_5p0", 4.5, 5.0),
    ("d_4p1", 4.0, 4.2),
    ("d_5p5", 5.4, 5.6),
    ("d_7p0", 6.8, 7.2),
    ("d_8p4", 8.2, 8.6),
]


def read_profile(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(
                {
                    "q_Ainv": float(row["q_center_inv_angstrom"]),
                    "d_A": float(row["d_center_angstrom"]),
                    "mean_intensity": float(row["mean_intensity"]),
                    "pixel_count": int(row["pixel_count"]),
                }
            )
    return rows


def normalize_profiles(profiles, normalize_q_min):
    normalized = {}
    for twist, rows in profiles.items():
        candidates = [
            row["mean_intensity"]
            for row in rows
            if row["q_Ainv"] >= normalize_q_min and row["pixel_count"] > 0
        ]
        max_intensity = max(candidates)
        normalized[twist] = [
            {
                **row,
                "normalized_intensity": row["mean_intensity"] / max_intensity,
            }
            for row in rows
        ]
    return normalized


def summarize_features(profiles):
    rows = []
    for twist, profile in profiles.items():
        summary = {"twist": twist}
        for name, d_min, d_max in FEATURE_WINDOWS:
            points = [row for row in profile if d_min <= row["d_A"] <= d_max]
            if points:
                best = max(points, key=lambda row: row["normalized_intensity"])
                summary[f"{name}_max_norm_intensity"] = best["normalized_intensity"]
                summary[f"{name}_q_Ainv"] = best["q_Ainv"]
                summary[f"{name}_d_A"] = best["d_A"]
            else:
                summary[f"{name}_max_norm_intensity"] = ""
                summary[f"{name}_q_Ainv"] = ""
                summary[f"{name}_d_A"] = ""
        rows.append(summary)
    return rows


def write_feature_csv(path, rows):
    fieldnames = ["twist"]
    for name, _, _ in FEATURE_WINDOWS:
        fieldnames.extend(
            [
                f"{name}_max_norm_intensity",
                f"{name}_q_Ainv",
                f"{name}_d_A",
            ]
        )
    with open(path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_profiles_q(path, profiles, q_min):
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    for twist, rows in sorted(profiles.items()):
        q = [row["q_Ainv"] for row in rows if row["q_Ainv"] >= q_min]
        intensity = [
            row["normalized_intensity"] for row in rows if row["q_Ainv"] >= q_min
        ]
        ax.plot(q, intensity, linewidth=1.4, label=f"{twist:g} deg")
    ax.set_xlabel("q_Ainv")
    ax.set_ylabel("normalized mean intensity")
    ax.set_title("Compact twist radial profiles")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_profiles_d(path, profiles, d_min, d_max):
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    for twist, rows in sorted(profiles.items()):
        selected = [row for row in rows if d_min <= row["d_A"] <= d_max]
        selected = sorted(selected, key=lambda row: row["d_A"])
        ax.plot(
            [row["d_A"] for row in selected],
            [row["normalized_intensity"] for row in selected],
            linewidth=1.4,
            label=f"{twist:g} deg",
        )
    ax.set_xlabel("d_A")
    ax.set_ylabel("normalized mean intensity")
    ax.set_title("Compact twist radial profiles by d-spacing")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_features(path, feature_rows):
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    twists = [row["twist"] for row in feature_rows]
    for name, _, _ in FEATURE_WINDOWS[:3]:
        values = [row[f"{name}_max_norm_intensity"] for row in feature_rows]
        ax.plot(twists, values, marker="o", linewidth=1.4, label=name)
    ax.set_xlabel("target twist angle, degrees")
    ax.set_ylabel("window max normalized intensity")
    ax.set_title("Feature response versus compact twist")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def best_twists(feature_rows):
    best = {}
    for name, _, _ in FEATURE_WINDOWS:
        populated = [
            row
            for row in feature_rows
            if row[f"{name}_max_norm_intensity"] != ""
        ]
        if populated:
            row = max(populated, key=lambda item: item[f"{name}_max_norm_intensity"])
            best[name] = (row["twist"], row[f"{name}_max_norm_intensity"])
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", nargs=2, action="append", metavar=("TWIST", "CSV"))
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--plot-dir", required=True)
    parser.add_argument("--normalize-q-min", type=float, default=0.4)
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    plot_dir = Path(args.plot_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    profiles = {
        float(twist): read_profile(path)
        for twist, path in args.profile
    }
    normalized = normalize_profiles(profiles, args.normalize_q_min)
    feature_rows = summarize_features(normalized)
    feature_rows.sort(key=lambda row: row["twist"])

    write_feature_csv(metrics_dir / "compact_twist_feature_summary.csv", feature_rows)
    plot_profiles_q(plot_dir / "compact_twist_radial_profiles_q.png", normalized, 0.15)
    plot_profiles_d(plot_dir / "compact_twist_radial_profiles_d.png", normalized, 2.5, 9.0)
    plot_features(plot_dir / "compact_twist_feature_response.png", feature_rows)

    print("Feature CSV:", metrics_dir / "compact_twist_feature_summary.csv")
    for name, (twist, value) in best_twists(feature_rows).items():
        print(f"Best {name}: twist={twist:g}, normalized_intensity={value:.6g}")


if __name__ == "__main__":
    main()
