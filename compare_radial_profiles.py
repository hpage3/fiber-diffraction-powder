import argparse
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_float(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_pixel_count(value):
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return None
    return parsed


def parse_guide_q(value):
    if not value:
        return []

    guide_values = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            guide_values.append(float(item))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Could not parse guide q value: {item}"
            ) from exc
    return guide_values


def read_profile(path):
    points = []
    with open(path, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            q = parse_float(row.get("q_center_inv_angstrom"))
            mean_intensity = parse_float(row.get("mean_intensity"))
            pixel_count = parse_pixel_count(row.get("pixel_count"))

            if q is None or mean_intensity is None:
                continue
            if pixel_count is None or pixel_count <= 0:
                continue

            points.append(
                {
                    "q_center_inv_angstrom": q,
                    "mean_intensity": mean_intensity,
                    "pixel_count": pixel_count,
                }
            )

    return points


def normalize_profile(points, plot_q_min, normalize_q_min):
    normalization_candidates = [
        point["mean_intensity"]
        for point in points
        if point["q_center_inv_angstrom"] >= normalize_q_min
    ]
    normalization_max = max(normalization_candidates, default=None)
    if normalization_max is None or normalization_max <= 0:
        raise ValueError(
            "No positive normalization maximum found for "
            f"q_center_inv_angstrom >= {normalize_q_min}"
        )

    plotted_points = []
    for point in points:
        if point["q_center_inv_angstrom"] < plot_q_min:
            continue
        normalized_intensity = point["mean_intensity"] / normalization_max
        plotted_points.append(
            {
                "q_center_inv_angstrom": point["q_center_inv_angstrom"],
                "normalized_intensity": normalized_intensity,
            }
        )

    return plotted_points, normalization_max


def plot_profiles(
    profile_a,
    profile_b,
    label_a,
    label_b,
    title,
    output_png,
    plot_log_y,
    guide_q,
):
    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=180)

    for profile, label in ((profile_a, label_a), (profile_b, label_b)):
        q_values = [point["q_center_inv_angstrom"] for point in profile]
        intensity_values = [point["normalized_intensity"] for point in profile]
        if plot_log_y:
            positive_points = [
                (q, intensity)
                for q, intensity in zip(q_values, intensity_values)
                if intensity > 0
            ]
            q_values = [q for q, _ in positive_points]
            intensity_values = [intensity for _, intensity in positive_points]

        ax.plot(q_values, intensity_values, linewidth=1.4, label=label)

    for q_value in guide_q:
        ax.axvline(q_value, color="0.35", linestyle="--", linewidth=0.8, alpha=0.65)

    ax.set_xlabel("q, inverse angstrom")
    ax.set_ylabel("normalized mean intensity")
    if plot_log_y:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two radial profile CSV files on one overlay plot."
    )
    parser.add_argument("--profile-a", required=True, help="First radial profile CSV.")
    parser.add_argument("--profile-b", required=True, help="Second radial profile CSV.")
    parser.add_argument("--label-a", required=True, help="Legend label for profile A.")
    parser.add_argument("--label-b", required=True, help="Legend label for profile B.")
    parser.add_argument("--output-png", required=True, help="Output PNG path.")
    parser.add_argument("--plot-q-min", type=float, default=0.15)
    parser.add_argument("--normalize-q-min", type=float, default=0.4)
    parser.add_argument(
        "--plot-log-y",
        action="store_true",
        help="Use a logarithmic y-axis.",
    )
    parser.add_argument(
        "--guide-q",
        type=parse_guide_q,
        default=[],
        help="Comma-separated q values for vertical guide lines.",
    )
    parser.add_argument("--title", default="Radial profile comparison")
    args = parser.parse_args()

    points_a = read_profile(args.profile_a)
    points_b = read_profile(args.profile_b)
    profile_a, normalization_max_a = normalize_profile(
        points_a,
        args.plot_q_min,
        args.normalize_q_min,
    )
    profile_b, normalization_max_b = normalize_profile(
        points_b,
        args.plot_q_min,
        args.normalize_q_min,
    )

    plot_profiles(
        profile_a,
        profile_b,
        args.label_a,
        args.label_b,
        args.title,
        args.output_png,
        args.plot_log_y,
        args.guide_q,
    )

    print(f"Profile A: {args.profile_a}")
    print(f"Profile B: {args.profile_b}")
    print(f"Label A: {args.label_a}")
    print(f"Label B: {args.label_b}")
    print(f"Plot q min, inverse angstrom: {args.plot_q_min}")
    print(f"Normalization q min, inverse angstrom: {args.normalize_q_min}")
    print(f"Plotted points A: {len(profile_a)}")
    print(f"Plotted points B: {len(profile_b)}")
    print(f"Normalization maximum A: {normalization_max_a}")
    print(f"Normalization maximum B: {normalization_max_b}")
    print(f"Output PNG: {args.output_png}")


if __name__ == "__main__":
    main()
