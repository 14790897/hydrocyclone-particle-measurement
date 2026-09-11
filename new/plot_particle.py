
import json
import os
import sys
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from process_utils import get_all_folders

BASE_PATH_INITIAL = "runs/eff1_new"
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
plot_output_dir = os.path.join(project_root, "plots-eff1-new")
if len(sys.argv) > 1 and sys.argv[1] == "--save":
    matplotlib.use("Agg")
    SAVE_MODE = True
    if len(sys.argv) > 2:
        BASE_PATH = os.path.normpath(sys.argv[2])
    else:
        BASE_PATH = BASE_PATH_INITIAL
else:
    SAVE_MODE = False
    BASE_PATH = BASE_PATH_INITIAL

print(f"Path:{BASE_PATH}")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16

def merge_stats(*folders):
    """
    Merge all_stats.json across flow-rate folders, suffixing later folders' keys with '-2', '-3', ...

    :param folders: list of folder paths to merge
    :return: merged JSON data
    """
    data = {}
    len_folders = len(folders)
    for index, folder in enumerate(folders):
        stats_path = os.path.join(folder, "initial_result", "all_stats.json")
        if index == 1:
            print("folder", folder, "number", len_folders)
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                stats_data = json.load(f)

            if index == 0:
                data.update(stats_data)
            else:
                for key, value in stats_data.items():
                    modified_key = f"{key}-{index+1}"
                    data[modified_key] = value
                    # if folder == r"runs/track\750" or folder == r"runs/track\750-2":
                    #     data[modified_key]["orbital_rev"] = 0
                    # if index == 1 and len_folders > 2:
                    #     data[modified_key]["orbital_rev"] = 0

    return data

base_path = BASE_PATH
folders = get_all_folders(base_path)

valid_flow_rates = ["450", "550", "650", "750", "850"]
folders = [
    f
    for f in folders
    if any(os.path.basename(f).startswith(flow) for flow in valid_flow_rates)
]
print(f"Folders after filtering: {len(folders)}")
print(f"Folder list: {[os.path.basename(f) for f in folders]}")

folder_groups = defaultdict(list)

for folder__ in folders:
    base_name = folder__.split("-")[0]
    folder_groups[base_name].append(folder__)

exp_indices = []
exp_avg_abs_rot = []
exp_avg_orb_rev = []

num_folders = len(folder_groups)
fig, axes = plt.subplots(
    num_folders, 1, figsize=(8, 3.3 * num_folders), gridspec_kw={"hspace": 0.25}
)

all_heights = []
for folder_ in folders:
    stats_file_path = os.path.join(folder_, "initial_result", "all_stats.json")
    if not os.path.exists(stats_file_path):
        continue

    with open(stats_file_path, "r") as stats_file:
        all_stats = json.load(stats_file)

    for key, value in all_stats.items():
        closest_point = value.get("closest_point", {})
        box = closest_point.get("Box", [0, 0, 0, 0])
        height = None
        if value.get("height") is None:
            height = (box[1] + box[3]) / 2 / 147 + 42 / 147
        else:
            height = value.get("height")
        inner_diameter = value.get("inner_diameter", 0)
        # 1
        # all_heights.append(height / inner_diameter * 147)
        all_heights.append(height)

min_height = min(all_heights)
max_height = max(all_heights)
print(len(folder_groups), folder_groups)

for i, (base_name, folder_list) in enumerate(folder_groups.items()):
    merged_stats = merge_stats(*folder_list)

    heights_abs_rot = []
    abs_rotations = []
    heights_orb_rev = []
    orbital_revs = []

    for key, value in merged_stats.items():
        try:
            abs_rotation = value.get("abs_rotation", 0)
            orbital_rev = value.get("orbital_rev", 0)
            closest_point = value.get("closest_point", {})

            box = closest_point.get("Box", [0, 0, 0, 0])
            height = None
            if value.get("height") is None:
                height = (box[1] + box[3]) / 2 / 147 + 42 / 147
            else:
                height = value.get("height")
            inner_diameter = value.get("inner_diameter", 0)
            if abs_rotation > 0:
                # 2
                # heights_abs_rot.append(height / inner_diameter * 147)  # h/D
                heights_abs_rot.append(height)
                abs_rotations.append(abs_rotation)

            if orbital_rev > 0:
                # 3
                # heights_orb_rev.append(height / inner_diameter * 147)
                heights_orb_rev.append(height)
                orbital_revs.append(orbital_rev)

        except Exception as e:
            print(f"Processing {key} error: {e}")
    data = pd.DataFrame(
        {
            "height": heights_orb_rev,
            "orb_rev": orbital_revs,
        }
    )
    # print("sampled_data:", sampled_data)
    avg_abs_rotation = np.mean(abs_rotations) if abs_rotations else 0
    avg_orbital_rev = np.mean(orbital_revs) if orbital_revs else 0
    # avg_orbital_rev = np.mean(sampled_data["orb_rev"]) if not sampled_data.empty else 0
    folder_name = os.path.basename(base_name)
    exp_indices.append(folder_name)
    exp_avg_abs_rot.append(avg_abs_rotation)
    exp_avg_orb_rev.append(avg_orbital_rev)

    ax = axes[i]

    y1_min, y1_max = min(abs_rotations), max(abs_rotations)
    y2_min, y2_max = min(orbital_revs), max(orbital_revs)

    all_speeds = abs_rotations + orbital_revs
    y_min, y_max = min(all_speeds), max(all_speeds)

    scatter1 = ax.scatter(
        heights_abs_rot,
        abs_rotations,
        alpha=0.7,
        color="blue",
        label="Rotation" if i == 0 else None,
    )

    scatter2 = ax.scatter(
        heights_orb_rev,
        orbital_revs,
        alpha=0.7,
        color="orange",
        label="Revolution" if i == 0 else None,
    )

    ax.set_xlabel(r"$h$ (cm)")
    ax.set_ylabel("Speed (rad/s)")
    ax.set_xlim(min_height, max_height)

    revolution_max = max(orbital_revs)
    rotation_min = min(abs_rotations)

    ax.set_ylim(0, 3000)

    if rotation_min > revolution_max:
        break_center = (revolution_max + rotation_min) / 2

        y_range = 3000
        break_pos = break_center / y_range

        d = 0.015
        kwargs = dict(transform=ax.transAxes, color="k", clip_on=False, linewidth=1.5)

        ax.plot((-d, +d), (break_pos - d, break_pos + d), **kwargs)
        ax.plot((-d, +d), (break_pos + 0.015 - d, break_pos + 0.015 + d), **kwargs)

    x_trend = np.linspace(min_height, max_height, 100)

    if len(heights_abs_rot) > 1:
        poly_coeffs = np.polyfit(heights_abs_rot, abs_rotations, 2)
        trend_line = np.poly1d(poly_coeffs)
        line1 = ax.plot(
            x_trend, trend_line(x_trend), color="blue", linestyle="--", alpha=0.5
        )

    if len(heights_orb_rev) > 1:
        poly_coeffs = np.polyfit(heights_orb_rev, orbital_revs, 2)
        trend_line = np.poly1d(poly_coeffs)
        line2 = ax.plot(
            x_trend, trend_line(x_trend), color="orange", linestyle="--", alpha=0.5
        )

    flow_text = f"{os.path.basename(base_name)} L/h"
    ax.text(
        0.98,
        0.98,
        flow_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=14,
    )

    subplot_labels = ["(a)", "(b)", "(c)", "(d)", "(e)"]
    if i < len(subplot_labels):
        ax.set_title(subplot_labels[i], loc="left", x=-0.13, y=0.87)

    if i == 0:
        ax.legend(loc="upper left", fontsize=12, framealpha=0.9, frameon=False)

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="minor", direction="in")
    ax.tick_params(which="major", direction="in")

    ax.set_yticks([0, 1000, 2000, 3000])

fig2, ax_summary = plt.subplots(1, 1, figsize=(7, 4.5))

line1 = ax_summary.plot(
    exp_indices,
    exp_avg_abs_rot,
    alpha=0.7,
    color="blue",
    marker="o",
    linestyle="-",
    linewidth=2,
    label="Rotation"
)

line2 = ax_summary.plot(
    exp_indices,
    exp_avg_orb_rev,
    alpha=0.7,
    color="orange",
    marker="s",
    linestyle="-",
    linewidth=2,
    label="Revolution"
)

ax_summary.set_xlabel("Inlet Flow Rate (L/h)")
ax_summary.set_ylabel("Speed (rad/s)")

ax_summary.set_ylim(0, 3000)
ax_summary.set_yticks([0, 1000, 2000, 3000])

ax_summary.legend(loc="upper left", fontsize=12, framealpha=0.9, frameon=False)

ax_summary.xaxis.set_minor_locator(AutoMinorLocator(2))
ax_summary.yaxis.set_minor_locator(AutoMinorLocator(2))
ax_summary.tick_params(which="minor", direction="in")
ax_summary.tick_params(which="major", direction="in")

if SAVE_MODE:
    os.makedirs(plot_output_dir, exist_ok=True)

    fig.savefig(
        os.path.join(plot_output_dir, "particle_analysis_detailed.png"),
        dpi=300,
        bbox_inches="tight",
    )
    fig2.savefig(
        os.path.join(plot_output_dir, "particle_analysis_summary.png"),
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Plot saved to: {plot_output_dir}")
    print("  - particle_analysis_detailed.png (detailed)")
    print("  - particle_analysis_summary.png (summary)")

    plt.close("all")
else:
    plt.show()
