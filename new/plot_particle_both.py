
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
plot_output_dir = os.path.join(project_root, "plots-eff1-new-both")

SAVE_MODE = False
REARRANGE_MODE = False
HIGH_SPEED_MODE = False
BASE_PATH = BASE_PATH_INITIAL

for i, arg in enumerate(sys.argv[1:]):
    if arg == "--save":
        matplotlib.use("Agg")
        SAVE_MODE = True
        if i + 1 < len(sys.argv) - 1 and not sys.argv[i+2].startswith("--"):
            BASE_PATH = os.path.normpath(sys.argv[i+2])
    elif arg == "--rearrange":
        REARRANGE_MODE = True
    elif arg == "--high-speed":
        HIGH_SPEED_MODE = True

print(f"Path:{BASE_PATH}")
if REARRANGE_MODE:
    print("Using rearranged layout: panel f first, then a,b,c,d,e")
    print("Panel f uses circle/triangle markers without error bars")
else:
    print("Using the original layout")
if HIGH_SPEED_MODE:
    print("Near/Far particle mode enabled")
else:
    print("Near/Far particle mode disabled")

config = {
    # "text.usetex": True,
    "font.family": 'serif',
    "font.serif": ['Times New Roman'],
    "mathtext.fontset": 'custom',
    "mathtext.rm": 'Times New Roman',
    "mathtext.it": 'Times New Roman:italic',
    "mathtext.bf": 'Times New Roman:bold',
    "font.size": 16,
    # "text.latex.preamble": r"\usepackage{mathptmx}"
}
plt.rcParams.update(config)

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
exp_sem_abs_rot = []
exp_sem_orb_rev = []

all_excel_data = []

num_folders = len(folder_groups)

if REARRANGE_MODE:
    num_rows = 3
    num_cols = 2
    create_separate_summary = False
else:
    num_rows = (num_folders + 1) // 2
    num_cols = 2
    create_separate_summary = num_folders >= num_rows * 2

fig, axes = plt.subplots(
    num_rows, num_cols, figsize=(12, 3.5 * num_rows), gridspec_kw={"hspace": 0.25, "wspace": 0.3, "left": 0.01, "right": 0.99, "bottom": 0.01, "top": 0.99}
)

if num_rows == 1:
    axes = axes.reshape(1, -1)

all_heights = []
for folder_ in folders:
    stats_file_path = os.path.join(folder_, "initial_result", "all_stats.json")
    if not os.path.exists(stats_file_path):
        continue

    with open(stats_file_path, "r") as stats_file:
        all_stats = json.load(stats_file)

    for key, value in all_stats.items():
        abs_rotation = value.get("abs_rotation", 0)
        orbital_rev = value.get("orbital_rev", 0)

        if abs_rotation > 0 and orbital_rev > 0:
            closest_point = value.get("closest_point", {})
            box = closest_point.get("Box", [0, 0, 0, 0])
            height = None
            if value.get("height") is None:
                height = (box[1] + box[3]) / 2 / 147 + 42 / 147
            else:
                height = value.get("height")  # assumed cm
            all_heights.append(height * 10)

min_height = min(all_heights) if all_heights else 0
max_height = max(all_heights) if all_heights else 1

print(len(folder_groups), folder_groups)

display_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

for i, (base_name, folder_list) in enumerate(folder_groups.items()):
    if REARRANGE_MODE:
        plot_index = i + 1
    else:
        plot_index = i
    
    row = plot_index // 2
    col = plot_index % 2
    
    merged_stats = merge_stats(*folder_list)

    heights_both = []
    abs_rotations_both = []
    orbital_revs_both = []
    is_high_speed = []

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
                height = value.get("height")  # assumed cm
            height_mm = height * 10

            if abs_rotation > 0 and orbital_rev > 0:
                inner_diameter = value.get("inner_diameter", 0)
                margin = value.get("margin", 0)
                
                # radius = (inner_diameter / 2 - margin * 147 / 101)
                # dz/2 = inner_diameter / 2
                # ratio = radius / (dz/2) = (inner_diameter / 2 - margin * 147 / 101) / (inner_diameter / 2)
                if inner_diameter > 0:
                    radius = inner_diameter / 2 - margin * 147 / 101
                    dz_half = inner_diameter / 2
                    ratio = radius / dz_half
                    high_speed = ratio > 0.9
                else:
                    high_speed = False
                
                heights_both.append(height_mm)
                abs_rotations_both.append(abs_rotation)
                orbital_revs_both.append(orbital_rev)
                is_high_speed.append(high_speed)

                all_excel_data.append({
                    'Flow_Rate_L_h': os.path.basename(base_name),
                    'Particle_ID': key,
                    'Height_mm': height_mm,
                    'Inner_Diameter_mm': (value.get("inner_diameter", 0)/147) * 10,
                    'Rotation_rad_s': abs_rotation,
                    'Revolution_rad_s': orbital_rev,
                    'Relative_Rotation_rad_s': value.get("rel_rotation", 0),
                    # 'Start_Frame_Revolution': value.get("start_frame_revolution", 0),
                    # 'End_Frame_Revolution': value.get("end_frame_revolution", 0),
                    'Total_Frames_Revolution': value.get("total_frames_revolution", 0),
                    # 'Start_Frame_Rotation': value.get("start_frame_rotation", 0),
                    # 'End_Frame_Rotation': value.get("end_frame_rotation", 0),
                    'Total_Frames_Rotation': value.get("total_frames_rotation", 0),
                    'Category_Changes': value.get("changes", 0),
                    'D1': value.get("d1_with_range_revolution", 0),
                    'D2': value.get("d2_with_range_revolution", 0),
                    'Margin': value.get("margin", 0),
                    "Inner_Radius_mm": (value.get("inner_diameter", 0)/147/2) * 10,
                    "Revolution_Radius_mm": (radius/147) * 10,
                    "radius / dz_half": ratio,
                    "High_Speed": high_speed,
                    
                    
                    # 'Closest_Point_Center_X': closest_point.get("Center", [0, 0])[0] if closest_point else 0,
                    # 'Closest_Point_Center_Y': closest_point.get("Center", [0, 0])[1] if closest_point else 0,
                    # 'Closest_Point_Box': str(closest_point.get("Box", [])) if closest_point else "",
                    # 'Not_Use': value.get("not_use", False),
                    # 'Must_Not_Use': value.get("must_not_use", False),
                    # 'Reason': value.get("reason", ""),
                    # 'Timestamp': value.get("timestamp", "")
                })

        except Exception as e:
            print(f"Processing {key} error: {e}")

    avg_abs_rotation = np.mean(abs_rotations_both) if abs_rotations_both else 0
    avg_orbital_rev = np.mean(orbital_revs_both) if orbital_revs_both else 0
    
    std_abs_rotation = np.std(abs_rotations_both, ddof=1) if len(abs_rotations_both) > 1 else 0
    std_orbital_rev = np.std(orbital_revs_both, ddof=1) if len(orbital_revs_both) > 1 else 0
    
    folder_name = os.path.basename(base_name)

    exp_indices.append(folder_name)
    exp_avg_abs_rot.append(avg_abs_rotation)
    exp_avg_orb_rev.append(avg_orbital_rev)
    exp_sem_abs_rot.append(std_abs_rotation)
    exp_sem_orb_rev.append(std_orbital_rev)

    ax = axes[row, col]

    if abs_rotations_both and orbital_revs_both:
        all_speeds = abs_rotations_both + orbital_revs_both
        y_min, y_max = min(all_speeds), max(all_speeds)

        if HIGH_SPEED_MODE:
            heights_rot_far = [h for h, hs in zip(heights_both, is_high_speed) if hs]
            abs_rot_far = [r for r, hs in zip(abs_rotations_both, is_high_speed) if hs]
            heights_rot_near = [h for h, hs in zip(heights_both, is_high_speed) if not hs]
            abs_rot_near = [r for r, hs in zip(abs_rotations_both, is_high_speed) if not hs]

            heights_rev_far = [h for h, hs in zip(heights_both, is_high_speed) if hs]
            orb_rev_far = [r for r, hs in zip(orbital_revs_both, is_high_speed) if hs]
            heights_rev_near = [h for h, hs in zip(heights_both, is_high_speed) if not hs]
            orb_rev_near = [r for r, hs in zip(orbital_revs_both, is_high_speed) if not hs]

            if heights_rot_near:
                ax.scatter(
                    heights_rot_near,
                    abs_rot_near,
                    facecolors="blue",
                    edgecolors="blue",
                    marker="o",
                    s=30,
                    label="Rotation (Near)" if i == 0 else None,
                )

            if heights_rot_far:
                ax.scatter(
                    heights_rot_far,
                    abs_rot_far,
                    facecolors="red",
                    edgecolors="red",
                    marker="o",
                    s=30,
                    label="Rotation (Far)" if i == 0 else None,
                )

            if heights_rev_near:
                ax.scatter(
                    heights_rev_near,
                    orb_rev_near,
                    facecolors="green",
                    edgecolors="green",
                    marker="s",
                    s=30,
                    label="Revolution (Near)" if i == 0 else None,
                )

            if heights_rev_far:
                ax.scatter(
                    heights_rev_far,
                    orb_rev_far,
                    facecolors="orange",
                    edgecolors="orange",
                    marker="s",
                    s=30,
                    label="Revolution (Far)" if i == 0 else None,
                )
        else:
            ax.scatter(
                heights_both,
                abs_rotations_both,
                facecolors="blue",
                edgecolors="blue",
                marker="o",
                s=30,
                label="Rotation",
            )

            ax.scatter(
                heights_both,
                orbital_revs_both,
                facecolors="orange",
                edgecolors="orange",
                marker="s",
                s=30,
                label="Revolution",
            )

        ax.set_xlabel(r"$h$ (mm)", labelpad=-5)
        ax.set_ylabel("Speed (rad/s)")
        ax.set_xlim(min_height, max_height)

        revolution_max = max(orbital_revs_both)
        rotation_min = min(abs_rotations_both)

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

        if len(heights_both) > 1:
            poly_coeffs = np.polyfit(heights_both, abs_rotations_both, 2)
            trend_line = np.poly1d(poly_coeffs)
            line1 = ax.plot(
                x_trend, trend_line(x_trend), color="blue", linestyle="--", alpha=0.5
            )

        if len(heights_both) > 1:
            poly_coeffs = np.polyfit(heights_both, orbital_revs_both, 2)
            trend_line = np.poly1d(poly_coeffs)
            line2 = ax.plot(
                x_trend, trend_line(x_trend), color="green", linestyle="--", alpha=0.5
            )

        flow_text = f"$Q_i$={os.path.basename(base_name)} L/h"

        high_x_threshold = min_height + (max_height - min_height) * 0.9
        high_y_threshold = 2400

        has_high_data = False
        for h, r in zip(heights_both, abs_rotations_both):
            if h > high_x_threshold and r > high_y_threshold:
                has_high_data = True
                break
        if not has_high_data:
            for h, r in zip(heights_both, orbital_revs_both):
                if h > high_x_threshold and r > high_y_threshold:
                    has_high_data = True
                    break

        if has_high_data:
            text_x, text_y = 0.02, 0.98
            h_align = 'left'
        else:
            text_x, text_y = 0.98, 0.98
            h_align = 'right'

        ax.text(
            text_x,
            text_y,
            flow_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment=h_align,
            fontsize=14,
        )

        if plot_index < len(display_labels):
            ax.set_title(display_labels[plot_index], loc="left", x=-0.19, y=0.9)

        if HIGH_SPEED_MODE:
            if i == 0:
                ax.legend(loc="upper left", fontsize=12, framealpha=0.9, frameon=False)
        else:
            legend_loc = "upper left" if h_align == 'right' else "upper right"
            if folder_name == '850':
                ax.legend(loc='lower left', bbox_to_anchor=(0.12, 0.18), fontsize=12, framealpha=0.9, frameon=False)
            else:
                ax.legend(loc=legend_loc, fontsize=12, framealpha=0.9, frameon=False)

        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which="minor", direction="in")
        ax.tick_params(which="major", direction="in")

        ax.set_yticks([0, 500, 1000, 1500, 2000, 2500, 3000])

if REARRANGE_MODE:
    ax_summary = axes[0, 0]
    ax_summary.set_visible(True)
    create_separate_summary = False
elif num_folders < num_rows * 2:
    last_row = (num_rows * 2 - 1) // 2
    last_col = (num_rows * 2 - 1) % 2
    ax_summary = axes[last_row, last_col]
    ax_summary.set_visible(True)
    create_separate_summary = False
else:
    fig2, ax_summary = plt.subplots(1, 1, figsize=(3.5, 2.5))
    create_separate_summary = True

if not REARRANGE_MODE:
    for j in range(num_folders, num_rows * 2):
        row = j // 2
        col = j % 2
        axes[row, col].set_visible(False)
else:
    for j in range(num_folders + 1, num_rows * 2):
        row = j // 2
        col = j % 2
        axes[row, col].set_visible(False)

if REARRANGE_MODE:
    
    ax_summary.plot(
        range(len(exp_indices)),
        exp_avg_abs_rot,
        color="blue",
        linestyle="-",
        linewidth=2,
        alpha=0.7
    )
    ax_summary.scatter(
        range(len(exp_indices)),
        exp_avg_abs_rot,
        color="blue",
        marker="o",
        s=80,
        alpha=0.7,
        label="Rotation"
    )

    ax_summary.plot(
        range(len(exp_indices)),
        exp_avg_orb_rev,
        color="orange",
        linestyle="-",
        linewidth=2,
        alpha=0.7
    )
    ax_summary.scatter(
        range(len(exp_indices)),
        exp_avg_orb_rev,
        color="orange",
        marker="^",
        s=80,
        alpha=0.7,
        label="Revolution"
    )

    ax_summary.set_xlabel("Inlet Flow Rate (L/h)")
    ax_summary.set_ylabel("Speed (rad/s)")
    ax_summary.set_xticks(range(len(exp_indices)))
    ax_summary.set_xticklabels(exp_indices)

    ax_summary.set_ylim(0, 3000)
    ax_summary.set_yticks([0, 500, 1000, 1500, 2000, 2500, 3000])

    summary_label = display_labels[0] if REARRANGE_MODE else display_labels[-1]
    ax_summary.set_title(summary_label, loc="left", x=-0.19, y=0.9)

    ax_summary.legend(loc="upper left", fontsize=12, framealpha=0.9, frameon=False)

    ax_summary.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax_summary.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax_summary.tick_params(which="minor", direction="in")
    ax_summary.tick_params(which="major", direction="in")

elif num_folders < num_rows * 2:

    ax_summary.errorbar(
        exp_indices,
        exp_avg_abs_rot,
        yerr=exp_sem_abs_rot,
        color="blue",
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=1,
        capsize=3,
        capthick=3,
        elinewidth=1,
        alpha=0.7,
        label="Rotation"
    )

    ax_summary.errorbar(
        exp_indices,
        exp_avg_orb_rev,
        yerr=exp_sem_orb_rev,
        color="orange",
        marker="s",
        linestyle="-",
        linewidth=2,
        markersize=1,
        capsize=3,
        capthick=3,
        elinewidth=1,
        alpha=0.7,
        label="Revolution"
    )

    ax_summary.set_xlabel("Inlet Flow Rate (L/h)")
    ax_summary.set_ylabel("Speed (rad/s)")

    ax_summary.set_ylim(0, 3000)
    ax_summary.set_yticks([0, 500, 1000, 1500, 2000, 2500, 3000])

    ax_summary.set_title("(f)", loc="left", x=-0.19, y=0.9)

    ax_summary.legend(loc="upper left", fontsize=12, framealpha=0.9, frameon=False)

    ax_summary.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax_summary.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax_summary.tick_params(which="minor", direction="in")
    ax_summary.tick_params(which="major", direction="in")
else:
    ax_summary.errorbar(
        exp_indices,
        exp_avg_abs_rot,
        yerr=exp_sem_abs_rot,
        color="blue",
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=5,
        capsize=3,
        capthick=3,
        elinewidth=1,
        alpha=0.7,
        label="Rotation"
    )

    ax_summary.errorbar(
        exp_indices,
        exp_avg_orb_rev,
        yerr=exp_sem_orb_rev,
        color="orange",
        marker="s",
        linestyle="-",
        linewidth=2,
        markersize=5,
        capsize=3,
        capthick=3,
        elinewidth=1,
        alpha=0.7,
        label="Revolution"
    )

    ax_summary.set_xlabel("Inlet Flow Rate (L/h)")
    ax_summary.set_ylabel("Speed (rad/s)")

    ax_summary.set_ylim(0, 3000)
    ax_summary.set_yticks([0, 500, 1000, 1500, 2000, 2500, 3000])

    ax_summary.legend(loc="upper left", fontsize=12, framealpha=0.9, frameon=False)

    ax_summary.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax_summary.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax_summary.tick_params(which="minor", direction="in")
    ax_summary.tick_params(which="major", direction="in")

df_detailed = pd.DataFrame(all_excel_data)

df_summary = pd.DataFrame({
    'Flow_Rate_L_h': exp_indices,
    'Avg_Rotation_rad_s': exp_avg_abs_rot,
    'Avg_Revolution_rad_s': exp_avg_orb_rev
})

os.makedirs(plot_output_dir, exist_ok=True)
excel_path = os.path.join(plot_output_dir, "particle_both_motion_data.xlsx")

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    df_detailed.to_excel(writer, sheet_name='Detailed_Data', index=False)
    df_summary.to_excel(writer, sheet_name='Summary_Data', index=False)

print(f"Data exported to Excel: {excel_path}")
print(f"  - total {len(df_detailed)} particles with both rotation and revolution")
print("  - Detailed_Data sheet: per-particle data")
print("  - Summary_Data sheet: per-flow-rate averages")

if SAVE_MODE:
    os.makedirs(plot_output_dir, exist_ok=True)

    fig.savefig(
        os.path.join(plot_output_dir, "particle_analysis_combined.png"),
        dpi=300,
        bbox_inches="tight",
    )
    
    if 'fig2' in locals():
        fig2.savefig(
            os.path.join(plot_output_dir, "particle_analysis_summary.png"),
            dpi=300,
            bbox_inches="tight",
        )

    print(f"Plot saved to: {plot_output_dir}")
    if num_folders < num_rows * 2:
        print("  - particle_analysis_combined.png (combined with means)")
    else:
        print("  - particle_analysis_combined.png (detailed)")
        print("  - particle_analysis_summary.png (summary)")

    plt.close("all")
else:
    plt.show()
