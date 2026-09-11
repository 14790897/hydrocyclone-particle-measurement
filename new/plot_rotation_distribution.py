
import json
import os
import sys
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from process_utils import get_all_folders

BASE_PATH_INITIAL = "runs/eff1_new"
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
plot_output_dir = os.path.join(project_root, "plots-eff1-new-distribution")
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
plt.rcParams["font.family"] = "SimHei"
plt.rcParams["font.size"] = 14
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['black', 'dimgray', 'gray', 'darkgray', 'lightgray'])
plt.style.use('grayscale')

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

flow_rotation_data = {}  # {flow_rate: [rotation_speeds]}
all_rotation_data = []

for base_name, folder_list in sorted(folder_groups.items()):
    merged_stats = merge_stats(*folder_list)
    
    rotation_speeds = []
    folder_name = os.path.basename(base_name)
    
    for key, value in merged_stats.items():
        abs_rotation = value.get("abs_rotation", 0)
        orbital_rev = value.get("orbital_rev", 0)
        
        if abs_rotation > 0 and orbital_rev > 0:
            rotation_speeds.append(abs_rotation)
            all_rotation_data.append({
                'Flow_Rate_L_h': folder_name,
                'Particle_ID': key,
                'Rotation_rad_s': abs_rotation
            })
    
    flow_rotation_data[folder_name] = rotation_speeds
    print(f"{folder_name} L/h: {len(rotation_speeds)} particles, "
          f"mean={np.mean(rotation_speeds):.1f} rad/s, "
          f"median={np.median(rotation_speeds):.1f} rad/s, "
          f"std={np.std(rotation_speeds):.1f} rad/s")

fig, ax = plt.subplots(figsize=(10, 6))

flow_rates_sorted = sorted(flow_rotation_data.keys(), key=lambda x: int(x))
data_for_boxplot = [flow_rotation_data[fr] for fr in flow_rates_sorted]

bp = ax.boxplot(data_for_boxplot, 
                labels=flow_rates_sorted,
                patch_artist=True,
                showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='black', markersize=6),
                medianprops=dict(color='black', linewidth=2),
                boxprops=dict(facecolor='white', edgecolor='black', alpha=0.7),
                whiskerprops=dict(linewidth=1.5, color='black'),
                capprops=dict(linewidth=1.5, color='black'),
                flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5))

ax.axhline(y=1500, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='1500 rad/s')

ax.set_xlabel('Inlet flow rate (L/h)', fontsize=14)
ax.set_ylabel('Rotation speed (rad/s)', fontsize=14)
ax.set_title('Particle rotation-speed distribution across flow rates', fontsize=16, pad=15)

ax.grid(True, axis='y', alpha=0.3, linestyle='--')

ax.set_ylim(0, max([max(data) for data in data_for_boxplot]) * 1.1)

ax.legend(loc='upper left', fontsize=12)

plt.tight_layout()

fig2, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, flow_rate in enumerate(flow_rates_sorted):
    if idx < len(axes):
        ax2 = axes[idx]
        data = flow_rotation_data[flow_rate]
        
        n, bins, patches = ax2.hist(data, bins=30, alpha=0.7, color='lightgray', edgecolor='black')
        
        ax2.axvline(x=1500, color='black', linestyle='--', linewidth=2, alpha=0.7, label='1500 rad/s')
        
        mean_val = np.mean(data)
        median_val = np.median(data)
        ax2.axvline(x=mean_val, color='dimgray', linestyle='-', linewidth=2, alpha=0.7, label=f'Mean: {mean_val:.0f}')
        ax2.axvline(x=median_val, color='gray', linestyle='-.', linewidth=2, alpha=0.7, label=f'Median: {median_val:.0f}')
        
        ax2.set_xlabel('Rotation speed (rad/s)', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title(f'Flow = {flow_rate} L/h (n={len(data)})', fontsize=13)
        ax2.legend(fontsize=9, loc='upper right')
        ax2.grid(True, alpha=0.3)

if len(flow_rates_sorted) < len(axes):
    axes[-1].set_visible(False)

plt.tight_layout()

fig3, ax3 = plt.subplots(figsize=(10, 6))

flow_rates_numeric = [int(fr) for fr in flow_rates_sorted]
mean_values = [np.mean(flow_rotation_data[fr]) for fr in flow_rates_sorted]
std_values = [np.std(flow_rotation_data[fr]) for fr in flow_rates_sorted]

coeffs = np.polyfit(flow_rates_numeric, mean_values, 1)
poly_func = np.poly1d(coeffs)
fitted_values = poly_func(flow_rates_numeric)

ss_res = np.sum((np.array(mean_values) - fitted_values) ** 2)
ss_tot = np.sum((np.array(mean_values) - np.mean(mean_values)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

ax3.errorbar(flow_rates_numeric, mean_values, yerr=std_values,
             marker='o', markersize=8, linewidth=2, capsize=5, capthick=2,
             color='black', ecolor='gray', 
             label='Mean ± SD')

ax3.axhline(y=1500, color='dimgray', linestyle='--', linewidth=1.5, alpha=0.7, label='1500 rad/s reference')

ax3.set_xlabel('Inlet flow rate (L/h)', fontsize=14)
ax3.set_ylabel('Rotation speed (rad/s)', fontsize=14)
ax3.set_title('Mean rotation speed vs. flow rate', fontsize=16, pad=15)

ax3.grid(True, alpha=0.3, linestyle='--')

y_min = min([m - s for m, s in zip(mean_values, std_values)])
y_max = max([m + s for m, s in zip(mean_values, std_values)])
ax3.set_ylim(max(0, y_min * 0.9), y_max * 1.1)

ax3.legend(loc='best', fontsize=12)

plt.tight_layout()

df_rotation = pd.DataFrame(all_rotation_data)
os.makedirs(plot_output_dir, exist_ok=True)
excel_path = os.path.join(plot_output_dir, "rotation_speed_distribution.xlsx")

summary_data = []
for flow_rate in flow_rates_sorted:
    data = flow_rotation_data[flow_rate]
    summary_data.append({
        'Flow_Rate_L_h': flow_rate,
        'Count': len(data),
        'Mean_rad_s': np.mean(data),
        'Median_rad_s': np.median(data),
        'Std_rad_s': np.std(data),
        'Min_rad_s': np.min(data),
        'Max_rad_s': np.max(data),
        'Q25_rad_s': np.percentile(data, 25),
        'Q75_rad_s': np.percentile(data, 75),
        'Percentage_above_1500': np.sum(np.array(data) > 1500) / len(data) * 100
    })

df_summary = pd.DataFrame(summary_data)

fit_info = pd.DataFrame({
    'Parameter': ['Slope', 'Intercept', 'R_squared'],
    'Value': [coeffs[0], coeffs[1], r_squared]
})

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    df_rotation.to_excel(writer, sheet_name='Raw_Data', index=False)
    df_summary.to_excel(writer, sheet_name='Statistics', index=False)
    fit_info.to_excel(writer, sheet_name='Fit_Info', index=False)

print(f"\nData exported to Excel: {excel_path}")
print(f"  - total {len(df_rotation)} particles' rotation data")
print("\nLinear fit:")
print(f"  - slope: {coeffs[0]:.4f}")
print(f"  - intercept: {coeffs[1]:.4f}")
print(f"  - R²: {r_squared:.4f}")
print("\nSummary statistics:")
print(df_summary.to_string(index=False))

if SAVE_MODE:
    os.makedirs(plot_output_dir, exist_ok=True)
    
    fig.savefig(
        os.path.join(plot_output_dir, "rotation_speed_boxplot.png"),
        dpi=300,
        bbox_inches="tight",
    )
    
    fig2.savefig(
        os.path.join(plot_output_dir, "rotation_speed_histogram.png"),
        dpi=300,
        bbox_inches="tight",
    )
    
    fig3.savefig(
        os.path.join(plot_output_dir, "rotation_speed_trend.png"),
        dpi=300,
        bbox_inches="tight",
    )
    
    print(f"\nPlot saved to: {plot_output_dir}")
    print("  - rotation_speed_boxplot.png (boxplot)")
    print("  - rotation_speed_histogram.png (histogram)")
    print("  - rotation_speed_trend.png (trend)")
    
    plt.close("all")
else:
    plt.show()
