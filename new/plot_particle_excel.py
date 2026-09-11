

import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from process_utils import get_all_folders

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16

def load_excel_true_data():
    """
    Load TRUE_rotation and TRUE_revolution from an Excel file,
    auto-detecting whether the table needs transposing.
    """
    data = pd.read_excel(
        "data/relative error.xlsx",
        sheet_name="Sheet1",
        header=None,
    )
    
    print("Raw data shape:", data.shape)
    print("First column (preview):", data.iloc[:10, 0].tolist())
    
    expected_fields = ['flow_velocity', 'TRUE_rotation', 'TRUE_revolution', 'height', 'diameter']
    first_column_values = data.iloc[:, 0].astype(str).str.lower().tolist()
    
    has_expected_fields = any(field.lower() in ' '.join(first_column_values) for field in expected_fields)
    
    if has_expected_fields:
        print("Detected transpose needed")
        data = data.T
        data.columns = data.iloc[0]
        data = data[1:]
    else:
        print("Table already in the right layout")
        data.columns = data.iloc[0]
        data = data[1:]
    
    data.columns = data.columns.astype(str).str.strip()
    print("Processed columns:", data.columns.tolist())
    
    column_mapping = {
        'height': ['height', 'Height', 'HEIGHT'],
        'diameter': ['diameter', 'Diameter', 'DIAMETER'],
        'TRUE_rotation': ['TRUE_rotation', 'true_rotation', 'True_rotation'],
        'TRUE_revolution': ['TRUE_revolution', 'true_revolution', 'True_revolution'],
        'flow_velocity': ['flow_velocity', 'flow_velocitiy', 'Flow_velocity']
    }
    
    actual_columns = {}
    for standard_name, variants in column_mapping.items():
        found = False
        for variant in variants:
            if variant in data.columns:
                actual_columns[standard_name] = variant
                found = True
                break
        if not found:
            print(f"Warning: column not found: {standard_name}")
    
    if len(actual_columns) < 5:
        print(f"Warning: only found {len(actual_columns)} required column(s)")
        return pd.DataFrame(columns=['H/D', 'TRUE_rotation', 'TRUE_revolution', 'flow_velocity'])
    
    rename_dict = {v: k for k, v in actual_columns.items()}
    data = data.rename(columns=rename_dict)
    
    numeric_columns = ['height', 'diameter', 'TRUE_rotation', 'TRUE_revolution']
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    
    if 'height' in data.columns and 'diameter' in data.columns:
        data["H/D"] = data["height"] / data["diameter"] * 147
    
    if 'flow_velocity' in data.columns:
        data['flow_velocity'] = data['flow_velocity'].astype(str)
    else:
        print("Warning: no 'flow_velocity' column found")
        data['flow_velocity'] = 'unknown'
    
    required_for_filter = [col for col in ['TRUE_rotation', 'TRUE_revolution', 'H/D'] if col in data.columns]
    if required_for_filter:
        valid_data = data.dropna(subset=required_for_filter)
        valid_data = valid_data[valid_data['flow_velocity'].notna()]
        valid_data = valid_data[valid_data['flow_velocity'] != 'nan']
        valid_data = valid_data[valid_data['flow_velocity'] != 'None']
    else:
        valid_data = data
    
    print(f"Valid rows: {len(valid_data)}")
    if len(valid_data) > 0 and 'flow_velocity' in valid_data.columns:
        print("Flow-rate groups:", valid_data['flow_velocity'].unique())
    
    return valid_data
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

base_path = "runs/track"
folders = get_all_folders(base_path)

true_data = load_excel_true_data()

folder_groups = defaultdict(list)

for folder__ in folders:
    base_name = folder__.split("-")[0]
    folder_groups[base_name].append(folder__)

exp_indices = []
exp_avg_abs_rot = []
exp_avg_orb_rev = []

num_folders = len(folder_groups)
fig, axes = plt.subplots(
    num_folders + 1, 2, figsize=(12, 6 * (num_folders + 1)), constrained_layout=True
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
        all_heights.append(height / inner_diameter * 147)

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
                heights_abs_rot.append(height / inner_diameter * 147)  # h/D
                abs_rotations.append(abs_rotation)

            if orbital_rev > 0:
                heights_orb_rev.append(height / inner_diameter * 147)
                orbital_revs.append(orbital_rev)

        except Exception as e:
            print(f"Processing {key} error: {e}")
    data = pd.DataFrame(
        {
            "height": heights_orb_rev,
            "orb_rev": orbital_revs,
        }
    )

    # kde = KernelDensity(kernel="gaussian", bandwidth=7).fit(
    #     np.array(data["height"]).reshape(-1, 1)
    # )
    # log_density = kde.score_samples(np.array(data["height"]).reshape(-1, 1))

    # weights = 1 / density

    # sampled_indices = np.random.choice(
    #     data.index, size=num_samples, replace=False, p=normalized_weights
    # )
    # sampled_data = data.loc[sampled_indices]
    # sampled_data = (
    #     data.groupby("bin")
    #     .apply(lambda x: x.sample(n=1, replace=True))
    #     .reset_index(drop=True)
    # )
    # print("sampled_data:", sampled_data)
    avg_abs_rotation = np.mean(abs_rotations) if abs_rotations else 0
    avg_orbital_rev = np.mean(orbital_revs) if orbital_revs else 0
    # avg_orbital_rev = np.mean(sampled_data["orb_rev"]) if not sampled_data.empty else 0
    folder_name = os.path.basename(base_name)
    exp_indices.append(folder_name)
    exp_avg_abs_rot.append(avg_abs_rotation)
    exp_avg_orb_rev.append(avg_orbital_rev)

    axes[i, 0].scatter(
        heights_abs_rot,
        abs_rotations,
        alpha=0.7,
        label=f"{os.path.basename(base_name)}",
    )
    
    folder_name = os.path.basename(base_name)
    
    matching_true_data = true_data[true_data['flow_velocity'].str.contains(folder_name, na=False)]
    
    if not matching_true_data.empty:
        all_h_d = []
        all_true_rotation = []
        
        for flow_vel, flow_data in matching_true_data.groupby('flow_velocity'):
            all_h_d.extend(flow_data['H/D'].tolist())
            all_true_rotation.extend(flow_data['TRUE_rotation'].tolist())
        
        axes[i, 0].scatter(
            all_h_d,
            all_true_rotation,
            alpha=0.8,
            marker='x',
            s=50,
            label=f"TRUE {folder_name}",
        )
    
    axes[i, 0].set_xlabel(r"$h/D$")
    axes[i, 0].set_ylabel("Rotation (rad/s)")
    axes[i, 0].set_xlim(min_height, max_height)
    if i == 0:
        axes[i, 0].set_title("Rotation vs Height", fontsize=22)
    # axes[i, 0].grid()
    axes[i, 0].legend(loc="upper right")
    x_trend = np.linspace(min_height, max_height, 100)
    if len(heights_abs_rot) > 1:
        poly_coeffs = np.polyfit(heights_abs_rot, abs_rotations, 2)
        trend_line = np.poly1d(poly_coeffs)
        axes[i, 0].plot(
            x_trend, trend_line(x_trend), color="blue", linestyle="--", label="Trend"
        )

    axes[i, 1].scatter(
        heights_orb_rev,
        orbital_revs,
        alpha=0.7,
        color="orange",
        label=f"{os.path.basename(base_name)}",
    )
    
    if not matching_true_data.empty:
        all_h_d_rev = []
        all_true_revolution = []
        
        for flow_vel, flow_data in matching_true_data.groupby('flow_velocity'):
            all_h_d_rev.extend(flow_data['H/D'].tolist())
            all_true_revolution.extend(flow_data['TRUE_revolution'].tolist())
        
        axes[i, 1].scatter(
            all_h_d_rev,
            all_true_revolution,
            alpha=0.8,
            marker='x',
            s=50,
            color='red',
            label=f"TRUE {folder_name}",
        )
    
    axes[i, 1].set_xlabel(r"$h/D$")
    axes[i, 1].set_ylabel("Revolution (rad/s)")
    if i == 0:
        axes[i, 1].set_title("Revolution vs Height", fontsize=22)
    # axes[i, 1].grid()
    axes[i, 1].legend(loc="upper right")
    if len(heights_orb_rev) > 1:
        poly_coeffs = np.polyfit(heights_orb_rev, orbital_revs, 2)
        trend_line = np.poly1d(poly_coeffs)
        axes[i, 1].plot(
            x_trend, trend_line(x_trend), color="red", linestyle="--", label="Trend"
        )

# axes[-1, 0].plot(
#     exp_indices,
#     exp_avg_abs_rot,
#     alpha=0.7,
#     color="blue",
#     label="Average Rotation",
#     marker="o",
#     linestyle="-",
# )
# axes[-1, 0].set_xlabel("Inlet Flow Rate (L/h)")
# axes[-1, 0].set_ylabel("Avg Rotation (rad/s)")
# axes[-1, 0].set_title("Inlet Flow Rate vs Absolute Rotation")
# # axes[-1, 0].grid()
# axes[-1, 0].legend()

# axes[-1, 1].plot(
#     exp_indices,
#     exp_avg_orb_rev,
#     alpha=0.7,
#     color="red",
#     label="Average Orbital Revolution",
#     marker="o",
#     linestyle="-",
# )
# axes[-1, 1].set_xlabel("Inlet Flow Rate (L/h)")
# axes[-1, 1].set_ylabel("Avg Orbital Revolution (rad/s)")
# axes[-1, 1].set_title("Inlet Flow Rate vs Orbital Revolution")
# # axes[-1, 1].grid()
# axes[-1, 1].legend()

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

axes2[0].plot(
    exp_indices,
    exp_avg_abs_rot,
    alpha=0.7,
    color="blue",
    label="Average Rotation",
    marker="o",
    linestyle="-",
)
axes2[0].set_xlabel("Inlet Flow Rate (L/h)")
axes2[0].set_ylabel("Avg Rotation (rad/s)")
axes2[0].set_title("Inlet Flow Rate vs Absolute Rotation", fontsize=18)
axes2[0].legend(loc="upper left")

axes2[1].plot(
    exp_indices,
    exp_avg_orb_rev,
    alpha=0.7,
    color="red",
    label="Average Orbital Revolution",
    marker="o",
    linestyle="-",
)
axes2[1].set_xlabel("Inlet Flow Rate (L/h)")
axes2[1].set_ylabel("Avg Orbital Revolution (rad/s)")
axes2[1].set_title("Inlet Flow Rate vs Orbital Revolution", fontsize=18)
axes2[1].legend(loc="upper left")

plt.show()
