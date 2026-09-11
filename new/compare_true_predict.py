import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator

if len(sys.argv) > 1 and sys.argv[1] == "--save":
    matplotlib.use('Agg')
    SAVE_MODE = True
else:
    SAVE_MODE = False

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

def load_comparison_data():
    """
    Load manual (true) and predicted data from an Excel file for comparison.
    """
    data = pd.read_excel(
        "data/relative error.xlsx",
        sheet_name="Sheet1",
        header=0,
    )
    
    print("Raw data shape:", data.shape)
    print("Columns:", data.columns.tolist())
    
    numeric_columns = ['height', 'diameter', 'TRUE_rotation', 'TRUE_revolution', 'predict_rotation', 'predict_revolution', 'H/D']
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    
    if 'H/D' not in data.columns and 'height' in data.columns and 'diameter' in data.columns:
        data["H/D"] = data["height"] / data["diameter"] * 147
    
    if 'flow_velocity' in data.columns:
        data['flow_velocity'] = data['flow_velocity'].astype(str)
    else:
        print("Warning: no 'flow_velocity' column found")
        return pd.DataFrame()
    
    valid_data = data[data['flow_velocity'].notna()]
    valid_data = valid_data[valid_data['flow_velocity'] != 'nan']
    valid_data = valid_data[valid_data['flow_velocity'] != 'None']
    
    print(f"Valid rows: {len(valid_data)}")
    if len(valid_data) > 0:
        print("Flow-rate groups:", valid_data['flow_velocity'].unique())
    
    return valid_data

def standardize_flow_velocity(flow_vel):
    """
    Normalise flow-rate labels to a common base name.
    """
    flow_str = str(flow_vel).replace('.0', '').split('-')[0].split('.')[0]
    return flow_str

def plot_comparison():
    """
    Plot the manual-vs-predicted comparison.
    """
    data = load_comparison_data()
    
    if data.empty:
        print("No valid data to plot")
        return
    
    data['flow_velocity_std'] = data['flow_velocity'].apply(standardize_flow_velocity)
    
    flow_groups = data.groupby('flow_velocity_std')
    unique_flows = sorted(data['flow_velocity_std'].unique())
    
    print(f"Found {len(unique_flows)} distinct flow-rate conditions: {unique_flows}")
    
    n_flows = len(unique_flows)
    ncols = 2
    nrows = (n_flows + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows),
                            gridspec_kw={'hspace': 0.25, 'wspace': 0.35})

    axes = axes.flatten() if n_flows > 1 else [axes]

    for i, flow_vel in enumerate(unique_flows):
        flow_data = flow_groups.get_group(flow_vel)

        print(f"Processing flow rate {flow_vel}: {len(flow_data)} data points")

        ax = axes[i]

        all_speeds = []
        rotation_speeds = []
        revolution_speeds = []

        rotation_data = flow_data.dropna(subset=['height','TRUE_rotation', 'predict_rotation'])

        if len(rotation_data) > 0:
            ax.scatter(
                rotation_data['height'] * 10,
                rotation_data['TRUE_rotation'],
                facecolors='blue',
                edgecolors='blue',
                marker='o',
                s=80,
                label='Rotation (MANUAL)' if i == 0 else None
            )
            ax.scatter(
                rotation_data['height'] * 10,
                rotation_data['predict_rotation'],
                facecolors='red',
                edgecolors='red',
                marker='^',
                s=80,
                label='Rotation (THIS STUDY)' if i == 0 else None
            )
            all_speeds.extend(rotation_data['TRUE_rotation'].tolist())
            all_speeds.extend(rotation_data['predict_rotation'].tolist())
            rotation_speeds.extend(rotation_data['TRUE_rotation'].tolist())
            rotation_speeds.extend(rotation_data['predict_rotation'].tolist())

            print(f"  - rotation points: {len(rotation_data)}")
        else:
            print("  - no rotation data")

        revolution_data = flow_data.dropna(subset=['height', 'TRUE_revolution', 'predict_revolution'])

        if len(revolution_data) > 0:
            ax.scatter(
                revolution_data['height'] * 10,
                revolution_data['TRUE_revolution'],
                facecolors='green',
                edgecolors='green',
                marker='s',
                s=80,
                label='Revolution (MANUAL)' if i == 0 else None
            )
            ax.scatter(
                revolution_data['height'] * 10,
                revolution_data['predict_revolution'],
                facecolors='orange',
                edgecolors='orange',
                marker='D',
                s=80,
                label='Revolution (THIS STUDY)' if i == 0 else None
            )

            all_speeds.extend(revolution_data['TRUE_revolution'].tolist())
            all_speeds.extend(revolution_data['predict_revolution'].tolist())
            revolution_speeds.extend(revolution_data['TRUE_revolution'].tolist())
            revolution_speeds.extend(revolution_data['predict_revolution'].tolist())

            print(f"  - revolution points: {len(revolution_data)}")
        else:
            print("  - no revolution data")

        ax.set_xlabel(r"$h$ (mm)")
        ax.set_ylabel("Speed (rad/s)")

        ax.set_xlim(0, 110)
        ax.set_ylim(0, 3000)

        ax.set_yticks([0, 1000, 2000, 3000])

        subplot_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']
        if i < len(subplot_labels):
            ax.set_title(subplot_labels[i], loc='left', x=-0.25, y=0.9)

        flow_text = f"$Q_i$={flow_vel} L/h"
        ax.text(0.98, 0.98, flow_text, transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='right',
        fontsize=14)

        if i == 0:
            ax.legend(loc="upper left", fontsize=12, framealpha=0.9, frameon=False)

        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(which='minor', direction='in')
        ax.tick_params(which='major', direction='in')

    if len(axes) > n_flows:
        ax_f = axes[n_flows]
        flows, rot_acc, rev_acc = [], [], []
        for flow_vel in unique_flows:
            fd = flow_groups.get_group(flow_vel)
            fr = fd.dropna(subset=['TRUE_rotation', 'predict_rotation'])
            if len(fr):
                rot_err = (
                    abs(fr['predict_rotation'] - fr['TRUE_rotation'])
                    / fr['TRUE_rotation'] * 100
                ).mean()
                rot_acc.append(100 - rot_err)
            else:
                rot_acc.append(float('nan'))
            fv = fd.dropna(subset=['TRUE_revolution', 'predict_revolution'])
            if len(fv):
                rev_err = (
                    abs(fv['predict_revolution'] - fv['TRUE_revolution'])
                    / fv['TRUE_revolution'] * 100
                ).mean()
                rev_acc.append(100 - rev_err)
            else:
                rev_acc.append(float('nan'))
            flows.append(str(flow_vel))

        x = np.arange(len(flows))
        w = 0.36
        bars_rot = ax_f.bar(x - w / 2, rot_acc, w, label='Rotation',
                            color='royalblue', edgecolor='black')
        bars_rev = ax_f.bar(x + w / 2, rev_acc, w, label='Revolution',
                            color='seagreen', edgecolor='black')
        ax_f.set_xticks(x)
        ax_f.set_xticklabels([f"${f}$" for f in flows])
        ax_f.set_xlabel(r"$Q_i$ (L/h)")
        ax_f.set_ylabel("Accuracy (%)")
        ax_f.set_ylim(80, 101)
        ax_f.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax_f.xaxis.set_minor_locator(plt.NullLocator())
        ax_f.tick_params(which='minor', direction='in')
        ax_f.tick_params(which='major', direction='in')
        ax_f.legend(loc="best", fontsize=11, frameon=True, framealpha=0.9)
        ax_f.set_title('(f)', loc='left', x=-0.25, y=0.9)
        for xi, (ra, rv) in enumerate(zip(rot_acc, rev_acc)):
            if not np.isnan(ra):
                ax_f.text(xi - w / 2, ra + 0.4, f"{ra:.1f}", ha='center', fontsize=9)
            if not np.isnan(rv):
                ax_f.text(xi + w / 2, rv + 0.4, f"{rv:.1f}", ha='center', fontsize=9)

    for i in range(min(n_flows + 1, len(axes)), len(axes)):
        axes[i].set_visible(False)

    print("\n" + "="*60)
    print("Relative error (|predicted - true| / true * 100%)")
    print("="*60)

    rotation_valid = data.dropna(subset=['TRUE_rotation', 'predict_rotation'])
    if len(rotation_valid) > 0:
        rotation_valid['rotation_error'] = (
            abs(rotation_valid['predict_rotation'] - rotation_valid['TRUE_rotation'])
            / rotation_valid['TRUE_rotation'] * 100
        )

        print("\n[Rotation relative error]")
        print(f"  total points: {len(rotation_valid)}")
        print(f"  mean relative error: {rotation_valid['rotation_error'].mean():.2f}%")
        print(f"  std: {rotation_valid['rotation_error'].std():.2f}%")
        print(f"  min error: {rotation_valid['rotation_error'].min():.2f}%")
        print(f"  max error: {rotation_valid['rotation_error'].max():.2f}%")
        print(f"  median error: {rotation_valid['rotation_error'].median():.2f}%")

        print("\n  by flow rate:")
        for flow_vel in sorted(rotation_valid['flow_velocity_std'].unique()):
            flow_rot = rotation_valid[rotation_valid['flow_velocity_std'] == flow_vel]
            print(f"    {flow_vel} L/h: mean={flow_rot['rotation_error'].mean():.2f}%, "
                  f"std={flow_rot['rotation_error'].std():.2f}%, n={len(flow_rot)}")

    revolution_valid = data.dropna(subset=['TRUE_revolution', 'predict_revolution'])
    if len(revolution_valid) > 0:
        revolution_valid['revolution_error'] = (
            abs(revolution_valid['predict_revolution'] - revolution_valid['TRUE_revolution'])
            / revolution_valid['TRUE_revolution'] * 100
        )

        print("\n[Revolution relative error]")
        print(f"  total points: {len(revolution_valid)}")
        print(f"  mean relative error: {revolution_valid['revolution_error'].mean():.2f}%")
        print(f"  std: {revolution_valid['revolution_error'].std():.2f}%")
        print(f"  min error: {revolution_valid['revolution_error'].min():.2f}%")
        print(f"  max error: {revolution_valid['revolution_error'].max():.2f}%")
        print(f"  median error: {revolution_valid['revolution_error'].median():.2f}%")

        print("\n  by flow rate:")
        for flow_vel in sorted(revolution_valid['flow_velocity_std'].unique()):
            flow_rev = revolution_valid[revolution_valid['flow_velocity_std'] == flow_vel]
            print(f"    {flow_vel} L/h: mean={flow_rev['revolution_error'].mean():.2f}%, "
                  f"std={flow_rev['revolution_error'].std():.2f}%, n={len(flow_rev)}")

    print("\n" + "="*60 + "\n")

    if SAVE_MODE:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_dir = os.path.join(project_root, "plots")
        os.makedirs(output_dir, exist_ok=True)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "comparison_by_flow.png"), dpi=300, bbox_inches='tight')

        print(f"Plot saved to: {output_dir}")
        print("  - comparison_by_flow.png (per flow rate)")

        plt.close('all')
    else:
        # plt.suptitle("TRUE vs PREDICT Data Comparison by Flow Velocity", fontsize=20, y=0.98)
        plt.show()
    
    # create_overall_comparison(data)

def create_overall_comparison(data):
    """
    Overall predicted-vs-true scatter plot (single y-axis).
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    all_speeds = []

    unique_flows = sorted(data['flow_velocity_std'].unique())
    colors_rotation = ['blue', 'cyan', 'navy', 'steelblue', 'dodgerblue']
    colors_revolution = ['orange', 'coral', 'darkorange', 'orangered', 'tomato']

    if 'TRUE_rotation' in data.columns and 'predict_rotation' in data.columns:
        valid_rotation = data.dropna(subset=['TRUE_rotation', 'predict_rotation'])

        for i, flow_vel in enumerate(unique_flows):
            flow_data = valid_rotation[valid_rotation['flow_velocity_std'] == flow_vel]
            if len(flow_data) > 0:
                ax.scatter(
                    flow_data['TRUE_rotation'],
                    flow_data['predict_rotation'],
                    alpha=0.7,
                    color=colors_rotation[i % len(colors_rotation)],
                    s=80,
                    marker='o'
                )
                all_speeds.extend(flow_data['TRUE_rotation'].tolist())
                all_speeds.extend(flow_data['predict_rotation'].tolist())

    if 'TRUE_revolution' in data.columns and 'predict_revolution' in data.columns:
        valid_revolution = data.dropna(subset=['TRUE_revolution', 'predict_revolution'])

        for i, flow_vel in enumerate(unique_flows):
            flow_data = valid_revolution[valid_revolution['flow_velocity_std'] == flow_vel]
            if len(flow_data) > 0:
                ax.scatter(
                    flow_data['TRUE_revolution'],
                    flow_data['predict_revolution'],
                    alpha=0.7,
                    color=colors_revolution[i % len(colors_revolution)],
                    s=80,
                    marker='^'
                )
                all_speeds.extend(flow_data['TRUE_revolution'].tolist())
                all_speeds.extend(flow_data['predict_revolution'].tolist())

    if len(all_speeds) > 0:
        min_val = min(all_speeds)
        max_val = max(all_speeds)
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2)
        ax.set_xlim(min_val * 0.95, max_val * 1.05)
        ax.set_ylim(min_val * 0.95, max_val * 1.05)

    ax.set_xlabel("MANUAL Speed (rad/s)")
    ax.set_ylabel("THIS STUDY Speed (rad/s)")

    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which='minor', direction='in')
    ax.tick_params(which='major', direction='in')
    ax.grid(True, alpha=0.3)

    if SAVE_MODE:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_dir = os.path.join(project_root, "plots")
        os.makedirs(output_dir, exist_ok=True)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "comparison_overall.png"), dpi=300, bbox_inches='tight')
        print("  - comparison_overall.png (overall)")
        plt.close('all')
    else:
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    plot_comparison()
