import json
import math
import os
import shutil
import sys

from process_utils import get_latest_folder

y_track_project = sys.argv[1] if len(sys.argv) > 1 else "runs/track"
y_camera_fps = 8000
if len(sys.argv) > 2:
    try:
        y_camera_fps = float(sys.argv[2])
    except ValueError:
        print(f"Invalid fps argument: {sys.argv[2]}, using default {y_camera_fps}")

base_path = os.path.normpath(y_track_project)
initial_result_directory = os.path.join(get_latest_folder(base_path), "initial_result")
stats_file_path = os.path.join(initial_result_directory, "all_stats.json")
calculation_results_path = os.path.join(
    initial_result_directory, "calculation_results.json"
)
if os.path.exists(stats_file_path):
    with open(stats_file_path, "r") as stats_file:
        all_stats = json.load(stats_file)
else:
    print("all_stats.json not found, exiting")
    exit()
calculation_results = []
keys_to_delete = []

for key, value in all_stats.items():
    try:
        revolution_notice = None
        not_use = value.get("not_use", False)
        not_use_rotation = value.get("not_use_rotation", False)
        not_use_revolution = value.get("not_use_revolution", False)
        must_not_use = value.get("must_not_use", False)
        # if not_use:
        #     continue
        changes = value.get("changes")
        closest_point = value.get("closest_point")
        total_frames_revolution = value.get("total_frames_revolution")
        total_frames_rotation = value.get("total_frames_rotation")
        d1_with_range_revolution = value.get(
            "d1_with_range_revolution", None
        )
        d2_with_range_revolution = value.get("d2_with_range_revolution", None)
        d1_origin = value.get("d1_origin", None)
        d2_origin = value.get("d2_origin", None)
        inner_diameter = value.get("inner_diameter", None)
        margin = value.get("margin", None)
        height = value.get("height", None)
        # origin_data = value.get("origin_data", [])
        not_use_revolution_margin_large = False
        if margin is None:
            margin = 0
            not_use_revolution_margin_large = True
        if margin > 40:
            # margin = 8
            # must_not_use = True
            print(f"{key}: margin > 40, margin: {margin}")
            not_use_revolution_margin_large = True
        if None in [
            d1_with_range_revolution,
            d2_with_range_revolution,
            d1_origin,
            d2_origin,
            inner_diameter,
            margin,
        ]:
            raise ValueError(f"Missing required fields for entry {key}, skipping.")
        radius = (
            inner_diameter / 2 - margin * 147 / 101
        )

        if (
            radius < d1_origin * 0.8
        ):
            # radius = max(d1, d2)
            if margin > 30: # 30/101 = 0.297cm

                print(
                    f"{key}: d1_origin > radius, radius: {radius}, d1_origin: {d1_origin}, margin: {margin}"
                )
                revolution_notice = (
                    f"d1_origin:{d1_origin * 0.8} too large, radius: {radius}"
                )
                not_use_revolution_margin_large = True
                # radius = max(
                #     d1_with_range_revolution, d2_with_range_revolution
        if radius < d2_origin * 0.8:
            # radius = max(d1, d2)
            # not_use_revolution = True
            if margin > 30:
                print(
                    f"{key}: d2_origin > radius, radius: {radius}, d2_origin: {d2_origin}, margin: {margin}"
                )
                revolution_notice = (
                    f"d2_origin0.9:{d1_origin * 0.9} too large, radius: {radius}"
                )
                not_use_revolution_margin_large = True
            # radius = max(d1_with_range_revolution, d2_with_range_revolution)
        value1 = d1_with_range_revolution / radius
        value2 = d2_with_range_revolution / radius

        # if not (-1 <= value1 <= 1) or not (-1 <= value2 <= 1):
        #     raise ValueError(
        #     )

        alpha1 = math.asin(value1)
        alpha2 = math.asin(value2)
        orbital_rev = (y_camera_fps * (alpha1 + alpha2)) / (
            total_frames_revolution - 1
        )
        if not_use_revolution_margin_large:
            orbital_rev = 0
            not_use_revolution = True
        
        should_delete = False
        if (not_use_rotation or not_use) and (not_use_revolution or not_use_revolution_margin_large):
            should_delete = True
            id_folder_path = os.path.join(initial_result_directory, str(key))
            print(f"ID {key}: both rotation and revolution unusable, removing folder")
            try:
                if os.path.exists(id_folder_path):
                    shutil.rmtree(id_folder_path)
                    print(f"  ✓ Removed folder: {id_folder_path}")
                    keys_to_delete.append(key)
                else:
                    print(f"  ⚠ Folder not found: {id_folder_path}")
            except Exception as e:
                print(f"  ✗ Removal failed: {id_folder_path}, error: {e}")
            continue
        
        if not must_not_use:
            if not_use_rotation or not_use:
                abs_rotation = 0
                rel_rotation = 0
                result = (
                    f"id: {key}, revolution: {orbital_rev:.2f} rad/s, height: {height}cm"
                    if not not_use_revolution
                    else f"id: {key}, revolution: {orbital_rev:.2f} rad/s, not_use_revolution, height: {height}cm"
                )
            else:
                abs_rotation = (
                    (changes * 3.1416 * y_camera_fps) / 2 / (total_frames_rotation - 1)
                )
                rel_rotation = orbital_rev + abs_rotation
                result = (
                    f"id: {key}, revolution: {orbital_rev:.2f}rad/s, rotation: {abs_rotation:.2f}rad/s, relative: {rel_rotation:.2f}rad/s, height: {height}cm"
                    if not not_use_revolution
                    else f"id: {key}, revolution: {orbital_rev:.2f}rad/s, rotation: {abs_rotation:.2f}rad/s, relative: {rel_rotation:.2f}rad/s, height: {height}cm, not_use_revolution"
                )
        else:
            orbital_rev = 0
            abs_rotation = 0
            rel_rotation = 0
            result = f"id: {key}, must_not_use, height: {height}cm"
        all_stats[key].update(
            {
                "orbital_rev": orbital_rev,
                "abs_rotation": abs_rotation,
                "rel_rotation": rel_rotation,
                "not_use_revolution": not_use_revolution,
                "revolution_notice": revolution_notice,
            }
        )
        print(result)
        calculation_results.append(result)

    except ValueError as e:
        print(f"Error processing entry with key {key}: {e}")
    except Exception as e:
        print(f"Unexpected error processing entry with key {key}: {e}")

for key in keys_to_delete:
    if key in all_stats:
        del all_stats[key]
        print(f"Removed record from all_stats: {key}")

with open(stats_file_path, "w") as stats_file:
    json.dump(all_stats, stats_file, indent=4)
with open(calculation_results_path, "w") as calc_file:
    json.dump(calculation_results, calc_file, indent=4)
