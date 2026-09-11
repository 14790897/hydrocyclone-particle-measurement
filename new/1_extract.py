import copy
import json
import os
import shutil
import sys
from collections import OrderedDict, defaultdict

from process_utils import (
    calculate_distance_and_draw,
    detect_frame_difference,
    find_changes_within_range,
    get_latest_folder,
    remove_empty,
)

y_track_project = sys.argv[1] if len(sys.argv) > 1 else "runs/track"

base_path = os.path.normpath(y_track_project)
initial_result_directory = os.path.join(get_latest_folder(base_path), "initial_result")
video_path = "my_process_particle_video.avi"
if not get_latest_folder(base_path).endswith("-2"):
    line_dict = {
        "1": (175, 31, 175, 437),
        "2": (175, 437, 209, 1021),
        "3": (551, 31, 551, 437),
        "4": (551, 437, 513, 1021),
        "5": (363, 31, 363, 1021),
    }
    y_min = 20
    y_max = 980
else:
    line_dict = {
        "1": (193, 0, 250, 1021),
        "2": (193, 0, 250, 1021),
        "3": (520, 0, 460, 1021),
        "4": (520, 0, 460, 1021),
        "5": (357, 0, 357, 1021),
    }
    y_min = 20
    y_max = 758

central_line_coords = line_dict["5"]
results = {}
results = defaultdict(
    lambda: {
        "changes": 0,
        "total_frames": 0,
        "category_changes": [],
        # "filter_data": [],
    }
)

exclude_last_frames = 8
exclude_first_frames = 0

all_stats = {}

def process_data():
    for i in os.listdir(initial_result_directory):
        if not os.path.isdir(os.path.join(initial_result_directory, i)):
            continue
        category_changes = []

        previous_category = None
        data_path = os.path.join(initial_result_directory, i, "initial_data.json")
        with open(data_path, "r") as file:
            initial_id_data = json.load(file)
        id_path = os.path.join(initial_result_directory, str(i))
        if not all(
            y_min <= (entry["Box"][1] + entry["Box"][3]) / 2 <= y_max
            for entry in initial_id_data
        ):
            print(f"Trajectory {i} out of Y range, skipping")
            # shutil.rmtree(id_path)
            results[i]["not_use"] = True
            results[i]["reason"] = "Out of Y coordinate range"
            continue

        # else:
        for count in range(
            0, len(initial_id_data)
        ):
            if count == 0:
                if (
                    len(initial_id_data) > 2
                    and initial_id_data[count + 1]["Category"]
                    == initial_id_data[count + 2]["Category"]
                ):
                    initial_id_data[count]["Category"] = initial_id_data[count + 1][
                        "Category"
                    ]
            elif count < len(initial_id_data) - 1:
                if (
                    initial_id_data[count - 1]["Category"]
                    == initial_id_data[count + 1]["Category"]
                    and initial_id_data[count]["Category"]
                    != initial_id_data[count - 1]["Category"]
                ):
                    initial_id_data[count]["Category"] = initial_id_data[count - 1][
                        "Category"
                    ]
            else:
                if count == len(initial_id_data) - 1:
                    if (
                        len(initial_id_data) > 2
                        and initial_id_data[count - 1]["Category"]
                        == initial_id_data[count - 2]["Category"]
                    ):
                        initial_id_data[count]["Category"] = initial_id_data[count - 1][
                            "Category"
                        ]
        previous_category = None
        category_start_frame = None
        category_changes = []
        # for count, entry in enumerate(initial_id_data):
        #     current_category = entry["Category"]
        #     if previous_category is None:
        #         previous_category = current_category
        #         continue

        #     if current_category != previous_category:
        #         category_changes.append(
        #             entry
        #         # results[id_]["changes"] += 1

        #     previous_category = current_category
        for count, entry in enumerate(initial_id_data):
            current_category = entry["Category"]
            current_frame = entry["Frame"]
            id_ = entry["ID"]

            if previous_category is None:
                previous_category = current_category
                category_start_frame = current_frame
                continue

            if current_category != previous_category:
                duration = current_frame - category_start_frame

                mid_frame = category_start_frame + duration // 2

                closest_entry = min(
                    initial_id_data,
                    key=lambda x: abs(x["Frame"] - mid_frame),
                )
                closest_entry["origin_frame"] = category_start_frame
                category_changes.append(closest_entry)

                previous_category = current_category
                category_start_frame = current_frame

        if category_start_frame is not None and previous_category is not None:
            duration = current_frame - category_start_frame
            mid_frame = category_start_frame + duration // 2

            closest_entry = min(
                initial_id_data, key=lambda x: abs(x["Frame"] - mid_frame)
            )
            closest_entry["origin_frame"] = category_start_frame
            category_changes.append(closest_entry)
        # if len(category_changes) == 0:
        #     break
        category_changes_with_all = copy.deepcopy(category_changes)
        true_last_appear_data = initial_id_data[-1]
        true_last_appear_data["origin_frame"] = initial_id_data[-1]["Frame"]
        if (
            true_last_appear_data["origin_frame"]
            - category_changes_with_all[-1]["origin_frame"]
            > 6
        ):
            category_changes_with_all.append(true_last_appear_data)
        origin_all_frame = (
            category_changes_with_all[-1]["origin_frame"]
            - category_changes_with_all[0]["origin_frame"]
        )
        all_frame = (
            category_changes_with_all[-1]["Frame"]
            - category_changes_with_all[0]["Frame"]
        )
        for id_category in range(len(category_changes_with_all) - 1):
            current_frame = category_changes_with_all[id_category]["origin_frame"]
            next_frame = category_changes_with_all[id_category + 1]["origin_frame"]
            origin_frame_diff = next_frame - current_frame
            frame_diff = (
                category_changes_with_all[id_category + 1]["Frame"]
                - category_changes_with_all[id_category]["Frame"]
            )
            print(f"{id_}Frame difference: {frame_diff}")
            # if frame_diff == 2:
            #     print(
            #     )
            #     results[i]["not_use"] = True
            #     results[i]["reason"] = f"Main change too fast at frame {current_frame}"
            # el
            if origin_frame_diff >= origin_all_frame / 2 or frame_diff >= all_frame / 2:
                print(
                    f"Main have a long time not change, id: {i}, at frame {current_frame}, not use"
                )
                results[id_]["not_use"] = True
                results[id_][
                    "reason"
                ] = f"Main have a long time not change，at frame {current_frame}"
        # if category_changes:
        #     category_changes = category_changes[:-1]
        # results[id_]["total_frames"] = (
        #     data[-exclude_last_frames - 1]["Frame"] - data[0]["Frame"] + 1
        # results[id_]["filter_data"] = initial_id_data[0:-exclude_last_frames]
        first_appear = initial_id_data[0]
        last_appear = initial_id_data[-1]
        first_appear_x_coord = (first_appear["Box"][0] + first_appear["Box"][2]) / 2
        first_appear_y_coord = (first_appear["Box"][1] + first_appear["Box"][3]) / 2
        last_appear_x_coord = (last_appear["Box"][0] + last_appear["Box"][2]) / 2
        last_appear_y_coord = (last_appear["Box"][1] + last_appear["Box"][3]) / 2
        # if last_appear_y_coord < first_appear_y_coord:
        #     print(f"ID {i}: Last frame is higher than the first frame, skipping.")
        #     continue
        first_appear_coordinates = (first_appear_x_coord, first_appear_y_coord)
        last_appear_coordinates = (last_appear_x_coord, last_appear_y_coord)
        if (first_appear_coordinates[0] - central_line_coords[0]) * (
            last_appear_coordinates[0] - central_line_coords[0]
        ) > 0:
            print(f"{id_} both points on the same side, removing")
            shutil.rmtree(id_path)
            # del results[id_]
            continue
        d1_origin, d2_origin = (
            calculate_distance_and_draw(p, central_line_coords)[0]
            for p in (first_appear_coordinates, last_appear_coordinates)
        )
        midpoint = [
            (first_appear_x_coord + last_appear_x_coord) / 2,
            (first_appear_y_coord + last_appear_y_coord) / 2,
        ]
        if midpoint[1] > 437.0:
            result_left = calculate_distance_and_draw(midpoint, line_dict["2"])
            result_right = calculate_distance_and_draw(midpoint, line_dict["4"])

            d_total_left = result_left[0]
            x_left = result_left[1]

            d_total_right = result_right[0]
            x_right = result_right[1]
            category_changes = find_changes_within_range(
                category_changes,
                x_left,
                x_right,
                range_ratio=0.6,
            )
            initial_id_data_with_range = find_changes_within_range(
                initial_id_data,
                x_left,
                x_right,
                range_ratio=0.6,
            )
        else:
            result_left = calculate_distance_and_draw(midpoint, line_dict["1"])
            result_right = calculate_distance_and_draw(midpoint, line_dict["3"])
            d_total_left = result_left[0]
            x_left = result_left[1]

            d_total_right = result_right[0]
            x_right = result_right[1]
            category_changes = find_changes_within_range(
                category_changes,
                x_left,
                x_right,
                range_ratio=0.6,
            )
            initial_id_data_with_range = find_changes_within_range(
                initial_id_data,
                x_left,
                x_right,
                range_ratio=0.6,
            )
        # category_changes = remove_long_time_not_change(category_changes,id_)
        first_appear = initial_id_data_with_range[
            0
        ]
        last_appear = initial_id_data_with_range[-1]
        if not category_changes:
            first_change = initial_id_data[0]
            first_change["origin_frame"] = initial_id_data[0]["Frame"]
            last_change = initial_id_data[-1]
            last_change["origin_frame"] = initial_id_data[-1]["Frame"]
        else:
            first_change = category_changes[0]
            last_change = category_changes[-1]
        first_appear_x_coord = (first_appear["Box"][0] + first_appear["Box"][2]) / 2
        first_appear_y_coord = (first_appear["Box"][1] + first_appear["Box"][3]) / 2
        last_appear_x_coord = (last_appear["Box"][0] + last_appear["Box"][2]) / 2
        last_appear_y_coord = (last_appear["Box"][1] + last_appear["Box"][3]) / 2
        first_appear_coordinates = (
            first_appear_x_coord,
            first_appear_y_coord,
        )
        last_appear_coordinates = (last_appear_x_coord, last_appear_y_coord)
        if (first_appear_coordinates[0] - central_line_coords[0]) * (
            last_appear_coordinates[0] - central_line_coords[0]
        ) > 0:
            print(f"{id_} both points on the same side within filter range, removing")
            shutil.rmtree(id_path)
            # del results[id_]
            continue
        d_total = d_total_left + d_total_right

        first_image = os.path.join(
            initial_result_directory,
            i,
            "images",
            f"frame_{first_appear['Frame']}.jpg",
        )
        last_image = os.path.join(
            initial_result_directory,
            i,
            "images",
            f"frame_{last_appear['Frame']}.jpg",
        )
        first_last_output_path = os.path.join(
            initial_result_directory, i, "first_last_output"
        )
        os.makedirs(first_last_output_path, exist_ok=True)

        shutil.copy(first_image, first_last_output_path)
        shutil.copy(last_image, first_last_output_path)

        new_image1 = calculate_distance_and_draw(
            first_appear_coordinates,
            central_line_coords,
            first_image,
        )[3]
        new_image2 = calculate_distance_and_draw(
            last_appear_coordinates, central_line_coords, last_image
        )[3]
        new_image1.save(os.path.join(first_last_output_path, "1_distance_result.jpg"))
        new_image2.save(os.path.join(first_last_output_path, "2_distance_result.jpg"))
        # stitched_image_path = os.path.join(
        #     initial_result_directory, i, "images", "stitched_image.jpg"
        # )
        # if not os.path.exists(stitched_image_path):
        #     stitched_image = extract_and_stitch_columns(
        #         video_path, category_changes
        #     )
        #     cv2.imwrite(
        #         stitched_image_path,
        #         stitched_image,
        #     )
        min_distance = float("inf")
        closest_point = None

        for item in initial_id_data:
            point = [
                (item["Box"][0] + item["Box"][2]) / 2,
                (item["Box"][1] + item["Box"][3]) / 2,
            ]
            distance = calculate_distance_and_draw(point, central_line_coords)[0]
            if distance < min_distance:
                min_distance = distance
                closest_point = point
                closest_point_data = item
        if closest_point:
            # print(
            # )
            frame_number = closest_point_data.get("Frame")
            if frame_number is not None:
                closest_image_filename = f"frame_{frame_number}.jpg"
                closest_image_path = os.path.join(
                    id_path, "images", closest_image_filename
                )
                if os.path.exists(closest_image_path):
                    saved_image_path = os.path.join(
                        first_last_output_path,
                        f"closest_point_{closest_image_filename}.jpg",
                    )
                    shutil.copy(closest_image_path, saved_image_path)
                else:
                    print(f"Frame image not found: {closest_image_path}")
            else:
                print("Closest-point frame number not found; cannot save image")
        else:
            print(f"{id_}: no point nearest to the central line found")
        d1_with_range_revolution, d2_with_range_revolution = (
            calculate_distance_and_draw(p, central_line_coords)[0]
            for p in (first_appear_coordinates, last_appear_coordinates)
        )
        with open(os.path.join(first_last_output_path, "output.json"), "w") as file:
            json.dump([first_appear, last_appear], file, indent=4)

        print(f"id {i}: Category changed {len(category_changes) - 1} times")
        total_frames_revolution = last_appear["Frame"] - first_appear["Frame"] + 1
        total_frames_rotation = last_change["Frame"] - first_change["Frame"] + 1
        total_frames_rotation_origin = (
            last_change["origin_frame"] - first_change["origin_frame"] + 1
        )
        box = closest_point_data.get("Box", [0, 0, 0, 0])
        if not get_latest_folder(base_path).endswith("-2"):
            height = (
                (box[1] + box[3]) / 2 / 147
            )
        else:
            height = (box[1] + box[3]) / 2 / 147 + 1024 / 147 - (40 + 147 + 17) / 147
        results[id_]["changes"] = (
            len(category_changes)
            - 1
        )
        results[id_]["category_changes"] = category_changes
        results[id_]["total_frames_revolution"] = total_frames_revolution
        results[id_]["start_frame_revolution"] = first_appear["Frame"]
        results[id_]["end_frame_revolution"] = last_appear["Frame"]
        results[id_]["total_frames_rotation"] = total_frames_rotation
        results[id_]["start_frame_rotation"] = first_change["Frame"]
        results[id_]["end_frame_rotation"] = last_change["Frame"]
        results[id_]["d1_with_range_revolution"] = d1_with_range_revolution
        results[id_]["d2_with_range_revolution"] = d2_with_range_revolution
        results[id_]["d1_origin"] = d1_origin
        results[id_]["d2_origin"] = d2_origin
        results[id_]["inner_diameter"] = d_total
        results[id_]["closest_point"] = closest_point_data
        results[id_]["height"] = height
        if len(category_changes_with_all) < 2:
            print(f"Not enough frames to process for id: {i}")
            results[id_]["not_use_rotation"] = True
            results[id_]["reason"] = "Not enough frames to process"
            continue
        if (
            min(d1_with_range_revolution, d2_with_range_revolution)
            / max(d1_with_range_revolution, d2_with_range_revolution)
            < 0.6
        ):
            results[id_]["must_not_use"] = True
            # results[id_]["not_use"] = True
            results[id_]["reason"] = "the difference between d1 and d2 is too large"
            print(f"id {i}: d1/d2 ratio too large, skipping")
        if (
            height > 10.5
        ):
            results[id_]["not_use_rotation"] = True
            print(f"id {i}: height > 10.5, computing revolution only")
        if len(category_changes) < 3:
            results[id_]["not_use_rotation"] = True
            print(f"id {i}: fewer than 3 changes in range, computing revolution only")
        else:
            print(f"id {i}: more than 2 changes in range")
        if total_frames_rotation_origin / total_frames_revolution < 0.4:
            results[id_]["not_use_rotation"] = True
            print(f"id {i}: rotation-time fraction < 0.4, computing revolution speed only")

    stats_filepath = os.path.join(initial_result_directory, "all_stats.json")

    if os.path.exists(stats_filepath):
        with open(stats_filepath, "r") as stats_file:
            try:
                all_stats = json.load(stats_file)
            except json.JSONDecodeError:
                all_stats = {}
    else:
        all_stats = {}

    for id_, result in results.items():
        if f"{id_}" not in all_stats:
            all_stats[f"{id_}"] = {}
        all_stats[f"{id_}"].update(
            OrderedDict(
                [
                    ("start_frame_revolution", result.get("start_frame_revolution")),
                    ("end_frame_revolution", result.get("end_frame_revolution")),
                    ("total_frames_revolution", result.get("total_frames_revolution")),
                    ("start_frame_rotation", result.get("start_frame_rotation")),
                    ("end_frame_rotation", result.get("end_frame_rotation")),
                    ("total_frames_rotation", result.get("total_frames_rotation")),
                    ("changes", result.get("changes")),
                    ("category_changes", result.get("category_changes")),
                    (
                        "d1_with_range_revolution",
                        result.get("d1_with_range_revolution"),
                    ),
                    (
                        "d2_with_range_revolution",
                        result.get("d2_with_range_revolution"),
                    ),
                    ("d1_origin", result.get("d1_origin")),
                    ("d2_origin", result.get("d2_origin")),
                    ("inner_diameter", result.get("inner_diameter")),
                    ("closest_point", result.get("closest_point")),
                    ("height", result.get("height")),
                    ("not_use", result.get("not_use")),
                    ("not_use_rotation", result.get("not_use_rotation")),
                    ("must_not_use", result.get("must_not_use")),
                    ("reason", result.get("reason")),
                ]
            )
        )
        # print("id", id_)
        # print(
        #     f"Category Changes: {result['changes']}, height："
        #     f"{height}cm"
        # )
    all_stats = remove_empty(all_stats)
    all_stats = detect_frame_difference(all_stats)
    with open(stats_filepath, "w") as stats_file:
        json.dump(all_stats, stats_file, indent=4)

if __name__ == "__main__":
    process_data()
