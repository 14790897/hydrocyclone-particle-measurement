import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime

import cv2
from process_utils import (
    calculate_distance_and_draw,
    draw_line_with_label,
    extract_frame,
    get_latest_folder,
)

y_track_project = sys.argv[1] if len(sys.argv) > 1 else "runs/track"
x_detect_project = sys.argv[2] if len(sys.argv) > 2 else "runs_x_me/detect"
x_camera_fps_ratio = 0.5
if len(sys.argv) > 3:
    try:
        x_camera_fps_ratio = float(sys.argv[3])
    except ValueError:
        print(f"Invalid fps-ratio argument: {sys.argv[3]}, using default {x_camera_fps_ratio}")

base_path = os.path.normpath(y_track_project)
initial_result_directory = os.path.join(get_latest_folder(base_path), "initial_result")
if not get_latest_folder(base_path).endswith("-2"):
    line_dict = {"1": (269, 49, 269, 328), "2": (269, 328, 250, 638)}
else:
    line_dict = {"1": (244, 0, 213, 639), "2": (244, 0, 213, 639)}

stats_file_path = os.path.join(initial_result_directory, "all_stats.json")
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if os.path.exists(stats_file_path):
    with open(stats_file_path, "r") as stats_file:
        all_stats = json.load(stats_file)
base_path = x_detect_project
x_detect_result_path = os.path.join(get_latest_folder(base_path), "detections.json")

with open(x_detect_result_path, "r") as f:
    x_data = json.load(f)
offset = 496

for k, v in all_stats.items():
    try:
        id_path = os.path.join(initial_result_directory, k)
        x_images_path = os.path.join(id_path, "x_images")
        if os.path.exists(x_images_path):
            shutil.rmtree(x_images_path)
        os.makedirs(x_images_path, exist_ok=True)
        if "closest_point" in v:
            closest_point = v["closest_point"]

        id = closest_point["ID"]
        frame = closest_point["Frame"]
        half_frame = int(round(frame * x_camera_fps_ratio))
        if half_frame < 1:
            half_frame = 1
        max_frame = max(map(int, x_data.keys()))
        min_frame = min(map(int, x_data.keys()))
        frame_range = list(
            range(max(min_frame, half_frame - 8), min(max_frame, half_frame + 8))
        )

        aggregated_results = defaultdict(list)

        for number in frame_range:
            frame_name = str(number)
            try:
                if frame_name in x_data:
                    aggregated_results[frame_name].extend(
                        x_data[frame_name]["detections"]
                    )
                else:
                    # print(f"Frame {frame_name} not found in detections")
                    pass

            except KeyError:
                print(f"No detections found for frame: {frame_name}")
            except json.JSONDecodeError:
                print(f"Error decoding JSON for file: {x_detect_result_path}")
        image_y_y_coord = (closest_point["Box"][1] + closest_point["Box"][3]) / 2
        if len(aggregated_results) < 10:
            available_frames = list(x_data.keys())
            available_frames = sorted(
                available_frames, key=lambda x: abs(int(x) - half_frame)
            )
            closest_frames = available_frames[:10]
            for closest_frame in closest_frames:
                if closest_frame in x_data:
                    aggregated_results[closest_frame].extend(
                        x_data[closest_frame]["detections"]
                    )
        min_margin = float("inf")
        min_margin_img_path = None
        for frame_name, detections in aggregated_results.items():
            frame_path = os.path.join(x_images_path, f"frame_{frame_name}.jpg")
            extract_frame(
                detections[0]["file_path"], str(int(frame_name) - 1), frame_path
            )
            frame_image = cv2.imread(frame_path)
            for detection in detections:
                x1, y1, x2, y2 = detection["bbox"]
                cv2.rectangle(
                    frame_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2
                )
            cv2.imwrite(frame_path, frame_image)
            for entry in detections:
                x1, y1, x2, y2 = entry["bbox"]
                # frame_image = cv2.imread(frame_image_path)
                x, y = (x1 + x2) / 2, (y1 + y2) / 2
                if not get_latest_folder(base_path).endswith("-2"):

                    image_x_y_coord = int(
                        (image_y_y_coord + 44) * 101 / 149 + 5
                    )
                else:
                    image_x_y_coord = int((image_y_y_coord) * 101 / 149)
                image_x_down_threshold = image_x_y_coord - 15
                image_x_up_threshold = image_x_y_coord + 15
                if image_x_down_threshold <= y <= image_x_up_threshold:
                    dist_to_line1 = (
                        calculate_distance_and_draw((x, y), line_dict["1"])[0]
                        if y < 328
                        else None
                    )
                    dist_to_line2 = (
                        calculate_distance_and_draw((x, y), line_dict["2"])[0]
                        if y >= 328
                        else None
                    )
                    distances = [
                        d for d in [dist_to_line1, dist_to_line2] if d is not None
                    ]

                    if distances:
                        margin = min(distances)
                        if margin < min_margin:
                            min_margin = margin
                            min_margin_img_path = (
                                x_images_path + "/frame_" + frame_name + ".jpg"
                            )
                            os.makedirs(
                                os.path.dirname(min_margin_img_path), exist_ok=True
                            )
                            # extract_frame(
                            #     entry["file_path"], frame_name, min_margin_img_path
                            # )
                            # min_margin_image = cv2.imread(min_margin_img_path)
                            # cv2.rectangle(
                            #     min_margin_image,
                            #     (int(x1), int(y1)),
                            #     (int(x2), int(y2)),
                            #     (255, 0, 0),
                            #     2,
                            # )
                            # cv2.imwrite(min_margin_img_path, min_margin_image)
                            # min_margin_img_path = entry["file_path"]
                            min_margin_coord = (x, y)
                # else:
        if min_margin_img_path is not None:
            print(f"{k} minimum distance: {min_margin}")
            all_stats[str(id)].update({"margin": min_margin, "timestamp": current_time})
            min_margin_image = cv2.imread(min_margin_img_path)
            color = (0, 255, 0)
            tl = 2

            line1 = line_dict["1"]
            line2 = line_dict["2"]

            draw_line_with_label(min_margin_image, line1[:2], line1[2:], color, tl, "1")
            draw_line_with_label(min_margin_image, line2[:2], line2[2:], color, tl, "2")
            cv2.line(
                min_margin_image,
                (0, image_x_down_threshold),
                (min_margin_image.shape[:2][1], image_x_down_threshold),
                color,
                tl,
            )
            cv2.line(
                min_margin_image,
                (0, image_x_up_threshold),
                (min_margin_image.shape[:2][1], image_x_up_threshold),
                color,
                tl,
            )

            image1 = (
                calculate_distance_and_draw(
                    min_margin_coord, line_dict["1"], min_margin_image
                )[3]
                if min_margin_coord[1] < 328
                else None
            )
            image2 = (
                calculate_distance_and_draw(
                    min_margin_coord, line_dict["2"], min_margin_image
                )[3]
                if min_margin_coord[1] >= 328
                else None
            )
            image = [d for d in [image1, image2] if d is not None]
            image_x_min_distance_path = os.path.splitext(
                os.path.basename(min_margin_img_path)
            )[0]
            output_image_path = os.path.join(
                id_path,
                "x_images",
                f"{image_x_min_distance_path}_min_margin_result.jpg",
            )
            cv2.imwrite(output_image_path, min_margin_image)
            print(f"Min-margin result image saved to {output_image_path}")
        else:
            print(f"{k}: no matching detection, set to 8")
            all_stats[str(id)].update(
                {
                    "margin": None,
                    "timestamp": current_time,
                    "not_use_revolution": False,
                }
            )
            # shutil.rmtree(id_path)
    except FileNotFoundError:
        print(f"Particle {k}: output.json not found, skipping.")

with open(stats_file_path, "w") as stats_file:
    all_stats_filtered = {k: v for k, v in all_stats.items() if "margin" in v}
    json.dump(all_stats_filtered, stats_file, indent=4)
    print(f"Updated data saved to {stats_file_path}")
