import cv2, os, shutil
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from natsort import natsorted

# for line_coords in line_dict.values():
#     draw.line(line_coords, fill=(255, 0, 0), width=2)

def calculate_distance_and_draw(point, line_coords, image=None):
    x, y = point
    x1, y1, x2, y2 = line_coords

    if not (min(y1, y2) <= y <= max(y1, y2)):
        print(f"Point {point} is outside the y-range of line {line_coords}; please check")

    if x1 != x2:
        m = (y2 - y1) / (x2 - x1)
        x0 = x1 + (y - y1) / m
    else:
        x0 = x1

    distance = abs(x - x0)
    if image is not None:
        if isinstance(image, (str, bytes, os.PathLike)):
            image = cv2.imread(image)

        if isinstance(image, np.ndarray):
            offset = 10

            cv2.line(
                image,
                (int(x), int(y - offset)),
                (int(x0), int(y - offset)),
                (0, 0, 255),
                2,
            )

            cv2.putText(
                image,
                f"{distance:.2f}   y:{ int(y)}",
                (int(x + offset), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        image_pillow = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return distance, x0, y, image_pillow
    else:
        return distance, x0, y

def draw_line_with_label(
    img, start, end, color, thickness, label, line_type=cv2.LINE_AA
):
    cv2.line(img, start, end, color, thickness=thickness, lineType=line_type)
    cv2.putText(
        img,
        label,
        (start[0] + 5, start[1] - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
        line_type,
    )

def extract_frame(video_path, frame_number, output_image_path):
    cap = cv2.VideoCapture(video_path)
    frame_number = int(frame_number)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(f"Total frames in video: {total_frames}")

    if frame_number < 0 or frame_number >= total_frames:
        print(
            f"Error: frame_number {frame_number} is out of range (0 to {total_frames - 1})"
        )
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()
    if not ret:
        print(f"Error: Cannot read frame {frame_number}")
        return

    cv2.imwrite(output_image_path, frame)
    # print(f"Frame {frame_number} saved as {output_image_path}")

    cap.release()
    return frame

def extract_and_stitch_columns(video_path, data):
    """
    Crop a fixed X-range column from each frame and stitch them into one image.

    Args:
        video_path (str): input video path.
        data (list): items with crop boxes, each of the form
                     {"Frame": int, "Box": [x1, y1, x2, y2]}; only x1/x2 are used.

    Returns:
        stitched_image (np.ndarray): stitched image on success.
        None: if no column could be cropped.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return None

    columns = []

    for entry in data:
        frame_index = entry["Frame"]
        box = entry["Box"]
        x1, x2 = box[0], box[2]

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1)
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Unable to read frame {frame_index}.")
            continue

        height, _, _ = frame.shape
        cropped_column = frame[0:height, x1:x2]

        columns.append(cropped_column)

    cap.release()

    if columns:
        columns = columns[::-1]
        stitched_image = np.hstack(columns)
        return stitched_image
    else:
        print("No columns were extracted.")
        return None

def find_changes_within_range(
    category_changes, left_line_x, right_line_x, range_ratio=0.6
):
    """
    Keep only category_changes inside the given fraction of the left/right lines.

    :param category_changes: List of changes (each with a "Box" defining its bounding box).
    :param left_line_x: X-coordinate of the left boundary line.
    :param right_line_x: X-coordinate of the right boundary line.
    :param range_ratio: Proportion of the width to define the central range (default is 0.5 for 50%).
    :return: List of changes within the specified range.
    """
    mid_left = left_line_x + (1 - range_ratio) / 2 * (right_line_x - left_line_x)
    mid_right = right_line_x - (1 - range_ratio) / 2 * (right_line_x - left_line_x)

    changes_in_range = []
    for change in category_changes:
        box = change["Box"]
        center_x = (box[0] + box[2]) / 2
        if mid_left <= center_x <= mid_right:
            changes_in_range.append(change)
    if not changes_in_range:
        print(f"No changes found within the range ({mid_left:.2f} to {mid_right:.2f}).")
    return changes_in_range

def get_box_center(x1, y1, x2, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2

def detect_frame_difference(data):
    """
    Flag adjacent category changes whose frame gap is 2.
    """
    for key, value in data.items():
        category_changes = value.get("category_changes", [])
        # start_frame_revolution = value.get("start_frame_revolution", 0)
        # end_frame_revolution = value.get("end_frame_revolution", 0)
        not_use_rotation = value.get("not_use_rotation", False)
        if not not_use_rotation and category_changes:
            origin_all_frame = (
                category_changes[-1]["origin_frame"]
                - category_changes[0]["origin_frame"]
            )
            all_frame = category_changes[-1]["Frame"] - category_changes[0]["Frame"]
            count_frame_3_diff = 0
            previous_frame_diff = None
            previous_origin_frame_diff = None
            # all_frame = end_frame_revolution - start_frame_revolution
            for i in range(len(category_changes) - 1):
                current_frame = category_changes[i]["origin_frame"]
                next_frame = category_changes[i + 1]["origin_frame"]
                origin_frame_diff = next_frame - current_frame

                frame_diff = (
                    category_changes[i + 1]["Frame"] - category_changes[i]["Frame"]
                )
                if (
                    previous_frame_diff == 3
                    and frame_diff == 3
                    or previous_origin_frame_diff == 3
                    and origin_frame_diff == 3
                ):
                    print(
                        f"Detected consecutive frame differences of 3 at frame {current_frame} of id: {key}, "
                        f"change too fast; consider removing"
                    )
                    data[key]["not_use"] = True
                    data[key][
                        "reason"
                    ] = f"Consecutive frame_diff = 3 at frame {current_frame}"
                    break

                previous_frame_diff = frame_diff
                previous_origin_frame_diff = origin_frame_diff
                # if origin_frame_diff or frame_diff == 3:
                if origin_frame_diff == 2:
                    print(
                        f"Detected a frame difference of 2 at frame {current_frame} of id {key}; change too fast to measure reliably, consider removing"
                    )
                    data[key]["not_use"] = True
                    data[key]["reason"] = f"change too fast at frame {current_frame}"
                    pass
                elif (
                    origin_frame_diff >= origin_all_frame / 2
                    or frame_diff >= all_frame / 2
                ):
                    print(
                        f"have a long time not change, id: {key}, at frame {current_frame}, not use"
                    )
                    data[key]["not_use"] = True
                    data[key][
                        "reason"
                    ] = f"have a long time not change,at frame {current_frame}"
            if count_frame_3_diff >= 2:
                print(
                    f"Detected at least two frame differences of 3 for id {key}; change too fast, consider removing"
                )
                data[key]["not_use"] = True
                data[key][
                    "reason"
                ] = f"Detected {count_frame_3_diff} times frame_diff = 3"
    return data

def remove_empty(data):
    return {
        key: value for key, value in data.items() if value.get("category_changes", [])
    }

def remove_long_time_not_change(category_changes, id):
    """
    Check for adjacent-frame gaps > 15 and drop the smaller side (unused).
    """
    all_frame = category_changes[-1]["Frame"] - category_changes[0]["Frame"]
    i = 0
    while i < len(category_changes) - 1:
        current_frame = category_changes[i]["Frame"]
        next_frame = category_changes[i + 1]["Frame"]
        frame_diff = next_frame - current_frame

        if frame_diff >= all_frame / 3:
            print(
                f"Detected a frame difference > 15 at frame {current_frame}; dropping the smaller side."
            )
            front_part = category_changes[: i + 1]
            back_part = category_changes[i + 1 :]

            if len(front_part) <= len(back_part):
                print(f"Dropping front part, length: {len(front_part)},id: {id}")
                category_changes = back_part
            else:
                print(f"Dropping back part, length: {len(back_part)},id: {id}")
                category_changes = front_part

            i = -1
            break

        i += 1

    return category_changes

# def get_latest_folder(base_path):
#     entries = [os.path.join(base_path, entry) for entry in os.listdir(base_path)]
#     folders = [entry for entry in entries if os.path.isdir(entry)]
#     if not folders:
#         raise FileNotFoundError(f"No folders found in {base_path}")
#     latest_folder = max(folders, key=os.path.getmtime)
#     return latest_folder

def get_latest_folder(base_path):
    """
    Return the newest folder under `base_path`.
    If env var LATEST_FOLDER is set, use it.
    """

    latest_folder_from_env = os.getenv("LATEST_FOLDER")
    if latest_folder_from_env:
        folder_name = os.path.basename(latest_folder_from_env)
        full_folder_path = os.path.join(base_path, folder_name)
        print(f"Using folder from environment variable: {full_folder_path}")
        return full_folder_path

    folders = [
        os.path.join(base_path, entry)
        for entry in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, entry))
    ]

    # if len(folders) < 2:
    #     raise ValueError(
    #         f"Not enough folders in {base_path} to get the second oldest one."
    #     )

    sorted_folders = sorted(folders, key=os.path.getmtime)
    print(f"Using folder from sorted logic: {sorted_folders[-1]}")
    return sorted_folders[-1]

def get_all_folders(base_path):
    """
    Return all sub-folder paths under base_path, newest first.
    """
    entries = [os.path.join(base_path, entry) for entry in os.listdir(base_path)]
    folders = [entry for entry in entries if os.path.isdir(entry)]

    if not folders:
        raise FileNotFoundError(f"No folders found in {base_path}")

    sorted_folders = natsorted(folders)

    return sorted_folders

def find_video_files(directory):
    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")

    video_files = []
    video_name = None
    for filename in os.listdir(directory):
        if filename.endswith(video_extensions):
            video_files.append(os.path.join(directory, filename))
            video_name = filename

    return video_files, video_name

def convert_to_mp4(input_video: str) -> None:
    """
    Convert an input video to .mp4 in place (no-op if already .mp4).

    :param input_video: input video path (with extension)
    :return: the .mp4 path
    """
    input_video = os.path.normpath(input_video)
    output_video = os.path.splitext(input_video)[0] + ".mp4"
    if not input_video.lower().endswith(".mp4"):
        print(f"The input video '{input_video}' is not in MP4 format.")
        print("Proceeding to convert the video to MP4 format.")

        if os.path.exists(output_video):
            print(f"The file '{output_video}' already exists. It will be replaced.")
            os.remove(output_video)

        cap = cv2.VideoCapture(input_video)

        if not cap.isOpened():
            print("Error: Couldn't open the video file.")
            error_code = cap.get(cv2.CAP_PROP_FOURCC)
            print(f"Error Code: {error_code}")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*"h264")
        out = cv2.VideoWriter(
            output_video, fourcc, frame_rate, (frame_width, frame_height)
        )

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            out.write(frame)

        cap.release()
        out.release()

        print(f"Video conversion complete. Saved as '{output_video}'.")

        os.remove(input_video)
        print("The input video has been replaced with the converted MP4 file.")

    else:
        print(
            f"The input video '{input_video}' is already in MP4 format. No conversion needed."
        )
    return output_video

def shorten_video_opencv(
    input_video: str, start_time: float, end_time: float, output_video: str
):
    """
    Clip a time range out of a video with OpenCV.

    :param input_video: input video path
    :param start_time: start time (s)
    :param end_time: end time (s)
    :param output_video: output video path
    """
    cap = cv2.VideoCapture(input_video)

    fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"h264")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame >= end_frame:
            break

        out.write(frame)
        current_frame += 1

    cap.release()
    out.release()
    print(f"Video clip done, saved to {output_video}")

def clear_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"Cannot delete {item_path}. Error: {e}")
    print(f"Cleared folder: {folder_path}")

if __name__ == "__main__":
    # shorten_video_opencv(
    #     "650-1-x1_particle_video.mp4", 0, 5, "x_video.mp4"
    # )
    convert_to_mp4(r"xy1-650-S14-2_particle_video.avi")
