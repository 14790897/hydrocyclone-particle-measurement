import json
import os
import sys

from process_utils import find_video_files, get_latest_folder

x_detect_project = sys.argv[1] if len(sys.argv) > 1 else "runs_x_me/detect"
video_output = sys.argv[2] if len(sys.argv) > 2 else "processed_video_gradio"

base_video_path = video_output
image_width = 360
image_height = 640
base_path = x_detect_project
track_data_path = os.path.join(get_latest_folder(base_path), "labels")
track_base_path = get_latest_folder(base_path)
# result
initial_result_path = os.path.join(track_base_path)
os.makedirs(initial_result_path, exist_ok=True)
# video
video_path, video_name = find_video_files(track_base_path)
base_name, _ = os.path.splitext(video_name)
new_video_path =os.path.join(base_video_path, f"{base_name}.mp4")

def convert_results(
    detection_results,
    image_width,
    image_height,
    initial_result_path,
    frame,
    classify=True,
    model_e=None,
    device=None,
    transform=None,
):
    """
    Convert detection results to JSON, one folder per track ID.
    """
    detections = []
    for index, result in enumerate(detection_results):
        category = int(result[0])
        x_center = float(result[1]) * image_width
        y_center = float(result[2]) * image_height
        width = float(result[3]) * image_width
        height = float(result[4]) * image_height
        x1 = round(x_center - width / 2)
        y1 = round(y_center - height / 2)
        x2 = round(x_center + width / 2)
        y2 = round(y_center + height / 2)

        detection = {
            "class": "particle",
            "confidence": 0,
            "bbox": [x1, y1, x2, y2],
            "Center": [round(x_center), round(y_center)],
            "file_path": new_video_path,
        }

        detections.append(detection)

    return {str(frame): {"detections": detections}}

def main(classify=False):
    result_dict = {}
    for index, filename in enumerate(
        sorted(
            os.listdir(track_data_path),
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
        )
    ):
        detection_results = []

        file_path = os.path.join(track_data_path, filename)
        with open(file_path, "r") as f:
            detection_results = [line.strip().split() for line in f.readlines()]
        # print(f"path: {file_path}")
        last_number = int(filename.split("_")[-1].split(".")[0])
        frame_result = convert_results(
            detection_results,
            image_width,
            image_height,
            initial_result_path,
            last_number,
            classify,
        )
        result_dict.update(frame_result)

    output_dir_path = os.path.join(initial_result_path)

    output_file = os.path.join(output_dir_path, "detections.json")
    with open(output_file, "w") as f:
        json.dump(result_dict, f)

if __name__ == "__main__":
    main()
