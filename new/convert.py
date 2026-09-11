import json
import os
import time

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import EfficientNet_B1_Weights, efficientnet_b1

from .process_utils import extract_frame, find_video_files, get_latest_folder

times = []

def convert_results(
    detection_results,
    image_width,
    image_height,
    initial_result_path,
    frame,
    new_video_path,
    classify=True,
    model_e=None,
    device=None,
    transform=None,
):
    """
    Convert detection results to JSON, one folder per track ID.
    """
    for index, result in enumerate(detection_results):
        category = int(result[0])
        x_center = float(result[1]) * image_width
        y_center = float(result[2]) * image_height
        width = float(result[3]) * image_width
        height = float(result[4]) * image_height
        try:
            id = int(result[5])
        except IndexError:
            id = 99999
            print("Detection result missing ID; please check")
        x1 = round(x_center - width / 2)
        y1 = round(y_center - height / 2)
        x2 = round(x_center + width / 2)
        y2 = round(y_center + height / 2)

        object_data = {
            "Frame": frame,
            "ID": id,
            "Category": category,
            "Box": [x1, y1, x2, y2],
            "Center": [round(x_center), round(y_center)],
            "WidthHeight": [round(width), round(height)],
        }

        output_dir_path = os.path.join(initial_result_path, str(id))
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)

        output_image_file_path = os.path.join(
            output_dir_path, "images", f"frame_{frame}.jpg"
        )
        os.makedirs(os.path.join(output_dir_path, "images"), exist_ok=True)
        output_image = extract_frame(
            new_video_path,
            str(int(frame) - 1),
            output_image_file_path,
        )
        classify_cat = None
        if classify:
            output_image_pil = Image.fromarray(
                cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            )

            cropped_image = output_image_pil.crop((x1, y1, x2, y2))
            transformed_image = transform(cropped_image).unsqueeze(0).to(device)
            start_time = time.time()
            outputs = model_e(transformed_image)
            end_time = time.time()
            elapsed_time = end_time - start_time
            times.append(elapsed_time)
            confidences = torch.softmax(outputs, dim=1)
            _, classify_cat = confidences.max(1)
            classify_cat = classify_cat.item()
            object_data["Category"] = classify_cat
            # print("classify_cat:", classify_cat)
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 0, 255), 1)

        text = f"id:{id} cat:{classify_cat}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            output_image,
            text,
            (x1, y1 - 10),
            font,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imwrite(output_image_file_path, output_image)
        output_file = os.path.join(output_dir_path, "initial_data.json")
        try:
            if os.path.exists(output_file):
                with open(output_file, "r") as f:
                    existing_data = json.load(f)
                existing_data.append(object_data)
            else:
                existing_data = [object_data]

            with open(output_file, "w") as f:
                json.dump(existing_data, f, indent=2)

        except json.JSONDecodeError:
            with open(output_file, "w") as f:
                json.dump([object_data], f, indent=2)
            print(f"Object {id} data saved to {output_file} (empty file initialised)")

def main_convert(classify=True, y_track_project=None, video_output=None):
    base_video_path = video_output if video_output else "processed_video_gradio"
    image_width = 768
    image_height = 1024
    base_path = y_track_project if y_track_project else "runs/track"
    track_data_path = os.path.join(get_latest_folder(base_path), "labels")
    track_base_path = get_latest_folder(base_path)
    # result
    initial_result_directory = "initial_result"
    initial_result_path = os.path.join(track_base_path, initial_result_directory)
    os.makedirs(initial_result_path, exist_ok=True)
    # video
    video_path, video_name = find_video_files(track_base_path)
    base_name, _ = os.path.splitext(video_name)
    new_video_path = os.path.join(base_video_path, f"{base_name}.mp4")
    if classify:
        model_e = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
        model_e.classifier[1] = nn.Linear(model_e.classifier[1].in_features, 2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_e = model_e.to(device)
        _weights_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights"
        )
        model_e.load_state_dict(
            torch.load(
                os.path.join(_weights_dir, "best_model_new_eff1.pth"),
                map_location=device,
            )
        )  # without_generate.pth  best_model (10).pth
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        model_e.eval()
    else:
        model_e = None
        device = None
        transform = None
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
        convert_results(
            detection_results,
            image_width,
            image_height,
            initial_result_path,
            last_number,
            new_video_path,
            classify,
            model_e,
            device,
            transform,
        )
        average_time = sum(times) / len(times)
        if index % 100 == 0:
            print(f"Mean classification time: {average_time:.6f} s")

if __name__ == "__main__":
    main_convert()
