import cv2
import os,shutil
from .batch import process_images_in_directory, rename_files_in_directory
import numpy as np

def delete_invalid_jpg_files(folder_path):
    """
    Delete all unreadable JPG files in a folder.
    :param folder_path: image folder path
    """
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".jpg"):
            file_path = os.path.join(folder_path, filename)
            try:
                img = cv2.imdecode(
                    np.fromfile(file=file_path, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                # img = cv2.imread(file_path)
                if img is None:
                    print(f"Deleting invalid JPG file: {file_path}")
                    os.remove(file_path)
            except Exception as e:
                print(f"Error reading file {file_path}. Deleting it. Error: {e}")
                os.remove(file_path)

def images_to_video(image_folder, output_video, frame_rate=25):
    images = [
        img
        for img in os.listdir(image_folder)
        if img.endswith((".png", ".jpg", ".jpeg"))
    ]
    images.sort()

    # if len(images) < 500:
    #     print(
    #         f"Error: Found only {len(images)} images in the range, need at least 500 images."
    #     )
    #     return

    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*"h264")
    video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        # print(f"Writing image {image_path} to video")
        img = cv2.imread(image_path)
        video.write(img)

    video.release()
    print(f"Video saved as {output_video}")

if __name__ == "__main__":
    input_directory = r"mot_particle\img1"
    output_directory = r"processed_y1_750"
    rename_files_in_directory(input_directory)
    delete_invalid_jpg_files(input_directory)
    if os.path.exists(output_directory):
        print(f"Deleting existing output directory: {output_directory}")
        shutil.rmtree(output_directory)
    process_images_in_directory(input_directory, output_directory)
    frame_rate = 1
    path_parts = os.path.normpath(input_directory).split(os.sep)
    last_two_parts = path_parts[-2:]
    output_name = "-".join(last_two_parts)

    output_video = os.path.join(input_directory, f"{output_name}_particle_video.mp4")

    images_to_video(output_directory, output_video, frame_rate)
