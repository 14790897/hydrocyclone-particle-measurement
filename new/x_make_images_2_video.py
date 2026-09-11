import cv2
import os, shutil
from .x_batch import (
    process_images_in_directory,
    rename_files_in_directory,
    tiff_to_jpeg,
)

def images_to_video(image_folder, output_video, frame_rate=30):
    images = [
        img
        for img in os.listdir(image_folder)
        if img.endswith((".png", ".jpg", ".jpeg"))
    ]
    images.sort()

    if len(images) < 250:
        print(
            f"Error: Found only {len(images)} images in the range, need at least 500 images."
        )
        return

    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)
    height, width, layers = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*"h264")
    video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        img = cv2.imread(image_path)
        video.write(img)

    video.release()
    print(f"Video saved as {output_video}")

if __name__ == "__main__":
    input_directory = r"650-1\x1"
    output_directory = r"processed_x1_750"
    jpeg_directory = "jpeg_x"
    if os.path.exists(output_directory):
        print(f"Deleting existing output directory: {output_directory}")
        shutil.rmtree(output_directory)
    if os.path.exists(jpeg_directory):
        print(f"Deleting existing output directory: {jpeg_directory}")
        shutil.rmtree(jpeg_directory)
    rename_files_in_directory(input_directory)
    tiff_to_jpeg(input_directory, jpeg_directory)
    process_images_in_directory(jpeg_directory, output_directory)
    frame_rate = 20
    output_name = input_directory.replace("\\", "-").replace("/", "-")

    output_video = f"{input_directory}/{output_name}_particle_video.mp4"

    images_to_video(output_directory, output_video, frame_rate)
