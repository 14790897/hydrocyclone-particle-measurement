# from sharp_ import process_image
import os

from PIL import Image, UnidentifiedImageError

from .contrast import process_image

def tiff_to_jpeg(directory, output_directory):
    """Convert all TIFF files in a directory to JPEG."""
    os.makedirs(output_directory, exist_ok=True)
    count = 0

    for filename in os.listdir(directory):
        if filename.lower().endswith(".tif") or filename.lower().endswith(".tiff"):
            tiff_path = os.path.join(directory, filename)
            jpeg_path = os.path.join(
                output_directory, os.path.splitext(filename)[0] + ".jpg"
            )

            try:
                with Image.open(tiff_path) as img:
                    rotated_img = img.convert("RGB").rotate(-90, expand=True)
                    rotated_img.save(jpeg_path, "JPEG")
                count += 1
                
                if count % 1000 == 0:
                    print(f"Converted {count} TIFF images")

            except UnidentifiedImageError:
                print(f"Skipping unrecognized TIFF file: {tiff_path}")
            except Exception as e:
                print(f"Processing {tiff_path} error: {e}")
    
    if count > 0:
        print(f"✓ TIFF conversion done, processed {count} images")
    else:
        print("No TIFF files found")

def process_images_in_directory(directory, output_directory, config=None):
    """Process all JPEG files in a directory."""
    os.makedirs(output_directory, exist_ok=True)
    count = 0
    for filename in os.listdir(directory):
        if filename.lower().endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            output_path = os.path.join(output_directory, f"x-{count + 1}.jpg")
            process_image(image_path, output_path, config)
            count += 1
            
            if count % 1000 == 0:
                print(f"Processed {count} images")
            #     break
    
    if count > 0:
        print(f"✓ Image processing done, processed {count} images")
    else:
        print("No JPG files found")

def rename_files_in_directory(directory):
    """Walk a directory and strip the Chinese 'camera' prefix from filenames."""
    for filename in os.listdir(directory):
        if "\u76f8\u673a" in filename:
            new_filename = filename.replace("\u76f8\u673a", "")
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            os.rename(old_file, new_file)
            print(f"Renamed {filename} to {new_filename}")

pipeline_config = {
    "gaussian_filter": {"kernel_size": (5, 5), "sigma_x": 1},
    "bilateral_filter": {"d": 9, "sigma_color": 75, "sigma_space": 75},
    "contrast_enhancement": {"clip_limit": 2.0, "tile_grid_size": (8, 8)},
    "thresholding": {"block_size": 11, "c_value": 2},
    "sharpening": {},
}
if __name__ == "__main__":
    input_directory = r"x-550\x1-550\Acq_A_001"
    jpeg_directory = "jpeg_x"
    processed_directory = "processed_x"

    rename_files_in_directory(input_directory)
    tiff_to_jpeg(input_directory, jpeg_directory)
    process_images_in_directory(
        jpeg_directory, processed_directory, pipeline_config
    )
