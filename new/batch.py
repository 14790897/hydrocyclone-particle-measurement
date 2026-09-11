# from sharp_ import process_image
import os

from .contrast import process_image

def process_images_in_directory(directory, output_directory, config=None, max_files=None):
    """
    Process all JPEG files in a directory.
    
    Args:
        directory: input directory
        output_directory: output directory
        config: processing config
        max_files: max files to process; None means all
    """
    os.makedirs(output_directory, exist_ok=True)
    
    jpg_files = [f for f in os.listdir(directory) if f.lower().endswith(".jpg")]
    
    if max_files is not None and max_files > 0:
        jpg_files = jpg_files[:max_files]
        print(f"Processing {len(jpg_files)} files (limited to {max_files}) from {directory}")
    else:
        print(f"Processing all {len(jpg_files)} files from {directory}")
    
    total_files = len(jpg_files)
    for file_idx, filename in enumerate(jpg_files, 1):
        if file_idx % 1000 == 0 or file_idx == total_files:
            print(f"  Progress: {file_idx}/{total_files} files processed ({file_idx*100//total_files}%)")
        image_path = os.path.join(directory, filename)
        # print(f'{image_path} contrast enhancement...')
        output_path = os.path.join(output_directory, "processed_" + filename)
        process_image(image_path, output_path, config)

def rename_files_in_directory(directory):
    """Walk a directory and strip the Chinese 'camera' prefix from filenames."""
    for filename in os.listdir(directory):
        try:
            if "\u76f8\u673a" in filename:
                new_filename = filename.replace("\u76f8\u673a", "")
                old_file = os.path.join(directory, filename)
                new_file = os.path.join(directory, new_filename)

                os.rename(old_file, new_file)
                print(f"Renamed {filename} to {new_filename}")
        except FileNotFoundError:
            print(f"File not found: {filename}")
        except PermissionError:
            print(f"Permission error: cannot rename file {filename}")
        except Exception as e:
            print(f"Failed to rename file {filename}: {e}")

if __name__ == "__main__":
    pipeline_config = {
        "gaussian_filter": {"kernel_size": (5, 5), "sigma_x": 1},
        "bilateral_filter": {"d": 9, "sigma_color": 75, "sigma_space": 75},
        "contrast_enhancement": {"clip_limit": 2.0, "tile_grid_size": (8, 8)},
        "thresholding": {"block_size": 11, "c_value": 2},
        "sharpening": {},
    }

    input_directory = r"550-y\Y2-550\No.1_C001H001S0001"
    output_directory = "processed_y2_550"

    rename_files_in_directory(input_directory)
    process_images_in_directory(input_directory, output_directory, pipeline_config)
