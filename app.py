import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import gradio as gr

from new.batch import process_images_in_directory, rename_files_in_directory
from new.convert import main_convert
from new.process_utils import clear_folder, get_latest_folder
from new.x_batch import tiff_to_jpeg
from new.y_make_images_2_video import delete_invalid_jpg_files, images_to_video

# Import delete_video_files function from 5_delete_videos.py
_delete_videos_spec = importlib.util.spec_from_file_location(
    "delete_videos_module",
    os.path.join(os.path.dirname(__file__), "new", "4_delete_videos.py")
)
if _delete_videos_spec and _delete_videos_spec.loader:
    _delete_videos_module = importlib.util.module_from_spec(_delete_videos_spec)
    _delete_videos_spec.loader.exec_module(_delete_videos_module)
    delete_video_files = _delete_videos_module.delete_video_files
else:
    raise ImportError("Failed to load 4_delete_videos.py module")

# Configuration file path
CONFIG_FILE = "config.json"

# Default configuration
DEFAULT_CONFIG = {
    "yolo_save_directories": {
        "y_track_project": "runs/track",
        "y_track_name": "exp",
        "x_detect_project": "runs_x_me/detect",
        "x_detect_name": "exp",
        "video_output": "processed_video_gradio"
    },
    "processing_options": {
        "max_files_per_folder": None,  # None means process all files
        "y_camera_fps": 8000,
        "x_camera_fps_ratio": 0.5
    },
    "batch_directories": [
        # Example format:
        # {"y": "path/to/y/dir", "x": "path/to/x/dir"}
    ]
}

def load_config():
    """Load configuration from config.json or create default"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
                # Ensure all required keys exist
                if "yolo_save_directories" not in config:
                    config = DEFAULT_CONFIG.copy()
                else:
                    # Fill in any missing keys with defaults
                    for key, value in DEFAULT_CONFIG["yolo_save_directories"].items():
                        if key not in config["yolo_save_directories"]:
                            config["yolo_save_directories"][key] = value
                    
                    # Ensure processing_options exist
                    if "processing_options" not in config:
                        config["processing_options"] = DEFAULT_CONFIG["processing_options"].copy()
                    else:
                        for key, value in DEFAULT_CONFIG["processing_options"].items():
                            if key not in config["processing_options"]:
                                config["processing_options"][key] = value
                    
                    # Ensure batch_directories exist
                    if "batch_directories" not in config:
                        config["batch_directories"] = DEFAULT_CONFIG["batch_directories"].copy()
                        
                return config
        except Exception as e:
            print(f"Error loading config: {e}, using default configuration")
            return DEFAULT_CONFIG
    else:
        # Create default config file
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG

def save_config(config):
    """Save configuration to config.json"""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

# Load initial configuration
config = load_config()
config["yolo_save_directories"]["y_track_name"] = DEFAULT_CONFIG["yolo_save_directories"]["y_track_name"]
config["yolo_save_directories"]["x_detect_name"] = DEFAULT_CONFIG["yolo_save_directories"]["x_detect_name"]
config["yolo_save_directories"]["video_output"] = DEFAULT_CONFIG["yolo_save_directories"]["video_output"]

base_path = config["yolo_save_directories"]["y_track_project"]
base_x_path = config["yolo_save_directories"]["x_detect_project"]
base_video_path = config["yolo_save_directories"]["video_output"]
plots_dir_path = "plots-eff1-new-both"
excel_output_path = os.path.join(plots_dir_path, "particle_both_motion_data.xlsx")

def update_config_values(
    y_track_proj,
    x_detect_proj,
    y_camera_fps,
    x_camera_fps_ratio,
    batch_directories_text,
):
    """Update configuration values"""
    global config, base_path, base_x_path, base_video_path
    try:
        if batch_directories_text:
            batch_directories = json.loads(batch_directories_text)
            if not isinstance(batch_directories, list):
                return "batch_directories must be a JSON array."
        else:
            batch_directories = []
    except json.JSONDecodeError as e:
        return f"Failed to parse batch_directories JSON: {e}"
    
    config["yolo_save_directories"]["y_track_project"] = y_track_proj
    config["yolo_save_directories"]["x_detect_project"] = x_detect_proj
    config["yolo_save_directories"]["y_track_name"] = DEFAULT_CONFIG["yolo_save_directories"]["y_track_name"]
    config["yolo_save_directories"]["x_detect_name"] = DEFAULT_CONFIG["yolo_save_directories"]["x_detect_name"]
    config["yolo_save_directories"]["video_output"] = DEFAULT_CONFIG["yolo_save_directories"]["video_output"]
    if y_camera_fps is None:
        y_camera_fps = DEFAULT_CONFIG["processing_options"]["y_camera_fps"]
    config["processing_options"]["y_camera_fps"] = y_camera_fps
    if x_camera_fps_ratio is None:
        x_camera_fps_ratio = DEFAULT_CONFIG["processing_options"]["x_camera_fps_ratio"]
    config["processing_options"]["x_camera_fps_ratio"] = x_camera_fps_ratio
    config["batch_directories"] = batch_directories
    
    if save_config(config):
        # Update global variables
        base_path = y_track_proj
        base_x_path = x_detect_proj
        base_video_path = config["yolo_save_directories"]["video_output"]
        return "Configuration saved successfully!"
    else:
        return "Error saving configuration!"

def reset_config():
    """Reset configuration to default values"""
    global config, base_path, base_x_path, base_video_path
    
    config = DEFAULT_CONFIG.copy()
    if save_config(config):
        base_path = config["yolo_save_directories"]["y_track_project"]
        base_x_path = config["yolo_save_directories"]["x_detect_project"]
        base_video_path = config["yolo_save_directories"]["video_output"]
        return (
            base_path,
            base_x_path,
            config["processing_options"]["y_camera_fps"],
            config["processing_options"]["x_camera_fps_ratio"],
            json.dumps(config["batch_directories"], ensure_ascii=False, indent=2),
            "Configuration reset to default successfully!"
        )
    else:
        return (
            base_path,
            base_x_path,
            config["processing_options"]["y_camera_fps"],
            config["processing_options"]["x_camera_fps_ratio"],
            json.dumps(config["batch_directories"], ensure_ascii=False, indent=2),
            "Error resetting configuration!"
        )

def get_batch_directories_from_config():
    """
    Get directory pairs from config.json batch_directories field.
    
    Returns:
        List of tuples: [(y_dir, x_dir), ...]
    """
    directory_pairs = []
    
    batch_dirs = config.get("batch_directories", [])
    
    for idx, entry in enumerate(batch_dirs, 1):
        if not isinstance(entry, dict):
            print(f"Warning: Entry {idx} is not a dictionary, skipping")
            continue
            
        y_dir = entry.get("y")
        x_dir = entry.get("x")
        
        if not y_dir or not x_dir:
            print(f"Warning: Entry {idx} missing 'y' or 'x' field, skipping")
            continue
        
        # Validate directories exist
        if os.path.exists(y_dir) and os.path.exists(x_dir):
            directory_pairs.append((y_dir, x_dir))
            print(f"Added directory pair: Y={y_dir}, X={x_dir}")
        else:
            if not os.path.exists(y_dir):
                print(f"Warning: Y directory does not exist: {y_dir}")
            if not os.path.exists(x_dir):
                print(f"Warning: X directory does not exist: {x_dir}")
    
    print(f"Loaded {len(directory_pairs)} directory pairs from config")
    return directory_pairs

def remove_chinese(text):
    return re.sub(r"[^\x00-\x7F]", "", text)

def auto_name_from_path(path):
    """
    Determine track/detect name automatically from a directory or file path.
    Extract the trailing sequence number (x and y separately).
    Rules:
      - Look for one of the flow prefixes: 450, 550, 650, 750, 850 in the parent directory name.
      - Extract the sequence number from the folder name (e.g., S0001 -> 001, Acq_A_002 -> 002).
      - If the parent directory name indicates a "second" set (contains y2, Y2, X2, x2, _2, -2),
        append "-2" suffix (e.g. "850-002-2").
      - Return format: "flow-seq" or "flow-seq-2" (e.g., "850-002" or "850-002-2").
      - Return None if no flow prefix is found.
    """
    if not path:
        return None
    try:
        # Get the parent directory name and the current folder name
        normalized_path = os.path.normpath(str(path))
        current_folder = os.path.basename(normalized_path)
        parent_dir = os.path.basename(os.path.dirname(normalized_path))
    except Exception:
        return None

    # Search for flow rate in parent directory name
    m = re.search(r"(450|550|650|750|850)", parent_dir)
    if not m:
        return None
    base = m.group(1)

    # Extract sequence number from current folder name
    # Pattern 1: S0001, S0002, etc. (Y-axis format)
    seq_match = re.search(r"S(\d{4})", current_folder)
    if seq_match:
        seq_num = seq_match.group(1)  # e.g., "0001"
    else:
        # Pattern 2: Acq_A_001, Acq_A_002, etc. (X-axis format)
        seq_match = re.search(r"_(\d{3})$", current_folder)
        if seq_match:
            seq_num = seq_match.group(1)  # e.g., "001"
        else:
            seq_num = "001"  # Default if no pattern matches

    # Build the name: flow-seq or flow-seq-2
    name = f"{base}-{seq_num}"
    
    # detect second dataset marker: Y2, y2, X2, x2, _2, -2 in parent directory name
    if re.search(r"(?:[yYxX]2|_2|-2)", parent_dir):
        name = f"{name}-2"
    
    return name

def process_with_subcommand(
    input_type,
    y_uploaded_video_path=None,
    x_uploaded_video_path=None,
    y_folder_path=None,
    x_folder_path=None,
    classify_checkbox=True,
    batch_file_path=None,
    max_files_per_folder=None,
):
    results = []
    
    # Convert max_files_per_folder: if 0 or None, set to None (process all)
    if max_files_per_folder is not None and max_files_per_folder <= 0:
        max_files_per_folder = None
    
    print(f"Max files per folder: {max_files_per_folder if max_files_per_folder else 'All files'}")
    
    # Check if we should load from config batch_directories
    if input_type == "batch file":
        directory_pairs = get_batch_directories_from_config()
        if not directory_pairs:
            return None, "No valid directory pairs found in config.json batch_directories.", None
        y_folder_list = [pair[0] for pair in directory_pairs]
        x_folder_list = [pair[1] for pair in directory_pairs]
    else:
        y_folder_list = (
            [p.strip() for p in y_folder_path.split(",") if p.strip()]
            if y_folder_path
            else []
        )
        x_folder_list = (
            [p.strip() for p in x_folder_path.split(",") if p.strip()]
            if x_folder_path
            else []
        )
    print(f"y_folder_list: {y_folder_list}")
    total_pairs = len(y_folder_list)
    print(f"\n{'='*60}")
    print(f"Starting batch processing: {total_pairs} directory pairs")
    print(f"{'='*60}\n")
    
    for idx, (y_video, x_video) in enumerate(zip(y_folder_list, x_folder_list), 1):
        print(f"\n{'='*60}")
        print(f"Processing pair {idx}/{total_pairs}")
        print(f"Y: {y_video}")
        print(f"X: {x_video}")
        print(f"{'='*60}\n")
        
        y_input_video_path = None
        x_input_video_path = None
        y_input_directory = None
        x_input_directory = None
        
        if input_type == "upload video":
            # guard against None to satisfy static checks
            y_input_video_path = str(Path(y_uploaded_video_path)) if y_uploaded_video_path else None
            x_input_video_path = str(Path(x_uploaded_video_path)) if x_uploaded_video_path else None

        elif input_type == "upload folder" or input_type == "batch file":
            y_input_directory = str(Path(y_video))
            y_output_directory = r"processed_y1_gradio"
            x_input_directory = str(Path(x_video))
            x_output_directory = r"processed_x1_gradio"
            rename_files_in_directory(y_input_directory)
            delete_invalid_jpg_files(y_input_directory)
            if os.path.exists(y_output_directory):
                print(f"Deleting existing output directory: {y_output_directory}")
                shutil.rmtree(y_output_directory)
            process_images_in_directory(y_input_directory, y_output_directory, max_files=max_files_per_folder)
            frame_rate = 25
            path_parts = os.path.normpath(y_input_directory).split(os.sep)
            processed_parts = [remove_chinese(part) for part in path_parts]

            last_two_parts = processed_parts[-2:]
            output_name = "-".join(last_two_parts)
            output_video = os.path.join(
                base_video_path, f"{output_name}_particle_video.mp4"
            )
            images_to_video(y_output_directory, output_video, frame_rate)
            y_input_video_path = output_video

            jpeg_directory = "jpeg_x"
            if os.path.exists(x_output_directory):
                print(f"Deleting existing output directory: {x_output_directory}")
                shutil.rmtree(x_output_directory)
            if os.path.exists(jpeg_directory):
                print(f"Deleting existing output directory: {jpeg_directory}")
                shutil.rmtree(jpeg_directory)
            rename_files_in_directory(x_input_directory)
            tiff_to_jpeg(x_input_directory, jpeg_directory)
            process_images_in_directory(jpeg_directory, x_output_directory, max_files=max_files_per_folder)
            x_output_video = os.path.join(
                base_video_path, f"x_{output_name}_particle_video.mp4"
            )
            images_to_video(x_output_directory, x_output_video, frame_rate)
            x_input_video_path = x_output_video

        # Try to automatically determine the --name values from the folder/file names
        # Prefer explicit processed/input directories if available, fall back to the original list entry
        y_candidate = None
        x_candidate = None
        # if upload video, prefer the uploaded path
        if input_type == "upload video":
            y_candidate = y_uploaded_video_path or y_input_video_path or y_video
            x_candidate = x_uploaded_video_path or x_input_video_path or x_video
        else:
            # for upload folder or batch file modes, use the input directory
            y_candidate = y_input_directory or y_input_video_path or y_video
            x_candidate = x_input_directory or x_input_video_path or x_video

        y_auto_name = auto_name_from_path(y_candidate)
        x_auto_name = auto_name_from_path(x_candidate)
        
        print(f"Auto-detected names: Y={y_auto_name}, X={x_auto_name} (from Y={y_candidate}, X={x_candidate})")

        y_name_for_cmd = y_auto_name if y_auto_name else config["yolo_save_directories"]["y_track_name"]
        x_name_for_cmd = x_auto_name if x_auto_name else config["yolo_save_directories"]["x_detect_name"]

        y_command = [
            sys.executable,
            "tracking/track.py",
            "--yolo-model",
            "weights/yolov8-particle-best.pt",
            "--source",
            y_input_video_path,
            "--save",
            "--save-txt",
            "--tracking-method",
            "bytetrack",
            "--conf",
            "0.01",
            "--iou",
            "0.01",
            "--project",
            config["yolo_save_directories"]["y_track_project"],
            "--name",
            y_name_for_cmd,
        ]
        x_command = [
            sys.executable,
            "tracking/track.py",
            "--yolo-model",
            "weights/yolov8_best.pt",
            "--source",
            x_input_video_path,
            "--save",
            "--save-txt",
            "--conf",
            "0.01",
            "--iou",
            "0.01",
            "--project",
            config["yolo_save_directories"]["x_detect_project"],
            "--name",
            x_name_for_cmd,
        ]
        try:
            project_root = os.path.dirname(os.path.abspath(__file__))
            subprocess.run(y_command, check=True, cwd=project_root)
            subprocess.run(x_command, check=True, cwd=project_root)
        except subprocess.CalledProcessError as e:
            error_msg = f"Error during processing: {e}"
            print(error_msg)
            return None, error_msg, None

        # Safely determine a base name for the output using available candidates
        name_src = None
        if y_input_video_path:
            name_src = y_input_video_path
        elif y_input_directory:
            # If we processed a directory, use that directory name
            name_src = y_input_directory
        else:
            name_src = y_video

        if name_src:
            base_name = os.path.basename(os.path.splitext(str(name_src))[0])
        else:
            base_name = "output"

        output_path = os.path.join(get_latest_folder(base_path), base_name + ".avi")
        # output_path = convert_to_mp4(output_path)
        txt_result, log_output = post_process(classify_checkbox)
        results.append((output_path, txt_result, log_output))
        
        print(f"\n{'='*60}")
        print(f"✓ Completed pair {idx}/{total_pairs}")
        print(f"{'='*60}\n")
        
    print(f"\n{'='*60}")
    print(f"Batch processing complete! Processed {total_pairs} directory pairs")
    print(f"{'='*60}\n")
    
    print("\nCleaning up temporary files...")
    temp_dirs = [
        "processed_y1_gradio",
        "processed_x1_gradio", 
        "jpeg_x"
    ]
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"✓ Removed temp directory: {temp_dir}")
            except Exception as e:
                print(f"✗ Failed to remove temp directory {temp_dir}: {e}")
    
    print(f"results: {results}")
    if len(results) == 1:
        return results[0][0], results[0][1], results[0][2]
    return None, results, None

def generate_both_plots():
    """Generate and return both rotation and revolution plot images"""
    y_track_proj = config["yolo_save_directories"]["y_track_project"]
    
    try:
        print("Generating particle rotation/revolution analysis plots...")
        plot_script = [
            sys.executable,
            "new/plot_particle_both.py",
            "--save",
            "--rearrange",
            y_track_proj,
        ]
        result = subprocess.run(
            plot_script, check=True, capture_output=True, text=True
        )
        print(f"Plot generation finished:\n{result.stdout}")
        
        output_dir = "plots-eff1-new-both"
        combined_plot = os.path.join(output_dir, "particle_analysis_combined.png")
        summary_plot = os.path.join(output_dir, "particle_analysis_summary.png")
        
        if os.path.exists(combined_plot):
            if os.path.exists(summary_plot):
                return combined_plot, summary_plot, "Rotation and revolution plots generated successfully!"
            else:
                return combined_plot, None, "Rotation and revolution plots generated successfully!"
        else:
            return None, None, "Plot files not found; ensure data has been processed."
            
    except subprocess.CalledProcessError as e:
        error_msg = f"Plot generation failed:\n{e.stderr}"
        print(error_msg)
        return None, None, error_msg
    except Exception as e:
        error_msg = f"Error while generating plots: {e}"
        print(error_msg)
        return None, None, error_msg

def post_process(classify):
    log_output = ""
    y_track_proj = config["yolo_save_directories"]["y_track_project"]
    x_detect_proj = config["yolo_save_directories"]["x_detect_project"]
    video_out = config["yolo_save_directories"]["video_output"]

    y_camera_fps = config["processing_options"]["y_camera_fps"]
    x_camera_fps_ratio = config["processing_options"]["x_camera_fps_ratio"]
    scripts = [
        [
            sys.executable,
            "new/detect_convert.py",
            x_detect_proj,
            video_out,
        ],
        [
            sys.executable,
            "new/1_extract.py",
            y_track_proj,
        ],
        [
            sys.executable,
            "new/2_images_x.py",
            y_track_proj,
            x_detect_proj,
            str(x_camera_fps_ratio),
        ],
        [
            sys.executable,
            "new/3_end.py",
            y_track_proj,
            str(y_camera_fps),
        ],
    ]
    print("Running script: main_convert")
    main_convert(
        classify,
        y_track_project=y_track_proj,
        video_output=video_out
    )
    for script in scripts:
        try:
            print(f"Running script: {script[1]}...")
            if script[1] == "new/1_extract.py":
                result = subprocess.run(
                    script, check=True, capture_output=True, text=True
                )
                log_output = f"{result.stdout}"
            else:
                result = subprocess.run(
                    script, check=True, capture_output=False, text=True
                )
            print(f"Script {script[1]} finished, output:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Script {script[1]} failed, error:\n{e.stderr}")
            break

    try:
        print("Generating particle motion analysis plots...")
        plot_script = [sys.executable, "new/plot_particle.py", "--save", y_track_proj]
        result = subprocess.run(
            plot_script, check=True, capture_output=True, text=True
        )
        print(f"Plot generation finished:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Plot generation failed:\n{e.stderr}")
    except Exception as e:
        print(f"Error while generating plots: {e}")

    try:
        print("\nAuto-deleting video files to free disk space...")

        # delete_result_y = delete_video_files(y_track_proj, dry_run=False)
        # print(delete_result_y)

        print(f"\n2. Cleaning video output directory: {video_out}")
        delete_result_video = delete_video_files(video_out, dry_run=False)
        print(delete_result_video)

    except Exception as e:
        print(f"Video deletion failed: {e}")

    latest_folder_path = get_latest_folder(base_path)
    initial_result_directory = os.path.join(latest_folder_path, "initial_result")
    calculation_results_path = os.path.join(
        initial_result_directory, "calculation_results.json"
    )
    # stats_file_path = os.path.join(initial_result_directory, "all_stats.json")
    if os.path.exists(calculation_results_path):
        try:
            with open(calculation_results_path, "r") as stats_file:
                content = json.load(stats_file)
                return json.dumps(content, ensure_ascii=False, indent=4), log_output
        except json.JSONDecodeError:
            return "JSON decoding error. The file content might be invalid."
    else:
        return (
            f"File {calculation_results_path} does not exist.",
            f"File {calculation_results_path} does not exist.",
        )

with gr.Blocks() as demo:
    gr.Markdown("# Particle Process Interface")
    os.makedirs(base_video_path, exist_ok=True)

    # Configuration section
    with gr.Accordion("YOLO Save Directory Configuration", open=False):
        gr.Markdown("""
        Configure the save directories for YOLO tracking and detection results.
        All paths are relative to the project root directory.
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Y-axis Tracking Configuration")
                y_track_project_input = gr.Textbox(
                    label="Y-axis Tracking Project Directory",
                    value=config["yolo_save_directories"]["y_track_project"],
                    placeholder="e.g., runs/track"
                )

            with gr.Column():
                gr.Markdown("### X-axis Detection Configuration")
                x_detect_project_input = gr.Textbox(
                    label="X-axis Detection Project Directory",
                    value=config["yolo_save_directories"]["x_detect_project"],
                    placeholder="e.g., runs_x_me/detect"
                )
        y_camera_fps_input = gr.Number(
            label="Y Camera FPS",
            value=config["processing_options"]["y_camera_fps"],
            precision=0,
            minimum=1,
        )
        x_camera_fps_ratio_input = gr.Number(
            label="X Camera FPS Ratio (to Y)",
            value=config["processing_options"]["x_camera_fps_ratio"],
            precision=3,
            minimum=0.01,
        )
        batch_directories_input = gr.Textbox(
            label="Batch Directories (JSON list)",
            value=json.dumps(config["batch_directories"], ensure_ascii=False, indent=2),
            lines=6,
        )

        with gr.Row():
            save_config_button = gr.Button("Save Configuration", variant="primary")
            reset_config_button = gr.Button("Reset to Default")

        config_status = gr.Textbox(label="Configuration Status", interactive=False)

        # Configuration button actions
        save_config_button.click(
            fn=update_config_values,
            inputs=[
                y_track_project_input,
                x_detect_project_input,
                y_camera_fps_input,
                x_camera_fps_ratio_input,
                batch_directories_input,
            ],
            outputs=config_status
        )

        reset_config_button.click(
            fn=reset_config,
            inputs=[],
            outputs=[
                y_track_project_input,
                x_detect_project_input,
                y_camera_fps_input,
                x_camera_fps_ratio_input,
                batch_directories_input,
                config_status
            ]
        )

    gr.Markdown("---")  # Separator
    gr.Markdown("## Processing Interface")

    def toggle_inputs(input_type):
        if input_type == "upload video":
            return (
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
            )
        elif input_type == "upload folder":
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
            )
        elif input_type == "batch file":
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )

    with gr.Row():
        with gr.Column(scale=1):
            input_type = gr.Radio(
                [ "upload folder", "batch file"], 
                label="select input type",
                value="batch file"
            )

            y_uploaded_video = gr.Video(label="y upload video", visible=False)
            x_uploaded_video = gr.Video(label="x upload video", visible=False)
            y_folder_input = gr.Textbox(label="y_folder_input", visible=False)
            x_folder_input = gr.Textbox(label="x_folder_input", visible=False)
            y_folder_input.change(
                fn=lambda y: y,
                inputs=y_folder_input,
                outputs=x_folder_input,
            )
            input_type.change(
                fn=toggle_inputs,
                inputs=[input_type],
                outputs=[
                    y_uploaded_video,
                    x_uploaded_video,
                    y_folder_input,
                    x_folder_input,
                ],
            )
        with gr.Column(scale=1):
            video_output = gr.Video(label="processed video")

    with gr.Row():
        with gr.Column():
            classify_checkbox = gr.Checkbox(label="classify", value=True)
        with gr.Column():
            max_files_input = gr.Number(
                label="Max Files Per Folder",
                value=config["processing_options"]["max_files_per_folder"],
                precision=0,
                minimum=1,
                info="Maximum number of files to process per folder (leave empty or 0 for all files)"
            )

    process_button = gr.Button("start process")
    # post_process_button = gr.Button("post process")
    text_output = gr.Textbox(label="result", type="text", lines=10)
    log_output = gr.Textbox(label="log", type="text", lines=10)

    gr.Markdown("---")  # Separator
    gr.Markdown("## Particle Analysis Plots (Rotation & Revolution)")

    with gr.Row():
        generate_both_plot_button = gr.Button("Generate and Display Plots", variant="primary")

    plot_status = gr.Textbox(label="Plot Status", interactive=False)

    both_combined_plot_output = gr.Image(label="Combined Analysis", type="filepath")
    # both_summary_plot_output = gr.Image(label="Summary Analysis", type="filepath")

    gr.Markdown("---")
    gr.Markdown("## Output Paths")
    excel_output_display = gr.Textbox(
        label="Excel Output",
        value=excel_output_path,
        interactive=False,
    )
    plots_dir_display = gr.Textbox(
        label="Plots Directory",
        value=plots_dir_path,
        interactive=False,
    )

    # clear folder
    gr.Markdown("---")  # Separator
    gr.Markdown("## Cleanup Operations")

    with gr.Row():
        clear_button = gr.Button("Clear Working Folders", variant="stop")
        delete_videos_button = gr.Button("Only Delete Videos", variant="stop")

    delete_videos_output = gr.Textbox(label="Video Deletion Status", lines=15, interactive=False)

    clear_button.click(
        fn=lambda: (
            clear_folder(base_path),
            clear_folder(base_x_path),
            clear_folder(base_video_path),
        ),
        inputs=[],
        outputs=[],
    )

    delete_videos_button.click(
        fn=lambda: delete_video_files(config["yolo_save_directories"]["y_track_project"], dry_run=False),
        inputs=[],
        outputs=delete_videos_output,
    )

    process_button.click(
        fn=process_with_subcommand,
        inputs=[
            input_type,
            y_uploaded_video,
            x_uploaded_video,
            y_folder_input,
            x_folder_input,
            classify_checkbox,
            gr.Textbox(visible=False),  # batch_file_path not used anymore
            max_files_input,
        ],
        outputs=[video_output, text_output, log_output],
    )

    generate_both_plot_button.click(
        fn=generate_both_plots,
        inputs=[],
        outputs=[both_combined_plot_output, plot_status],
    )
    # post_process_button.click(
    #     fn=post_process, inputs=classify_checkbox, outputs=text_output
    # )
    # y_uploaded_video = gr.File(
    # )
    # y_uploaded_video.change(
    #     fn=lambda x: x, inputs=y_uploaded_video, outputs=video_output
    # )

if __name__ == "__main__":
    demo.launch(debug=True)
