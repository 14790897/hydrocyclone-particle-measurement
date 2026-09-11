"""
End-to-end particle pipeline runner (Gradio-free).

Runs the complete hydrocyclone particle measurement pipeline on a pair of
image-sequence directories (Y view + X view):

    1) image enhancement + frames -> video
    2) YOLOv8 detection + ByteTrack tracking  (Y view and X view)
    3) EfficientNet-B1 motion-state classification -> per-ID JSON
    4) kinematic model (revolution / rotation velocity, height)
    5) motion plots

This mirrors the folder-mode branch of app.py, but is driven from the command
line and does not require the Gradio UI.

Usage
-----
    python run_pipeline.py --y path/to/Y_images --x path/to/X_images

    # optional
    --name exp            run name (default: auto-derived, else "exp")
    --fps 8000            Y-camera frame rate
    --x-fps-ratio 0.5     X/Y frame-rate ratio
    --no-classify         skip EfficientNet-B1 classification
    --max-files 500       process only the first N frames per folder
    --skip-plots          skip the final plotting stage

All outputs are written under the project root (runs/, runs_x_me/,
processed_video_gradio/, plots/).
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

# Ensure the project root is importable / is the working directory.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from new.batch import process_images_in_directory, rename_files_in_directory
from new.x_batch import tiff_to_jpeg
from new.y_make_images_2_video import images_to_video
from new.convert import main_convert

FRAME_RATE = 25  # fps used when packing enhanced frames into a video


def remove_chinese(text: str) -> str:
    return re.sub(r"[^\x00-\x7F]", "", text)


def auto_name_from_path(path: str):
    """
    Derive "flow-seq" (e.g. "850-001") from a Y/X folder path.

    Looks for a flow prefix (450/550/650/750/850) in the PARENT directory name
    and a sequence number (S0001 / Acq_A_001) in the folder name itself.
    Returns None when no flow prefix is found.
    """
    if not path:
        return None
    norm = os.path.normpath(str(path))
    current_folder = os.path.basename(norm)
    parent_dir = os.path.basename(os.path.dirname(norm))

    m = re.search(r"(450|550|650|750|850)", parent_dir)
    if not m:
        return None
    base = m.group(1)

    seq = re.search(r"S(\d{4})", current_folder)
    if seq:
        seq_num = seq.group(1)
    else:
        seq = re.search(r"_(\d{3})$", current_folder)
        seq_num = seq.group(1) if seq else "001"

    name = f"{base}-{seq_num}"
    if re.search(r"(?:[yYxX]2|_2|-2)", parent_dir):
        name = f"{name}-2"
    return name


def _run(cmd, **kw):
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n", flush=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, **kw)


def build_videos(y_dir, x_dir, video_output):
    """Enhance Y/X frames and pack them into videos. Returns (y_video, x_video)."""
    print("\n" + "=" * 60)
    print("[1] image enhancement + frames -> video")
    print("=" * 60)

    # ---- Y view ----
    y_out = "processed_y1_gradio"
    shutil.rmtree(y_out, ignore_errors=True)
    process_images_in_directory(y_dir, y_out)
    parts = [remove_chinese(p) for p in os.path.normpath(y_dir).split(os.sep)][-2:]
    output_name = "-".join(parts)
    y_video = os.path.join(video_output, f"{output_name}_particle_video.mp4")
    images_to_video(y_out, y_video, FRAME_RATE)

    # ---- X view ----
    x_out = "processed_x1_gradio"
    jpeg_dir = "jpeg_x"
    shutil.rmtree(x_out, ignore_errors=True)
    shutil.rmtree(jpeg_dir, ignore_errors=True)
    rename_files_in_directory(x_dir)
    tiff_to_jpeg(x_dir, jpeg_dir)
    process_images_in_directory(jpeg_dir, x_out)
    x_video = os.path.join(video_output, f"x_{output_name}_particle_video.mp4")
    images_to_video(x_out, x_video, FRAME_RATE)

    print(f"\nY video: {y_video}\nX video: {x_video}")
    return y_video, x_video, output_name


def detect_and_track(y_video, x_video, name, project_y, project_x):
    """Stage [2]: YOLOv8 + ByteTrack for both views."""
    print("\n" + "=" * 60)
    print("[2] YOLOv8 detection + ByteTrack tracking")
    print("=" * 60)
    _run([
        sys.executable, "tracking/track.py",
        "--yolo-model", "weights/yolov8-particle-best.pt",   # Y-view detector
        "--source", y_video,
        "--save", "--save-txt",
        "--tracking-method", "bytetrack",
        "--conf", "0.01", "--iou", "0.01",
        "--project", project_y, "--name", name,
    ])
    _run([
        sys.executable, "tracking/track.py",
        "--yolo-model", "weights/yolov8_best.pt",            # X-view detector
        "--source", x_video,
        "--save", "--save-txt",
        "--conf", "0.01", "--iou", "0.01",
        "--project", project_x, "--name", name,
    ])


def post_process(classify, project_y, project_x, video_output, fps, x_ratio):
    """Stages [3]-[4]: classification + kinematic model."""
    print("\n" + "=" * 60)
    print("[3] EfficientNet-B1 classification -> per-ID JSON")
    print("=" * 60)
    main_convert(classify, y_track_project=project_y, video_output=video_output)

    print("\n" + "=" * 60)
    print("[4] kinematic model (revolution / rotation velocity)")
    print("=" * 60)
    _run([sys.executable, "new/detect_convert.py", project_x, video_output])
    _run([sys.executable, "new/1_extract.py", project_y])
    _run([sys.executable, "new/2_images_x.py", project_y, project_x, str(x_ratio)])
    _run([sys.executable, "new/3_end.py", project_y, str(fps)])


def make_plots(project_y):
    print("\n" + "=" * 60)
    print("[5] motion plots")
    print("=" * 60)
    try:
        _run([sys.executable, "new/plot_particle.py", "--save", project_y])
    except subprocess.CalledProcessError:
        # Plots need at least one valid trajectory; with too little data (or an
        # empty run) there is nothing to draw. Treat as non-fatal.
        print("WARNING: plotting skipped -- no valid trajectories in this run.")


def main():
    ap = argparse.ArgumentParser(description="Run the full particle pipeline.")
    ap.add_argument("--y", required=True, help="Y-view image-sequence directory")
    ap.add_argument("--x", required=True, help="X-view image-sequence directory")
    ap.add_argument("--name", default=None, help="run name (default: auto / 'exp')")
    ap.add_argument("--project-y", default="runs/track")
    ap.add_argument("--project-x", default="runs_x_me/detect")
    ap.add_argument("--video-output", default="processed_video_gradio")
    ap.add_argument("--fps", type=float, default=8000, help="Y-camera frame rate")
    ap.add_argument("--x-fps-ratio", type=float, default=0.5)
    ap.add_argument("--no-classify", action="store_true")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.video_output, exist_ok=True)

    name = args.name or auto_name_from_path(args.y) or "exp"

    print(f"Y dir : {args.y}")
    print(f"X dir : {args.x}")
    print(f"name  : {name}")

    y_video, x_video, _ = build_videos(args.y, args.x, args.video_output)
    detect_and_track(y_video, x_video, name, args.project_y, args.project_x)
    post_process(
        not args.no_classify,
        args.project_y, args.project_x, args.video_output,
        args.fps, args.x_fps_ratio,
    )
    if not args.skip_plots:
        make_plots(args.project_y)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Results: {args.project_y}/{name}/initial_result/calculation_results.json")


if __name__ == "__main__":
    main()
