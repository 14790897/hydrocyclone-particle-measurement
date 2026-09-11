"""
Delete all video files under the runs/track directory.
Frees disk space; videos are no longer needed after processing.
"""
import os
import sys

def delete_video_files(base_path, dry_run=False):
    """
    Delete all _particle_video .mp4/.avi files under the given directory.
    
    Args:
        base_path: base path (e.g. runs/track)
        dry_run: If True, only list the files that would be deleted.
    
    Returns:
        str: Human-readable result of the deletion (for the Gradio UI).
    """
    base_path = os.path.normpath(base_path)
    
    if not os.path.exists(base_path):
        return f"✗ Error: path does not exist: {base_path}"
    
    video_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith("_particle_video.mp4") or file.endswith("_particle_video.avi"):
                video_files.append(os.path.join(root, file))
    
    if not video_files:
        return "✓ No video files found"
    
    result_lines = [f"Found {len(video_files)} video file(s):", "=" * 60]
    
    total_size = 0
    deleted_count = 0
    failed_count = 0
    
    for video_file in video_files:
        try:
            file_size = os.path.getsize(video_file)
            total_size += file_size
            size_mb = file_size / (1024 * 1024)
            
            rel_path = os.path.relpath(video_file, base_path)
            
            if dry_run:
                result_lines.append(f"[preview] {rel_path} ({size_mb:.2f} MB)")
            else:
                try:
                    os.remove(video_file)
                    result_lines.append(f"✓ Deleted: {rel_path} ({size_mb:.2f} MB)")
                    deleted_count += 1
                except Exception as e:
                    result_lines.append(f"✗ Deletion failed: {rel_path} - {e}")
                    failed_count += 1
        except Exception as e:
            result_lines.append(f"✗ Failed to read file info: {video_file} - {e}")
            failed_count += 1
    
    result_lines.append("=" * 60)
    total_size_mb = total_size / (1024 * 1024)
    total_size_gb = total_size / (1024 * 1024 * 1024)
    
    if dry_run:
        result_lines.append(f"\n[preview mode] will delete {len(video_files)} video file(s)")
        result_lines.append(f"Total size: {total_size_mb:.2f} MB ({total_size_gb:.2f} GB)")
        result_lines.append("\nClick 'Confirm video deletion' to actually delete")
    else:
        result_lines.append("\nDeletion complete!")
        result_lines.append(f"  succeeded: {deleted_count}")
        result_lines.append(f"  failed: {failed_count}")
        result_lines.append(f"  freed: {total_size_mb:.2f} MB ({total_size_gb:.2f} GB)")
    
    return "\n".join(result_lines)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python new/4_delete_videos.py <track_base_path> [--confirm]")
        print("Example: python new/4_delete_videos.py runs/track")
        print("      python new/4_delete_videos.py runs/track --confirm  # confirm deletion")
        sys.exit(1)
    
    track_base_path = sys.argv[1]
    confirm = "--confirm" in sys.argv
    
    if confirm:
        print("⚠ Warning: about to delete ALL video files!")
        result = delete_video_files(track_base_path, dry_run=False)
    else:
        print("Preview mode (no files are deleted)")
        result = delete_video_files(track_base_path, dry_run=True)
    
    print(result)
