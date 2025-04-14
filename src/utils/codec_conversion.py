import subprocess
import shutil
import os

def ensure_h264_compliance(input_path: str) -> str:
    """
    Checks if the input video is using H.264. If not, re-encodes it.
    Returns the path to a compliant file (could be same as input if already compliant).
    """
    # Use ffprobe to check the video codec
    ffprobe_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_path
    ]
    
    try:
        codec = subprocess.check_output(ffprobe_cmd).decode().strip()
    except subprocess.CalledProcessError:
        raise RuntimeError("Could not determine codec. Is ffmpeg installed?")

    if codec.lower() == "h264":
        return input_path  # Already compliant

    # If not H.264, convert
    output_path = input_path.replace(".mp4", "_converted.mp4")
    
    # Check if the output file already exists and delete it
    if os.path.exists(output_path):
        os.remove(output_path)

    ffmpeg_cmd = [
        "ffmpeg",
        "-i", input_path,
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "fast",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        output_path
    ]

    subprocess.run(ffmpeg_cmd, check=True)
    return output_path