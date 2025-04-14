import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from utils.codec_conversion import ensure_h264_compliance
from classical.optical_flow.utils import smooth_trajectory, fix_border
from .utils import track_motion_block_matching

def block_matching(input_path, output_filename, smoothing_radius=30, block_size=16, search_area=16, use_kalman=False):
    """
    Stabilize a video using block matching for motion estimation.
    """
    # Ensure input video is H.264 compliant
    input_path = ensure_h264_compliance(input_path)

    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Could not open video")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video Properties - Width: {width}, Height: {height}, FPS: {fps}, Frame Count: {num_frames}")

    # Define output path
    current_dir = os.path.abspath(os.path.dirname(__file__))
    outputs_dir = os.path.join(current_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    output_path = os.path.join(outputs_dir, output_filename)

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read first frame")
        return

    # Store all frames for later processing
    print("Reading all frames...")
    frames = [prev_frame]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    print(f"Processing {len(frames)} frames")

    # Initialize transformation arrays
    transforms = np.zeros((len(frames) - 1, 2), np.float32)  # dx, dy only

    # Calculate motion between consecutive frames
    print("Calculating frame-to-frame motion...")
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda i: track_motion_block_matching(frames[i], frames[i + 1], block_size, search_area), range(len(frames) - 1)))

    for i, (dx, dy) in enumerate(results):
        transforms[i] = [dx, dy]

        if i % 10 == 0:
            print(f"Processed motion for frame {i}/{len(frames) - 1}")

    # Compute trajectory using accumulated motion
    trajectory = np.cumsum(transforms, axis=0)

    # Smooth trajectory with moving average or Kalman filter
    print("Smoothing trajectory...")
    smoothed_trajectory = smooth_trajectory(np.column_stack([trajectory, np.zeros(len(trajectory))]), smoothing_radius=smoothing_radius, use_kalman=use_kalman)[:, :2]

    # Calculate difference between smoothed and original trajectory
    difference = smoothed_trajectory - trajectory

    # Calculate smoother transforms
    transforms_smooth = transforms + difference

    # Apply smoothed transforms to frames
    print("Applying stabilization transforms...")
    for i in range(len(frames) - 1):
        # Get current frame
        frame = frames[i + 1]

        # Get smoothed transform
        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]

        # Create transformation matrix (translation only)
        m = np.float32([[1, 0, dx], [0, 1, dy]])

        # Apply transformation
        frame_stabilized = cv2.warpAffine(frame, m, (width, height))
        frame_stabilized = fix_border(frame_stabilized)

        # Write frame to output video
        out.write(frame_stabilized)

        # Show progress
        if i % 10 == 0:
            print(f"Stabilized frame {i}/{len(frames) - 1}")

    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video stabilization complete. Output saved to {output_path}")
    return output_path

