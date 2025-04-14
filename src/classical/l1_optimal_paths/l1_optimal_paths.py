import cv2
import numpy as np
import os
from .L1optimal_lpp import stabilize
from .L1optimal import get_inter_frame_transforms, write_output

def l1_optimal_stabilization(input_path, output_filename, crop_ratio=0.8):
    """
    Perform L1 optimal stabilization on the input video.

    Args:
        input_path (str): Path to the input video file.
        output_filename (str): Name of the output stabilized video file.
        crop_ratio (float): Crop ratio to avoid black borders (default: 0.8).

    Returns:
        str: Path to the stabilized video file.
    """
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise IOError(f"Could not open video file: {input_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    current_dir = os.path.abspath(os.path.dirname(__file__))
    outputs_dir = os.path.join(current_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    output_path = os.path.join(outputs_dir, output_filename)

    # Initialize transform array with identity matrices
    F_transforms = np.zeros((num_frames, 3, 3), np.float32)
    F_transforms[:, :, :] = np.eye(3)

    # Read first frame
    ret, first_frame = cap.read()
    if not ret:
        raise IOError("Failed to read first frame.")

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Compute inter-frame transforms
    get_inter_frame_transforms(cap, F_transforms, prev_gray)

    # Compute stabilization transforms using L1 optimization
    B_transforms = stabilize(F_transforms, first_frame.shape, crop_ratio=crop_ratio)

    # Reset capture to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Apply transformations and write output video
    write_output(cap, out, B_transforms, (width, height), crop_ratio)

    cap.release()
    out.release()

    return output_path
