# video_stabilizer.py
import numpy as np
import cv2
import os
import time
from .frameCollect import read_video, read_frame
from .utils import (
    blockSearchBody, 
    extract_bitplanes, 
    gray_code_transform,
    hierarchical_gcbpm,
    gcbpm_search
)
from .bitplane_config import MATCHING

def calculate_moving_average(curve, radius):
    """Calculate the moving average of a curve using a given radius"""
    window_size = 2 * radius + 1
    kernel = np.ones(window_size) / window_size
    curve_padded = np.pad(curve, (radius, radius), 'edge')
    smoothed_curve = np.convolve(curve_padded, kernel, mode='same')
    return smoothed_curve[radius:-radius]

def smooth_trajectory(trajectory, radius=50):
    """Smooth the trajectory using moving average on each dimension"""
    smoothed_trajectory = np.copy(trajectory)
    for i in range(2):  # Only X/Y translation for bitplane matching
        smoothed_trajectory[:, i] = calculate_moving_average(trajectory[:, i], radius=radius)
    return smoothed_trajectory

def fix_border(frame, scale=1.04):
    h, w = frame.shape[:2]
    T = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
    frame = cv2.warpAffine(frame, T, (w, h))
    return frame

def bitplane_motion_estimation(prev_frame, curr_frame):
    # Convert to grayscale if necessary
    if prev_frame.ndim == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame
    
    if curr_frame.ndim == 3:
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    else:
        curr_gray = curr_frame
    
    # Preprocess for bitplane matching
    prev_bp = extract_bitplanes(gray_code_transform(prev_gray))
    curr_bp = extract_bitplanes(gray_code_transform(curr_gray))
    
    # Hierarchical motion estimation
    if MATCHING['hierarchical']['enabled']:
        motion_vector = hierarchical_gcbpm(prev_bp, curr_bp)
    else:
        motion_vectors = gcbpm_search(prev_bp, curr_bp, 
                                    MATCHING['block_size'], 
                                    MATCHING['search_radius'])
        motion_vector = np.median(motion_vectors, axis=(0,1))
    
    return motion_vector

def bitplane_matching(filepath, output_filename, smoothing_radius=50, scale=1.04):
    """Main stabilization function with bitplane matching"""

    current_dir = os.path.abspath(os.path.dirname(__file__))
    outputs_dir = os.path.join(current_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    video_name, video, frame_width, frame_height, fps, frame_count = read_video(filepath)
    
    # Video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # stabilized_path = f"{outputs_dir}/{video_name}_stabilized.mp4"
    stabilized_path = os.path.join(outputs_dir, output_filename)
    # if side_by_side:
    #     out = cv2.VideoWriter(stabilized_path, fourcc, fps, (frame_width*2, frame_height))
    # else:
    out = cv2.VideoWriter(stabilized_path, fourcc, fps, (frame_width, frame_height))
    
    # Initialize transformations
    transforms = np.zeros((frame_count-1, 3), np.float32)  # [dx, dy, da]
    trajectory = np.zeros((frame_count-1, 3), np.float32)
    
    # Read first frame
    prev_frame = read_frame(video)[1]
    
    for i in range(frame_count-1):
        ok, curr_frame = read_frame(video)
        if not ok: break
        
        # Bitplane motion estimation
        dx, dy = bitplane_motion_estimation(prev_frame, curr_frame)
        
        # Store transformation
        transforms[i] = [dx, dy, 0]
        trajectory[i] = np.cumsum(transforms[:i+1], axis=0)[-1]
        
        prev_frame = curr_frame.copy()
    
    # Smooth trajectory
    smoothed_trajectory = smooth_trajectory(trajectory, smoothing_radius)
    transforms_smooth = smoothed_trajectory - trajectory
    
    # Reset video capture
    video.release()
    video = cv2.VideoCapture(filepath)
    prev_frame = read_frame(video)[1]
    
    # Apply smoothed transformations
    for i in range(frame_count-1):
        ok, frame = read_frame(video)
        if not ok: break
        
        dx, dy, _ = transforms_smooth[i]
        
        # Build transformation matrix
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        frame_stabilized = cv2.warpAffine(frame, M, (frame_width, frame_height))
        
        # Fix border artifacts
        frame_stabilized = fix_border(frame_stabilized, scale)
        
        # Block matching analysis
        # if block_matching and (i % pframeSet == 0):
        #     iframe = frame_stabilized.copy()
        #     residual_metric, _ = blockSearchBody(iframe, frame_stabilized, 
        #                                        blockSize=MATCHING['block_size'])
        
        # Write output
        # if side_by_side:
        #     comparison = cv2.hconcat([frame, frame_stabilized])
        #     out.write(comparison)
        # else:
        out.write(frame_stabilized)
        
        # if show_result:
        #     cv2.imshow("Stabilized", frame_stabilized)
        #     if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    # Cleanup
    out.release()
    video.release()
    cv2.destroyAllWindows()

    # Verify the video was written correctly
    if os.path.exists(stabilized_path) and os.path.getsize(stabilized_path) > 0:
        print(f"Stabilized video saved to: {stabilized_path}")
    else:
        print("Error: Failed to save video or file is empty")
    return stabilized_path


# def bitplane_matching(input_path):
#     start_time = time.time()
#     stabilize_video(
#         input_path,
#         smoothing_radius=30,
#         scale=1.04,
#         side_by_side=True,
#         block_matching=True
#     )
#     print(f"Processing completed in {time.time()-start_time:.2f} seconds")
