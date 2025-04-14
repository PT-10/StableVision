import cv2
import numpy as np

def track_motion_block_matching(prev_frame, curr_frame, block_size, search_area):
    """
    Optimized block matching with downscaling and coarser search.
    """
    # Convert to grayscale
    if len(prev_frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame
        curr_gray = curr_frame

    # Downscale for faster matching
    scale = 0.5
    prev_gray = cv2.resize(prev_gray, (0, 0), fx=scale, fy=scale)
    curr_gray = cv2.resize(curr_gray, (0, 0), fx=scale, fy=scale)

    h, w = prev_gray.shape
    grid_step = block_size * 3  # Coarser grid

    motion_vectors = []

    for y in range(block_size, h - block_size, grid_step):
        for x in range(block_size, w - block_size, grid_step):
            curr_block = curr_gray[y:y + block_size, x:x + block_size]

            y_start = max(0, y - search_area)
            y_end = min(h - block_size, y + search_area)
            x_start = max(0, x - search_area)
            x_end = min(w - block_size, x + search_area)

            best_x, best_y = x, y
            min_sad = float('inf')

            for sy in range(y_start, y_end + 1, 4):  # Coarser step
                for sx in range(x_start, x_end + 1, 4):
                    prev_block = prev_gray[sy:sy + block_size, sx:sx + block_size]
                    sad = np.sum(np.abs(curr_block - prev_block))
                    if sad < min_sad:
                        min_sad = sad
                        best_x, best_y = sx, sy

            conf_threshold = block_size * block_size * 10
            if min_sad < conf_threshold:
                dx = (best_x - x) / scale  # Rescale back
                dy = (best_y - y) / scale
                confidence = 1.0 / (min_sad + 1.0)
                motion_vectors.append((dx, dy, confidence))

    if not motion_vectors:
        return 0, 0

    motion_vectors = np.array(motion_vectors)
    weights = motion_vectors[:, 2]

    sorted_indices_x = np.argsort(motion_vectors[:, 0])
    cumsum_x = np.cumsum(weights[sorted_indices_x])
    median_x_idx = np.searchsorted(cumsum_x, cumsum_x[-1] * 0.5)
    dx = motion_vectors[sorted_indices_x[median_x_idx], 0]

    sorted_indices_y = np.argsort(motion_vectors[:, 1])
    cumsum_y = np.cumsum(weights[sorted_indices_y])
    median_y_idx = np.searchsorted(cumsum_y, cumsum_y[-1] * 0.5)
    dy = motion_vectors[sorted_indices_y[median_y_idx], 1]

    return dx, dy


def read_video(filename):
    """Returns video object with its properties."""
    video = cv2.VideoCapture(filename)
    if not video.isOpened():
        raise IOError("Could not open video")
    return video