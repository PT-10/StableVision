import cv2
import numpy as np
import os

from .horn_schunck import calcOpticalFlowHS
from .utils import smooth_trajectory, fix_border


def optical_flow(input_path: str, output_filename: str,
                 smoothing_radius: int = 30,
                 use_kalman: bool = False,
                 method: str = 'lk',  # 'lk', 'farneback', 'horn-schunck'
                 maxCorners=200,
                 qualityLevel=0.01,
                 minDistance=30,
                 blockSize=3) -> str:

    cap = cv2.VideoCapture(input_path)

    current_dir = os.path.abspath(os.path.dirname(__file__))
    outputs_dir = os.path.join(current_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    output_path = os.path.join(outputs_dir, output_filename)

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    _, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    transforms = np.zeros((num_frames - 1, 3), np.float32)

    for i in range(num_frames - 2):
        success, curr_frame = cap.read()
        if not success:
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        if method == 'lk':
            prev_points = cv2.goodFeaturesToTrack(prev_gray, maxCorners, qualityLevel, minDistance, blockSize)
            curr_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_points, None)
            idx = np.where(status == 1)[0]
            prev_points = prev_points[idx]
            curr_points = curr_points[idx]

            matrix, _ = cv2.estimateAffine2D(prev_points, curr_points)
            dx = matrix[0, 2]
            dy = matrix[1, 2]
            da = np.arctan2(matrix[1, 0], matrix[0, 0])

        elif method == 'farneback':
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            dx = np.mean(flow[..., 0])
            dy = np.mean(flow[..., 1])
            da = 0

        elif method == 'horn-schunck':
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
            flow = calcOpticalFlowHS(prev_gray, curr_gray, alpha=0.001, num_iter=100, criteria=criteria)
            dx = np.mean(flow[..., 0])
            dy = np.mean(flow[..., 1])
            da = 0

        else:
            raise ValueError(f"Unsupported method: {method}")

        transforms[i] = [dx, dy, da]
        prev_gray = curr_gray

    # Smooth trajectory
    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth_trajectory(trajectory, smoothing_radius, use_kalman)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)

    for i in range(num_frames - 2):
        success, frame = cap.read()
        if not success:
            break

        dx, dy, da = transforms_smooth[i]
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        stabilized_frame = cv2.warpAffine(frame, m, (width, height))
        stabilized_frame = fix_border(stabilized_frame)
        out.write(stabilized_frame)

    cap.release()
    out.release()
    return output_path
