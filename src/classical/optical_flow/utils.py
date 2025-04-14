import numpy as np
import cv2

def calculate_moving_average(curve, radius):
    # Calculate the moving average of a curve using a given radius
    window_size = 2 * radius + 1
    kernel = np.ones(window_size) / window_size
    curve_padded = np.pad(curve, (radius, radius), mode='edge')
    smoothed_curve = np.convolve(curve_padded, kernel, mode='same')
    smoothed_curve = smoothed_curve[radius:-radius]
    return smoothed_curve

def smooth_trajectory_kalman(trajectory):
    """
    Smooth the trajectory using a Kalman filter.
    """
    smoothed_trajectory = np.zeros_like(trajectory)
    n = trajectory.shape[1]  # Number of dimensions (e.g., dx, dy, da)

    # Initialize Kalman filter parameters
    for i in range(n):
        # State variables: position and velocity
        x = np.array([trajectory[0, i], 0])  # Initial state [position, velocity]
        P = np.eye(2)  # Initial covariance matrix
        F = np.array([[1, 1], [0, 1]])  # State transition matrix
        Q = np.array([[1, 0], [0, 1]]) * 0.01  # Process noise covariance
        H = np.array([[1, 0]])  # Observation matrix
        R = np.array([[1]]) * 0.1  # Measurement noise covariance

        for t in range(len(trajectory)):
            # Prediction step
            x = F @ x
            P = F @ P @ F.T + Q

            # Update step
            z = np.array([trajectory[t, i]])  # Measurement
            y = z - H @ x  # Measurement residual
            S = H @ P @ H.T + R  # Residual covariance
            K = P @ H.T @ np.linalg.inv(S)  # Kalman gain
            x = x + K @ y
            P = (np.eye(2) - K @ H) @ P

            # Store the smoothed position
            smoothed_trajectory[t, i] = x[0]

    return smoothed_trajectory


def smooth_trajectory(trajectory, smoothing_radius=30, use_kalman=False):
    """
    Smooth the trajectory using either moving average or Kalman filter.
    
    Args:
        trajectory (np.ndarray): The original trajectory (dx, dy, da).
        smoothing_radius (int): Radius for moving average smoothing.
        use_kalman (bool): If True, use Kalman filter; otherwise, use moving average.
    
    Returns:
        np.ndarray: Smoothed trajectory.
    """
    if use_kalman:
        return smooth_trajectory_kalman(trajectory)
    else:
        smoothed_trajectory = np.copy(trajectory)
        for i in range(3):
            smoothed_trajectory[:, i] = calculate_moving_average(
                trajectory[:, i],
                radius=smoothing_radius
            )
        return smoothed_trajectory


def fix_border(frame):
    # Fix the border of a frame by applying rotation and scaling transformation
    frame_shape = frame.shape
    
    matrix = cv2.getRotationMatrix2D(
        (frame_shape[1] / 2, frame_shape[0] / 2),
        0,
        1.04
    )

    frame = cv2.warpAffine(frame, matrix, (frame_shape[1], frame_shape[0]))
    return frame