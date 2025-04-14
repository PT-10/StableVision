import numpy as np
import cv2

def calcOpticalFlowHS(prev_gray, next_gray, flow=None, alpha=0.001, num_iter=100, criteria=None):
    """
    Compute Horn-Schunck optical flow using a faster, vectorized approach.

    Args:
        prev_gray (np.ndarray): First grayscale frame (2D array).
        next_gray (np.ndarray): Second grayscale frame (2D array).
        flow (np.ndarray): Pre-allocated flow array to store results.
        alpha (float): Regularization constant.
        num_iter (int): Number of iterations.
        criteria (tuple): Termination criteria for iterative refinement.

    Returns:
        flow (np.ndarray): Optical flow with shape (H, W, 2) containing (u, v).
    """
    prev_gray = prev_gray.astype(np.float32)
    next_gray = next_gray.astype(np.float32)

    # Compute gradients using Sobel (optimize by computing for both images at once)
    fx = cv2.Sobel(prev_gray, cv2.CV_32F, 1, 0, ksize=3) + cv2.Sobel(next_gray, cv2.CV_32F, 1, 0, ksize=3)
    fy = cv2.Sobel(prev_gray, cv2.CV_32F, 0, 1, ksize=3) + cv2.Sobel(next_gray, cv2.CV_32F, 0, 1, ksize=3)
    ft = next_gray - prev_gray

    # Initialize flow field
    if flow is None:
        u = np.zeros_like(prev_gray)
        v = np.zeros_like(prev_gray)
    else:
        u, v = flow[..., 0], flow[..., 1]

    # Pre-compute filters
    kernel = np.array([[1/12, 1/6, 1/12],
                       [1/6,   0,  1/6],
                       [1/12, 1/6, 1/12]], dtype=np.float32)

    # Iterate to refine flow estimate
    for _ in range(num_iter):
        # Convolve u and v with the kernel
        u_avg = cv2.filter2D(u, -1, kernel)
        v_avg = cv2.filter2D(v, -1, kernel)

        # Compute the ratio term
        numerator = fx * u_avg + fy * v_avg + ft
        denominator = alpha**2 + fx**2 + fy**2
        ratio = numerator / denominator

        # Update u and v
        u = u_avg - fx * ratio
        v = v_avg - fy * ratio

        # If we need to stop early (based on criteria), we can implement it here
        if criteria is not None:
            pass  # Add termination condition here if needed (like norm of change < threshold)

    # Stack u and v into a single array for (u, v) flow format
    flow = np.stack((u, v), axis=-1)
    return flow
