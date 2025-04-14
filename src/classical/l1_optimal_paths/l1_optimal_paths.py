import cv2
import numpy as np
import pulp as lpp
import os
from utils.codec_conversion import ensure_h264_compliance

# Constants (moved to tuples for immutability where possible)
W = (10, 1, 100)
C = ([1, 1, 100, 100, 100, 100],) * 3  # c1, c2, c3
N = 6

def transform_product(F_t, p, t):
    f20, f21 = F_t[2, 0], F_t[2, 1]
    return [
        p[t, 0] + f20 * p[t, 2] + f21 * p[t, 3],
        p[t, 1] + f20 * p[t, 4] + f21 * p[t, 5],
        F_t[0, 0] * p[t, 2] + F_t[0, 1] * p[t, 3],
        F_t[1, 0] * p[t, 2] + F_t[1, 1] * p[t, 3],
        F_t[0, 0] * p[t, 4] + F_t[0, 1] * p[t, 5],
        F_t[1, 0] * p[t, 4] + F_t[1, 1] * p[t, 5]
    ]

def get_crop_window(shape, ratio):
    h, w = shape
    ch, cw = round(h * ratio), round(w * ratio)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return [(x0, y0), (x0+cw, y0), (x0, y0+ch), (x0+cw, y0+ch)]

def stabilize(F_transforms, frame_shape, crop_ratio=0.8):
    n_frames = len(F_transforms)
    prob = lpp.LpProblem("stabilize", lpp.LpMinimize)

    e = [lpp.LpVariable.dicts(f"e{i+1}", ((t, j) for t in range(n_frames) for j in range(N)), lowBound=0.0)
         for i in range(3)]
    p = lpp.LpVariable.dicts("p", ((t, j) for t in range(n_frames) for j in range(N)))

    # Objective
    prob += lpp.lpSum(
        W[i] * lpp.lpSum(e[i][t, j] * C[i][j] for t in range(n_frames) for j in range(N))
        for i in range(3)
    )

    # Constraints
    for t in range(n_frames - 3):
        prod = [transform_product(F_transforms[t + i + 1], p, t + i + 1) for i in range(3)]
        res = [[prod[i][j] - p[t + i, j] for j in range(N)] for i in range(3)]

        for j in range(N):
            prob += -e[0][t, j] <= res[0][j]
            prob += e[0][t, j] >= res[0][j]
            prob += -e[1][t, j] <= res[1][j] - res[0][j]
            prob += e[1][t, j] >= res[1][j] - res[0][j]
            prob += -e[2][t, j] <= res[2][j] - 2 * res[1][j] + res[0][j]
            prob += e[2][t, j] >= res[2][j] - 2 * res[1][j] + res[0][j]

    corner_points = get_crop_window(frame_shape, crop_ratio)
    w, h = frame_shape[1], frame_shape[0]

    for t in range(n_frames):
        a, b, c, d, e_, f = (p[t, i] for i in range(N))

        # Affine bounds
        prob += 0.9 <= c <= 1.1
        prob += -0.1 <= d <= 0.1
        prob += -0.1 <= e_ <= 0.1
        prob += 0.9 <= f <= 1.1
        prob += -0.1 <= d + e_ <= 0.1
        prob += -0.05 <= c - f <= 0.05

        for cx, cy in corner_points:
            x_trans = a + c * cx + d * cy
            y_trans = b + e_ * cx + f * cy
            prob += 0 <= x_trans <= w
            prob += 0 <= y_trans <= h

    # Solve
    prob.solve()

    # Construct transforms
    B_transforms = np.repeat(np.eye(3, dtype=np.float32)[np.newaxis, :, :], n_frames, axis=0)

    if lpp.LpStatus[prob.status] == 'Optimal':
        for t in range(n_frames):
            B_transforms[t, :, :2] = np.array([
                [p[t, 2].varValue, p[t, 4].varValue],
                [p[t, 3].varValue, p[t, 5].varValue],
                [p[t, 0].varValue, p[t, 1].varValue]
            ])
    else:
        print("Optimization failed:", lpp.LpStatus[prob.status])

    return B_transforms

def l1_optimal_stabilization(input_path, output_filename, crop_ratio=0.8):
    input_path = ensure_h264_compliance(input_path)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError("Could not open video")

    width, height = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    F_transforms = np.tile(np.eye(3, dtype=np.float32), (num_frames, 1, 1))

    _, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    for i in range(1, num_frames):
        ret, frame = cap.read()
        if not ret: break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, 200, 0.01, 30, blockSize=3)
        if prev_pts is None: continue

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        if curr_pts is None or status is None: continue

        valid = status.ravel() == 1
        m = cv2.estimateAffine2D(curr_pts[valid], prev_pts[valid])[0]
        if m is not None:
            F_transforms[i, :, :2] = m.T
        prev_gray = curr_gray

    B_transforms = stabilize(F_transforms, (height, width), crop_ratio)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    out_path = os.path.join(os.path.dirname(input_path), output_filename)
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret: break
        M = B_transforms[i, :, :2].T
        out.write(cv2.warpAffine(frame, M, (width, height)))

    cap.release()
    out.release()
    return out_path
