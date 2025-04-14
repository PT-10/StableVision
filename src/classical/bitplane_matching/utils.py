# main.py
import cv2
import numpy as np
import os
from .bitplane_config import (
    GRAY_CODE, 
    BITPLANES,
    MATCHING,
    STABILIZATION,
    COLOR,
    validate_config
)

# Validate configuration on import
validate_config()

# ========== CORE CONFIGURATION ==========
debug = True  # Set to False to disable debug prints

# ========== COLOR SPACE CONVERSIONS ==========
def YCrCb2BGR(image):
    """Converts numpy image from YCrCb to BGR color space"""
    return cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)

def BGR2YCrCb(image):
    """Converts numpy image from BGR to YCrCb color space"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

# ========== BITPLANE MATCHING CORE ==========
def gray_code_transform(image):
    """Convert image to Gray code using XOR-based transformation"""
    if GRAY_CODE['pre_blur']:
        image = cv2.GaussianBlur(image, GRAY_CODE['pre_blur'], 0)
    gray = cv2.bitwise_xor(image, image >> 1)
    return gray

def extract_bitplanes(frame, num_planes=BITPLANES['num_planes']):
    """Extract MSB bitplanes from Gray-coded image with post-processing"""
    gray = gray_code_transform(frame)
    planes = [(gray >> (7 - i)) & 1 for i in BITPLANES['plane_indices']]
    stack = np.stack(planes, axis=2).astype(np.uint8)
    
    if GRAY_CODE['post_blur']:
        stack = cv2.GaussianBlur(stack, GRAY_CODE['post_blur'], 0)
    
    return stack * 255  # Scale to 0-255

def bitplane_hamming_distance(bp1, bp2):
    """Calculate Hamming distance between bit-plane blocks"""
    xor_result = cv2.bitwise_xor(bp1, bp2)
    
    # Convert multi-channel to single channel for counting
    if bp1.ndim == 3:
        # For 3D bitplane arrays (H,W,C), flatten to 2D (H*W*C, 1)
        xor_flat = xor_result.reshape(-1, 1)
    else:
        xor_flat = xor_result
        
    return cv2.countNonZero(xor_flat) / (bp1.size / 8)  # Normalized

# ========== IMAGE SEGMENTATION ==========
def segmentImage(anchor, blockSize=16):
    """Determines macroblock segmentation of image"""
    h, w = anchor.shape[:2] if anchor.ndim == 3 else anchor.shape
    return (h // blockSize, w // blockSize)

# ========== HIERARCHICAL BLOCK MATCHING ==========
def hierarchical_gcbpm(prev_frame, curr_frame, levels=MATCHING['hierarchical']['levels']):
    """Multi-resolution bitplane matching"""
    prev_pyramid = [prev_frame]
    curr_pyramid = [curr_frame]
    
    # Build pyramids
    for _ in range(levels-1):
        prev_pyramid.append(cv2.pyrDown(prev_pyramid[-1]))  # Directly process bitplanes
        curr_pyramid.append(cv2.pyrDown(curr_pyramid[-1]))
    
    motion_vector = np.zeros(2)
    for level in reversed(range(levels)):
        # Get spatial dimensions (ignore channels)
        h, w = prev_pyramid[level].shape[:2]  # <-- FIX HERE
        dsize = (w, h)  # <-- FIX HERE
        
        # Apply motion vector from previous level
        scaled_vector = 2 * motion_vector
        M = np.float32([[1, 0, scaled_vector[0]], [0, 1, scaled_vector[1]]])
        warped = cv2.warpAffine(prev_pyramid[level], M, dsize)  # <-- USE FIXED DSIZE
        
        # Refine at current level
        level_vectors = gcbpm_search(warped, curr_pyramid[level])
        motion_vector += np.median(level_vectors, axis=(0,1))
        
    return motion_vector

def gcbpm_search(anchor_bp, target_bp, block_size=MATCHING['block_size'], search_radius=MATCHING['search_radius']):
    """3-step search with bitplane matching and early termination"""
    h, w = target_bp.shape[:2]
    motion_vectors = np.zeros((h//block_size, w//block_size, 2), dtype=int)
    
    for y in range(0, h-block_size, block_size):
        for x in range(0, w-block_size, block_size):
            target_block = target_bp[y:y+block_size, x:x+block_size]
            
            min_cost = float('inf')
            best_dx, best_dy = 0, 0
            
            # Three-step search pattern
            for step in [4, 2, 1]:
                for dy in range(-step, step+1, step):
                    for dx in range(-step, step+1, step):
                        px = x + dx
                        py = y + dy
                        
                        if px < 0 or px > w-block_size: continue
                        if py < 0 or py > h-block_size: continue
                        
                        anchor_block = anchor_bp[py:py+block_size, px:px+block_size]
                        cost = bitplane_hamming_distance(target_block, anchor_block)
                        
                        # Early termination check
                        if (MATCHING['early_termination']['enabled'] and 
                            cost < (block_size**2 * MATCHING['early_termination']['threshold'])):
                            best_dx, best_dy = dx, dy
                            min_cost = cost
                            break  # Exit step loop early
                        
                        if cost < min_cost:
                            min_cost = cost
                            best_dx, best_dy = dx, dy
                            
            motion_vectors[y//block_size, x//block_size] = [best_dx, best_dy]
    
    return motion_vectors

# ========== PREPROCESSING ==========
def preprocess(anchor, target, blockSize):
    """Preprocess frames for bitplane matching"""
    # Convert to grayscale (handle both BGR and existing grayscale)
    if isinstance(anchor, str) and isinstance(target, str):
        anchor_gray = cv2.cvtColor(cv2.imread(anchor), cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(cv2.imread(target), cv2.COLOR_BGR2GRAY)
    elif isinstance(anchor, np.ndarray) and isinstance(target, np.ndarray):
        # Handle existing grayscale or BGR inputs
        if anchor.ndim == 3 and anchor.shape[2] == 3:  # BGR image
            anchor_gray = cv2.cvtColor(anchor, cv2.COLOR_BGR2GRAY)
        elif anchor.ndim == 2:  # Already grayscale
            anchor_gray = anchor.astype(np.uint8)
        else:
            raise ValueError("Unsupported anchor image format")
        
        if target.ndim == 3 and target.shape[2] == 3:
            target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        elif target.ndim == 2:
            target_gray = target.astype(np.uint8)
        else:
            raise ValueError("Unsupported target image format")
    else:
        raise ValueError("Invalid input types")

    # Rest of the function remains unchanged
    hSegments, wSegments = segmentImage(anchor_gray, blockSize)
    new_size = (wSegments * blockSize, hSegments * blockSize)
    
    anchor_resized = cv2.resize(anchor_gray, new_size)
    target_resized = cv2.resize(target_gray, new_size)

    return (
        extract_bitplanes(gray_code_transform(anchor_resized)),
        extract_bitplanes(gray_code_transform(target_resized))
    )

# ========== BLOCK MATCHING BODY ==========
def blockSearchBody(anchor, target, blockSize=16, searchArea=7):
    """
    Bitplane-based block matching core function
    
    Parameters:
    -----------
    anchor : numpy.ndarray
        Reference frame (bitplane representation)
    target : numpy.ndarray
        Current frame to match against reference (bitplane representation)
    blockSize : int
        Size of blocks for matching (default: 16)
    searchArea : int
        Search radius around each block (default: 7)
        
    Returns:
    --------
    tuple: (predicted_frame, motion_info)
        predicted_frame: Motion-compensated prediction of the target frame
        motion_info: Dictionary containing motion vectors and statistics
    """
    # Store motion information for analysis
    motion_info = {}
    
    # Choose motion estimation method based on configuration
    if MATCHING['hierarchical']['enabled']:
        # Hierarchical method returns a single global motion vector
        motion_vector = hierarchical_gcbpm(anchor, target)
        
        # Store the motion vector
        motion_info['global_vector'] = motion_vector
        motion_info['method'] = 'hierarchical'
        
        # Apply global motion compensation
        h, w = target.shape[:2]
        dx, dy = motion_vector
        
        # Create transformation matrix
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        predicted = cv2.warpAffine(anchor, M, (w, h))
        
    else:
        # Block matching returns a field of motion vectors
        motion_vectors = gcbpm_search(anchor, target, blockSize, searchArea)
        
        # Store motion vectors for analysis
        motion_info['motion_field'] = motion_vectors
        motion_info['method'] = 'block_matching'
        
        # Calculate global motion for overall frame adjustment
        median_dx = int(np.round(np.median(motion_vectors[:,:,0])))
        median_dy = int(np.round(np.median(motion_vectors[:,:,1])))
        motion_info['global_vector'] = [median_dx, median_dy]
        
        # Apply motion compensation using global shift
        # This is simpler and often works as well as block-by-block warping
        predicted = np.roll(anchor, shift=(median_dy, median_dx), axis=(0, 1))
    
    if debug:
        print(f"Motion vector: {motion_info['global_vector']}")
    
    # Calculate prediction quality metrics
    if target.dtype != predicted.dtype:
        predicted = predicted.astype(target.dtype)
    
    # Ensure shapes match (handle boundary effects)
    h, w = min(predicted.shape[0], target.shape[0]), min(predicted.shape[1], target.shape[1])
    predicted = predicted[:h, :w]
    target_crop = target[:h, :w]
    
    # Calculate average difference as metric
    abs_diff = np.abs(predicted.astype(np.float32) - target_crop.astype(np.float32))
    motion_info['mean_abs_diff'] = np.mean(abs_diff)
    
    return predicted, motion_info

# ========== RESIDUAL CALCULATIONS ==========
def getResidual(target, predicted):
    """Calculate residual frame (single-channel output)"""
    residual = np.subtract(target, predicted)
    return cv2.cvtColor(residual, cv2.COLOR_BGR2GRAY) if COLOR['y_only'] else residual

def getReconstructTarget(residual, predicted):
    """Reconstruct target from residual"""
    return np.add(residual, predicted)

# ========== METRIC CALCULATIONS ==========
def getResidualMetric(residualFrame):
    """Calculate residual metric"""
    return np.sum(np.abs(residualFrame)) / (residualFrame.shape[0] * residualFrame.shape[1])

# ========== MAIN PROCESSING PIPELINE ==========
def main(anchorFrame, targetFrame, outfile="OUTPUT", saveOutput=True, blockSize=16):
    """Main processing pipeline"""
    # Preprocess inputs
    anchor_bp, target_bp = preprocess(anchorFrame, targetFrame, blockSize)
    
    # Perform block matching
    predictedFrame, _ = blockSearchBody(anchor_bp, target_bp, blockSize)
    
    # Calculate residual frame
    residualFrame = getResidual(target_bp, predictedFrame)
    
    # Calculate metrics
    residualMetric = getResidualMetric(residualFrame)
    naiveResidualFrame = getResidual(anchor_bp, target_bp)
    naiveResidualMetric = getResidualMetric(naiveResidualFrame)

    # Save outputs
    if saveOutput:
        os.makedirs(outfile, exist_ok=True)
        cv2.imwrite(f"{outfile}/targetFrame.png", target_bp)
        cv2.imwrite(f"{outfile}/predictedFrame.png", predictedFrame)
        cv2.imwrite(f"{outfile}/residualFrame.png", residualFrame)
        cv2.imwrite(f"{outfile}/naiveResidualFrame.png", naiveResidualFrame)
        with open(f"{outfile}/results.txt", "w") as f:
            f.write(f"Residual Metric: {residualMetric:.2f}\n")
            f.write(f"Naive Residual Metric: {naiveResidualMetric:.2f}\n")

    print(f"Residual Metric: {residualMetric:.2f}")
    print(f"Naive Residual Metric: {naiveResidualMetric:.2f}")
    
    return residualMetric, residualFrame
