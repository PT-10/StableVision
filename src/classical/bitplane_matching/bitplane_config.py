# bitplane_config.py
"""
Configuration for Gray-Coded Bitplane Matching Video Stabilization
"""

# ========== GRAY CODE CONVERSION ==========
GRAY_CODE = {
    'enabled': True,   # Enable Gray code transformation
    'pre_blur': (3,3), # Gaussian blur kernel before Gray coding (width, height)
    'post_blur': (1,1) # Blur after bitplane extraction (width, height)
}

# ========== BITPLANE EXTRACTION ==========
BITPLANES = {
    'num_planes': 4,       # Use 4 MSB planes (4-7)
    'plane_indices': (4,5,6,7),  # Which planes to use (0=LSB,7=MSB)
    'normalize': True,     # Scale bitplanes to 0-255 range
    'pack_bits': False     # Pack 8 planes into 1 uint8 channel (experimental)
}

# ========== BLOCK MATCHING ==========
MATCHING = {
    'block_size': 16,      # Macroblock size in pixels
    'search_radius': 7,    # Search window radius
    'hierarchical': {
        'enabled': True,   # Enable 3-level hierarchical search
        'levels': 3        # Number of pyramid levels
    },
    'early_termination': {
        'enabled': True,   # Stop search if threshold met
        'threshold': 0.05  # 5% of total bits mismatched
    }
}

# ========== STABILIZATION ==========
STABILIZATION = {
    'smoothing_radius': 30,  # Moving average window size (frames)
    'border_scale': 1.04,    # Scaling factor for border fix
    'median_filter': True    # Use median of motion vectors
}

# ========== PERFORMANCE ==========
PERFORMANCE = {
    'multiprocessing': True,  # Use parallel processing
    'num_workers': 4,         # CPU cores to use
    'gpu_acceleration': False,  # Use CUDA optimizations
    'downsample': {           # Input downsampling
        'enabled': False,
        'factor': 0.5         # 1.0 = original, 0.5 = half resolution
    }
}

# ========== DEBUG/ANALYTICS ==========
DEBUG = {
    'show_bitplanes': False,  # Display bitplane visualizations
    'log_metrics': True,      # Record residual metrics
    'save_vectors': False     # Save motion vector data
}

# ========== COLOR MODES ==========
COLOR = {
    'process_chroma': False,  # Process color channels (experimental)
    'y_only': True            # Use only luma channel (recommended)
}

def validate_config():
    """Ensure configuration values are valid"""
    assert 1 <= BITPLANES['num_planes'] <= 8, "Invalid number of bitplanes (1-8)"
    assert MATCHING['block_size'] in {4,8,16,32}, "Unsupported block size"
    assert 0 < STABILIZATION['border_scale'] <= 1.2, "Invalid border scale (0-1.2)"
    assert MATCHING['hierarchical']['levels'] in {1,2,3}, "Hierarchical levels must be 1-3"
    assert 0 <= MATCHING['early_termination']['threshold'] <= 1, "Threshold must be 0-1"

# Validate configuration on import
validate_config()
