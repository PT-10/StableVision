import os
import logging
import tempfile

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import PathSmoothUNet
from image_warper import FlowWarper
from metrices import MetricAnalyzer
from dataset import NUSDataset
from utils import getAllFileInDir, load_checkpoint, concatImagesHorizon, RunningAverage, checkAndMakeDir


class VideoStabilizer:
    def __init__(self, model_path=None, net_radius=15, scale_factor=8, latency=0, 
                 num_workers=4, bs=1, con_num=1, motion_dir=None):
        """
        Initialize the Video Stabilizer.
        
        Args:
            model_path: Path to pretrained model weights
            net_radius: Radius for the smoothing network
            scale_factor: Scale factor for upsampling
            latency: Latency parameter for stabilization
            num_workers: Number of workers for data loading
            bs: Batch size
            con_num: Connection number
            motion_dir: Directory containing motion files (.npy)
        """
        self.net_radius = net_radius
        self.scale_factor = scale_factor
        self.latency = latency
        self.num_workers = num_workers
        self.bs = bs
        self.con_num = con_num
        
        # Set up motion directory
        self.motion_dir = motion_dir
        
        # Load model
        self.net = None
        if model_path:
            self.load_model(model_path)
            
        # Set up other components
        self.image_warper = FlowWarper()
        self.bilinear_upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_model(self, model_path):
        """Load the pretrained stabilization model"""
        self.logger.info(f"Loading Pretrained Model From: {model_path}")
        self.net = PathSmoothUNet(4 * self.net_radius)
        self.net = nn.DataParallel(self.net)
        torch.backends.cudnn.benchmark = False
        self.net = self.net.cuda()
        load_checkpoint(model_path, self.net)
        return self.net
    
    def prepare_dataloader(self, motion_path):
        """Create dataloader from motion file"""
        dataset = NUSDataset(
            [motion_path], 
            None,  # No ground truth needed for inference
            self.net_radius, 
            self.con_num, 
            2 * self.net_radius - self.latency
        )
        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=self.bs, 
            shuffle=False, 
            num_workers=self.num_workers
        )
        return dataloader
    
    def get_warp_transformation(self, video_path, dataloader):
        """Compute warp transformation for stabilization"""
        if not self.net:
            raise ValueError("Model not loaded. Please load model first with load_model()")
            
        # Open video to get properties
        capture = cv2.VideoCapture()
        capture.open(video_path)
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        
        self.net.eval()
        warp_trans = np.zeros(
            (len(dataloader.dataset) + 2 * self.net_radius + self.con_num - 1,
             frame_height, frame_width, 2),
            dtype=np.float32
        )
        
        run_num = 0
        self.logger.info("Computing warp transformations...")
        with torch.no_grad():
            for flows, _ in tqdm(dataloader, desc="Processing frames"):
                flows = -flows.cuda()
                Bi = self.net(flows[:, 0, :, :, :])
                Bi = self.bilinear_upsample(Bi)
                Bi = np.squeeze(Bi.detach().cpu().numpy().transpose(0, 2, 3, 1))
                warp_trans[(run_num + 2 * self.net_radius - self.latency):(run_num + 2 * self.net_radius - self.latency + flows.shape[0])] = Bi
                run_num += flows.shape[0]
                
        return warp_trans
    
    def render_stabilized_video(self, input_path, output_path, warp_trans, create_comparison=False):
        """Render stabilized video using warp transformations"""
        # Open input video
        capture = cv2.VideoCapture()
        capture.open(input_path)
        
        # Get video properties
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize image warper
        self.image_warper.initialize(frame_width, frame_height)
        
        # Determine transformation length
        trans_length = min(frame_count, warp_trans.shape[0])
        
        # Prepare output video
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Decide if we need to crop the video (80% of original size)
        cropped_width = int(frame_width * 0.8)
        cropped_height = int(frame_height * 0.8)
        
        # Create video writer
        writer = cv2.VideoWriter(output_path, fourcc, fps, (cropped_width, cropped_height))
        
        # Create comparison video if requested
        comparison_path = None
        writer_comparison = None
        if create_comparison:
            comparison_path = os.path.splitext(output_path)[0] + "_comparison.mp4"
            writer_comparison = cv2.VideoWriter(comparison_path, fourcc, fps, (cropped_width * 2, cropped_height))
        
        self.logger.info(f"Rendering stabilized video to {output_path}")
        frame_processed = 0
        
        # Process frames
        for i in range(trans_length):
            ok, frame = capture.read()
            if not ok:
                break
                
            # Apply stabilization
            new_frame = self.image_warper.warp_image(frame, warp_trans[i])
            writer.write(new_frame)
            
            # Create comparison frame if requested
            if writer_comparison:
                cat_frame = concatImagesHorizon([frame, new_frame])
                writer_comparison.write(cat_frame)
                
            frame_processed += 1
            
        # Release resources
        capture.release()
        writer.release()
        if writer_comparison:
            writer_comparison.release()
            
        self.logger.info(f"Stabilized {frame_processed} frames")
        return output_path, comparison_path

    def stabilize_video(self, input_path, output_filename, motion_file=None, create_comparison=False):
        """
        Main function to stabilize a video
        
        Args:
            input_path: Path to input video
            output_filename: Name of output file
            motion_file: Optional path to motion file (.npy). If None, will look in motion_dir or create a temp one
            create_comparison: Whether to create a side-by-side comparison video
            
        Returns:
            Path to stabilized video
        """
        if not self.net:
            raise ValueError("Model not loaded. Please load model first with load_model()")
            
        # Handle output path
        output_dir = os.path.dirname(output_filename) if os.path.dirname(output_filename) else "."
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Determine motion file path
        if motion_file is None:
            if self.motion_dir:
                # Try to find motion file in motion_dir
                video_name = os.path.splitext(os.path.basename(input_path))[0]
                motion_file = os.path.join(self.motion_dir, f"{video_name}.npy")
                if not os.path.exists(motion_file):
                    # Generate motion file (you need to implement this part depending on your workflow)
                    self.logger.warning(f"Motion file not found for {video_name}. Need to generate one.")
                    # Here you would generate the motion file
                    # For example: motion_file = self.generate_motion_file(input_path)
                    raise ValueError("Motion file not found. Please provide motion file or implement motion generation.")
            else:
                raise ValueError("No motion file provided and no motion directory set")
        
        # Prepare dataloader
        dataloader = self.prepare_dataloader(motion_file)
        
        # Get warp transformation
        warp_trans = self.get_warp_transformation(input_path, dataloader)
        
        # Render stabilized video
        output_path, comparison_path = self.render_stabilized_video(
            input_path, 
            output_filename, 
            warp_trans, 
            create_comparison
        )
        
        return output_path


def stabilize_video(input_path, output_filename, model_path, motion_dir=None, create_comparison=False):
    """
    Easy-to-use function to stabilize a video.
    
    Args:
        input_path: Path to input video
        output_filename: Path to output stabilized video
        model_path: Path to pretrained model weights
        motion_dir: Directory containing motion files (.npy)
        create_comparison: Whether to create a side-by-side comparison video
        
    Returns:
        Path to stabilized video
    """
    stabilizer = VideoStabilizer(
        model_path=model_path,
        motion_dir=motion_dir
    )
    
    return stabilizer.stabilize_video(
        input_path=input_path,
        output_filename=output_filename,
        create_comparison=create_comparison
    )