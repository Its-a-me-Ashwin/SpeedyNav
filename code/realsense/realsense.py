## Env used: python=3.9


import pyrealsense2 as rs
import numpy as np
import json
import cv2
import os
from datetime import datetime
from skimage.restoration import denoise_tv_chambolle

# Modify the Camera class to add a method for capturing frames
class Camera:
    def __init__(self, config, filters=None):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.alpha = config.get('alpha', 0.75)
        self.previous_frame = None
        self.filters = self.default_filters
        self.custom_filters = None if filters == None else filters
        self.width = config.get('width', 640)
        self.height = config.get('height', 480)
        self.color_width = config.get('color_width', 1280)
        self.color_height = config.get('color_height', 720)

        self.min_distance = config.get('MinDistance', 0.1)
        self.max_distance = config.get('MaxDistance', 5.0)
        
        self._setup_camera()
        
        self.recording = False
        self.video_writer = None
        self.base_dir = "../recordings"

        self.enableFilters = False
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def _setup_camera(self):
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        
        found_rgb = any(s.get_info(rs.camera_info.name) == 'RGB Camera' for s in device.sensors)
        if not found_rgb:
            raise RuntimeError("The camera does not have an RGB sensor.")
        
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, self.color_width, self.color_height, rs.format.bgr8, 30)

        self.pipeline.start(self.config)

        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

    def default_filters(self, depth_frame):
        # Apply default filters: decimation, spatial, temporal, and hole-filling
        decimation = rs.decimation_filter()
        spatial = rs.spatial_filter()
        temporal = rs.temporal_filter()
        hole_filling = rs.hole_filling_filter()
        
        depth_frame = decimation.process(depth_frame)
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)
        
        return depth_frame

    def process_depth_frame(self, depth_frame):
        # Apply the custom or default filters
        depth_frame = self.filters(depth_frame)
        
        # Convert to numpy array
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert min_distance and max_distance from meters to millimeters
        min_distance_mm = int(self.min_distance * 1000)  # Convert meters to millimeters
        max_distance_mm = int(self.max_distance * 1000)  # Convert meters to millimeters

        # Clamp the pixel values to be between min_distance and max_distance in millimeters
        depth_image = np.clip(depth_image, min_distance_mm, max_distance_mm)

        # Apply custom filters if provided
        if self.custom_filters is not None:
            depth_image = self.custom_filters(depth_image)
        
        # Crop 2.5% from each side
        crop_h = int(self.height * 0.025)
        crop_w = int(self.width * 0.025)
        depth_image_cropped = depth_image[crop_h:self.height-crop_h, crop_w:self.width-crop_w]
        
        # Remove salt-and-pepper noise using morphological operation
        kernel = np.ones((3, 3), np.uint8)
        depth_image_cleaned = cv2.morphologyEx(depth_image_cropped, cv2.MORPH_OPEN, kernel)
        
        # Apply median filter
        depth_image_filtered = cv2.medianBlur(depth_image_cleaned, 5)
        
        # Apply temporal averaging
        if self.previous_frame is None:
            self.previous_frame = depth_image_filtered.copy()
        else:
            depth_image_filtered = cv2.addWeighted(depth_image_filtered, self.alpha, self.previous_frame, 1 - self.alpha, 0)
            self.previous_frame = depth_image_filtered.copy()
        
        return depth_image_filtered

    def capture_frame(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        
        # Align the depth frame to the color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None

        # Process the depth frame
        depth_image_filtered = None
        if self.enableFilters:
            depth_image_filtered = self.process_depth_frame(depth_frame)
        else:
            depth_image_filtered = np.asanyarray(depth_frame.get_data())

        # Convert the color frame to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image_filtered


    def long_exposure(self, exposure_time=1):
        num_frames = int(exposure_time * 30) ## It is set to 30FPS. I dont know how to change it. Alos depend on CPU useage.
        color_accumulator = None
        depth_accumulator = None
        for i in range(num_frames):
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert frames to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Initialize accumulators
            if color_accumulator is None:
                color_accumulator = np.zeros_like(color_image, dtype=np.float32)
            if depth_accumulator is None:
                depth_accumulator = np.zeros_like(depth_image, dtype=np.float32)

            # Accumulate frames
            color_accumulator += color_image
            depth_accumulator += depth_image

        # Compute the average
        color_avg = (color_accumulator / num_frames).astype(np.uint8)
        depth_avg = (depth_accumulator / num_frames).astype(np.uint16)

        print("Long exposure completed.")
        return color_avg, depth_avg 

    def save_images(color_filename, depth_filename, color_image, depth_image):
        cv2.imwrite(color_filename, color_image)
        np.save(depth_filename, depth_image)

    def load_images(color_filename, depth_filename):
        color_image = cv2.imread(color_filename)
        depth_image = np.load(depth_filename)
        return color_image, depth_image

    def invert_frame(color_image, depth_image, flip_code=0):
        inverted_color = cv2.flip(color_image, flip_code)
        inverted_depth = cv2.flip(depth_image, flip_code)
        return inverted_color, inverted_depth

    def run(self):
        try:
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()
                
                # Align the depth frame to the color frame
                aligned_frames = self.align.process(frames)

                # Get aligned frames
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                # Process the depth frame
                depth_image_filtered = None
                if self.enableFilters:
                    depth_image_filtered = self.process_depth_frame(depth_frame)
                else:
                    depth_image_filtered = np.asanyarray(depth_frame.get_data())

                # Convert the color frame to numpy array
                color_image = np.asanyarray(color_frame.get_data())

                # Apply colormap on depth image (for display purposes)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_filtered, alpha=0.03), cv2.COLORMAP_JET)

                # Display the RGB image in one window
                cv2.imshow('RGB Frame', color_image)

                # Display the depth colormap in another window
                cv2.imshow('Depth Frame', depth_colormap)

                # Wait for a key press
                key = cv2.waitKey(1)

                # Press 'R' to start/stop recording
                if key == ord('r'):
                    self.toggle_recording(color_image)

                # Press 'S' to take a screenshot
                if key == ord('s'):
                    self.take_screenshot(color_image, depth_image_filtered)

                # Press 'Q' to quit
                if key == ord('q'):
                    break

        finally:
            self.pipeline.stop()
            if self.video_writer is not None:
                self.video_writer.release()
            cv2.destroyAllWindows()

    def toggle_recording(self, color_image):
        if not self.recording:
            # Start recording
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            recording_dir = os.path.join(self.base_dir, timestamp)
            os.makedirs(recording_dir, exist_ok=True)

            self.video_writer = cv2.VideoWriter(os.path.join(recording_dir, 'recording.avi'),
                                                cv2.VideoWriter_fourcc(*'XVID'), 30,
                                                (color_image.shape[1], color_image.shape[0]))

            self.recording = True
            print(f"Recording started: {recording_dir}")
        else:
            # Stop recording
            self.recording = False
            self.video_writer.release()
            self.video_writer = None
            print(f"Recording stopped")

    def take_screenshot(self, color_image, depth_image_filtered):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_filename = os.path.join(self.base_dir, f'screenshot_{timestamp}')
        cv2.imwrite(screenshot_filename + ".png", color_image)
        np.save(screenshot_filename, depth_image_filtered, allow_pickle=True)
        print(f"Screenshot saved: {screenshot_filename}")

def custom_filters(depth_frame):
    return depth_frame 
