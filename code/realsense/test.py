### File used to test the realsense camera class. 
from realsense import Camera
import json
import time

def custom_filters(depth_frame):
    # depth_frame = np.array(depth_frame, dtype=np.float64)
    # depth_frame = denoise_tv_chambolle(depth_frame, weight=0.1)
    return depth_frame 

## Test time complexity of all the filters.

if __name__ == "__main__":
    # Load the configuration from the JSON file
    with open('d435.config', 'r') as f:
        config = json.load(f)

    # Instantiate the Camera class with the loaded configuration
    camera = Camera(config=config, filters=custom_filters)
    camera.run()
