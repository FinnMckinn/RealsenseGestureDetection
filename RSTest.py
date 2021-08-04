# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from scipy import ndimage as sp

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)


# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# Subtraction
clipping_distance_in_meters = 0.5 # Meters
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Create white image for binary image
white_image = np.float64(np.full((480,640,3), 255))
fN = 0

# Draw frame data
current_frame = 0
draw_frame_cords = (0, 0)

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Remove background - Set pixels further than clipping_distance to black
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), 0, white_image)
        
        # Convert to binary image
        bg_removed = np.array(bg_removed)
        bg_removed[bg_removed == 255] = 1
        matrix = bg_removed[:,:,0]

        # Save binary iamge
        '''
        #plt.imsave('frame{}.png'.format(fN), matrix, cmap=cm.gray)
        #fN += 1
        '''

        # Distance Transform applied to matrix

        # Format for cv2
        img = matrix.astype(np.uint8)

        # Perform the distance transform algorithm
        dist = cv2.distanceTransform(img, cv2.DIST_L2, 3)

        # Save distance values for palm estimation 
        dist_values = np.copy(dist)

        # Normalize the distance image for range = {0.0, 1.0}
        # so we can visualize and threshold it
        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        cv2.imshow('Distance Transform Image', dist)

        # Find highest value and index in distance transfrom
        highest_val_index = np.unravel_index(dist_values.argmax(), dist_values.shape)
        highest_val = dist_values[highest_val_index[0], highest_val_index[1]]
        
        # Draw circle centered on highest value with radius relative to distance
        # Only draw circle every 5 frames to reduce stutter
        if current_frame == 5:
            x = highest_val_index[1]
            y = highest_val_index[0]
            highest_val_index = (x,y)

            draw_frame_cords = highest_val_index
            draw_frame = False
            current_frame = 0

        image = cv2.circle(bg_removed, draw_frame_cords, int(highest_val), (0, 0, 255), 1)
        current_frame += 1
        
        # Display view
        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', image)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()