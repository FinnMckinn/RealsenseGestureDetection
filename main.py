import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

from RealSesneConfig.RSConfig import RealSenseConfig, RealSenseFilters

mpHands = mp.solutions.hands
hands = mpHands.Hands(False, 1, 0.5, 0.5)
mpDraw = mp.solutions.drawing_utils

rsConfig = RealSenseConfig()
rsFilter = RealSenseFilters(rs)

rsConfig.setup(rs)

# Create white image for binary image
white_image = np.float64(np.full((480,640,3), 255))
fN = 0

# Draw frame data
current_frame = 0
draw_frame_cords = (0, 0)

# Mp Landmark Data
keyLandmarks = [2, 4, 5, 8, 9, 12, 13, 16, 17, 20]
fingerLandmarks = {
    "palm" :   [0,0],
    "pinky" :  [[0,0], [0,0]],
    "ring" :   [[0,0], [0,0]],
    "middle" : [[0,0], [0,0]],
    "index" :  [[0,0], [0,0]],
    "thumb" :  [[0,0], [0,0]]
}

def depth_threshold(background, foreground):
    filtered_image = np.where((depth_image_3d > rsConfig.clipping_distance) | (depth_image_3d <= 0), background, foreground)
    return filtered_image

def distance_transform(binary_image):
    # Flatten
    matrix = binary_image[:,:,0]

    # Format for cv2
    cvtMatrix = matrix.astype(np.uint8)

    # Perform the distance transform algorithm
    dist = cv2.distanceTransform(cvtMatrix, cv2.DIST_L2, 3)
    return dist

def estimate_palm_point(highest_val_index):
    x = highest_val_index[1]
    y = highest_val_index[0]
    highest_val_index = (x,y)
    
    return highest_val_index

def draw_finger_keypoints(processed_image):
    # Mp Landmarks
    imgRGB = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:

            # Segment landmarks
            for id, lm in enumerate(handLandmarks.landmark):
                h, w, c = bg_removed.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
            
                # Find fingertips
                if id in keyLandmarks:
                    if id == 2:
                        cv2.circle(processed_image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                        fingerLandmarks['thumb'][0] = (cx, cy)
                        cv2.line(processed_image, fingerLandmarks['palm'], fingerLandmarks['thumb'][0], color=(255, 0, 0), thickness=3, lineType=8)
                    if id == 4:
                        cv2.circle(processed_image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                        fingerLandmarks['thumb'][1] = (cx, cy)
                        cv2.line(processed_image, fingerLandmarks['thumb'][0], fingerLandmarks['thumb'][1], color=(255, 0, 0), thickness=3, lineType=8)


                    if id == 5:
                        cv2.circle(processed_image, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                        fingerLandmarks['index'][0] = (cx, cy)
                        cv2.line(processed_image, fingerLandmarks['palm'], fingerLandmarks['index'][0], color=(0, 255, 0), thickness=3, lineType=8)
                    if id == 8:
                        cv2.circle(processed_image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                        fingerLandmarks['index'][1] = (cx, cy)
                        cv2.line(processed_image, fingerLandmarks['index'][0], fingerLandmarks['index'][1], color=(0, 255, 0), thickness=3, lineType=8)


                    if id == 9:
                        cv2.circle(processed_image, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
                        fingerLandmarks['middle'][0] = (cx, cy)
                        cv2.line(processed_image, fingerLandmarks['palm'], fingerLandmarks['middle'][0], color=(0, 0, 255), thickness=3, lineType=8)
                    if id == 12:
                        cv2.circle(processed_image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                        fingerLandmarks['middle'][1] = (cx, cy)
                        cv2.line(processed_image, fingerLandmarks['middle'][0], fingerLandmarks['middle'][1], color=(0, 0, 255), thickness=3, lineType=8)


                    if id == 13:
                        cv2.circle(processed_image, (cx, cy), 5, (255, 100, 0), cv2.FILLED)
                        fingerLandmarks['ring'][0] = (cx, cy)
                        cv2.line(processed_image, fingerLandmarks['palm'], fingerLandmarks['ring'][0], color=(255, 100, 0), thickness=3, lineType=8)
                    if id == 16:
                        cv2.circle(processed_image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                        fingerLandmarks['ring'][1] = (cx, cy)
                        cv2.line(processed_image, fingerLandmarks['ring'][0], fingerLandmarks['ring'][1], color=(255, 100, 0), thickness=3, lineType=8)


                    if id == 17:
                        cv2.circle(processed_image, (cx, cy), 5, (255, 0, 100), cv2.FILLED)
                        fingerLandmarks['pinky'][0] = (cx, cy)
                        cv2.line(processed_image, fingerLandmarks['palm'], fingerLandmarks['pinky'][0], color=(255, 0, 100), thickness=3, lineType=8)
                    if id == 20:
                        cv2.circle(processed_image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                        fingerLandmarks['pinky'][1] = (cx, cy)
                        cv2.line(processed_image, fingerLandmarks['pinky'][0], fingerLandmarks['pinky'][1], color=(255, 0, 0), thickness=3, lineType=8)


if __name__ == "__main__":
    try:
        while True:  
            # Get frameset of color and depth
            frames = rsConfig.pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = rsConfig.align.process(frames)

            # Get aligned frames
            aligned_depth_frame = rsFilter.apply_filters(aligned_frames)
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Remove background - Set pixels further than clipping_distance to black
            depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
            bg_removed = depth_threshold(0, color_image)
            binary_image = depth_threshold(0, white_image)
            
            # Convert to binary image
            binary_image = np.array(binary_image)
            binary_image[binary_image == 255] = 1

            # Distance Transfrom
            dist = distance_transform(binary_image)
            
            # Find highest value and index in distance transfrom
            highest_val_index = np.unravel_index(dist.argmax(), dist.shape)
            highest_val = dist[highest_val_index[0], highest_val_index[1]]
            
            # Normalize range to {0.0 - 1.0} for visulisation
            cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        
            # Draw circle centered on highest value with radius relative to distance
            # Only draw circle every 5 frames to reduce stutter
            if current_frame == 5:
                draw_frame_cords = np.flipud(highest_val_index)
                fingerLandmarks['palm'] = draw_frame_cords
                current_frame = 0

            # Draw keypoints
            processed_image = cv2.circle(binary_image, draw_frame_cords, int(highest_val), (0, 0, 255), 1)
            draw_finger_keypoints(processed_image)

            current_frame += 1
            
            # Display view
            cv2.imshow('Virtual Glove Marker', processed_image)
            cv2.imshow("Depth Threshold", bg_removed)
            cv2.imshow('Distance Transform Image', dist)

            # Press esc or 'q' to close the image window
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        rsConfig.pipeline.stop()