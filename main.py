import cv2
import mediapipe as mp
import numpy as np
import pyrealsense2 as rs

from RealSesneConfig.RSConfig import RealSenseConfig, RealSenseFilters
from HandData.HandData import FingerData, PalmData

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

PALMCOLOUR =  (0, 1, 1)

PINKYCOLOUR = (0, 1, 0)
RINGCOLOUR = (0, 0, 1)
MIDDLECOLOUR = (1, 1, 0)
INDEXCOLOUR = (1, 0, 0)
THUMBCOLOUR = (1, 0, 1)

# Mp Landmark Data
keyLandmarks = [2, 4, 5, 8, 9, 12, 13, 16, 17, 20]

palmData = PalmData(PALMCOLOUR)

pinkyData = FingerData(PINKYCOLOUR)
ringData = FingerData(RINGCOLOUR)
middleData = FingerData(MIDDLECOLOUR)
indexData = FingerData(INDEXCOLOUR)
thumbData = FingerData(THUMBCOLOUR)

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

def draw_knuckle(fingerData, cords):
    fingerData.setknucklePoint(cords)
    cv2.circle(processed_image, cords, 5, fingerData.colour, cv2.FILLED)
    cv2.line(processed_image, palmData.centrePoint, fingerData.knucklePoint, color=fingerData.colour, thickness=5, lineType=8)

def draw_tip(fingerData, cords):
    fingerData.setTipPoint(cords)
    cv2.circle(processed_image, cords, 5, fingerData.colour, cv2.FILLED)
    cv2.line(processed_image, fingerData.knucklePoint, fingerData.tipPoint, color=fingerData.colour, thickness=15, lineType=8)


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
            
                # Draw Skeleton Glove
                if id in keyLandmarks:
                    if id == 2: draw_knuckle(thumbData, (cx, cy))
                    if id == 4: draw_tip(thumbData, (cx, cy))

                    if id == 5: draw_knuckle(indexData, (cx, cy))
                    if id == 8: draw_tip(indexData, (cx, cy))

                    if id == 9: draw_knuckle(middleData, (cx, cy))
                    if id == 12: draw_tip(middleData, (cx, cy))

                    if id == 13: draw_knuckle(ringData, (cx, cy))
                    if id == 16: draw_tip(ringData, (cx, cy))

                    if id == 17: draw_knuckle(pinkyData, (cx, cy))
                    if id == 20: draw_tip(pinkyData, (cx, cy))


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
            norm_dist = np.copy(dist)
            cv2.normalize(norm_dist, norm_dist, 0, 1.0, cv2.NORM_MINMAX)
        
            # Draw circle centered on highest value with radius relative to distance
            # Only draw circle every 5 frames to reduce stutter
            if current_frame == 5:
                palmData.setCentrePoint(np.flipud(highest_val_index))
                palmData.calcAverageCord()
                current_frame = 0

            # Draw keypoints
            processed_image = cv2.circle(binary_image, palmData.averagePoint, int(highest_val), palmData.colour, cv2.FILLED)
            draw_finger_keypoints(processed_image)

            current_frame += 1
            
            # Display view
            cv2.imshow('Virtual Glove Marker', processed_image)
            cv2.imshow("Depth Threshold", bg_removed)
            cv2.imshow('Distance Transform Image', norm_dist)

            # Press esc or 'q' to close the image window
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        rsConfig.pipeline.stop()