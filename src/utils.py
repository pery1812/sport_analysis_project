import cv2
import numpy as np

SKELETON = [
    (0, 1), (1, 3), (0, 2), (2, 4),  # Face (nose to eyes and ears)
    (5, 6),                          # Shoulders
    (5, 7), (7, 9),                  # Left arm
    (6, 8), (8, 10),                 # Right arm
    (11, 12),                        # Hips
    (5, 11), (6, 12),                # Body
    (11, 13), (13, 15),              # Left leg
    (12, 14), (14, 16)               # Right leg
]
def draw_bbox(image, bbox, score):
    """
    Draw bounding box and score on the image.

    Parameters:
        image (numpy.ndarray): Input image.
        bbox (list): Bounding box [x1, y1, x2, y2].
        score (float): Confidence score for the bounding box.

    Returns:
        image: Image with bounding box and score drawn.
    """
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(image, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image

def draw_keypoints(image, keypoints):
    """
    Draw keypoints on the image.

    Parameters:
        image (numpy.ndarray): Input image.
        keypoints (numpy.ndarray): Keypoints for a single person, expected shape (N, 3).

    Returns:
        image: Image with keypoints drawn.
    """
    # Extract keypoint data
    keypoints_array = keypoints.data[0]  # Shape: (17, 3)

    # Calculate circle size based on image dimensions
    height, width, _ = image.shape
    circle_radius = max(2, min(height, width) // 150)  # Adjust this factor (e.g., 150) as needed

    # Draw keypoints
    for kp in keypoints_array:
        if len(kp) == 3:  # Ensure there are (x, y, confidence)
            x, y, confidence = kp
            if confidence > 0.5:  # Draw only confident keypoints
                cv2.circle(image, (int(x), int(y)), circle_radius, (0, 255, 0), -1)
    return image

def draw_keypoints_and_skeleton(image, keypoints, skeleton=SKELETON):
    """
    Draw keypoints and connect them with lines to form the skeleton.

    Parameters:
        image (numpy.ndarray): Input image.
        keypoints (numpy.ndarray): Keypoints for a single person, expected shape (N, 3).
        skeleton (list of tuple): List of keypoint index pairs defining the skeleton.

    Returns:
        image: Image with keypoints and skeleton drawn.
    """
    # Extract keypoint data
    keypoints_array = keypoints.data[0]  # Shape: (17, 3)

    # Calculate circle size based on image dimensions
    height, width, _ = image.shape
    circle_radius = max(2, min(height, width) // 150)  # Adjust this factor (e.g., 150) as needed

    # Draw keypoints
    for kp in keypoints_array:
        x, y, confidence = kp
        if confidence > 0.5:  # Draw only confident keypoints
            cv2.circle(image, (int(x), int(y)), circle_radius, (0, 255, 0), -1)

    # Connect keypoints with lines
    for start_idx, end_idx in skeleton:
        if start_idx < len(keypoints_array) and end_idx < len(keypoints_array):
            x1, y1, conf1 = keypoints_array[start_idx]
            x2, y2, conf2 = keypoints_array[end_idx]
            if conf1 > 0.5 and conf2 > 0.5:  # Draw lines only if both keypoints are confident
                cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    return image

def visualize_basketball(frame, basketball_boxes, hoop_boxes):
    """
    Create a new visualization image showing only the basketball and hoop positions.

    Parameters:
        frame (ndarray): Input frame to determine size.
        basketball_boxes (list): List of basketball bounding boxes in YOLO format [x1, y1, x2, y2].
        hoop_boxes (list): List of hoop bounding boxes in YOLO format [x1, y1, x2, y2].

    Returns:
        visualized_frame (ndarray): New visualization frame.
    """
    # Create a blank white image with the same size as the input frame
    height, width, _ = frame.shape
    visualized_frame = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Draw basketball positions
    for ball_box in basketball_boxes:
        x1, y1, x2, y2 = ball_box
        # Calculate the center of the ball
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        # Calculate the radius as half the width of the box
        radius = max(2, int((x2 - x1) / 2))
        # Draw the ball as a purple circle
        cv2.circle(visualized_frame, (center_x, center_y), radius, (255, 0, 255), thickness=-1)

    # Draw hoop positions
    for hoop_box in hoop_boxes:
        x1, y1, x2, y2 = hoop_box
        # Draw the hoop as an orange rectangle
        cv2.rectangle(visualized_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), thickness=3)

    return visualized_frame