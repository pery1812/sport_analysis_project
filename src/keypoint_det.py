import os
import numpy as np
import cv2
from ultralytics import YOLO


class yolo_keypoint(object):
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

    def __init__(self, model_path, device='cpu'):
        """
        Initialize the YOLO Keypoint Detection model.
        
        Parameters:
            model_path (str): Path to the YOLOv8/YOLOv11 pose estimation model.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.device = device
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        """
        Load the YOLO model for pose estimation.

        Parameters:
            model_path (str): Path to the YOLO pose estimation model.

        Returns:
            model: Loaded YOLO model.
        """
        model = YOLO(model_path)  # Load YOLOv8 or YOLOv11 model
        model.to(self.device)
        return model

    def preprocess_image(self, image):
        """
        Preprocess the input image using OpenCV.

        Parameters:
            image (np.ndarray): Numpy array image

        Returns:
            image: Preprocessed image ready for model inference.
        """
        # image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image error")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #change color channel
        return image

    def predict(self, image):
        """
        Predict keypoints and bounding boxes from the input image using the model.

        Parameters:
            image (np.ndarray): Numpy array image

        Returns:
            results: List of dictionaries containing bounding box, keypoints, and scores for each detected person.
        """
        image = self.preprocess_image(image)

        # Run inference
        results = self.model(image)

        # Check for bounding boxes
        # has_person = results.boxes is not None and len(results.boxes) > 0

        # Parse results
        parsed_results = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
            scores = result.boxes.conf.cpu().numpy()  # Confidence scores for boxes
            keypoints = result.keypoints.cpu().numpy()  # Keypoints (x, y, confidence)
            for box, score, kp in zip(boxes, scores, keypoints):
                parsed_results.append({
                    "bounding_box": box,  # [x1, y1, x2, y2]
                    "score": score,
                    "keypoints": kp  # [[x1, y1, conf], [x2, y2, conf], ...]
                })
        return parsed_results, True

    def draw_keypoints(self, image, keypoints):
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
    
    def draw_keypoints_and_skeleton(self, image, keypoints, skeleton):
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

    def draw_bbox(self, image, bbox, score):
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

    def draw_results(self, image, output_path=""):
        """
        Draw keypoints and bounding boxes on the image and save the output.

        Parameters:
            image (np.ndarray): Numpy array image
            output_path (str): Path to save the output image.
        """
        if image is None:
            raise FileNotFoundError(f"Image not found at {image}")
        results, has_person = self.predict(image)

        for result in results:
            bbox = result["bounding_box"]
            score = result["score"]
            keypoints = result["keypoints"]

            image = self.draw_bbox(image, bbox, score)
            image = self.draw_keypoints_and_skeleton(image, keypoints, self.SKELETON)

        if output_path:
            cv2.imwrite(output_path, image)
        
        return image, results, has_person

if __name__ == "__main__":
    # Example usage
    model_path = "./weights/yolo11s-pose.pt"
    yolo_pose = yolo_keypoint(model_path, device='cpu')

    input_image = "./data_test/img1.png"
    save_dir = "./outputs"
    os.makedirs(save_dir, exist_ok=True)
    output_image = os.path.join(save_dir, os.path.basename(input_image)[:-4] + ".jpg")

    # Predict and visualize results
    image = cv2.imread(input_image)
    yolo_pose.draw_results(image, output_image)
