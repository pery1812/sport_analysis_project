import os
import numpy as np
import cv2
from ultralytics import YOLO
# import ipdb

class yolo_basketball(object):
    CLASS_NAMES = ["basketball", "person", "board"]
    def __init__(self, model_path, threshold=0.5, device='cpu'):
        """
        Initialize the YOLO Keypoint Detection model.
        
        Parameters:
            model_path (str): Path to the YOLOv8/YOLOv11 pose estimation model.
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.device = device
        self.model = self.load_model(model_path)
        self.thres = threshold

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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
        # ipdb.set_trace()

        # Check for bounding boxes
        # has_person = results.boxes is not None and len(results.boxes) > 0

        # Parse results into two separate lists for each class
        ball_results = []  # For class 0
        hoop_results = []  # For class 1

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
            scores = result.boxes.conf.cpu().numpy()  # Confidence scores for boxes
            class_ids = result.boxes.cls.cpu().numpy()  # Class IDs

            for box, score, class_id in zip(boxes, scores, class_ids):
                if score >= self.thres:
                    parsed_box = {
                        "bounding_box": box,  # [x1, y1, x2, y2]
                        "score": score,
                        }
                    if class_id == 0:
                        ball_results.append(parsed_box)
                    elif class_id == 2:
                        hoop_results.append(parsed_box)
        return ball_results, hoop_results

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
        ball_results, hoop_results = self.predict(image)

        for results in [ball_results, hoop_results]:
            for result in results:
                bbox = result["bounding_box"]
                score = result["score"]

                image = self.draw_bbox(image, bbox, score)

        if output_path:
            cv2.imwrite(output_path, image)
        
        return image, ball_results, hoop_results

if __name__ == "__main__":
    # Example usage
    model_path = "./weights/dets_best.pt"
    yolo_det = yolo_basketball(model_path, device='cpu')

    input_image = "./data_test/images/img02.jpg"
    save_dir = "./outputs"
    os.makedirs(save_dir, exist_ok=True)
    output_image = os.path.join(save_dir, os.path.basename(input_image)[:-4] + ".jpg")

    # Predict and visualize results
    image = cv2.imread(input_image)

    yolo_det.draw_results(image, output_image)
