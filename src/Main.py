# coding=utf-8
import os
import json, time
import threading
import cv2
import numpy as np
from PIL import Image
from PyQt5 import QtGui
from PyQt5.QtWidgets import QLabel, QSizePolicy
from qt_thread_updater import get_updater
from src import config as co
from src.keypoint_det import yolo_keypoint
from src.basketball import yolo_basketball
from src.utils import draw_bbox, draw_keypoints, draw_keypoints_and_skeleton, visualize_basketball

def text_size(frame):
    frame_height, frame_width = frame.shape[:2]

    # Define the text and initial font settings
    text = "Computer"
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Calculate font scale and thickness based on frame size
    font_scale = frame_width / 600  # Adjust this divisor to scale text size (smaller values = larger text)
    font_thickness = max(2, int(frame_height / 200))  # Adjust divisor for thickness (ensure thickness >= 1)
    text_color = (0, 165, 255)  # Orange in BGR

    # Calculate text size with the dynamic scale and thickness
    (_, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

    # Define text position in the top-left corner with padding
    padding = int(0.02 * frame_height)  # 2% of the frame height as padding
    text_x = padding
    text_y = text_height + padding  # Position Y to account for text height

    return (text_x, text_y), font, font_scale, text_color, font_thickness

class Main:
    def __init__(self, MainGUI):
        self.MainGUI = MainGUI
        self.camera = None
        self.ret = False
        self.start_camera = True
        self.keypoint_det = yolo_keypoint("./weights/poses_best.pt", device='cpu')
        self.basketball_det = yolo_basketball("./weights/dets_best.pt", device='cpu')
        self.init_text_size()

    def img_cv_2_qt(self, img_cv):
        height, width, channel = img_cv.shape
        bytes_per_line = channel * width
        img_qt = QtGui.QImage(img_cv, width, height, bytes_per_line, QtGui.QImage.Format_RGB888).rgbSwapped()
        return QtGui.QPixmap.fromImage(img_qt)
    
    def init_devices(self, url_camera):
        self.camera = cv2.VideoCapture(url_camera) 
        self.ret, frame = self.camera.read() 
        if not self.ret:
            self.start_camera = False
            self.MainGUI.MessageBox_signal.emit("Có lỗi xảy ra ! \n Không tìm thấy camera/video", "error")
        else:
            self.start_camera = True
            (self.text_x, self.text_y), self.font, self.font_scale, self.text_color, self.font_thickness = text_size(frame)
    
    def auto_camera(self):
        url_camera = co.CAMERA_DEVICE
        self.init_devices(url_camera)
        while self.ret and self.start_camera:
            try:
                ret, frame = self.camera.read()
                self.ret = ret
                if self.ret and self.start_camera:
                    keypoints_results = self.keypoint_det.predict(frame.copy())
                    has_person = True if len(keypoints_results) > 0 else False
                    basketball_results, hoop_results = self.basketball_det.predict(frame.copy())
                    has_basketball = True if len(basketball_results) > 0 else False
                    has_hoop = True if len(hoop_results) > 0 else False
                    image = frame.copy()
                    basketball_boxes = []
                    hoop_boxes = []
                    if has_person:
                        for kp in keypoints_results:
                            bbox = kp["bounding_box"]
                            score = kp["score"]
                            keypoints = kp["keypoints"]
                            image = draw_bbox(image, bbox, score)
                            image = draw_keypoints_and_skeleton(image, keypoints)

                    if has_basketball:
                        for ball in basketball_results:
                            bbox = ball["bounding_box"]
                            score = ball["score"]
                            image = draw_bbox(image, bbox, score)
                            basketball_boxes.append(bbox)

                    if has_hoop:
                        for hoop in hoop_results:
                            bbox = hoop["bounding_box"]
                            score = hoop["score"]
                            image = draw_bbox(image, bbox, score)
                            hoop_boxes.append(bbox)

                    get_updater().call_latest(self.MainGUI.label_Image.setPixmap, self.img_cv_2_qt(image))
                    if (has_basketball or has_hoop):
                        # Get basketball and hoop boxes:
                        # Visualize output
                        image_view = visualize_basketball(frame.copy(), basketball_boxes, hoop_boxes)
                        get_updater().call_latest(self.MainGUI.label_View.setPixmap, self.img_cv_2_qt(image_view))
                        get_updater().call_latest(self.MainGUI.text_result.setText, "Pose")
                        get_updater().call_latest(self.MainGUI.text_result.setStyleSheet,"background-color: rgb(0, 255, 0);")
                    else:
                        get_updater().call_latest(self.MainGUI.text_result.setText, "None")
                        get_updater().call_latest(self.MainGUI.text_result.setStyleSheet,"background-color: rgb(0, 0, 255);")
                else:
                    break
            except Exception as e:
                print("Bug: ", e)
        self.close_camera()

    def auto_video(self, path_video):
        url_camera = path_video
        self.init_devices(url_camera)
        while self.ret and self.start_camera:
            try:
                ret, frame = self.camera.read()
                self.ret = ret
                if self.ret and self.start_camera:
                    keypoints_results = self.keypoint_det.predict(frame.copy())
                    has_person = True if len(keypoints_results) > 0 else False
                    basketball_results, hoop_results = self.basketball_det.predict(frame.copy())
                    has_basketball = True if len(basketball_results) > 0 else False
                    has_hoop = True if len(hoop_results) > 0 else False
                    image = frame.copy()
                    basketball_boxes = []
                    hoop_boxes = []
                    if has_person:
                        for kp in keypoints_results:
                            bbox = kp["bounding_box"]
                            score = kp["score"]
                            keypoints = kp["keypoints"]
                            image = draw_bbox(image, bbox, score)
                            image = draw_keypoints_and_skeleton(image, keypoints)

                    if has_basketball:
                        for ball in basketball_results:
                            bbox = ball["bounding_box"]
                            score = ball["score"]
                            image = draw_bbox(image, bbox, score)
                            basketball_boxes.append(bbox)

                    if has_hoop:
                        for hoop in hoop_results:
                            bbox = hoop["bounding_box"]
                            score = hoop["score"]
                            image = draw_bbox(image, bbox, score)
                            hoop_boxes.append(bbox)

                    get_updater().call_latest(self.MainGUI.label_Image.setPixmap, self.img_cv_2_qt(image))
                    if (has_basketball or has_hoop):
                        # Get basketball and hoop boxes:
                        # Visualize output
                        image_view = visualize_basketball(frame.copy(), basketball_boxes, hoop_boxes)
                        get_updater().call_latest(self.MainGUI.label_View.setPixmap, self.img_cv_2_qt(image_view))
                        get_updater().call_latest(self.MainGUI.text_result.setText, "Pose")
                        get_updater().call_latest(self.MainGUI.text_result.setStyleSheet,"background-color: rgb(0, 255, 0);")
                    else:
                        get_updater().call_latest(self.MainGUI.text_result.setText, "None")
                        get_updater().call_latest(self.MainGUI.text_result.setStyleSheet,"background-color: rgb(0, 0, 255);")
                else:
                    break
            except Exception as e:
                print("Bug: ", e)
        self.close_camera()

    def manual_image(self, image_file):
        frame = cv2.imread(image_file)
        # Call keypoin model
        keypoints_results = self.keypoint_det.predict(frame.copy())
        has_person = True if len(keypoints_results) > 0 else False
        basketball_results, hoop_results = self.basketball_det.predict(frame.copy())
        has_basketball = True if len(basketball_results) > 0 else False
        has_hoop = True if len(hoop_results) > 0 else False
        image = frame.copy()
        basketball_boxes = []
        hoop_boxes = []
        if has_person:
            for kp in keypoints_results:
                bbox = kp["bounding_box"]
                score = kp["score"]
                keypoints = kp["keypoints"]
                image = draw_bbox(image, bbox, score)
                image = draw_keypoints_and_skeleton(image, keypoints)

        if has_basketball:
            for ball in basketball_results:
                bbox = ball["bounding_box"]
                score = ball["score"]
                image = draw_bbox(image, bbox, score)
                basketball_boxes.append(bbox)

        if has_hoop:
            for hoop in hoop_results:
                bbox = hoop["bounding_box"]
                score = hoop["score"]
                image = draw_bbox(image, bbox, score)
                hoop_boxes.append(bbox)

        get_updater().call_latest(self.MainGUI.label_Image.setPixmap, self.img_cv_2_qt(image))
        if (has_basketball or has_hoop):
            # Get basketball and hoop boxes:
            # Visualize output
            image_view = visualize_basketball(frame.copy(), basketball_boxes, hoop_boxes)
            get_updater().call_latest(self.MainGUI.label_View.setPixmap, self.img_cv_2_qt(image_view))
            get_updater().call_latest(self.MainGUI.text_result.setText, "Pose")
            get_updater().call_latest(self.MainGUI.text_result.setStyleSheet,"background-color: rgb(0, 255, 0);")
        else:
            get_updater().call_latest(self.MainGUI.text_result.setText, "None")
            get_updater().call_latest(self.MainGUI.text_result.setStyleSheet,"background-color: rgb(0, 0, 255);")

    def close_camera(self):
        try:
            self.start_camera = False
            if self.ret:
                self.camera.release()
            self.camera = None
            self.ret = False
            
            time.sleep(1)
            self.MainGUI.label_Image.clear()

        except Exception as e:
                print("Bug: ", e)

    def init_text_size(self):
        self.text_x = 20
        self.text_y = 20
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.text_color = (0, 255, 0)
        self.font_thickness = 2