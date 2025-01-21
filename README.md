# sport_analysis_project
AI Sports Motion Analysis Application
This project is an AI application created to analyze motion and form during sport activities. Using OpenCV, PyTorch, and PyQT5, the application can process data in real time and generate visualizations. In addition to detecting and evaluating movements, it can also provide feedback to enhance athletic performance. By utilizing person detection and key point detection model, the application offers valuable insights for athletes. 

# How to reproduce my code

## 1.Training
About dataset: https://universe.roboflow.com/cv-8scak/cv-cnfd4
There are 2566 train images, 733 valid images, 367 test images.

Data Structure
```
.
├── datasets/
│   ├── train/
│   │   ├── images/
│   │   ├── labels/
```

### 1. `datasets/train/images`
This folder contains all the training images used to train the YOLO model. Each image file is named using a consistent naming convention (e.g., `image_001.jpg`, `image_002.jpg`, etc.).

- **Supported Formats**: `.jpg`, `.png`
- **Example Path**: `datasets/train/images/image_001.jpg`

### 2. `datasets/train/labels`
This folder contains the label files corresponding to the images in the `images` folder. Each label file is a text file with the same base name as the corresponding image (e.g., `image_001.txt` for `image_001.jpg`).

#### Label File Format
Each label file contains annotations for the objects in the image. The format for each line in the label file is as follows:
```
<class_id> <x_center> <y_center> <width> <height>
```
- `<class_id>`: Integer ID representing the class of the object (e.g., `0`:Basketball, `1`:person, `2`:rim, etc.).
- `<x_center>`: Normalized x-coordinate of the object's center (value between 0 and 1).
- `<y_center>`: Normalized y-coordinate of the object's center (value between 0 and 1).
- `<width>`: Normalized width of the object (value between 0 and 1).
- `<height>`: Normalized height of the object (value between 0 and 1).

#### Example Label File (`image_001.txt`):
```
0 0.5 0.5 0.2 0.3
1 0.7 0.8 0.1 0.1
```
This example contains two objects:
1. An object of class `0` centered at `(0.5, 0.5)` with a width of `0.2` and height of `0.3`.
2. An object of class `1` centered at `(0.7, 0.8)` with a width of `0.1` and height of `0.1`.

### Example
```
datasets/
├── train/
│   ├── images/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── image_003.jpg
│   ├── labels/
│       ├── image_001.txt
│       ├── image_002.txt
│       └── image_003.txt
```

### Train command
```python train.py```
You can modify hyperparameters like `epochs`, `imgsz` and `batch` to optimize your training process. 