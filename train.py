from ultralytics import YOLO

if __name__ == "__main__":

    # Load a model
    model = YOLO('./weights/yolo11s.pt')

    # Train the model
    results = model.train(data='./datasets/data.yaml', epochs=100, imgsz=640, batch=32, device="cuda")