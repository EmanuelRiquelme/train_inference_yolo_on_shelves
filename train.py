from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='SKU-110K.yaml', epochs=100, imgsz=640,batch = 12)
