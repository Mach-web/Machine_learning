from ultralytics import YOLO

model = YOLO("yolov8n.pt")
# use camera
results = model(source=0,show = True, conf = 0.4, save = True)
# use image in same folder
# results = model(source="bus.jpg",show=True, conf = 0.4,save = False)