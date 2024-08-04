from ultralytics import YOLO 

model = YOLO('models/yolov8_trained_best_model.pt')

results = model.predict('input videos/1.mp4',save=True)
print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)
