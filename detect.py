import cv2
from utils.hubconf import custom
from utils.plots import plot_one_box
import random


path_to_video = 'video.mp4'
path_to_model = 'yolov7.pt'

class_names = open('class.txt').read().splitlines()
colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

model = custom(path_or_model=path_to_model, gpu=True)
cap = cv2.VideoCapture(path_to_video)
while True:
	success, img = cap.read()
	if not success:
		break
	bbox_list = []
	results = model(img)
	# Bounding Box
	box = results.pandas().xyxy[0]
	class_list = box['class'].to_list()
	for i in box.index:
		xmin, ymin, xmax, ymax, conf, class_id = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), \
            int(box['ymax'][i]), box['confidence'][0], box['class'][i]
		bbox_list.append([xmin, ymin, xmax, ymax])
		bbox = [xmin, ymin, xmax, ymax]
		plot_one_box(bbox, img, label=class_names[class_id], color=colors[class_id], line_thickness=2)

	cv2.imshow('img', img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		cv2.destroyAllWindows()
		break
