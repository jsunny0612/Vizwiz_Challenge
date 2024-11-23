import json
import os
import cv2

with open("./datasets/annotations.json","r") as f:
    file_name = json.load(f)

sorted_data_path = "./datasets/challenge/8900/"

if not os.path.exists(sorted_data_path):
    os.makedirs(sorted_data_path)

for i in file_name['images']:
    file_path = os.path.join("./datasets/images/", i)
    image = cv2.imread(file_path)
    file_write = os.path.join(sorted_data_path, i)
    cv2.imwrite(file_write, image)
