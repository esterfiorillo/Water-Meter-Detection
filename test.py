import pandas as pd
import numpy as np
import cv2
import os
import re
import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse


device = torch.device('cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default='.', help='Path to the image files.')
parser.add_argument('--model', type=str, default='.', help='Path to the bounding box files.')
parser.add_argument('--output_path', type=str, default='.', help='Path to the image files.')
opts = parser.parse_args()

DIR_TEST = opts.img_path
test_images = os.listdir(DIR_TEST)
print(f"Test instances: {len(test_images)}")


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  
# get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# fine-tune head
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

os.makedirs(opts.output_path, exist_ok=True)
model.load_state_dict(torch.load(opts.model))
model.to(device)


detection_threshold = 0.9
img_num = 0
results = []
model.eval()
with torch.no_grad():
    for i, image in tqdm(enumerate(test_images), total=len(test_images)):

        orig_image = cv2.imread(f"{DIR_TEST}/{test_images[i]}", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image = image/255.0
        image = np.transpose(image, (2, 0, 1)).astype(np.float)
        image = torch.tensor(image, dtype=torch.float)
        image = torch.unsqueeze(image, 0)

        cpu_device = torch.device("cpu")

        outputs = model(image)
        
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        if len(outputs[0]['boxes']) != 0:
            for counter in range(len(outputs[0]['boxes'])):
                boxes = outputs[0]['boxes'].data.cpu().numpy()
                scores = outputs[0]['scores'].data.cpu().numpy()
                boxes = boxes[scores >= detection_threshold].astype(np.int32)
                draw_boxes = boxes.copy()
                boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
                boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            
            for box in draw_boxes:
                cv2.rectangle(orig_image,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            (0, 0, 255), 3)
        
            plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.savefig(f"{opts.output_path}/{test_images[i]}")
            plt.close()
                
