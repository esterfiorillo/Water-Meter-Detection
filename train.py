import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
import dataset
import skimage
from skimage.io import imread, imsave
import argparse
import os

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN

import engine
import matplotlib.pyplot as plt
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default='.', help='Path to the image files.')
parser.add_argument('--bbx_path', type=str, default='.', help='Path to the bounding box files.')
opts = parser.parse_args()


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, 
                                                             min_size=1024)
# one class is water meter, and the other is background
num_classes = 2
# get the input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace pre-trained head with our features head
# the head layer will classify the images based on our data input features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# use our dataset
train_dataset = dataset.WaterMeters('train', opts.img_path, opts.bbx_path)
test_dataset = dataset.WaterMeters('test', opts.img_path, opts.bbx_path)


# define training and validation data loaders
train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=engine.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    test_dataset, batch_size=2, shuffle=False, num_workers=4,
    collate_fn=engine.collate_fn)


# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

#10 epochs
num_epochs = 10


# images, targets, image_ids = next(iter(train_data_loader))
# images = list(image.to(device) for image in images)
# targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
# boxes = targets[0]['boxes'].cpu().numpy().astype(np.int)
# print(boxes)
# sample = images[0].permute(1,2,0).cpu().numpy()
# fig, ax = plt.subplots(1, 1, figsize=(16, 8))

# for box in boxes:
#     cv2.rectangle(sample,
#                     (box[0], box[1]),
#                     (box[2], box[3]),
#                     (220, 0, 0), 3)


# cv2.imwrite('sample.png', sample)

# initialize the Averager
loss_hist = engine.Averager()

iou_thresholds = [x for x in np.arange(0.5, 0.76, 0.05)]

train_loss = []
precision = []
for epoch in range(num_epochs):
    itr = 1
    train_loss_hist, end, start = engine.train(train_data_loader, lr_scheduler,
                                        model, optimizer, device,
                                        epoch, loss_hist, itr)
    valid_prec = engine.validate(data_loader_test, model, device, iou_thresholds)
    print(f"Took {(end-start)/60:.3f} minutes for epoch# {epoch} to train")
    print(f"Epoch #{epoch} Train loss: {train_loss_hist.value}")  
    print(f"Epoch #{epoch} Validation Precision: {valid_prec}")  
    train_loss.append(train_loss_hist.value)
    precision.append(valid_prec)
    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()

# plot and save the training loss
plt.figure()
plt.plot(train_loss, label='Training loss')
plt.legend()
plt.show()
plt.savefig('loss.png')

# plot and save the validation precision
plt.figure()
plt.plot(precision, label='Validation precision')
plt.legend()
plt.show()
plt.savefig('precision.png')


torch.save(model.state_dict(), 'fasterrcnn_resnet50_fpn.pth')

