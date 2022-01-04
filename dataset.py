import numpy as np
import torch
import skimage
from skimage.io import imread, imsave
from torch.utils.data import Dataset
import os
import cv2

def get_dataset(mode, imgs_path):
	imgs = os.listdir(imgs_path + '/' + str(mode))
	bbxs = [i[:-3]+'txt' for i in imgs]

	return imgs, bbxs

class WaterMeters(Dataset):

	def __init__ (self, mode, imgs_path, bbx_path, transform=None):

		self.mode = mode
		self.imgs_path = imgs_path
		self.bbx_path = bbx_path
		self.imgs, self.bbxs = get_dataset(self.mode, self.imgs_path)
		self.len = len(self.imgs)
		self.transform = transform

	def __getitem__(self, idx):
		bbx = []
		image = cv2.imread(self.imgs_path + '/' + self.mode + '/' + self.imgs[idx], cv2.IMREAD_COLOR)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
		image = image/255
		image = np.transpose(image, (2, 0, 1)).astype(np.float)
		image = torch.tensor(image, dtype=torch.float)

		bbx2 = open(self.bbx_path + '/' + self.mode + '/' + self.bbxs[idx], "r").read()
		bbx2 = bbx2.split(' ')
		bbx2 = list(map(int, bbx2))
		bbx.append(bbx2)

		num_box = len(bbx)

		
		if num_box>0:
			bbx = torch.as_tensor(bbx, dtype=torch.float32)
		else:
			bbx = torch.zeros((0, 4), dtype=torch.float32)

		labels = torch.ones((num_box,), dtype=torch.int64)

		image_id = torch.tensor([idx])
		area = (bbx[:,3] - bbx[:,1])*(bbx[:,2] - bbx[:,0])
		iscrowd = torch.zeros((num_box,), dtype=torch.int64)
		target = {}
		target["boxes"] = bbx
		target["labels"] = labels
		target["image_id"] = image_id
		target["area"] = area
		target["iscrowd"] = iscrowd
	

		if self.transform is not None:
			image, target = self.transform(image, target)

		

		return image, target, image_id

	def __len__(self):
		return self.len




