import numpy as np
import cv2
from skimage.measure import label, regionprops
from skimage.io import imread
import os
import shutil
import argparse

def get_bbxs(path):
	masks = os.listdir(path + '/masks')

	exist_bbx = os.path.exists(path + '/bbxs')
	print(exist_bbx)
	if not exist_bbx:
		os.mkdir(path + '/bbxs')

	for i in masks:

		im = cv2.imread(path + '/masks/' + i)

		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (5, 5), 0)
		thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]

		contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		for c in contours:
		    rect = cv2.boundingRect(c)
		    x,y,w,h = rect
		    print(x,y,x+w,y+h)

		    f = open(path + '/bbxs/' + i[:-3] + 'txt', "a")
		    f.write(str(x) + ' ' + str(y) + ' ' + str(x+w) + ' ' + str(y+h))
		    f.close()


def train_test_val_split(output_path, path):
	total = os.listdir(path + '/images')
	num_total = len(total)
	train = int(0.7*num_total)
	test = int(train+0.15*num_total)

	train_split = total[0:train]
	test_split = total[train:test]
	val_split = total[test:]

	exist_path = os.path.exists(output_path + '/images')
	if not exist_path:
		os.mkdir(output_path + '/images')
	exist_path = os.path.exists(output_path + '/bbxs')
	if not exist_path:
		os.mkdir(output_path + '/bbxs')
	exist_path = os.path.exists(output_path + '/images/train')
	if not exist_path:
		os.mkdir(output_path + '/images/train')
	exist_path = os.path.exists(output_path + '/images/test')
	if not exist_path:
		os.mkdir(output_path + '/images/test')
	exist_path = os.path.exists(output_path + '/images/val')
	if not exist_path:
		os.mkdir(output_path + '/images/val')
	exist_path = os.path.exists(output_path + '/bbxs/train')
	if not exist_path:
		os.mkdir(output_path + '/bbxs/train')
	exist_path = os.path.exists(output_path + '/bbxs/test')
	if not exist_path:
		os.mkdir(output_path + '/bbxs/test')
	exist_path = os.path.exists(output_path + '/bbxs/val')
	if not exist_path:
		os.mkdir(output_path + '/bbxs/val')

	for i in train_split:
		shutil.copyfile(path + '/images/'+i, output_path + '/images/train/' + i)

	for i in test_split:
		shutil.copyfile(path + '/images/'+i, output_path + '/images/test/' + i)

	for i in val_split:
		shutil.copyfile(path + '/images/'+i, output_path + '/images/val/' + i)


	for i in train_split:
		shutil.copyfile(path + '/bbxs/'+i[:-3] + 'txt', output_path + '/bbxs/train/'+i[:-3] + 'txt')

	for i in test_split:
		shutil.copyfile(path + '/bbxs/'+i[:-3] + 'txt', output_path + '/bbxs/test/'+i[:-3] + 'txt')

	for i in val_split:
		shutil.copyfile(path + '/bbxs/'+i[:-3] + 'txt', output_path + '/bbxs/val/'+i[:-3] + 'txt')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='.', help='Path to the dataset file.')
parser.add_argument('--output_path', type=str, default='.', help='Path to the dataset file.')
opts = parser.parse_args()

get_bbxs(opts.dataset_path)
train_test_val_split(opts.output_path, opts.dataset_path)