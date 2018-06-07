import numpy as np
import os
from shutil import copyfile
from PIL import Image


def build_image_sets(val_ratio=0.1):
	max_height = 0
	max_width = 0
	source_dir = '/home/jinwoop/cs231n/project/MVD/training/images/'
	source_filename_list = os.listdir(source_dir)
	images_filename_list = []
	i = 1
	for filename in source_filename_list:
		print(i)
		i += 1
		if os.path.splitext(filename)[0] == '.DS_Store':
			continue
		im = Image.open(source_dir + os.path.splitext(filename)[0] + '.jpg')
		width, height = im.size
		# if width < 3264 and height < 3264:
		if width > 3263 and width < 3265 and height > 2000 and height < 2449:
			if height > max_height:
				max_height = height
			if width > max_width:
				max_width = width
			images_filename_list.append(os.path.splitext(filename)[0])
			# src = source_dir + os.path.splitext(filename)[0] + '.jpg'
			# dst = '/Users/shengli/Desktop/models/research/deeplab/datasets/mvd/mvd_raw/JPEGImages/' \
			# 	+ os.path.splitext(filename)[0] + '.jpg'
			# copyfile(src, dst)



	
	# images_dir = '/home/shengli/models/research/deeplab/datasets/mvd/mvd_raw/JPEGImages/'
	# images_filename_list = os.listdir(images_dir)

	# for filename in images_filename_list:
	# 	if os.path.splitext(filename)[0] == '.DS_Store':
	# 		continue
	# 	src = '/Users/shengli/Downloads/mapillary-vistas-dataset_public_v1.0/training/labels/' \
	# 		+ os.path.splitext(filename)[0] + '.png'
	# 	dst = '/Users/shengli/Desktop/models/research/deeplab/datasets/mvd/mvd_raw/SegmentationClass/' \
	# 		+ os.path.splitext(filename)[0] + '.png'
	# 	copyfile(src, dst)

	np.random.shuffle(images_filename_list)

	# split 10% val, 90% train
	val_images_filename_list = images_filename_list[:int(val_ratio*len(images_filename_list))]
	train_images_filename_list = images_filename_list[int(val_ratio*len(images_filename_list)):]

	num_train = 0
	num_val = 0

	with open("train.txt", "w") as f:
		for filename in train_images_filename_list:
			if os.path.splitext(filename)[0] == '.DS_Store':
				continue
			f.write(os.path.splitext(filename)[0] + "\n")
			num_train += 1

	with open("val.txt", "w") as f:
		for filename in val_images_filename_list:
			if os.path.splitext(filename)[0] == '.DS_Store':
				continue
			f.write(os.path.splitext(filename)[0] + "\n")
			num_val += 1

	filenames = ['train.txt', 'val.txt']
	with open('trainval.txt', 'w') as outfile:
		for fname in filenames:
			with open(fname) as infile:
				outfile.write(infile.read())

	print('Max dims of MVD is h = {}, w = {}'.format(max_height, max_width))
	print('num_train = {}'.format(num_train))
	print('num_val = {}'.format(num_val))
	print('num_trainval = {}'.format(num_train + num_val))


build_image_sets(val_ratio=0.1)