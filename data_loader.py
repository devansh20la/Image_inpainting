import os
from PIL import Image
import random
from torchvision import transforms
import torch
from torch.utils.data import Dataset
import random
from torchvision import transforms
from torchvision.transforms import functional as F
import numbers 
import PIL.ImageOps as ImageOps
import PIL.ImageChops as ImageChops
import numpy as np 
import time

# My random crop for multiple images 
class RandomCrop(object):
	def __init__(self, size, padding=0):
		if isinstance(size, numbers.Number):
			self.size = (int(size), int(size))
		else:
			self.size = size
		self.padding = padding

	@staticmethod
	def get_params(img, output_size):
		w, h = img.size
		th, tw = output_size
		if w == tw and h == th:
			return 0, 0, h, w

		i = random.randint(0, h - th)
		j = random.randint(0, w - tw)
		return i, j, th, tw

	def __call__(self, img1,img2):

		if self.padding > 0:
			img1 = F.pad(img1, self.padding)
			img2 = F.pad(img2, self.padding)

		assert(img1.size == img2.size)

		i, j, h, w = self.get_params(img1, self.size)

		return F.crop(img1, i, j, h, w), F.crop(img2, i, j, h, w)

	def __repr__(self):
		return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

resize = transforms.Resize(300, interpolation=2)
random_crop = RandomCrop(256)
to_tensor = transforms.ToTensor()
center_crop = transforms.CenterCrop(256)
normalize = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])

#########################################
# Masking the input image with segmentation mask
#########################################
def mask_image(img_input,img_mask):

	img_mask = np.array(img_mask)
	img_mask = np.concatenate((np.expand_dims(img_mask,axis=2),np.expand_dims(img_mask,axis=2),np.expand_dims(img_mask,axis=2)),axis=2)
	img_mask = Image.fromarray(img_mask,mode='RGB')

	return Image.composite(img_input,img_mask,img_mask.convert('1'))



class imageandlabel(Dataset):

	def __init__ (self,root_dir,training):
		self.root_dir = root_dir
		self.training = training
		self.files = [fn for fn in os.listdir(root_dir) if (('target' in fn) and fn.endswith('.png'))]

	def __len__(self):
		return len(self.files)

	def __getitem__(self,idx):
		imgname = os.path.join(self.root_dir,self.files[idx])
		image_input = Image.open(imgname)

		if self.training == True:
			r = random.random()

			####################################################
			# I generate three kinds of masking procedures:
			# 1) Segmentation mask of the image- I use to help model learn
			# 2) Mask for hairs(occlusion) on the images
			# 3) Randomly generate mask 
			####################################################
			if r < 0.3:
				try:
					image_seg_target = Image.open(imgname.split('_target.png')[0] + '_segmentation.png')
					image_seg_target = image_seg_target.convert('1')
					image_seg_target = ImageChops.invert(image_seg_target)
				except:
					image_seg_target = Image.open(imgname.split('_target.png')[0] + '_mask.png')
					image_seg_target = image_seg_target.convert('1')
			elif r > 0.7:
				image_seg_target = torch.randn(1,32,32)
				image_seg_target = torch.nn.functional.dropout(image_seg_target,p=0.6,training=True)
				image_seg_target[image_seg_target != 0] = 1
				to_pil = transforms.ToPILImage()
				image_seg_target = to_pil(image_seg_target).convert('1')
			else:
				image_seg_target = Image.open(imgname.split('_target.png')[0] + '_mask.png')
				image_seg_target = image_seg_target.convert('1')
		else:
			image_seg_target = Image.open(imgname.split('_target.png')[0] + '_mask.png')
			image_seg_target = image_seg_target.convert('1')
			
		image_input = resize(image_input)
		image_seg_target = resize(image_seg_target)

		###################################
		# Data Augmentation Blocks you can add more if you want to 
		###################################
		if self.training == True:
			if random.random() < 0.5:
				image_input = image_input.transpose(Image.FLIP_TOP_BOTTOM)
				image_seg_target = image_seg_target.transpose(Image.FLIP_TOP_BOTTOM)

			if random.random() < 0.5:
				image_input = image_input.transpose(Image.FLIP_LEFT_RIGHT)
				image_seg_target = image_seg_target.transpose(Image.FLIP_LEFT_RIGHT)

			image_input, image_seg_target = random_crop(image_input,image_seg_target)        
		else:
			image_input = center_crop(image_input)
			image_seg_target = center_crop(image_seg_target)

		#image_target = self.mask_image(image_input.copy(), image_seg_target.copy())
		image_seg_target = ImageChops.invert(image_seg_target)
		image_missed_input = self.mask_image(image_input, image_seg_target)
		image_missed_input = np.array(image_missed_input)

		##########################################
		# Change masked region to white. This helps model learn better.
		#########################################
		for i in range(image_missed_input.shape[0]):
			for j in range(image_missed_input.shape[1]):
				if (image_missed_input[i,j,:] == [0,0,0]).all():
					image_missed_input[i,j,:] = [255,255,255]

		image_missed_input = Image.fromarray(image_missed_input)

		##############################################
		# Converting to tensor and normalization
		###############################################
		image_seg_target = to_tensor(image_seg_target)
		image_missed_input = to_tensor(image_missed_input)
		image_missed_input = normalize(image_missed_input)

		image_input = to_tensor(image_input)
		image_input = normalize(image_input)

		image_seg_target = image_seg_target.expand_as(image_missed_input)
		
		sample = {'input': image_missed_input, 'mask':image_seg_target, 'target':image_input, 'name':self.files[idx]}
		return sample


if __name__ == '__main__':
	main()

def main():
	m = imageandlabel('data/val/',1)
	dataloader = torch.utils.data.DataLoader(m, batch_size=1, num_workers=0, shuffle=False)

	for i in dataloader:
		data = i['mask']


