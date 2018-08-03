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

resize = transforms.Resize(300, interpolation=2)
to_tensor = transforms.ToTensor()
center_crop = transforms.CenterCrop(256)
normalize = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])

class imageandlabel(Dataset):

	def __init__ (self,root_dir,training):
		self.root_dir = root_dir
		self.training = training
		self.files = [fn for fn in os.listdir(root_dir) 
		if (('_segmentation' not in fn and '_mask' not in fn) 
			and (fn.endswith('.jpg') or fn.endswith('.png') or fn.endswith('.bmp')))]

	def __len__(self):
		return len(self.files)

	def mask_image(self,img_input,img_mask):

		img_mask = np.array(img_mask)
		img_mask = np.concatenate((np.expand_dims(img_mask,axis=2),np.expand_dims(img_mask,axis=2),np.expand_dims(img_mask,axis=2)),axis=2)
		img_mask = Image.fromarray(img_mask,mode='RGB')

		return Image.composite(img_input,img_mask,img_mask.convert('1'))


	def __getitem__(self,idx):
		imgname = os.path.join(self.root_dir,self.files[idx])
		image_input = Image.open(imgname)

		try:
			image_seg_target = Image.open(imgname.split('.')[0] + '_mask.png')
			image_seg_target = image_seg_target.convert('1')
		except:
			image_seg_target = Image.new(mode='1', size=image_input.size, color=0)

		image_input = resize(image_input)
		image_seg_target = resize(image_seg_target)

		image_input = center_crop(image_input)
		image_seg_target = center_crop(image_seg_target)

		image_seg_target = ImageChops.invert(image_seg_target)
		image_missed_input = self.mask_image(image_input, image_seg_target)
		image_missed_input = np.array(image_missed_input)

		for i in range(image_missed_input.shape[0]):
			for j in range(image_missed_input.shape[1]):
				#print(image_missed_input[i,j,:])
				if (image_missed_input[i,j,:] == [0,0,0]).all():
					image_missed_input[i,j,:] = [255,255,255]

		image_missed_input = Image.fromarray(image_missed_input)
		image_seg_target = to_tensor(image_seg_target)
		
		image_missed_input = to_tensor(image_missed_input)
		image_missed_input = normalize(image_missed_input)

		image_input = to_tensor(image_input)
		image_input = normalize(image_input)

		image_seg_target = image_seg_target.expand_as(image_missed_input)
		
		sample = {'input': image_missed_input, 'mask':image_seg_target, 'target':image_input, 'name':self.files[idx]}
		return sample


# m = imageandlabel('data/val/',1)
# dataloader = torch.utils.data.DataLoader(m, batch_size=1, num_workers=0, shuffle=False)

# for i in dataloader:
# 	data = i['mask']
# 	from torchvision import utils as vutils
# 	vutils.save_image(data,'mask.png')
	
# 	import time
# 	time.sleep(1)

