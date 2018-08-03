from tqdm import tqdm 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from data_loader import imageandlabel
import random
import shutil
import argparse
from torchvision import models,transforms as models, transforms
from torchvision import utils as vutils 
from logger_imp import viz_logger
import model_Unet as model  

def main():

	parser = argparse.ArgumentParser(description='PyTorch Skin Lesion Testing')
	parser.add_argument('--cp','--checkpoint',type=str,default='')
	parser.add_argument('--bs','--batch_size',type=int,default=2)

	args = parser.parse_args()
	use_cuda = torch.cuda.is_available()
	print ("....Initializing data sampler.....")
	data_dir = 'data/'

	dsets = imageandlabel(data_dir + 'val/', training=False)
	loader = torch.utils.data.DataLoader(dsets, batch_size=args.bs, num_workers=10)
	
	print ("....Loading Model.....")

	Gen = nn.DataParallel(model.gen_256())
	Gen.eval()

	if use_cuda:
		Gen = Gen.cuda()

	state = torch.load(args.cp)
	Gen.load_state_dict(state['Gen'])

	for inp_data in tqdm(loader):

		mis_inputs = inp_data['input']
		mask = inp_data['mask']
		name = inp_data['name']
			
		if use_cuda:
			mis_inputs, mask = mis_inputs.cuda(), mask.cuda()

		with torch.set_grad_enabled(False):
			#...............Train Generator..............
			# ........Calculate reconstruction loss.......
			outputs = Gen(mis_inputs, mask)
			comp = mask*mis_inputs + (1-mask)*outputs 

		for tensor,ten_name in zip(comp,name):
			vutils.save_image(tensor,'results/' + ten_name.split('.')[0] + '_removed.png', normalize=True)

if __name__ == '__main__':
	main()
