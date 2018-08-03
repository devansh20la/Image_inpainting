import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torchvision import models,transforms
import os
import copy
from data_loader import imageandlabel
import pandas as pd
import random
import shutil
import argparse
import pickle
import numpy as np
from torchvision import utils as vutils 
from logger_imp import viz_logger
import model_Unet as model  

parser = argparse.ArgumentParser(description='PyTorch Skin Lesion Training')
parser.add_argument('--lr','--learning_rate',type=float,default=1e-4,help='learning rate')
parser.add_argument('--cp','--checkpoint',type=str,default='')
parser.add_argument('--wd','--weightdecay',type=float,default=0)
parser.add_argument('--ms','--manualSeed',type=int,default=123)
parser.add_argument('--bs','--batch_size',type=int,default=2)
parser.add_argument('--ep','--epochs',type=int,default=500)
parser.add_argument('--n','--name',type=str,default='main')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
random.seed(args.ms)
torch.manual_seed(args.ms)
np.random.seed(args.ms)

if use_cuda:
   torch.cuda.manual_seed_all(args.ms)

def main():

	print ("....Initializing data sampler.....")
	data_dir = 'data/'

	dsets = {'train': imageandlabel(data_dir + 'train/', training=True), 'val': imageandlabel(data_dir + 'val/', training=False)}

	train_loader = torch.utils.data.DataLoader(dsets['train'], batch_size=args.bs, num_workers=10, shuffle=True)
	val_loader = torch.utils.data.DataLoader(dsets['val'], batch_size=25, num_workers=5, shuffle=False)
	

	print ("....Loading Model.....")

	net = nn.DataParallel(model.network())
	vgg = nn.DataParallel(model.vgg_ext())
	vgg.eval()

	if use_cuda:
		net = net.cuda()
		vgg =vgg.cuda()

	MSEloss = nn.MSELoss()
	L1loss = nn.L1Loss()

	if use_cuda:
		MSEloss = MSEloss.cuda()
		L1loss = L1loss.cuda()

	opti = optim.Adam([param for param in net.parameters() if param.requires_grad==True], lr=args.glr, betas = (0.5,0.99))

	if args.cp:
		state = torch.load(args.cp)
		net.load_state_dict(state['net'])

		opti.load_state_dict(state['optimizer'])

		logger = state['logger']
		logger.viz.env = args.n
		start_epoch = state['epoch']

		print("checkpoint Loaded Epoch= {0}".format(start_epoch))
		del state
		start_epoch+=1

	else:
		logger = viz_logger(env=args.n)
		logger.add_scalar(name='train_loss',title='Train Loss',xlabel='epochs',ylabel='Loss')
		logger.add_image(name='recon_img',title='Reconstructed Image')
		logger.add_image(name='input_img',title='Input image')
		logger.add_image(name='target_img',title='Target image')
		start_epoch = 0

	for epoch in range(start_epoch,args.ep):

		train_loss = []

		print('\nEpoch: [%d | %d]' % (epoch, args.ep))

		for batch_idx, inp_data in enumerate(train_loader,1):

			mis_inputs = inp_data['input']
			target_img = inp_data['target']
			img_mask = inp_data['mask']
			
			if use_cuda:
				mis_inputs, target_img, img_mask = mis_inputs.cuda(), target_img.cuda(), img_mask.cuda()

			
			with torch.set_grad_enabled(True):
				#...............Train Generator..............
				# ........Calculate reconstruction loss.......
				opti.zero_grad()
				outputs = net(mis_inputs, img_mask)

			p_loss1, style_loss1 = percept_style_loss(vgg, L1loss, outputs, target_img)

			comp = img_mask*target_img + (1-img_mask)*outputs

			p_loss2, style_loss2 = percept_style_loss(vgg, L1loss, comp, target_img)
			mse_loss = MSEloss(outputs,target_img)

			valid = L1loss(img_mask*outputs, img_mask*target_img)
			hole = L1loss((1-img_mask)*outputs, (1-img_mask)*target_img)
			temp = comp.clone()
			temp = temp.detach()

			tvd = L1loss(comp[:, :, :, :-1],temp[:, :, :, 1:]) + L1loss(comp[:, :, :-1, :],temp[:, :, 1:, :])
			
			#.........Train Generator with Recon_Loss + Adver_Loss
			Gen_loss = 0.02*(p_loss1 + p_loss2) + mse_loss + valid + 6*hole + 0.1*tvd + 120*(style_loss1 + style_loss2)
			Gen_loss.backward()
			opti.step()

			train_loss.append(Gen_loss.cpu().item())
		
		val(val_loader, net, L1loss, MSEloss, logger, use_cuda)

		logger.update_summary(train_loss = torch.Tensor([np.mean(train_loss)]))

		print ('Gen_Loss = {0}'.format(np.mean(train_loss)))
		
		state = {'epoch': epoch,'net': net.state_dict(), 'optimizer': opti.state_dict(), 'logger':logger}

		torch.save(state,'checkpoints/{0}/checkpoint{1}.pth.tar'.format(args.n,epoch))


def percept_style_loss(vgg, L1loss, x, target):

	with torch.no_grad():
		x1,x2,x3 = vgg(target)
		x1 = x1.detach()
		x2 = x2.detach()
		x3 = x3.detach()


	with torch.set_grad_enabled(True):
		x11,x22,x33 = vgg(x)

	perp_loss = L1loss(x11,x1) + L1loss(x22,x2) + L1loss(x33,x3)

	x1, x2, x3 = gram_matrix(x1), gram_matrix(x2),gram_matrix(x3)
	x11, x22, x33 = gram_matrix(x11), gram_matrix(x22),gram_matrix(x33)

	style_loss = L1loss(x11,x1) + L1loss(x22,x2) + L1loss(x33,x3)

	return perp_loss, style_loss

def gram_matrix(input):
    a, b, c, d = input.size()  
    # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

def val(val_loader, net, L1loss, MSEloss, logger, use_cuda):

	save_val_loss = []

	net.train()

	for batch_idx, inp_data in enumerate(val_loader,1):
		mis_inputs = inp_data['input']
		target_img = inp_data['target']
		mask = inp_data['mask']

		if use_cuda:
			mis_inputs, target_img, mask  = mis_inputs.cuda(), target_img.cuda(), mask.cuda()


		with torch.set_grad_enabled(False):

			#..........Generate fake data from Generator..........
			outputs = net(mis_inputs, mask)

			loss1 = MSEloss(outputs,target_img)

			loss2 = L1loss(mask*outputs, mask*target_img)
			loss3 = L1loss((1-mask)*outputs, (1-mask)*target_img)

		save_val_loss.append(loss1.item() + loss2.item() + loss3.item())

	logger.update_summary(gen_val_loss = torch.Tensor([np.mean(save_val_loss)]), 
		input_img = mis_inputs, 
		recon_img = outputs, 
		target_img= target_img)

if __name__ == '__main__':
	main()
