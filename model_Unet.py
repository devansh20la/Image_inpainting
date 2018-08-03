import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn


class par_conv(nn.Module):
	
	def __init__(self, **kwargs):
		super(par_conv, self).__init__()
		self.main = nn.Conv2d(**kwargs)
		self.mask_conv = nn.Conv2d(**kwargs)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
				nn.init.normal_(m.bias, 0.02)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.normal_(m.weight, 0.02)
				nn.init.normal_(m.bias, 0)

		for param in self.mask_conv.parameters():
			param.requires_grad=False

		self.mask_conv.weight.fill_(1)
		self.mask_conv.bias.fill_(0)


	def forward(self, x, mask):

		x = mask * x
		x = self.main(x)

		with torch.no_grad():
			mask = self.mask_conv(mask)

		mask = mask.expand_as(x)

		z_ind = (mask ==0 )
		nz_ind = (mask !=0 )

		x[nz_ind] = x[nz_ind] / mask[nz_ind]
		x[z_ind] = 0

		# exp_bias = self.main.bias.view(1,x.size()[1],1,1).expand_as(x)
		# x[nz_ind] = x[nz_ind] + exp_bias[nz_ind] * (temp_mask[nz_ind] - 1) / temp_mask[nz_ind]

		mask[mask != 0] = 1
		mask[mask == 0 ] = 0

		return x,mask


class down_module(nn.Module):

	def __init__(self, channels, kernel_size, padding, bn_flag=True):
		super(down_module, self).__init__()
		self.conv = par_conv(in_channels=channels[0],out_channels=channels[1],kernel_size=kernel_size,stride=2,padding=padding)
		if bn_flag:
			self.bn = nn.BatchNorm2d(channels[1])
		self.bn_flag = bn_flag
		self.act = nn.ReLU()

		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d):
				nn.init.normal_(m.weight, 0.02)
				nn.init.normal_(m.bias, 0)


	def forward(self, x, mask):

		x, mask = self.conv(x, mask)
		if self.bn_flag:
			x = self.bn(x)
		x = self.act(x)

		return x, mask

class up_module(nn.Module):

	def __init__(self, channels, act='leaky', bn_flag=True):
		super(up_module, self).__init__()
		self.upsample = nn.Upsample(scale_factor=2,mode='nearest')

		if act == 'tanh':
			self.act = nn.Tanh()
		else:
			self.act = nn.LeakyReLU(0.2)

		self.conv = par_conv(in_channels=channels[0],out_channels=channels[1],kernel_size=3,stride=1,padding=1)
		self.bn_flag = bn_flag
		if self.bn_flag:
			self.bn = nn.BatchNorm2d(channels[1])
		
		for m in self.modules():
			if isinstance(m, nn.BatchNorm2d):
				nn.init.normal_(m.weight, 0.02)
				nn.init.normal_(m.bias, 0)

	def forward(self, x, prev_x, mask, prev_mask):

		x = self.upsample(x)
		mask = self.upsample(mask)

		x = torch.cat([x, prev_x], dim=1)
		mask = torch.cat([mask, prev_mask], dim=1)

		x, mask = self.conv(x, mask)
		if self.bn_flag:
			x = self.bn(x)
		x = self.act(x)

		return x, mask

class gen(nn.Module):
	def __init__(self) :
		super(gen, self).__init__()
		self.layer1 = down_module((3,64), kernel_size=7, padding=3, bn_flag=False)
		self.layer2 = down_module((64,128), kernel_size=5, padding=2)
		self.layer3 = down_module((128,256), kernel_size=5, padding=2)
		self.layer4 = down_module((256,512), kernel_size=3, padding=1)
		self.layer5 = down_module((512,512), kernel_size=3, padding=1)
		self.layer6 = down_module((512,512), kernel_size=3, padding=1)
		self.layer7 = down_module((512,512), kernel_size=3, padding=1)
		self.layer8 = down_module((512,512), kernel_size=3, padding=1)

		self.layer9  = up_module((512 + 512,512))
		self.layer10 = up_module((512 + 512,512))
		self.layer11 = up_module((512 + 512,512))
		self.layer12 = up_module((512 + 512,512))
		self.layer13 = up_module((512 + 256,256))
		self.layer14 = up_module((256 + 128,128))
		self.layer15 = up_module((128 + 64,64))
		self.layer16 = up_module((64 + 3,3), bn_flag=False, act='tanh')



	def forward(self, x, mask):
		org_x = x
		org_mask = mask
		x1, mask1 = self.layer1(x,mask)
		x2, mask2 = self.layer2(x1,mask1)
		x3, mask3 = self.layer3(x2,mask2)
		x4, mask4 = self.layer4(x3,mask3)
		x5, mask5 = self.layer5(x4,mask4)
		x6, mask6 = self.layer6(x5,mask5)
		x7, mask7 = self.layer7(x6,mask6)
		x8, mask8 = self.layer8(x7,mask7)

		x, mask = self.layer9(x8, x7, mask8, mask7)
		x, mask = self.layer10(x, x6, mask, mask6)
		x, mask = self.layer11(x, x5, mask, mask5)
		x, mask = self.layer12(x, x4, mask, mask4)
		x, mask = self.layer13(x, x3, mask, mask3)
		x, mask = self.layer14(x, x2, mask, mask2)
		x, mask = self.layer15(x, x1, mask, mask1)
		x, _ = self.layer16(x, org_x, mask, org_mask)

		return x

########################################################
# Model for 25x256 image sizes
########################################################
class network(nn.Module):
	def __init__(self) :
		super(network, self).__init__()
		self.layer1 = down_module((3,64), kernel_size=7, padding=3, bn_flag=False)
		self.layer2 = down_module((64,128), kernel_size=5, padding=2)
		self.layer3 = down_module((128,256), kernel_size=5, padding=2)
		self.layer4 = down_module((256,512), kernel_size=3, padding=1)
		self.layer5 = down_module((512,512), kernel_size=3, padding=1)
		self.layer6 = down_module((512,512), kernel_size=3, padding=1)
		self.layer7 = down_module((512,512), kernel_size=3, padding=1)

		self.layer10 = up_module((512 + 512,512))
		self.layer11 = up_module((512 + 512,512))
		self.layer12 = up_module((512 + 512,512))
		self.layer13 = up_module((512 + 256,256))
		self.layer14 = up_module((256 + 128,128))
		self.layer15 = up_module((128 + 64,64))
		self.layer16 = up_module((64 + 3,3), bn_flag=False, act='tanh')



	def forward(self, x, mask):
		org_x = x
		org_mask = mask
		x1, mask1 = self.layer1(x,mask)
		x2, mask2 = self.layer2(x1,mask1)
		x3, mask3 = self.layer3(x2,mask2)
		x4, mask4 = self.layer4(x3,mask3)
		x5, mask5 = self.layer5(x4,mask4)
		x6, mask6 = self.layer6(x5,mask5)
		x7, mask7 = self.layer7(x6,mask6)

		x, mask = self.layer10(x7, x6, mask7, mask6)
		x, mask = self.layer11(x, x5, mask, mask5)
		x, mask = self.layer12(x, x4, mask, mask4)
		x, mask = self.layer13(x, x3, mask, mask3)
		x, mask = self.layer14(x, x2, mask, mask2)
		x, mask = self.layer15(x, x1, mask, mask1)
		x, _ = self.layer16(x, org_x, mask, org_mask)

		return x

from torchvision.models import *


#########################################################
# VGG model to extract features for perceptual Loss
#########################################################
class vgg_ext(nn.Module):
	def __init__(self):
		super(vgg_ext, self).__init__()
		m = vgg.vgg16(pretrained=True)
		self.pool1 = nn.Sequential(*m.features[:5])
		self.pool2 = nn.Sequential(*m.features[5:10])
		self.pool3 = nn.Sequential(*m.features[10:17])

	def forward(self, x):
		x1 = self.pool1(x)
		x2 = self.pool2(x1)
		x3 = self.pool3(x2)
		return x1,x2,x3






# 
# m = vgg_ext()
# print(m)

# A = torch.randn(1,3,256,256)
# mask = torch.randn(1,3,256,256).fill_(1)
# m = gen_256()
# print(m)
# out = m(A,mask)
# print(out.size())