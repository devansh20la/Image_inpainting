from visdom import Visdom
import numpy as np
import math
import os.path
import getpass
import time
from sys import platform as _platform
from six.moves import urllib
import pickle
import torch

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


#####################################################
# Logger class for histogram 
#####################################################
class histogram():

	def __init__(self,winname,title,bins=30):
		self.data = []
		self.winname = winname
		self.opts = dict(title=title,numbins=bins)
		self.fig = plt.figure()
		self.ax = self.fig.add_subplot(111, projection='3d')
		# self.ax.view_init(40, 90)
		self.colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())[5:35]

	def update(self,tensor,viz):
		s = len(self.data)
		if s > 30:
			self.data.pop(0)
		self.data.append(tensor)
		s+=1

		for tensor,c, z in zip(self.data,self.colors[:s], [i for i in range(0,100,int(100/s))]):
			hist, bins = np.histogram(tensor.numpy(), bins=self.opts['numbins'])
			xs = (bins[:-1] + bins[1:])/2
			self.ax.bar(xs, hist, zs=z, zdir='y', color=c, ec=c, alpha=0.8)

		viz.matplot(self.fig,win=self.winname)

#####################################################
# Logger class for Image 
#####################################################
class Image():
	def __init__(self, winname, title):
		self.winname = winname
		self.data = None
		self.opts = dict(title=title)

	def normalize(self,img,min,max):
		img.clamp_(min=min, max=max)
		img.add_(-min).div_(max - min + 1e-5)
		return img

	def update(self,value,viz):
		for t in value:
			self.normalize(t,t.min(),t.max())
		self.data = value
		viz.images(self.data, opts=self.opts, win=self.winname)


#####################################################
# Logger class for Scalar values 
#####################################################
class Scalar():

	def __init__(self, winname, title, xlabel, ylabel):
		self.winname = winname
		self.data = None
		self.opts = dict(title=title,xlabel=xlabel,ylabel=ylabel)

	def update(self,value,viz):

		if self.data is not None:
			self.data = torch.cat([self.data,value],dim=0)
		else:
			self.data = value

		viz.line(X =torch.arange(self.data.size()[0]), Y = self.data, opts=self.opts, win=self.winname)


#####################################################
# Logger class
#####################################################
class viz_logger():

	def __init__(self,env):
		self.viz = Visdom(env =env)
		self.states = {}

	def add_scalar(self, name, title, xlabel, ylabel):
		self.states[name] = Scalar(winname=name,title=title,xlabel=xlabel,ylabel=ylabel)
	
	def add_image(self, name, title):
		self.states[name] = Image(winname=name,title=title)

	def add_histogram(self,name,title):
		self.states[name] = histogram(winname=name,title=title,bins=30)

	def update_summary(self,**kwargs):
		for key,value in kwargs.items():
			self.states[key].update(value,self.viz)

	def __getstate__(self):
		state = self.__dict__.copy()
		del state['viz']
		return state

	def __setstate__(self,state):
		self.__dict__.update(state)
		self.viz = Visdom()

# logger = viz_logger()

# logger.add_scalar(name='train_loss',title='Training Loss',xlabel='epochs',ylabel='Loss')
# # # # # logger.add_image(name='in_img',title='Input Image')
# # # # # # logger.add_histogram(name='weight1',title='weights')

# logger.update_summary(train_loss=torch.Tensor([1]))
# logger.update_summary(train_loss=torch.Tensor([1]))

# import torch
# state = {'logger':logger,'a':1}
# torch.save(state,'save.pth')
# state = torch.load('save.pth')
# logger = state['logger']
# # pickle.dump(logger,open('save.pkl','wb'))

# # logger = pickle.load(open( "save.pkl", "rb" ) )
# logger.update_summary(train_loss=torch.Tensor([100]))
# logger.update_summary(train_loss=torch.Tensor([1]),weight1=torch.randn(100))
# logger.update_summary(train_loss=torch.Tensor([1]),weight1=torch.randn(100))
# logger.update_summary(train_loss=torch.Tensor([1]),weight1=torch.randn(100))
# logger.update_summary(train_loss=torch.Tensor([1]),weight1=torch.randn(100))
# logger.update_summary(train_loss=torch.Tensor([1]),weight1=torch.randn(100))

# logger.add_histogram('')
# print(logger.states['train_loss'].data)

# print(logger.get_states())
