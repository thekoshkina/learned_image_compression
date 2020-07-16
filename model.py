"""
Implementation of the model from the paper

Minnen, David, Johannes Ballé, and George D. Toderici.
["Joint autoregressive and hierarchical priors for learned image compression."](http://papers.nips.cc/paper/8275-joint-autoregressive-and-hierarchical-priors-for-learned-image-compression.pdf
) Advances in Neural Information Processing Systems. 2018.
"""

import torch
from torch import nn
from gdn import GDN
from masked_conv import MaskedConv2d


class Encoder(nn.Module):
	def __init__(self, dim_in, device):
		super(Encoder, self).__init__()
		
		self.first_conv = nn.Conv2d(in_channels=dim_in, out_channels=192, kernel_size=5, stride=2)
		self.conv = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=2)
		self.gdn = GDN(192, inverse=False, device=device, beta_min=1e-6, gamma_init=.1, reparam_offset=2 ** -18)
	
	def forward(self, x):
		x = self.first_conv(x)
		x = self.gdn(x)
		x = self.conv(x)
		x = self.gdn(x)
		x = self.conv(x)
		x = self.gdn(x)
		x = self.conv(x)
		return x


class Decoder(nn.Module):
	def __init__(self, dim_in, device):
		super(Decoder, self).__init__()
		
		self.deconv = nn.ConvTranspose2d(in_channels=dim_in, out_channels=192, kernel_size=5, stride=2)
		self.last_deconv = nn.ConvTranspose2d(in_channels=192, out_channels=3, kernel_size=5, stride=2)
		self.igdn = GDN(192, inverse=True, device=device, beta_min=1e-6, gamma_init=.1, reparam_offset=2 ** -18)
	
	def forward(self, x):
		x = self.deconv(x)
		x = self.igdn(x)
		x = self.deconv(x)
		x = self.igdn(x)
		x = self.deconv(x)
		x = self.igdn(x)
		x = self.last_deconv(x)
		return x


class HyperEncoder(nn.Module):
	def __init__(self, dim_in):
		super(HyperEncoder, self).__init__()
		
		self.conv1 = nn.Conv2d(in_channels=dim_in, out_channels=192, kernel_size=3, stride=1)
		self.conv2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=2, padding=1)
		self.conv3 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, stride=2)
	
	def forward(self, x):
		x = self.conv1(x)
		x = nn.LeakyReLU()(x)
		x = self.conv2(x)
		x = nn.LeakyReLU()(x)
		x = self.conv3(x)
		return x


class HyperDecoder(nn.Module):
	def __init__(self, dim_in):
		super(HyperDecoder, self).__init__()
		
		self.deconv1 = nn.ConvTranspose2d(in_channels=dim_in, out_channels=192, kernel_size=5, stride=2)
		self.deconv2 = nn.ConvTranspose2d(in_channels=192, out_channels=288, kernel_size=5, stride=2, padding=1)
		self.deconv3 = nn.ConvTranspose2d(in_channels=288, out_channels=384, kernel_size=3, stride=1)
	
	def forward(self, x):
		x = self.deconv1(x)
		x = nn.LeakyReLU()(x)
		x = self.deconv2(x)
		x = nn.LeakyReLU()(x)
		x = self.deconv3(x)
		return x


class ContextPrediction(nn.Module):
	def __init__(self, dim_in):
		super(ContextPrediction, self).__init__()
		self.masked = MaskedConv2d("A", in_channels=dim_in, out_channels=384, kernel_size=5, stride=1, padding=2)
	
	def forward(self, x):
		return self.masked(x)


class EntropyParameters(nn.Module):
	def __init__(self, dim_in):
		super(EntropyParameters, self).__init__()
		
		self.conv1 = nn.Conv2d(in_channels=dim_in, out_channels=640, kernel_size=1, stride=1)
		self.conv2 = nn.Conv2d(in_channels=640, out_channels=512, kernel_size=1, stride=1)
		self.conv3 = nn.Conv2d(in_channels=512, out_channels=384, kernel_size=1, stride=1)
	
	def forward(self, x):
		x = self.conv1(x)
		x = nn.LeakyReLU()(x)
		x = self.conv2(x)
		x = nn.LeakyReLU()(x)
		x = self.conv3(x)
		return x


class Model(nn.Module):
	def __init__(self, device):
		super(Model, self).__init__()
		self.encoder = Encoder(3, device)
		self.decoder = Decoder(192, device)
		self.hyper_encoder = HyperEncoder(192)
		self.hyper_decoder = HyperDecoder(192)
		self.entropy = EntropyParameters(768)
		self.context = ContextPrediction(192)
		
	def quantize(self, x):
		"""
		Quantize function - restores the value from previous rescaling (multiply by 255), quintises it to the nearest integer and scales it between 0 and 1 (devide by 255)
		:param x: Tensor
		:return: Tensor
		"""
		return (x * 255).round() / 255.0

	def forward(self, x):
		y = self.encoder(x)
		y_hat = self.quantize(y)
		z = self.hyper_encoder(y)
		z_hat = self.quantize(z)
		phi = self.context(y_hat)
		psi = self.hyper_decoder(z_hat)
		phi_psi = torch.cat([phi, psi], dim=1)
		sigma_mu = self.entropy(phi_psi)
		sigma, mu = torch.split(sigma_mu, y_hat.shape[1], dim=1)
		x_hat = self.decoder(y_hat)
		return x_hat, sigma, mu, y_hat, z_hat
