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
from torch.nn.parameter import Parameter

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


class HyperRate(nn.Module):
	def __init__(self, channels, w, h):
		super(HyperRate, self).__init__()
		
		self.H1 = Parameter(torch.ones(channels * w * h, channels * w * h), requires_grad=True)
		self.b1 = Parameter(torch.randn(channels * w * h), requires_grad=True)
		self.a1 = Parameter(torch.zeros(channels * w * h), requires_grad=True)
		
		self.H2 = Parameter(torch.ones(channels * w * h, channels * w * h), requires_grad=True)
		self.b2 = Parameter(torch.randn(channels * w * h), requires_grad=True)
		self.a2 = Parameter(torch.zeros(channels * w * h), requires_grad=True)
		
		self.H3 = Parameter(torch.ones(channels * w * h, channels * w * h), requires_grad=True)
		self.b3 = Parameter(torch.rand(channels * w * h), requires_grad=True)
		self.a3 = Parameter(torch.zeros(channels * w * h), requires_grad=True)
		
		self.H4 = Parameter(torch.ones(channels * w * h, channels * w * h), requires_grad=True)
		self.b4 = Parameter(torch.rand(channels * w * h), requires_grad=True)
		self.a4 = Parameter(torch.zeros(channels * w * h), requires_grad=True)
	
	def forward(self, x):
		initial_shape = x.shape
		x = x.view(x.shape[0], -1)
		
		self.H1.data = nn.functional.softplus(self.H1)
		self.a1.data = torch.tanh(self.a1)
		x = torch.matmul(x, self.H1) + self.b1
		x = x + self.a1 * x
		
		self.H2.data = nn.functional.softplus(self.H2)
		self.a2.data = torch.tanh(self.a2)
		x = torch.matmul(x, self.H2) + self.b2
		x += self.a2 * x
		
		self.H3.data = nn.functional.softplus(self.H3)
		self.a3.data = torch.tanh(self.a3)
		x = torch.matmul(x, self.H3) + self.b3
		x += self.a3 * x
		
		self.H4.data = nn.functional.softplus(self.H4)
		x = torch.matmul(x, self.H4) + self.b4
		x = nn.functional.sigmoid(x)
		
		x = x.view(initial_shape)
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

		# z_hat = z_hat.view(z_hat.shape[0], -1)
		hyperlatent_rate = self.hyperlatent_rate(z_hat)

		latent_rate = torch.mean(self.latent_rate(mu, sigma, y_hat))
		distortion = torch.mean((x - x_hat).pow(2))
		loss = self.lam * distortion + latent_rate + hyperlatent_rate
		return x_hat, loss


	def cumulative(self, mu, sigma, x):
		half = 0.5
		const = (2 ** -0.5)
		return half*(1 + torch.erf( (torch.tanh(x) - mu) / (const * sigma)))


	def latent_rate(self, mu, sigma, y):
		upper = self.cumulative(mu, sigma, (y + .5))
		lower = self.cumulative(mu, sigma, (y - .5))
		return -torch.sum(torch.log2(torch.abs(upper - lower)), dim=(1))

	def hyperlatent_rate(self, z):
		# calculate rate for the hyperprior
		upper = self.hyper_cdf(z + 0.5)
		lower = self.hyper_cdf(z - 0.5)

		return -torch.sum(torch.log2(torch.abs(upper - lower)), dim=(1))

