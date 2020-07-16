# from https://www.codeproject.com/Articles/5061271/PixelCNN-in-Autoregressive-Models
from torch import nn


class MaskedConv2d(nn.Conv2d):
	'''

	Implementation of the Masked convolution from the paper
	Van den Oord, Aaron, et al. "Conditional image generation with pixelcnn decoders." Advances in neural information processing systems. 2016.
	https://arxiv.org/pdf/1606.05328.pdf


	
	'''
	def __init__(self, mask_type, *args, **kwargs):
		super().__init__(*args, **kwargs)
		assert mask_type in ('A', 'B')
		self.register_buffer('mask', self.weight.data.clone())
		_, _, kH, kW = self.weight.size()
		self.mask.fill_(1)
		self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
		self.mask[:, :, kH // 2 + 1:] = 0

	def forward(self, x):
		self.weight.data *= self.mask
		return super(MaskedConv2d, self).forward(x)
