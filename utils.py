from PIL import Image
import torchvision
import torch
from model import Model


def run_on_image(model_path, image_path, device):
	"""
	Run the pretrained model stored at model_path on an image
	:param model_path: path to the model weights
	:param image_path: path to the image
	:return:
	"""
	model = Model(device)
	
	checkpoint = torch.load(model_path)
	model.load_state_dict(checkpoint['state_dict'])
	model.to(device)
	model.eval()
	
	transform = torchvision.transforms.Compose([torchvision.transforms.Resize((765, 765)), torchvision.transforms.ToTensor()])
	image = Image.open(image_path)
	inputs = transform(image)
	inputs = torch.unsqueeze(inputs, 0)
	inputs = inputs.to(device)
	
	x_hat, _, _, _, _ = model(inputs)
	reconstructed_image = torchvision.transforms.ToPILImage(mode='RGB')(x_hat.squeeze)
	result_image = concat_images(image, reconstructed_image)
	result_image.show()


def concat_images(image1, image2):
	"""
	Concatenates two images together
	"""
	result_image = Image.new('RGB', (image1.width + image2.width, image1.height))
	result_image.paste(image1, (0, 0))
	result_image.paste(image2, (image1.width, 0))
	return result_image


class AverageMeter(object):
	"""Stores current value of statistics and computes average"""
	
	def __init__(self):
		self.reset()
	
	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
	
	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

