
import torch
import argparse
import torchvision

from torch import optim
from torch.autograd import Variable
from torchsummary import summary
from PIL import Image
from model import Model
from ratedistortionloss import RateDistortionLoss
from utils import concat_images, AverageMeter


def train_epoch(model, criterion, optimiser, train_dataloader, epoch, epochs, lam):
	"""
	Train model for one epoch
	"""
	losses = AverageMeter()
	model.train()  # Set model to training mode
	
	for batch, (inputs, _) in enumerate(train_dataloader):
		inputs = Variable(inputs)
		inputs = inputs.to(args.device)
		
		# forward
		x_hat, sigma, mu, y_hat, z_hat = model(inputs)
		loss, lam = criterion(inputs, x_hat, sigma, mu, y_hat, z_hat, lam)
		optimiser.zero_grad()
		# backward
		loss.backward()
		optimiser.step()
		
		# keep track of loss
		losses.update(loss.item(), inputs.size(0))
		
		# print out loss and visualise results
		if batch % 10 == 0:
			print('Epoch {}/{}:[{}]/[{}] Loss: {:.4f}'.format(epoch, epochs, batch, len(train_dataloader), losses.avg))
			reconstructed_image = torchvision.transforms.ToPILImage(mode='RGB')(x_hat[0].to('cpu'))
			image = torchvision.transforms.ToPILImage(mode='RGB')(inputs[0].to('cpu'))
			result_image = concat_images(image, reconstructed_image)
			result_image.save("train_images/epoch{}batch{}.png".format(epoch, batch))

	return losses.avg


def train(args):
	
	# load dataset
	train_data = torchvision.datasets.SBU(args.root, transform=torchvision.transforms.Compose([torchvision.transforms.Resize((765, 765)), torchvision.transforms.ToTensor()]), target_transform=None, download=True)
	
	# create data loader
	train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
	
	model = Model(args.device)
	criterion = RateDistortionLoss()
	optimiser = optim.Adam(model.parameters(), lr=args.lr)

	# load model and continue training
	if args.continue_training:
		checkpoint = torch.load(args.checkpoint)
		model.load_state_dict(checkpoint['state_dict'])
		optimiser.load_state_dict(checkpoint['optimiser'])
		start_epoch = checkpoint['epoch']
	else:
		start_epoch = 0
	
	# move model to gpu and show structure
	model = model.to(args.device)
	summary(model, input_size=train_data[0][0].shape)

	# initial value of lambda for trainig lagrangian multiplier
	lam = 0.025
	for epoch in range(start_epoch, args.epochs):
		train_loss = train_epoch(model, criterion, optimiser, train_dataloader, epoch, args.epochs, lam)
		
		# save the model
		state = {'epoch': epoch,
				 'state_dict': model.state_dict(),
				 'optimizer': optimiser.state_dict(),
				 'loss': train_loss}
		torch.save(state, args.checkpoint)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Process some integers.')
	parser.add_argument('-epochs', type=int, default=1000, help='number of epoch for training')
	parser.add_argument('-device', type=str, default="cuda:0", help='which device to run the training on (cpu or gpu)')
	parser.add_argument('-batch_size', type=int, default=4, help='number of epoch for training')
	parser.add_argument('-continue_training', type=bool, default=False, help='whether to use pretrained model from the checkpoint file')
	parser.add_argument('-checkpoint', type=str, default='compression_model.pth', help='path where to save checkpoint during training')
	parser.add_argument('-root', type=str, default='data/', help='path to the folder with images')
	parser.add_argument('-lr', type=float, default=1e-4, help='path to the folder with grayscale images')
	
	args = parser.parse_args()
	train(args)
	print("Done.")
