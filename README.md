# Joint autoregressive and hierarchical priors for learned image compression

PyTorch implementation of the paper

Minnen, David, Johannes Ballé, and George D. Toderici. 
["Joint autoregressive and hierarchical priors for learned image compression."](http://papers.nips.cc/paper/8275-joint-autoregressive-and-hierarchical-priors-for-learned-image-compression.pdf
) Advances in Neural Information Processing Systems. 2018.

![Model structure](https://github.com/thekoshkina/learned_image_compression/blob/master/images/model_structure.png)

### Required packages: 
```
torch~=1.3.1
torchvision~=0.4.2
torchsummary~=1.5.1
Pillow~=5.1.0
scipy~=1.4.0
```

## Implemented model summary
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1        [-1, 192, 381, 381]          14,592
               GDN-2        [-1, 192, 381, 381]               0
            Conv2d-3        [-1, 192, 189, 189]         921,792
               GDN-4        [-1, 192, 189, 189]               0
            Conv2d-5          [-1, 192, 93, 93]         921,792
               GDN-6          [-1, 192, 93, 93]               0
            Conv2d-7          [-1, 192, 45, 45]         921,792
           Encoder-8          [-1, 192, 45, 45]               0
            Conv2d-9          [-1, 192, 43, 43]         331,968
           Conv2d-10          [-1, 192, 21, 21]         921,792
           Conv2d-11            [-1, 192, 9, 9]         921,792
     HyperEncoder-12            [-1, 192, 9, 9]               0
     MaskedConv2d-13          [-1, 384, 45, 45]       1,843,584
ContextPrediction-14          [-1, 384, 45, 45]               0
  ConvTranspose2d-15          [-1, 192, 21, 21]         921,792
  ConvTranspose2d-16          [-1, 288, 43, 43]       1,382,688
  ConvTranspose2d-17          [-1, 384, 45, 45]         995,712
     HyperDecoder-18          [-1, 384, 45, 45]               0
           Conv2d-19          [-1, 640, 45, 45]         492,160
           Conv2d-20          [-1, 512, 45, 45]         328,192
           Conv2d-21          [-1, 384, 45, 45]         196,992
EntropyParameters-22          [-1, 384, 45, 45]               0
  ConvTranspose2d-23          [-1, 192, 93, 93]         921,792
              GDN-24          [-1, 192, 93, 93]               0
  ConvTranspose2d-25        [-1, 192, 189, 189]         921,792
              GDN-26        [-1, 192, 189, 189]               0
  ConvTranspose2d-27        [-1, 192, 381, 381]         921,792
              GDN-28        [-1, 192, 381, 381]               0
  ConvTranspose2d-29          [-1, 3, 765, 765]          14,403
          Decoder-30          [-1, 3, 765, 765]               0
================================================================
Total params: 13,896,419
Trainable params: 13,896,419
```
## Preliminary results
Training of the model with a simplified hyperlatent rate show the potential for the model to train: 

Example of original/reconstructed image at epoch 0
![Example of original/reconstructed image at epoch 0](https://github.com/thekoshkina/learned_image_compression/blob/master/images/epoch0batch0.png )
Example of original/reconstructed image at epoch 1 
![xample of original/reconstructed image at epoch 1](https://github.com/thekoshkina/learned_image_compression/blob/master/images/epoch1batch0.png) 
Example of original/reconstructed image at epoch 2 
![xample of original/reconstructed image at epoch 2](https://github.com/thekoshkina/learned_image_compression/blob/master/images/epoch2batch0.png)

## Future work 
- add padding so the model can work with imaging of any resolution
- optimise updates of dynamic Lagrange multipliers vs constants
- add Compress function encodes an image into a bitstream
- add Decompress function that reconstructs 
 
### References: 
 Minnen, David, Johannes Ballé, and George D. Toderici. 
["Joint autoregressive and hierarchical priors for learned image compression."](http://papers.nips.cc/paper/8275-joint-autoregressive-and-hierarchical-priors-for-learned-image-compression.pdf
) Advances in Neural Information Processing Systems. 2018.
 
 Ballé, D. Minnen, S. Singh, S. J. Hwang, and N. Johnston,
[“Variational image compression with a scale hyperprior,”](https://openreview.net/forum?id=rkcQFMZRb) 6th Int. Conf. on Learning Representations, 2018. 
		
 Implementation of Masked convolution was taken from: https://www.codeproject.com/Articles/5061271/PixelCNN-in-Autoregressive-Models
 Implementation of the GDN non-linearity was from https://github.com/jorge-pessoa/pytorch-gdn/blob/master/pytorch_gdn/__init__.py
