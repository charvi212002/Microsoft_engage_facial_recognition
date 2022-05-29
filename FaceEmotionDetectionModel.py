
import torch.nn as nn

# Fully-convolutional neural network architecture -> Mini-Xception

#Depth-wise separable convolutions 
class Conv2d_Separable(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(Conv2d_Separable, self).__init__() 

        self.pointwise_seperable = nn.Conv2d(in_channels, 
                                             out_channels, 
                                             1, 1, 0, 1, 1, 
                                             bias=bias) 
        
        self.depthwise_seperable = nn.Conv2d(in_channels, in_channels, 
                                             kernel_size, stride, 
                                             padding, dilation, 
                                             groups=in_channels, bias=bias)
        

    def forward(self, layer):
        layer = self.depthwise_seperable(layer)      
        layer = self.pointwise_seperable(layer)
        return layer


# Four residual depthwise separable convolutions 
# Each convolution is followed by a batch normalization operation
# And a ReLU activation function

class Block(nn.Module):  #residualblocks

    def __init__(self, in_channeld, out_channels):
        super(Block, self).__init__()

        self.residual_conv = nn.Conv2d(in_channels=in_channeld,
                                       out_channels=out_channels,
                                       kernel_size=1, stride=2,
                                       bias=False)
        
        self.residual_batchnorm = nn.BatchNorm2d(out_channels,
                                                 momentum=0.99, eps=1e-3)

        self.sep_conv2D_1 = Conv2d_Separable(in_channels=in_channeld, 
                                             out_channels=out_channels, 
                                             kernel_size=3, bias=False,
                                             padding=1)
        
        self.batchnorm_1 = nn.BatchNorm2d(out_channels, 
                                          momentum=0.99, eps=1e-3)
        
        self.relu = nn.ReLU()

        self.sep_conv2D_2 = Conv2d_Separable(in_channels=out_channels, 
                                             out_channels=out_channels, 
                                             kernel_size=3, bias=False,
                                             padding=1)
        
        self.batchnorm_2 = nn.BatchNorm2d(out_channels, 
                                          momentum=0.99, eps=1e-3)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, layer):
        res = self.residual_conv(layer)
        res = self.residual_batchnorm(res)

        layer = self.sep_conv2D_1(layer)
        layer = self.batchnorm_1(layer)
        layer = self.relu(layer)

        layer = self.sep_conv2D_2(layer)
        layer = self.batchnorm_2(layer)
        layer = self.maxpool(layer)
        return res + layer

# Our initial proposed architecture
# is a standard fully-convolutional neural network composed of
# 9 convolution layers, ReLUs [5], Batch Normalization [7]
# and Global Average Pooling

class Model(nn.Module):

    def __init__(self, num_classes):
        super(Model, self).__init__()

        self.conv2D_1 = nn.Conv2d(in_channels=1, 
                                  out_channels=8, 
                                  kernel_size=3, stride=1, 
                                  bias=False)
        
        self.batchnorm_1 = nn.BatchNorm2d(8, affine=True, 
                                          momentum=0.99, 
                                          eps=1e-3)
        self.relu1 = nn.ReLU()

        self.conv2D_2 = nn.Conv2d(in_channels =8, 
                                  out_channels=8, 
                                  kernel_size =3, 
                                  stride=1, bias=False)
        
        self.batchnorm_2 = nn.BatchNorm2d(8, momentum=0.99, 
                                          eps=1e-3)
        
        self.relu2 = nn.ReLU()

        self.module1 = Block(in_channeld=8, out_channels=16)
        self.module2 = Block(in_channeld=16, out_channels=32)
        self.module3 = Block(in_channeld=32, out_channels=64)
        self.module4 = Block(in_channeld=64, out_channels=128)

        self.last_conv2D = nn.Conv2d(in_channels=128, 
                                     out_channels=num_classes, 
                                     kernel_size=3, padding=1)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input):
        layer = input

        layer = self.conv2D_1(layer)
        layer = self.batchnorm_1(layer)
        layer = self.relu1(layer)
        
        layer = self.conv2D_2(layer)
        layer = self.batchnorm_2(layer)
        layer = self.relu2(layer)
        
        layer = self.module1(layer)
        layer = self.module2(layer)
        layer = self.module3(layer)   
        layer = self.module4(layer)
        
        layer = self.last_conv2D(layer)
        layer = self.avgpool(layer)
        layer = layer.view((layer.shape[0], -1))
        return layer