import torch
import torch.nn as nn

from libs.U_Net_components import DoubleConvolution
from libs.U_Net_components import DownSample
from libs.U_Net_components import UpSample


class UNet(nn.Module):
    """
    A U-Net implementation for semantic segmentation.
    This class assembles the single components (Modules) to define the architecture 
    of the model.

    Args:
        in_channels (int): Number of channels in the input image.
        num_classes (int): Number of output channels representing the number of classes to segment.

    Returns:
        torch.Tensor: A tensor of size (batch_size, num_classes, height, width) containing the segmentation mask.
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        #Encoder component
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)
        self.down_convolution_5 = DownSample(512, 1024)

        #Bottleneck
        self.bottle_neck = DoubleConvolution(1024, 2048)

        #Decoder component
        self.up_convolution_1 = UpSample(2048, 1024)
        self.up_convolution_2 = UpSample(1024, 512)
        self.up_convolution_3 = UpSample(512, 256)
        self.up_convolution_4 = UpSample(256, 128)
        self.up_convolution_5 = UpSample(128, 64)

        #Output layer
        self.out = nn.Conv2d(in_channels = 64, out_channels = num_classes, kernel_size = 1)

    
    def forward(self, x):
        """
        Performs a forward pass through the U-Net.

        Args:
            x (torch.Tensor): Input image tensor of size (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Segmentation mask tensor of size (batch_size, num_classes, height, width).
        """
        convolved_1, pooled_1 = self.down_convolution_1(x)
        convolved_2, pooled_2 = self.down_convolution_2(pooled_1)
        convolved_3, pooled_3 = self.down_convolution_3(pooled_2)
        convolved_4, pooled_4 = self.down_convolution_4(pooled_3)
        convolved_5, pooled_5 = self.down_convolution_5(pooled_4)
        
        b_neck = self.bottle_neck(pooled_5)

        upsampled_1 = self.up_convolution_1(b_neck, convolved_5)
        upsampled_2 = self.up_convolution_2(upsampled_1, convolved_4)
        upsampled_3 = self.up_convolution_3(upsampled_2, convolved_3)
        upsampled_4 = self.up_convolution_4(upsampled_3, convolved_2)
        upsampled_5 = self.up_convolution_5(upsampled_4, convolved_1)

        output = self.out(upsampled_5)
        output = nn.Sigmoid()(output)
        
        return output
        