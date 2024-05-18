import torch 
import torch.nn as nn

class DoubleConvolution(nn.Module):
    """
    A PyTorch module that performs a double convolution operation on an input tensor.

    This module applies two successive 3x3 convolutions with ReLU activation
    between them. The input and output channels are specified during initialization.
    """

    def __init__(self, in_channels, out_channels):
        """ 
        Initializes the `DoubleConvolution` module.

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output tensor.

        Attributes:
            double_conv_operation (nn.Sequential): A sequential container holding
                the two convolutional layers with ReLU activations. This attribute
                performs the double convolution operation on the input tensor.
        """
        super().__init__()
        self.double_conv_operation = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.ReLU()
        )


    def forward(self, x):
        """
        Performs the double convolution operation on the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        return self.double_conv_operation(x)
    


class DownSample(nn.Module):
    """
    A PyTorch module that performs a downsampling operation on an input tensor.

    This module first applies a `DoubleConvolution`, then, it performs a max pooling 
    operation with a kernel size of 2 and stride of 2 to downsample the spatial 
    dimensions of the feature map. 

    The module returns both the convolved and pooled outputs, allowing for the implementation
    of skip connections in the model.
    """
     
    def __init__(self, in_channels, out_channels):
        """
        Initializes the `DownSample` module.

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output of the `DoubleConvolution`.
        
        Attributes:
            conv: Creates an instance of the DoubleConvolution class
            pool: Defines the pooling method for downsampling as a MaxPool with a 2x2 kernel
                and stride of 2.
        """
        super().__init__()
        self.conv = DoubleConvolution(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)


    def forward(self, x):
        """
        Performs the downsampling operation on the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - convolved: Output tensor of the `DoubleConvolution` of shape:
                  (batch_size, out_channels, height, width).
                - pooled: Downsampled output after max pooling of shape:
                  (batch_size, out_channels, height // 2, width // 2).
        """
        convolved = self.conv(x)
        pooled = self.pool(convolved)

        return convolved, pooled



class UpSample(nn.Module):
    """
    A PyTorch module that performs the upsampling operation on an input tensor.

    This module first applies a 2x2 transposed convolution to perform the upsampling, then, it 
    performs a 'DoubleConvolution' operation. 

    The module returns the output of double convolution computed on the upsampled tensor
    concatenated with the respective feature maps of the encoder (Skip connections).
    """

    def __init__(self, in_channels, out_channels):
        """ 
        Initializes the `DoubleConvolution` module.

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output tensor.

        Attributes:
            up: Defines upsampling method as a 2D transposed convolution that returns
                a tensor with half the number of initial channels, a 2x2 kernel and stride of 2
            conv: Creates an instance of the DoubleConvolution class
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size = 2, stride = 2)
        self.conv = DoubleConvolution(in_channels, out_channels)


    def forward(self, x1, x2):
        """
        Performs the upsampling and the double convolution operation 
        on the input tensor with skip connections.

        Args:
            x1 (torch.Tensor): Input tensor to upsample.
            x2 (torch.Tensor): Skip connection to concatenate.

        Returns:
            torch.Tensor: Obtained applying the double convolution to
                        the tensor given by the concatenation of x2 with
                        the upsampled x1 along the channel dimension.
        """
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim = 1)

        return self.conv(x)