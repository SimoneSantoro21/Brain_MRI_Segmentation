import torch
from libs.U_Net_components import DoubleConvolution
from libs.U_Net_components import DownSample
from libs.U_Net_components import UpSample

def test_DoubleConvolution_output():
    """
    Testing the shape of the output of the DoubleConvolution operation

    GIVEN: A torch tensor of shape (batch_size, in_channels, height, width)
    WHEN: DoubleConvolution operation is performed
    THEN: The output is a tensor of shape (batch_size, out_channels, height, width)
    """
    seed = 42
    torch.manual_seed(seed)

    batch_size = 2
    height, width = 256, 256
    in_channels = 1
    out_channels = 64

    test_tensor = torch.randn(batch_size, in_channels, height, width).float()
    doubleconv = DoubleConvolution(in_channels, out_channels)

    output = doubleconv(test_tensor)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, out_channels, height, width)


def test_DoubleConvolution_data_type():
    """
    Testing the datatype output of the DoubleConvolution operation

    GIVEN: A float32 torch tensor
    WHEN: DoubleConvolution operation is performed
    THEN: The output is still a float32 torch tensor
    """
    seed = 42
    torch.manual_seed(seed)

    batch_size = 2
    height, width = 256, 256
    in_channels = 1
    out_channels = 64

    test_tensor = torch.randn(batch_size, in_channels, height, width).float()
    doubleconv = DoubleConvolution(in_channels, out_channels)

    output = doubleconv(test_tensor)

    assert output.dtype == torch.float32


def test_DownSample_output_type():
    """
    Testing that the type of the output of the DownSample operation is correct.

    GIVEN: A torch tensor of shape (batch_size, in_channels, height, width)
    WHEN: DownSample operation is performed 
    THEN: The output is a tuple containing the convolved tensor and the pooled one
    """
    seed = 42
    torch.manual_seed(seed)

    batch_size = 2
    height, width = 256, 256
    in_channels = 1
    out_channels = 64

    test_tensor = torch.randn(batch_size, in_channels, height, width).float()
    downsample = DownSample(in_channels, out_channels)

    output = downsample(test_tensor)

    assert isinstance(output, tuple)
    assert len(output) == 2


def test_DownSample_pooled_dimension():
    """
    Testing that the pooled output of the DownSample operation has the correct dimensions.

    GIVEN: A torch tensor of shape (batch_size, in_channels, height, width)
    WHEN: DownSample operation is performed 
    THEN: The pooled output is a tensor with shape (batch_size, out_channels, height // 2, width // 2)
    """
    seed = 42
    torch.manual_seed(seed)

    batch_size = 2
    height, width = 256, 256
    in_channels = 1
    out_channels = 64

    test_tensor = torch.randn(batch_size, in_channels, height, width).float()
    downsample = DownSample(in_channels, out_channels)

    output = downsample(test_tensor)
    pooled_output = output[1]

    assert isinstance(pooled_output, torch.Tensor)
    assert pooled_output.shape == (batch_size, out_channels, height // 2, width // 2)


def test_UpSample_output_dimension():
    """
    Testing the output dimension of the UpSampling operation

    GIVEN: Two float32 torch tensor
    WHEN: UpSample operation is performed 
    THEN: The number of channels of the output is equal to out_channels.
    """
    seed = 42
    torch.manual_seed(seed)

    batch_size = 2
    height, width = 256, 256
    in_channels = 128
    out_channels = 64

    test_tensor1 = torch.randn(batch_size, in_channels, height, width).float()
    test_tensor2 = torch.randn(batch_size, out_channels, height * 2, width * 2).float()
    upsample = UpSample(in_channels, out_channels)

    output = upsample(test_tensor1, test_tensor2)

    assert output.shape == (batch_size, out_channels, height * 2, width * 2)


def test_UpSample_output_type():
    """
    Testing the datatype output of the UpSample operation

    GIVEN: Two float32 torch tensor
    WHEN: UpSample operation is performed 
    THEN: The output is still a float32 torch tensor
    """
    seed = 42
    torch.manual_seed(seed)

    batch_size = 2
    height, width = 256, 256
    in_channels = 128
    out_channels = 64

    test_tensor1 = torch.randn(batch_size, in_channels, height, width).float()
    test_tensor2 = torch.randn(batch_size, out_channels, height * 2, width * 2).float()
    upsample = UpSample(in_channels, out_channels)

    output = upsample(test_tensor1, test_tensor2)

    assert output.dtype == torch.float32





