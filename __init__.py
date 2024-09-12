import torch

def numpy_to_torch(image):
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3: #检查输入图像的维度是否为3
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2: #检查输入图像的维度是否为2，即是否为灰度图像。
        image = image[None]  # add channel axis 用[None]操作在图像数组中添加一个额外的维度，以表示通道
    else:
        raise ValueError(f'Not an image: {image.shape}') #抛出一个ValueError异常，说明图像不是有效的格式。
    return torch.from_numpy(image / 255.).float() #归一化到0到1的范围，并使用torch.from_numpy()函数将其转换为Torch张量。同时，将数据类型转换为float类型。
