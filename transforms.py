"""Contains transform classes definitions"""

from torch import nn, device, Tensor

class ToDevice(nn.Module):
    """Used for moving the tensor to device."""

    def __init__(self, device: device):
        """Initialize the ToDevice transform with the given device

        Args:
            device (device): Device used for the transformation
        """
        super().__init__()
        self.device = device

    def forward(self, img: Tensor) -> Tensor:
        """Moves the tensor to self.device.

        Args:
            img (Tensor): Tensor to move

        Returns:
            Tensor: Tensor moved to the specified device
        """
        return img.to(self.device)


class ImageByteToFloat(nn.Module):
    """Converts the image from byte (0-255) format to floating point format (0.0-1.0)."""

    def forward(self, img: Tensor) -> Tensor:
        """Converts the image from byte-format to floating-point-format.

        Args:
            img (Tensor): Image tensor to convert

        Returns:
            Tensor: Image converted to float-representation
        """
        return img.float() / 255.0


class ImageFloatToByte(nn.Module):
    """Converts the image from floating point format (0.0-1.0) to byte format (0-255)."""

    def forward(self, img: Tensor) -> Tensor:
        """Converts the image from floating-point-format to byte-format.

        Args:
            img (Tensor): Image tensor to convert

        Returns:
            Tensor: Image converted to byte-representation
        """
        return (img * 255).byte()