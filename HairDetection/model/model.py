import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from HairDetection.base import ModelBase
from HairDetection.utils import setup_logger


log = setup_logger(__name__)


class ResNet50Model(ModelBase):
    def __init__(self, num_classes):
        """
        Initialize the ResNet50 model.

        Args:
            num_classes (int): Number of output classes.
        """
        super(ResNet50Model, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(in_features, num_classes)
        log.info(f'<init>: \n{self}')


    def forward(self, x):
        """
        Forward pass of the ResNet50 model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.resnet50(x)


class ResNet101Model(ModelBase):
    def __init__(self, num_classes):
        """
        Initialize the ResNet101 model.

        Args:
            num_classes (int): Number of output classes.
        """
        super(ResNet101Model, self).__init__()
        self.resnet101 = models.resnet101(pretrained=True)
        in_features = self.resnet101.fc.in_features
        self.resnet101.fc = nn.Linear(in_features, num_classes)
        log.info(f'<init>: \n{self}')


    def forward(self, x):
        """
        Forward pass of the ResNet101 model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.resnet101(x)


class ConvNeXtModel(ModelBase):
    def __init__(self, num_classes):
        """
        Initialize the ConvNeXt model.

        Args:
            num_classes (int): Number of output classes.
        """
        super(ConvNeXtModel, self).__init__()
        # You can use a different pre-trained model as the backbone here
        self.convnext = models.resnext50_32x4d(pretrained=True)
        in_features = self.convnext.fc.in_features
        self.convnext.fc = nn.Linear(in_features, num_classes)
        log.info(f'<init>: \n{self}')


    def forward(self, x):
        """
        Forward pass of the ConvNeXt model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.convnext(x)


class ResNeXtModel(ModelBase):
    def __init__(self, num_classes):
        """
        Initialize the ResNeXt model.

        Args:
            num_classes (int): Number of output classes.
        """
        super(ResNeXtModel, self).__init__()
        self.resnext = models.resnext101_32x8d(pretrained=True)
        in_features = self.resnext.fc.in_features
        self.resnext.fc = nn.Linear(in_features, num_classes)
        log.info(f'<init>: \n{self}')


    def forward(self, x):
        """
        Forward pass of the ResNeXt model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.resnext(x)


class VITModel(ModelBase):
    def __init__(self, num_classes, patch_size, embed_dim, num_heads, num_layers):
        """
        Initialize the VIT (Vision Transformer) model.

        Args:
            num_classes (int): Number of output classes.
            patch_size (int): Size of the image patches.
            embed_dim (int): Dimension of the token embeddings.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
        """
        super(VITModel, self).__init__()
        self.vit = models.vit_base_patch16_224(pretrained=True)
        in_features = self.vit.head.in_features
        self.vit.head = nn.Linear(in_features, num_classes)
        log.info(f'<init>: \n{self}')


    def forward(self, x):
        """
        Forward pass of the VIT model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.vit(x)