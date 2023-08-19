import abc
import torch
import torchvision.transforms as T

class AugmentationFactoryBase(abc.ABC):
    def build_transforms(self, train):
        return self.build_train() if train else self.build_test()

    @abc.abstractmethod
    def build_train(self):
        pass

    @abc.abstractmethod
    def build_test(self):
        pass

class Transforms(AugmentationFactoryBase):
    MEANS = [0]
    STDS = [1]

    def __init__(self):
        self.train_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.1),
            T.RandomVerticalFlip(p=0.1),
            T.RandomApply([T.RandomRotation(degrees=90)], p=0.1),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomApply([T.RandomAffine(degrees=0, shear=15, scale=(0.8, 1.2))], p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=5)], p=0.2),
            T.ToTensor(),
            T.Normalize(self.MEANS, self.STDS)
        ])
        self.test_transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(self.MEANS, self.STDS)
        ])

    def build_train(self):
        return self.train_transforms

    def build_test(self):
        return self.test_transforms
