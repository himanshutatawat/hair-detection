# import numpy as np
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.transforms import transforms
# from torch.utils.data.sampler import SubsetRandomSampler
# from HairDetection.base import DataLoaderBase

# # Update and use our own custom dataset

# # class CustomDataLoader(DataLoaderBase):
# #     def __init__(self, transforms, data_dir, batch_size, shuffle, nworkers, train=True):
# #         self.data_dir = data_dir

# #         self.train_dataset = ImageFolder(
# #             os.path.join(self.data_dir, 'train'),  # Path to the training data
# #             transform=transforms.build_transforms(train=True)
# #         )
# #         self.valid_dataset = ImageFolder(
# #             os.path.join(self.data_dir, 'validation'),  # Path to the validation data
# #             transform=transforms.build_transforms(train=False)
# #         ) if train else None

# #         super().__init__(self.train_dataset, shuffle=shuffle, num_workers=nworkers, batch_size=batch_size)

# # class CustomDataLoader(DataLoaderBase):
# #     def __init__(self, transforms, data_dir, batch_size, shuffle, nworkers):
# #         self.data_dir = data_dir

# #         dataset_dir = os.path.join(self.data_dir, 'content', 'sampled_images')

# #         self.train_dataset = ImageFolder(
# #             dataset_dir,  # Path to the main dataset directory
# #             transform=transforms.build_transforms(train=True)
# #         )

# #         super().__init__(self.train_dataset, shuffle=shuffle, num_workers=nworkers, batch_size=batch_size)
# import os
# import numpy as np
# import torch
# from torchvision.datasets import ImageFolder
# from torch.utils.data import random_split

# class CustomDataLoader(DataLoaderBase):
#     def __init__(self, transforms, data_dir, batch_size, shuffle, nworkers, validation_ratio=0.1):
#         self.data_dir = data_dir

#         dataset_dir = self.data_dir
#         print(dataset_dir)

#         full_dataset = ImageFolder(
#             dataset_dir,  # Path to the main dataset directory
#             transform=transforms.build_transforms(train=True)
#         )

#         # Calculate the sizes of training and validation subsets
#         num_samples = len(full_dataset)
#         num_validation = int(np.floor(validation_ratio * num_samples))
#         num_training = num_samples - num_validation

#         # Split the dataset into training and validation subsets
#         self.train_dataset, self.valid_dataset = random_split(full_dataset, [num_training, num_validation])

#         super().__init__(self.train_dataset, shuffle=shuffle, num_workers=nworkers, batch_size=batch_size)


#     def split_validation(self):
#         if self.valid_dataset is None:
#             return None
#         else:
#             return DataLoader(self.valid_dataset, shuffle=False, batch_size=self.init_kwargs['batch_size'])
# # class CustomDataLoader(DataLoaderBase):
#     """
#     Hair damage detection data loading using DataLoaderBase
#     """
#     def __init__(self, transforms, data_dir, batch_size, shuffle, validation_split, nworkers,
#                  train=True):
#         self.data_dir = data_dir

#         self.train_dataset = datasets.ImageFolder(
#             self.data_dir,
#             transform=transforms.build_transforms(train=True)
#         )
#         self.valid_dataset = None
#         if train and validation_split > 0.0:
#             num_train = len(self.train_dataset)
#             print(num_train)
#             indices = list(range(num_train))
#             split = int(np.floor(validation_split * num_train))
#             # np.random.shuffle(indices)
#             train_indices, valid_indices = indices[split:], indices[:split]

#             self.train_sampler = SubsetRandomSampler(train_indices)
#             self.valid_sampler = SubsetRandomSampler(valid_indices)
#             self.valid_dataset = datasets.ImageFolder(
#                 self.data_dir,
#                 transform=transforms.build_transforms(train=False)
#             )

#         self.init_kwargs = {
#             'batch_size': batch_size,
#             'num_workers': nworkers,
#         }

#         super().__init__(
#             dataset=self.train_dataset if self.valid_dataset is None else None,
#             batch_size=batch_size,
#             shuffle=False,
#             sampler=self.train_sampler if self.valid_dataset is None else self.valid_sampler,
#             num_workers=nworkers
#         )

#     def split_validation(self):
#         if self.valid_dataset is None:
#             return None
#         else:
#             return DataLoader(self.valid_dataset, shuffle=False, **self.init_kwargs)
import os
import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from HairDetection.base import DataLoaderBase

class CustomDataLoader(DataLoaderBase):
    """
    Hair damage detection data loading using DataLoaderBase
    """
    def __init__(self, transforms, data_dir, batch_size, shuffle, validation_split, nworkers, train=True):
        self.data_dir = data_dir

        # Create the training dataset using ImageFolder
        self.train_dataset = datasets.ImageFolder(
            self.data_dir,
            transform=transforms.build_transforms(train=True)
        )

        if train and validation_split > 0.0:
            num_train = len(self.train_dataset)
            split = int(np.floor(validation_split * num_train))
            
            # Split the training dataset into training and validation subsets
            self.train_dataset, self.valid_dataset = torch.utils.data.random_split(
                self.train_dataset, [num_train - split, split]
            )

        self.init_kwargs = {
            'batch_size': batch_size,
            'num_workers': nworkers,
        }

        super().__init__(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=nworkers
        )

    def split_validation(self):
        if self.valid_dataset is None:
            return None
        else:
            valid_loader = DataLoader(
                self.valid_dataset, shuffle=False, **self.init_kwargs
            )
            return valid_loader
