# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from timm.data import transforms_factory as timm_transforms
import torch

from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
# from sklearn.model_selection import train_test_split
import numpy as np

def pil_loader(path): # to solve the numpy issue
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class IndexedDataset(Dataset):
    """
    Wrap an existing dataset.
    By default returns (img, target) so it is compatible with the training loop.
    If you need the original index (rare), set return_index=True to have __getitem__ return (idx, img, target).
    """
    def __init__(self, dataset, return_index: bool = False):
        self.dataset = dataset
        self.samples = dataset.samples
        self.targets = dataset.targets
        self.return_index = return_index

    def __getitem__(self, idx):
        img, target = self.dataset[idx]   # assume underlying returns (img, target)
        if self.return_index:
            return idx, img, target
        return img, target

    def __len__(self):
        return len(self.dataset)

class SplitToFourDataset(Dataset):
    """
    Wrap a base dataset (e.g., ImageFolder) and split each image into 4 overlapping crops.
    """
    def __init__(self, base_dataset, transform=None, return_index=False):
            self.base_dataset = base_dataset
            self.transform = transform
            self.return_index = return_index

    def __len__(self):
        return len(self.base_dataset)

    def _split_image(self, img):
        crop_boxes = [
            (0, 0, 372, 256),
            (372 - 100, 0, 640, 256),
            (0, 256 - 50, 372, 440),
            (372 - 100, 256 - 50, 640, 440)
        ]
        return [img.crop(box) for box in crop_boxes]

    def __getitem__(self, idx):
        img, target = self.base_dataset[idx]

        crops = self._split_image(img)

        if self.transform:
            crops = [self.transform(c) for c in crops]

        # return 4 crops + target + original index
        if self.return_index:
            return idx, crops, target  # crops is a list of 4 tensors
        else:
            return crops, target

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


class GREEN30Dataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        """
        root: path to data_split (e.g. .../green30/data_split)
        split: 'train', 'val', or 'test'
        Each of split/0/ and split/1/ contains a data.csv with image paths.
        """
        self.samples = []
        self.transform = transform

        for label in [0, 1]:
            csv_path = os.path.join(root, split, str(label), 'data.csv')
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path, header=None)
            for img_path in df[0].tolist():
                self.samples.append((img_path, label))

        self.nb_classes = 2

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
        # return img, label, idx

    def __len__(self):
        return len(self.samples)

# for the combined multi-label training (singleLane + crosswalks + notsinglefamilyhome + green30 + streetlight)
class ChicagoCombinedDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        # Save image paths
        self.image_paths = self.df['filepath'].tolist()

        # Select the multi-label columns in the same order each time
        label_columns = ['singleLane', 'crosswalk', 'green30', 'notsinglehome', 'streetlight']
        self.labels = self.df[label_columns].astype(int).values  # shape: (N, 5)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Convert labels to tensor (multi-label: shape [5])
        labels = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, labels


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform,loader=pil_loader)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'STREET': # Xiaoya: added for GSV streetlight
        if args.eval:
            split = 'test'
        else:
            split = 'train' if is_train else 'val'
        # root = os.path.join(args.data_path, 'train' if is_train else 'val')
        root = os.path.join(args.data_path, split)
        if args.gsv_split_to_four == True:
            print('splitted to four.')
            ds = datasets.ImageFolder(root, transform=None,loader=pil_loader)
            dataset = SplitToFourDataset(ds, transform=transform, return_index=True) 
        else:
            ds = datasets.ImageFolder(root, transform=transform,loader=pil_loader)
            dataset = IndexedDataset(ds, return_index=True) # wrap with index-tracking, for choosing samples in attention visualization
        nb_classes = 2
    elif args.data_set == 'NotSinFamily': # Xiaoya: added for GSV NotASingleFamily
        if args.eval:
            split = 'test'
        else:
            split = 'train' if is_train else 'val'
        # root = os.path.join(args.data_path, 'train' if is_train else 'val')
        root = os.path.join(args.data_path, split)
        dataset = datasets.ImageFolder(root, transform=transform,loader=pil_loader)
        nb_classes = 3
    elif args.data_set == 'GREEN30': # Xiaoya: added for GSV NotASingleFamily 
        if args.eval:
            split = 'test'
        else:
            split = 'train' if is_train else 'val'
        root = args.data_path

        dataset = GREEN30Dataset(root, split=split, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'SIDEWALKS': # Xiaoya: added for GSV NotASingleFamily 
        if args.eval:
            split = 'test'
        else:
            split = 'train' if is_train else 'val'
        root = args.data_path
        print(f"Loading SIDEWALKS dataset from {root}, split={split}")

        dataset = GREEN30Dataset(root, split=split, transform=transform)
        nb_classes = dataset.nb_classes
        
    elif args.data_set == 'COMBINED': # Xiaoya: added for supervised fine-tuning(5 objects)
        if is_train:
            csv_path = args.train_csv_path  # e.g., args.train_csv = '/path/to/train.csv'
        else:
            csv_path = args.val_csv_path    # e.g., args.val_csv = '/path/to/val.csv'

        dataset = ChicagoCombinedDataset(csv_path,transform=transform)
        nb_classes = 5

    elif args.data_set == 'SPEEDBUMPS':
        if args.eval:
            split = 'test'
        else:
            split = 'train' if is_train else 'val'
        root = args.data_path
        print(f"Loading SPEEDBUMPS dataset from {root}, split={split}")

        dataset = GREEN30Dataset(root, split=split, transform=transform)
        nb_classes = 2
        
    return dataset, nb_classes


# def build_transform(is_train, args):
#     resize_im = args.input_size > 32  
#     if is_train:
#         # if args.data_set in ['STREET', 'NotSinFamily']:   # Manually build the training transform with custom aspect ratio
#         #     transform = timm_transforms.transforms_imagenet_train(
#         #         img_size=args.input_size,
#         #         color_jitter=args.color_jitter,
#         #         auto_augment=args.aa,
#         #         interpolation=args.train_interpolation,
#         #         re_prob=args.reprob,
#         #         re_mode=args.remode,
#         #         re_count=args.recount,
#         #         ratio=(1.0, 1.6),
#         #     )
#         #     if not resize_im:
#         #         transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
#         #     return transform
#         # else:
#         transform = create_transform(
#             input_size=args.input_size,
#             is_training=True,
#             color_jitter=args.color_jitter,
#             auto_augment=args.aa,
#             interpolation=args.train_interpolation,
#             re_prob=args.reprob,
#             re_mode=args.remode,
#             re_count=args.recount,
#         )
#         if not resize_im:
#             transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
#         return transform
            
#     # val transform
#     t = []
#     if resize_im:
#         size = int(args.input_size / args.eval_crop_ratio)
#         t.append(
#             transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
#         )
#         t.append(transforms.CenterCrop(args.input_size))

#     t.append(transforms.ToTensor())
#     t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
#     return transforms.Compose(t)

class EnsurePILImage:
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            return Image.fromarray(img)
        return img

def build_transform(is_train, args):
    resize_im = args.input_size > 32  
    if is_train:
        if args.data_set in ['STREET', 'NotSinFamily']:   # Manually build the training transform with custom aspect ratio
            transform = timm_transforms.transforms_imagenet_train(
                img_size=args.input_size,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation=args.train_interpolation,
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
                ratio=(1.0, 1.6),
            )
            transform = transforms.Compose([
            EnsurePILImage(),  # 关键修复
            transform
            ])
            if not resize_im:
                transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
            return transform
        else:
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation=args.train_interpolation,
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
            )
            if not resize_im:
                transform.transforms[0] = transforms.RandomCrop(args.input_size, padding=4)
            return transform
            
    # val transform
    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

