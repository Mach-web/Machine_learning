import os
import numpy as np
import pandas as pd
import torch
from PIL import Image
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import functional as F
import torchvision_starter.transforms as T
from tqdm import tqdm
import multiprocessing
from pathlib import Path
import math
from torchvision_starter import utils


def compute_mean_and_std(folder):
    """
    Compute per-channel mean and std of the dataset (to be used in transforms.Normalize())
    """

    ds = UdacitySelfDrivingDataset(
        folder, transform=T.Compose([T.ToTensor()]), train=True
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=1, num_workers=multiprocessing.cpu_count()
    )

    mean = 0.0
    for images, _ in tqdm(dl, total=len(ds), desc="Computing mean", ncols=80):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(dl.dataset)

    var = 0.0
    npix = 0
    for images, _ in tqdm(dl, total=len(ds), desc="Computing std", ncols=80):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
        npix += images.nelement()

    std = torch.sqrt(var / (npix / 3))

    return mean, std


class UdacitySelfDrivingDataset(torch.utils.data.Dataset):
    
    # Mean and std of the dataset to be used in nn.Normalize
    mean = torch.tensor([0.3680, 0.3788, 0.3892])
    
    std = torch.tensor([0.2902, 0.3069, 0.3242])
    
    def __init__(self, root, transform, train=True, thinning=None):
        
        super().__init__()
        
        self.root = os.path.abspath(os.path.expandvars(os.path.expanduser(root)))
        self.transform = transform
        
        # load datasets
        if train:
            self.df = pd.read_csv(os.path.join(self.root, "labels_train.csv"))
        else:
            self.df = pd.read_csv(os.path.join(self.root, "labels_test.csv"))
        
        # Index by file id (i.e., a sequence of the same length as the number of images)
        codes, uniques = pd.factorize(self.df['frame'])
        
        if thinning:
            # Take every n-th rows. This makes sense because the images are
            # frames of videos from the car, so we are essentially reducing
            # the frame rate
            thinned = uniques[::thinning]
            idx = self.df['frame'].isin(thinned)
            print(f"Keeping {thinned.shape[0]} of {uniques.shape[0]} images")
            print(f"Keeping {idx.sum()} objects out of {self.df.shape[0]}")
            self.df = self.df[idx].reset_index(drop=True)
            
            # Recompute codes
            codes, uniques = pd.factorize(self.df['frame'])
        
        self.n_images = len(uniques)
        self.df['image_id'] = codes
        self.df.set_index("image_id", inplace=True)
        
        self.classes = ['car', 'truck', 'pedestrian', 'bicyclist', 'light']
        self.colors = ['cyan', 'blue', 'red', 'purple', 'orange']
    
    @property
    def n_classes(self):
        return len(self.classes)
    
    def __getitem__(self, idx):
        
        if idx in self.df.index:
            row = self.df.loc[[idx]]
        else:
            return KeyError(f"Element {idx} not in dataframe")
        
        # load images fromm file
        img_path = os.path.join(self.root, "images", row['frame'].iloc[0])
        img = Image.open(img_path).convert("RGB")
        
        # Exclude bogus boxes with 0 height or width
        h = row['ymax'] - row['ymin']
        w = row['xmax'] - row['xmin']
        filter_idx = (h > 0) & (w > 0)
        row = row[filter_idx]
        
        # get bounding box coordinates for each mask
        boxes = row[['xmin', 'ymin', 'xmax', 'ymax']].values

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # get the labels
        labels = torch.as_tensor(row['class_id'].values, dtype=int)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # assume no crowd for everything
        iscrowd = torch.zeros((row.shape[0],), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return self.n_images
    
    def plot(self, idx, renormalize=True, predictions=None, threshold=0.5, ax=None):
        
        image, label_js = self[idx]
        
        if renormalize:
            
            # Invert the T.Normalize transform
            unnormalize = T.Compose(
                [
                    T.Normalize(mean = [ 0., 0., 0. ], std = 1 / type(self).std),
                    T.Normalize(mean = -type(self).mean, std = [ 1., 1., 1. ])
                ]
            )

            image, label_js = unnormalize(image, label_js)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        _ = ax.imshow(torch.permute(image, [1, 2, 0]))
        
        for i, box in enumerate(label_js['boxes']):
            
            xy = (box[0], box[1])
            h, w = (box[2] - box[0]), (box[3] - box[1])
            r = patches.Rectangle(xy, h, w, fill=False, color=self.colors[label_js['labels'][i]-1], lw=2, alpha=0.5)
            ax.add_patch(r)
        
        if predictions is not None:
            
            # Make sure the predictions are on the CPU
            for k in predictions:
                predictions[k] = predictions[k].detach().cpu().numpy()
            
            for i, box in enumerate(predictions['boxes']):
                
                if predictions['scores'][i] > threshold:
                    xy = (box[0], box[1])
                    h, w = (box[2] - box[0]), (box[3] - box[1])
                    r = patches.Rectangle(xy, h, w, fill=False, color=self.colors[predictions['labels'][i]-1], lw=2, linestyle=':')
                    ax.add_patch(r)
        
        _ = ax.axis("off")
        
        return ax


def get_data_loaders(
    folder, batch_size: int = 2, valid_size: float = 0.2, num_workers: int = -1, limit: int = -1, thinning: int = None
):
    """
    Create and returns the train_one_epoch, validation and test data loaders.

    :param foder: folder containing the dataset
    :param batch_size: size of the mini-batches
    :param valid_size: fraction of the dataset to use for validation. For example 0.2
                       means that 20% of the dataset will be used for validation
    :param num_workers: number of workers to use in the data loaders. Use -1 to mean
                        "use all my cores"
    :param limit: maximum number of data points to consider
    :param thinning: take every n-th frame, instead of all frames
    :return a dictionary with 3 keys: 'train_one_epoch', 'valid' and 'test' containing respectively the
            train_one_epoch, validation and test data loaders
    """

    if num_workers == -1:
        # Use all cores
        num_workers = multiprocessing.cpu_count()

    # We will fill this up later
    data_loaders = {"train": None, "valid": None, "test": None}

    # create 3 sets of data transforms: one for the training dataset,
    # containing data augmentation, one for the validation dataset
    # (without data augmentation) and one for the test set (again
    # without augmentation)
    data_transforms = {
        "train": get_transform(UdacitySelfDrivingDataset.mean, UdacitySelfDrivingDataset.std, train=True),
        "valid": get_transform(UdacitySelfDrivingDataset.mean, UdacitySelfDrivingDataset.std, train=False),
        "test": get_transform(UdacitySelfDrivingDataset.mean, UdacitySelfDrivingDataset.std, train=False),
    }

    # Create train and validation datasets
    train_data = UdacitySelfDrivingDataset(
        folder, 
        transform=data_transforms["train"], 
        train=True,
        thinning=thinning
    )
    
    # The validation dataset is a split from the train_one_epoch dataset, so we read
    # from the same folder, but we apply the transforms for validation
    valid_data = UdacitySelfDrivingDataset(
        folder, 
        transform=data_transforms["valid"], 
        train=True,
        thinning=thinning
    )

    # obtain training indices that will be used for validation
    n_tot = len(train_data)
    indices = torch.randperm(n_tot)

    # If requested, limit the number of data points to consider
    if limit > 0:
        indices = indices[:limit]
        n_tot = limit

    split = int(math.ceil(valid_size * n_tot))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)  # =

    # prepare data loaders
    data_loaders["train"] = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=utils.collate_fn
    )
    data_loaders["valid"] = torch.utils.data.DataLoader(
        valid_data,  # -
        batch_size=batch_size,  # -
        sampler=valid_sampler,  # -
        num_workers=num_workers,  # -
        collate_fn=utils.collate_fn
    )

    # Now create the test data loader
    test_data = UdacitySelfDrivingDataset(
        folder, 
        transform=data_transforms["test"], 
        train=False,
        thinning=thinning
    )

    if limit > 0:
        indices = torch.arange(limit)
        test_sampler = torch.utils.data.SubsetRandomSampler(indices)
    else:
        test_sampler = None

    data_loaders["test"] = torch.utils.data.DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        sampler=test_sampler, 
        collate_fn=utils.collate_fn  # -
    )

    return data_loaders

    
def get_transform(mean=0.5, std=0.5, train=True):
    
    transforms = [T.ToTensor()]

    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    
    transforms.append(T.Normalize(mean, std))
    
    return T.Compose(transforms)    


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
