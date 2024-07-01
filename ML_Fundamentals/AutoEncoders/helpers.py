import torch
from torchvision import datasets
import torchvision.transforms as transforms
import multiprocessing
import random
import numpy as np
import matplotlib.pyplot as plt

DATA = "C:\\Users\\mkand\\Documents\\Data"
def get_data_loaders(batch_size, val_fraction=0.2):
    
    transform = transforms.ToTensor()
    
    num_workers = multiprocessing.cpu_count()
    
    # Get train, validation and test
    # Let's start with train and validation
    trainval_data = datasets.MNIST(
        root=DATA, train=True, transform=transform, download = True
    )

    # Split in train and validation
    # NOTE: we set the generator with a fixed random seed for reproducibility
    train_len = int(len(trainval_data) * (1 - val_fraction))
    val_len = len(trainval_data) - train_len
    print(f"Using {train_len} examples for training and {val_len} for validation")
    train_subset, val_subset = torch.utils.data.random_split(
        trainval_data, [train_len, val_len], generator=torch.Generator().manual_seed(42)
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_subset, shuffle=True, batch_size=batch_size, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_subset, shuffle=False, batch_size=batch_size, num_workers=num_workers
    )

    # Get test data
    test_data = datasets.MNIST(root=DATA, train=False, transform=transform, download = True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, num_workers=num_workers
    )
    print(f"Using {len(test_data)} for testing")
    
    return {
        'train': train_loader,
        'valid': val_loader,
        'test': test_loader
    }


def seed_all(seed=42):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def anomaly_detection_display(df):
    
    df.sort_values(by='loss', ascending=False, inplace=True)
    
    fig, sub = plt.subplots()
    df['loss'].hist(bins=100)
    sub.set_yscale('log')
    sub.set_xlabel("Score (loss)")
    sub.set_ylabel("Counts per bin")
    fig.suptitle("Distribution of score (loss)")
    
    fig, subs = plt.subplots(2, 20, figsize=(20, 3))

    for img, sub in zip(df['image'].iloc[:20], subs[0].flatten()):
        sub.imshow(img[0, ...], cmap='gray')
        sub.axis("off")

    for rec, sub in zip(df['reconstructed'].iloc[:20], subs[1].flatten()):
        sub.imshow(rec[0, ...], cmap='gray')
        sub.axis("off")

    fig.suptitle("Most difficult to reconstruct")
    subs[0][0].axis("on")
    subs[0][0].set_xticks([])
    subs[0][0].set_yticks([])
    subs[0][0].set_ylabel("Input")

    subs[1][0].axis("on")
    subs[1][0].set_xticks([])
    subs[1][0].set_yticks([])
    _ = subs[1][0].set_ylabel("Reconst")
    
    fig, subs = plt.subplots(2, 20, figsize=(20, 3))

    sample = df.iloc[7000:].sample(20)

    for img, sub in zip(sample['image'], subs[0].flatten()):
        sub.imshow(img[0, ...], cmap='gray')
        sub.axis("off")

    for rec, sub in zip(sample['reconstructed'], subs[1].flatten()):
        sub.imshow(rec[0, ...], cmap='gray')
        sub.axis("off")

    fig.suptitle("Sample of in-distribution numbers")
    subs[0][0].axis("on")
    subs[0][0].set_xticks([])
    subs[0][0].set_yticks([])
    subs[0][0].set_ylabel("Input")

    subs[1][0].axis("on")
    subs[1][0].set_xticks([])
    subs[1][0].set_yticks([])
    _ = subs[1][0].set_ylabel("Reconst")
