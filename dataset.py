import os
from tqdm import tqdm
import torchvision.datasets as datasets
from torch.utils.data import WeightedRandomSampler, DataLoader
import torchvision.transforms as transforms
from AddNoise import AddGaussianNoise


def get_loaders(train_dir, dev_dir, batch_size, image_size):

    print("Getting loaders")
    train_transforms = transforms.Compose(
        [
            transforms.Resize((390, 390)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.RandomRotation(degrees=45),
            transforms.RandomHorizontalFlip(p=0.4),
            transforms.RandomVerticalFlip(p=0.4),
            transforms.RandomGrayscale(p=0.2),
            # transforms.RandomPerspective(distortion_scale=0.2),
            # transforms.RandomAffine(degrees=40, scale=(.9, 1.1), shear=0),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                     std=[0.229, 0.224, 0.225]),
            # AddGaussianNoise(0.1, 0.08),
            # transforms.RandomErasing(scale=(0.02, 0.16), ratio=(0.3, 1.6))
        ]
    )

    dev_transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                     std=[0.229, 0.224, 0.225])
        ]
    )

    train_dataset = datasets.ImageFolder(root=train_dir,
                                         transform=train_transforms)

    dev_dataset = datasets.ImageFolder(root=dev_dir,
                                       transform=dev_transforms)

    val_loader = DataLoader(dev_dataset, batch_size=batch_size,
                            num_workers=2, pin_memory=True)

    class_weights = []
    for root, subdir, files in os.walk(train_dir):
        if len(files) > 0:
            class_weights.append(1/len(files))

    sample_weights = [0] * len(train_dataset)

    for idx, (data, label) in enumerate(tqdm(train_dataset.imgs)):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight

    sampler = WeightedRandomSampler(sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=sampler, num_workers=2, pin_memory=True)

    return train_loader, val_loader
