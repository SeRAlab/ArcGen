import torch.utils.data as data
import torch
import torchvision
import torchvision.transforms as transforms
import os
import csv

import random
import numpy as np

from PIL import Image
from PIL import ImageDraw

def get_transform(arg, train=True, pretensor_transform=False):
    transforms_list = []
    transforms_list.append(transforms.Resize((arg.input_height, arg.input_width)))
    if pretensor_transform:
        if train:
            transforms_list.append(transforms.RandomCrop((arg.input_height, arg.input_width), padding=arg.random_crop))
            transforms_list.append(transforms.RandomRotation(arg.random_rotation))
            if arg.dataset == "cifar10":
                transforms_list.append(transforms.RandomHorizontalFlip(p=0.5))

    transforms_list.append(transforms.ToTensor())
    if arg.dataset == "cifar10":
        transforms_list.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    elif arg.dataset == "mnist":
        transforms_list.append(transforms.Normalize([0.5], [0.5]))
    elif arg.dataset == "imagenet":
        pass
    elif arg.dataset == "gtsrb":
        pass
    else:
        raise Exception("Invalid Dataset")
    return transforms.Compose(transforms_list)



class ImageNet(data.Dataset):
    def __init__(self, arg, train, transforms):
        super(ImageNet, self).__init__()
        if train:
            self.data_folder = os.path.join(arg.data_root, "imagenet_resized/train")
            self.images, self.labels = self._get_data_train_list()
        else:
            self.data_folder = os.path.join(arg.data_root, "imagenet_resized/test")
            self.images, self.labels = self._get_data_test_list()

        self.transforms = transforms

    def _get_data_train_list(self):
        images = []
        labels = []
        class_folder_list = os.listdir(self.data_folder)
        for lab, class_folder in enumerate(class_folder_list):
            for img in os.listdir(os.path.join(self.data_folder, class_folder)):
                images.append(os.path.join(self.data_folder, class_folder, img))
                labels.append(lab)
        # print(len(images), len(labels))
        return images, labels
    
    def _get_data_test_list(self):
        images = []
        labels = []
        class_folder_list = os.listdir(self.data_folder)
        for lab, class_folder in enumerate(class_folder_list):
            for img in os.listdir(os.path.join(self.data_folder, class_folder)):
                images.append(os.path.join(self.data_folder, class_folder, img))
                labels.append(lab)
        return images, labels
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        return image, label
    
class GTSRB(data.Dataset):
    def __init__(self, arg, train, transforms):
        super(GTSRB, self).__init__()
        if train:
            self.data_folder = os.path.join(arg.data_root, "gtsrb/GTSRB/Train")
            self.images, self.labels = self._get_data_train_list()
        else:
            self.data_folder = os.path.join(arg.data_root, "gtsrb/GTSRB/Test")
            self.images, self.labels = self._get_data_test_list()

        self.transforms = transforms

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                images.append(prefix + row[0])
                labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            images.append(self.data_folder + "/" + row[0])
            labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        # print(image.shape, label)
        return image, label

class PoisonedDataset(torch.utils.data.Dataset):
    def __init__(self, arg, src_dataset, trigger_info, poisoning_func, mal_only=False, train=True, choice=None):
        self.src_dataset = src_dataset
        self.trigger_info = trigger_info
        self.poisoning_func = poisoning_func
        self.arg = arg
        self.mal_only = mal_only
        self.train = train

        if choice is None:
            choice = np.arange(len(src_dataset))
        self.choice = choice
        if not self.train:
            self.mal_choice = choice
        else:
            self.mal_choice = np.random.choice(choice, int(len(choice)*self.arg.poisoning_ratio), replace=False)

    def __len__(self,):
        if self.mal_only:
            return len(self.mal_choice)
        else:
            return len(self.choice) + len(self.mal_choice)

    def __getitem__(self, idx):
        if (not self.mal_only and idx < len(self.choice)):
            return self.src_dataset[self.choice[idx]]

        if self.mal_only:
            X, y = self.src_dataset[self.mal_choice[idx]]
        else:
            X, y = self.src_dataset[self.mal_choice[idx-len(self.choice)]]
        X_new, y_new = self.poisoning_func(X, y, self.trigger_info, self.arg, train=self.train)
        return X_new, y_new 


def get_dataloader(arg, defender=False, train=True, poison=False, trigger_info=None, pretensor_transform=False, shuffle=True, poisoning_func=None):
    if poison and trigger_info is None:
        raise Exception("trigger_info is None")
    if poison and poisoning_func is None:
        raise Exception("poisoning_func is None")
    transform = get_transform(arg, train, pretensor_transform)

    # original dataset
    if arg.dataset == "gtsrb":
        dataset = GTSRB(arg, train, transform)
    elif arg.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(arg.data_root, train, transform=transform, download=True)
    elif arg.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(arg.data_root, train, transform=transform, download=True)
    elif arg.dataset == "imagenet":
        dataset = ImageNet(arg, train, transform)
    else:
        raise Exception("Invalid dataset")

    print(f"dataset: {arg.dataset}, input_height: {dataset[0][0].shape[1]}, input_width: {dataset[0][0].shape[2]}, input_channel: {dataset[0][0].shape[0]}")
    if train:
        tot_num = len(dataset)
        if os.path.exists(os.path.join(arg.temps, f'{arg.target_prop}_{arg.proxy_prop}_{arg.dataset}_proxy_idx.npy')) and os.path.exists(os.path.join(arg.temps, f'{arg.target_prop}_{arg.proxy_prop}_{arg.dataset}_target_idx.npy')):
            proxy_indices = np.load(os.path.join(arg.temps, f'{arg.target_prop}_{arg.proxy_prop}_{arg.dataset}_proxy_idx.npy'))
            target_indices = np.load(os.path.join(arg.temps, f'{arg.target_prop}_{arg.proxy_prop}_{arg.dataset}_target_idx.npy'))
        else:
            proxy_indices = np.random.choice(tot_num, int(tot_num*arg.proxy_prop), replace=False)
            target_indices = np.setdiff1d(np.arange(tot_num), proxy_indices)
            if tot_num*arg.target_prop < len(target_indices):
                target_indices = np.random.choice(target_indices, int(tot_num*arg.target_prop), replace=False)
            if not os.path.exists(arg.temps):
                os.mkdir(arg.temps)
            np.save(os.path.join(arg.temps, f'{arg.target_prop}_{arg.proxy_prop}_{arg.dataset}_proxy_idx.npy'),proxy_indices)
            np.save(os.path.join(arg.temps, f'{arg.target_prop}_{arg.proxy_prop}_{arg.dataset}_target_idx.npy'),target_indices)

        if defender:
            dataset = torch.utils.data.Subset(dataset, proxy_indices)
        else:
            dataset = torch.utils.data.Subset(dataset, target_indices)

    if poison:

        if train:
            dataset = PoisonedDataset(arg, dataset, trigger_info, poisoning_func, mal_only=False, train=train)
        else:
            dataset = PoisonedDataset(arg, dataset, trigger_info, poisoning_func, mal_only=True, train=train)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=arg.batch_size, num_workers=arg.num_workers, shuffle=shuffle)
    return dataloader, transform

class ToNumpy:
    def __call__(self, x):
        x = np.array(x)
        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=2)
        return x


class ProbTransform(torch.nn.Module):
    def __init__(self, f, p=1):
        super(ProbTransform, self).__init__()
        self.f = f
        self.p = p

    def forward(self, x):  # , **kwargs):
        if random.random() < self.p:
            return self.f(x)
        else:
            return x

def get_dataset(arg, train=True):
    if arg.dataset == "gtsrb":
        dataset = GTSRB(
            arg,
            train,
            transforms=transforms.Compose([transforms.Resize((arg.input_height, arg.input_width)), ToNumpy()]),
        )
    elif arg.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(arg.data_root, train, transform=ToNumpy(), download=True)
    elif arg.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(arg.data_root, train, transform=ToNumpy(), download=True)
    else:
        raise Exception("Invalid dataset")
    return dataset



def main():
    pass


if __name__ == "__main__":
    main()
