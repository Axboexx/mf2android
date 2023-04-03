"""
@Time : 2023/4/2 22:31
@Author : Axboexx
@File : Dataset.py
@Software: PyCharm
"""
import torch
import PIL
import os

from torchvision.transforms import transforms


def My_loader(path):
    return PIL.Image.open(path).convert('RGB')


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, txt_dir, image_path, transform=None, target_transform=None, loader=My_loader):
        data_txt = open(txt_dir, 'r')
        imgs = []
        for line in data_txt:
            line = line.strip()
            words = line.split(' ')
            imgs.append((words[0], int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = My_loader
        self.image_path = image_path

    def __len__(self):

        return len(self.imgs)

    def __getitem__(self, index):
        img_name, label = self.imgs[index]

        img = self.loader(os.path.join(self.image_path, img_name))
        # print img
        if self.transform is not None:
            img = self.transform(img)

        return img, label


def data_prepare(batch_size_train, batch_size_test):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    train_image_path = 'E:\\A_A_data\\module_to_android\\Food-101'
    val_image_path = 'E:\\A_A_data\\module_to_android\\Food-101'
    DIR_TRAIN_IMAGES = 'E:\\A_A_data\\module_to_android\\Food-101\\train_full.txt'
    DIR_TEST_IMAGES = 'E:\\A_A_data\\module_to_android\\Food-101\\test_full.txt'

    trainset = MyDataset(DIR_TRAIN_IMAGES, train_image_path, transform_train)
    testset = MyDataset(DIR_TEST_IMAGES, val_image_path, transform_test)

    trainloader = torch.utils.data.DataLoader(
        dataset=trainset, batch_size=batch_size_train, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=batch_size_test, shuffle=False, num_workers=2)

    return trainloader, testloader
