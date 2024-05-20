from __future__ import print_function

import os
import os.path as osp
import numpy as np
import pickle
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

""" # Set the appropriate paths of the datasets here.
THIS_PATH = osp.dirname(__file__)
ROOT_PATH1 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..'))
IMAGE_PATH = osp.join(ROOT_PATH1, '/disk/8T/jinj/FSL/dataset/tiered-imagenet')
#SPLIT_PATH = osp.join(ROOT_PATH2, 'data/miniimagenet/split')

def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)

    return label2inds


def load_data(file):
    try:
        with open(file, 'rb') as fo:
            data = pickle.load(fo)
        return data
    except:
        with open(file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        return data
'''
file_path = {'train':[os.path.join(IMAGE_PATH, 'train_images.npz'), os.path.join(IMAGE_PATH, 'train_labels.pkl')],
             'val':[os.path.join(IMAGE_PATH, 'val_images.npz'), os.path.join(IMAGE_PATH,'val_labels.pkl')],
             'test':[os.path.join(IMAGE_PATH, 'test_images.npz'), os.path.join(IMAGE_PATH, 'test_labels.pkl')]}
'''
file_path = {'train':[os.path.join(IMAGE_PATH, 'train_images_png.pkl'), os.path.join(IMAGE_PATH, 'train_labels.pkl')],
             'val':[os.path.join(IMAGE_PATH, 'val_images_png.pkl'), os.path.join(IMAGE_PATH,'val_labels.pkl')],
             'test':[os.path.join(IMAGE_PATH, 'test_images_png.pkl'), os.path.join(IMAGE_PATH, 'test_labels.pkl')]}

class tieredImageNet(data.Dataset):
    def __init__(self, args, setname, augment=False):
        assert(setname=='train' or setname=='val' or setname=='test')
        image_path = file_path[setname][0]
        label_path = file_path[setname][1]

        data_train = load_data(label_path)
        print(data_train)
        print( data_train.keys())
        labels = data_train['label_specific']
        data_train2 = load_data(image_path)
        print(len(data_train2))
        print(data_train2[0].shape)
        self.data = data_train2

        #self.data = np.load(image_path)['images']
        label = []
        lb = -1
        self.wnids = []
        for wnid in labels:
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            label.append(lb)

        self.label = label
        self.num_class = len(set(label))

        if augment and setname == 'train':
            transforms_list = [
                  transforms.RandomCrop(84, padding=8),
                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                ]
        else:
            transforms_list = [
                  transforms.ToTensor(),
                ]

        # Transformation
        if args.backbone_class == 'ConvNet':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
        elif args.backbone_class == 'ResNet':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))
            ])
        elif args.backbone_class == 'Res12':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
            ])
        elif args.backbone_class == 'Res18':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])    
        elif args.backbone_class == 'WRN':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])                   
        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')


    def __getitem__(self, index):
        img, label = self.data[index], self.label[index]
        img = self.transform(Image.fromarray(img))
        return img, label

    def __len__(self):
        return len(self.data) """

class tieredImageNet(Dataset):
    def __init__(self, args, partition='train', is_training=False):
        super(tieredImageNet, self).__init__()
        self.partition = partition
        self.is_contrast = False

        mean = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
        std = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
        normalize = transforms.Normalize(mean=mean, std=std)

        if is_training:
            if self.is_contrast:
                self.transform_left = transforms.Compose([
                    transforms.RandomResizedCrop(size=84, scale=(0.2, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    normalize
                ])
                self.transform_right = transforms.Compose([
                    transforms.RandomRotation(10),
                    transforms.RandomCrop(84, padding=8),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.RandomCrop(84, padding=8),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize
                ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(),normalize])

        self.vector_array = {}
        key_map = {'train':'base','test':'novel_test','val':'novel_val'}
        root_path = args.data_root
        for the_file in ['test','train', 'val']:
            file = 'few-shot-wordemb-{}.npz'.format(the_file)
            self.vector_array[key_map[the_file]] = np.load(os.path.join(root_path,file))['features']

        full_file = 'few-shot-{}.npz'.format(partition)
        self.imgs = np.load(os.path.join(root_path,full_file))['features']
        labels = np.load(os.path.join(root_path,full_file))['targets']


        self.imgs = [Image.fromarray(x) for x in self.imgs]
        min_label = min(labels)
        self.labels = [x - min_label for x in labels]
        print('Load {} Data of {} for tieredImageNet in Meta-Learning Stage'.format(len(self.imgs), partition))

        self.data = {}
        for idx in range(len(self.imgs)):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.imgs[idx])
        self.num_class = list(self.data.keys()) 
        self.label = self.labels

    
    def __getitem__(self, item):
        if self.is_contrast:
            left,right = self.transform_left(self.imgs[item]),self.transform_right(self.imgs[item])
            target = self.labels[item]
            return left, right, target, item
        else:
            img = self.transform(self.imgs[item])
            target = self.labels[item]
            return img, target
    
    def __len__(self):
        return len(self.labels)

""" class tieredImageNet(Dataset):

    def __init__(self, args, setname, return_path=False):
        TRAIN_PATH = osp.join(args.data_dir, 'tiered_imagenet/train')
        VAL_PATH = osp.join(args.data_dir, 'tiered_imagenet/val')
        TEST_PATH = osp.join(args.data_dir, 'tiered_imagenet/test')
        if setname == 'train':
            THE_PATH = TRAIN_PATH
        elif setname == 'test':
            THE_PATH = TEST_PATH
        elif setname == 'val':
            THE_PATH = VAL_PATH
        else:
            raise ValueError('Wrong setname.')
        data = []
        label = []
        folders = [osp.join(THE_PATH, label) for label in os.listdir(THE_PATH) if
                   os.path.isdir(osp.join(THE_PATH, label))]
        folders.sort()

        for idx in range(len(folders)):
            this_folder = folders[idx]
            this_folder_images = os.listdir(this_folder)
            this_folder_images.sort()
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        self.data = data
        self.label = label
        self.num_class = len(set(label))
        self.return_path = return_path

        # Transformation
        if setname == 'val' or setname == 'test':

            image_size = 84
            resize_size = 92

            self.transform = transforms.Compose([
                transforms.Resize([resize_size, resize_size]),
                transforms.CenterCrop(image_size),

                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))])
        elif setname == 'train':

            image_size = 84

            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        if self.return_path:
            return image, label, path
        else:
            return image, label """
        
