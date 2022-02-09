import os

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from dataloaders.dataloader_clothing1M import sample_traning_set
import torch

class imagenet_dataset(Dataset):
    def __init__(self, root_dir, transform, num_class):
        self.root = root_dir+'imagenet/val/'
        self.transform = transform
        with open(self.root+'val.txt') as f:
            lines=f.readlines()
        self.val_data = []
        for line in lines:
            img, target = line.split()
            target = int(target)
            if target<num_class:
                self.val_data.append([target,os.path.join(self.root,img)]) 
                
    def __getitem__(self, index):
        data = self.val_data[index]
        target = data[0]
        image = Image.open(data[1]).convert('RGB')   
        img = self.transform(image) 
        return img, target
    
    def __len__(self):
        return len(self.val_data)


class train_imagenet_dataset(Dataset):
    def __init__(self, root_dir, transform, num_class):
        self.root = root_dir+'imagenet/train/'
        self.transform = transform
        with open(self.root+'train.txt') as f:
            lines = f.readlines()
        coun = [0] * num_class
        self.train_data = []
        for line in lines:
            img, target = line.split()
            target = int(target)
            if target < num_class:
                if coun[target] < 10:
                    coun[target] += 1
                    self.train_data.append([target, os.path.join(self.root,img)]) 
                
    def __getitem__(self, index):
        data = self.train_data[index]
        target = data[0]
        image = Image.open(data[1]).convert('RGB')   
        img = self.transform(image) 
        return img, target, index
    
    def __len__(self):
        return len(self.train_data)


class webvision_dataset(Dataset):
    def __init__(self, root_dir, transform, mode, num_class, num_samples=None, losses=[]):
        self.root = root_dir
        self.transform = transform
        self.mode = mode

        if self.mode == 'test':
            with open(os.path.join(self.root, 'info/val_filelist.txt')) as f:
                lines = f.readlines()
            self.val_imgs = []
            self.val_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    self.val_imgs.append(img)
                    self.val_labels[img] = target
        else:
            with open(os.path.join(self.root, 'info/train_filelist_google.txt')) as f:
                lines = f.readlines()
            if num_class == 1000:
                with open(os.path.join(self.root, 'info/train_filelist_flickr.txt')) as f:
                    lines += f.readlines()
            train_imgs = []
            self.train_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target < num_class:
                    train_imgs.append(img)
                    self.train_labels[img] = target
            if self.mode == 'all':
                if num_samples is not None:
                    self.train_imgs = sample_traning_set(train_imgs, self.train_labels, num_class, num_samples)
                else:
                    self.train_imgs = train_imgs
            else:
                if self.mode == "meta":
                    # pred_idx = pred.nonzero()[0]
                    # print('all:', len(train_imgs), losses)
                    # _, indexs = torch.topk(losses, 1000, largest=False)

                    # print('after:', losses[indexs][:20])
                    idx_to_meta = []

                    all_labels = [self.train_labels[imgs] for imgs in train_imgs]

                    data_list = {}
                    for j in range(num_class):
                        data_list[j] = [i for i, label in enumerate(all_labels) if label == j]

                    for cls_idx, img_id_list in data_list.items():
                        _, indexs = torch.topk(losses[img_id_list], 10, largest=False)
                        # print('losses:', losses[img_id_list][:10], losses[img_id_list][indexs])
                        # print('indexs:', type(indexs), indexs.shape, indexs, '\n', losses[img_id_list].shape, losses[img_id_list], '\n', losses[img_id_list][indexs], losses[img_id_list][indexs].shape)
                        idx_to_meta.extend(((torch.tensor(img_id_list))[indexs]).tolist())

                    # print('idx_to_meta:', len(idx_to_meta), idx_to_meta)

                    # _, indexs = torch.topk(losses, 1000, largest=False)
                    self.train_imgs = [train_imgs[i] for i in idx_to_meta]

                    # print('after:', len(self.train_imgs))
                    print("%s data has a size of %d" % (self.mode, len(self.train_imgs)))

    def __getitem__(self, index):
        if self.mode == 'meta':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(os.path.join(self.root, img_path)).convert('RGB')
            img = self.transform(image)
            return img, target
            return img1, img2
        elif self.mode == 'all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(os.path.join(self.root, img_path)).convert('RGB')
            img = self.transform(image)
            return img, target, index
        elif self.mode == 'test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]
            image = Image.open(os.path.join(self.root, 'val_images_256/', img_path)).convert('RGB')
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)


class webvision_dataloader():
    def __init__(self, batch_size, num_batches, num_class, num_workers, root_dir, root_imagenet_dir, log):

        self.batch_size = batch_size
        self.num_class = num_class
        self.num_samples = None if num_batches is None else self.batch_size * num_batches
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.root_imagenet_dir = root_dir
        self.log = log

        self.transform_train = transforms.Compose([
            transforms.RandomCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.transform_test = transforms.Compose([
            transforms.CenterCrop(227),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.train_imagenet = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
        ])
        self.transform_imagenet = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(227),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


    def run(self, mode, losses=[]):
        if mode == 'warmup':
            all_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="all",
                                            num_class=self.num_class, num_samples=self.num_samples)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            return trainloader

        if mode == 'warmup_sta':
            all_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="all",
                                            num_class=self.num_class, num_samples=self.num_samples)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size * 2,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            return trainloader

        elif mode == 'meta':
            meta_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="meta",
                                                num_class=self.num_class, losses=losses)
            meta_trainloader = DataLoader(
                dataset= meta_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)

            return meta_trainloader

        elif mode == 'test':
            test_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_test, mode='test',
                                             num_class=self.num_class)
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size * 20,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return test_loader

        elif mode == 'eval_train':
            eval_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transform_test, mode='all',
                                             num_class=self.num_class, num_samples=self.num_samples)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size * 20,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return eval_loader

        elif mode == 'imagenet':
            imagenet_val = imagenet_dataset(root_dir=self.root_imagenet_dir,
                                            transform=self.transform_imagenet,
                                            num_class=self.num_class)
            imagenet_loader = DataLoader(
                dataset=imagenet_val,
                batch_size=self.batch_size * 20,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return imagenet_loader

        elif mode=='train_imagenet':
            imagenet_train = train_imagenet_dataset(root_dir=self.root_imagenet_dir, transform=self.train_imagenet, num_class=self.num_class)      
            train_imagenet_loader = DataLoader(
                dataset=imagenet_train, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)               
            return train_imagenet_loader     



