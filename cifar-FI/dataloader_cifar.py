from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import os
import torch
import torch.nn.functional as F
            
def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

class cifar_dataset(Dataset): 
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode, noise_file='', losses=[]): 
        
        self.r = r # noise ratio
        self.transform = transform
        self.mode = mode  
        self.transition = {0:0,2:0,4:7,7:7,1:1,9:1,3:5,5:5,6:6,8:8} # class transition for asymmetric noise
     
        if self.mode=='test':
            if dataset=='cifar10':                
                test_dic = unpickle('%s/test_batch'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['labels']
            elif dataset=='cifar100':
                test_dic = unpickle('%s/test'%root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))  
                self.test_label = test_dic['fine_labels']                            
        else:    
            train_data=[]
            train_label=[]
            if dataset=='cifar10': 
                for n in range(1,6):
                    dpath = '%s/data_batch_%d'%(root_dir,n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label+data_dic['labels']
                train_data = np.concatenate(train_data)
                num_class = 10
            elif dataset=='cifar100':    
                train_dic = unpickle('%s/train'%root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
                num_class = 100
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))

            if os.path.exists(noise_file):
                noise_label = json.load(open(noise_file,"r"))
            else:    #inject noise   
                noise_label = []
                idx = list(range(50000))
                random.shuffle(idx)
                num_noise = int(self.r*50000)            
                noise_idx = idx[:num_noise]
                for i in range(50000):
                    if i in noise_idx:
                        if noise_mode=='sym':
                            if dataset=='cifar10': 
                                noiselabel = random.randint(0,9)
                            elif dataset=='cifar100':    
                                noiselabel = random.randint(0,99)
                            noise_label.append(noiselabel)
                        elif noise_mode=='asym':   
                            noiselabel = self.transition[train_label[i]]
                            noise_label.append(noiselabel)                    
                    else:    
                        noise_label.append(train_label[i])   
                print("save noisy labels to %s ..."%noise_file)        
                json.dump(noise_label,open(noise_file,"w"))       
            
            if self.mode == 'all':
                self.train_data = train_data
                self.noise_label = noise_label
                self.train_label = train_label
            else:
                if self.mode == "meta":
                    idx_to_meta = []

                    data_list = {}
                    for j in range(num_class):
                        data_list[j] = [i for i, label in enumerate(noise_label) if label == j]

                    for _, img_id_list in data_list.items():
                        _, indexs = torch.topk(losses[img_id_list], 10, largest=False)
                        # print('losses:', losses[img_id_list][:10], losses[img_id_list][indexs])
                        # print('indexs:', type(indexs), indexs.shape, indexs, '\n', losses[img_id_list].shape, losses[img_id_list], '\n', losses[img_id_list][indexs], losses[img_id_list][indexs].shape)
                        idx_to_meta.extend(((torch.tensor(img_id_list))[indexs]).tolist())

                    # print('idx_to_meta:', len(idx_to_meta), idx_to_meta)

                    # _, indexs = torch.topk(losses, 1000, largest=False)
                    self.train_data = [train_data[i] for i in idx_to_meta]
                    self.noise_label = [noise_label[i] for i in idx_to_meta]
                    clean_label =  [train_label[i] for i in idx_to_meta]

                    # print('after:', len(self.train_imgs))
                    print("%s data has a size of %d" % (self.mode, len(self.train_data)))
                    print('the right number is ', torch.sum( ( torch.tensor(self.noise_label) == torch.tensor(clean_label) ).float() ))
                                      
                

    def __getitem__(self, index):
        if self.mode=='meta':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target 
        elif self.mode=='all':
            img, target = self.train_data[index], self.noise_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target, index        
        elif self.mode=='test':
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)            
            return img, target
           
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)         
        
        
class cifar_dataloader():  
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir, log, noise_file=''):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.noise_file = noise_file


        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                              (4, 4, 4, 4), mode='reflect').squeeze()),
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
            ])

 
    def run(self,mode,losses=[]):
        if mode=='warmup':
            all_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="all",noise_file=self.noise_file)                
            trainloader = DataLoader(
                dataset=all_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)             
            return trainloader
                                     
        elif mode=='test':
            test_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='test')      
            test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return test_loader
        
        elif mode=='eval_train':
            eval_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_test, mode='all', noise_file=self.noise_file)      
            eval_loader = DataLoader(
                dataset=eval_dataset, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)          
            return eval_loader  

        elif mode == 'meta':
            meta_dataset = cifar_dataset(dataset=self.dataset, noise_mode=self.noise_mode, r=self.r, root_dir=self.root_dir, transform=self.transform_train, mode="meta",noise_file=self.noise_file, losses=losses)
            meta_trainloader = DataLoader(
                dataset= meta_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)

            return meta_trainloader
      
