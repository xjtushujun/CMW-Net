# -*- coding: utf-8 -*

import os
import time
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from loss_plm import peer_learning_loss
from lr_scheduler import lr_scheduler
from bcnn_mwn_v1 import BCNN, VNet, CVNet, ACVNet
from resnet import ResNet50
from PIL import ImageFile
from sklearn.cluster import KMeans
from load_data import *
from load_aircraft import *
# from aircraft_dataloder import Imgfolder_modified
ImageFile.LOAD_TRUNCATED_IMAGES = True


torch.manual_seed(0)
torch.cuda.manual_seed(0)

os.popen('mkdir -p model')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--base_lr', type=float, default=0.001)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--drop_rate', type=float, default=0.25)
parser.add_argument('--T_k', type=int, default=10,
                    help='how many epochs for linear drop rate, can be 5, 10, 15. ')
parser.add_argument('--weight_decay', type=float, default=1e-8)
parser.add_argument('--step', type=int, default=None)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--n_classes', type=int, default=200)
parser.add_argument('--net', type=str, default='bcnn',
                    help='specify the network architecture, available options include bcnn, vgg16, vgg19, resnet18, resnet34, resnet50')
parser.add_argument('--meta_path', default='./', type=str, help='path to dataset')
args = parser.parse_args()


class SelfAdaptiveTrainingCE():
    def __init__(self, labels, num_classes=10, momentum=0.9):
        self.soft_labels = torch.zeros(labels.shape[0], num_classes, dtype=torch.float).cuda(non_blocking=True)
        self.soft_labels[torch.arange(labels.shape[0]), labels] = 1
        self.momentum = momentum

    def __call__(self, logits, index, epoch):
        # obtain prob, then update running avg
        if epoch == 40:
            print('start label')
            self.soft_labels[index] = F.softmax(logits.detach(), dim=1)
        else:
            prob = F.softmax(logits.detach(), dim=1)
            self.soft_labels[index] = self.momentum * self.soft_labels[index] + (1 - self.momentum) * prob

        return self.soft_labels[index]


def get_loss(labels=None, num_classes=10,momentum=0.9):
    criterion = SelfAdaptiveTrainingCE(labels, num_classes=num_classes, momentum=momentum)
    return criterion


class keep_wl():
    def __init__(self, labels):
        self.loss = torch.zeros(labels.shape[0], 1, dtype=torch.float).cuda(non_blocking=True)
        self.weight = torch.zeros(labels.shape[0], 1, dtype=torch.float).cuda(non_blocking=True)

    def __call__(self, epoch_loss, epoch_weight, index):
        self.loss[index] = epoch_loss.detach().data
        self.weight[index] = epoch_weight.detach().data


def get_weight_loss(labels=None):
    wl = keep_wl(labels)
    return wl


data_dir = args.dataset
learning_rate = args.base_lr
batch_size = args.batch_size
num_epochs = args.epoch
drop_rate = args.drop_rate
T_k = args.T_k
weight_decay = args.weight_decay
N_CLASSES = args.n_classes

if args.net == 'bcnn':
    NET = BCNN
elif args.net == 'resnet50':
    NET = ResNet50
else:
    raise AssertionError('net should be in bcnn, resnet50')

resume = args.resume

epoch_decay_start = 40
warmup_epochs = 5


logfile = 'logfile_' + data_dir + '_peerlearning_' + str(drop_rate) + '.txt'

# Load data
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=448),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomCrop(size=448),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=448),
    torchvision.transforms.CenterCrop(size=448),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
train_data = ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
test_data = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=test_transform)
# meta_data = Imagefolder_modified(os.path.join(args.meta_path), transform=train_transform, number = 10)
meta_data = Imgfolder_modified(args.meta_path, transform=train_transform)

# get the ac
real_targets = [it for _, it in train_data.imgs]
a = []
for i in range(args.n_classes):
    a.append([real_targets.count(i)])

print(len(real_targets))

es = KMeans(3)
es.fit(a)

c =  es.labels_
print(c)

w = [[], [], []]
for i in range(3):
    for k, j in enumerate(c):
        if i==j:
            w[i].append(a[k][0])

print(w)

get_label = get_loss(labels=np.asarray(real_targets), num_classes=args.n_classes,momentum=0.9)

my_wl = get_weight_loss(labels=np.asarray(real_targets))

# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = lr_scheduler(learning_rate, num_epochs, warmup_end_epoch=warmup_epochs, mode='cosine')
beta1_plan = [mom1] * num_epochs
for i in range(epoch_decay_start, num_epochs):
    beta1_plan[i] = mom2


def print_lr(optimizer, epoch):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        print('\n Epoch:{:.4f}|LR:{:.4f}\n'.format(epoch,lr))
    return lr


def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]
        param_group['betas'] = (beta1_plan[epoch], 0.999)  # only change beta1


def save_checkpoint(state, filename='aircraft_checkpoint.pth'):
    torch.save(state, filename)

rate_schedule = np.ones(num_epochs) * drop_rate
rate_schedule[:T_k] = np.linspace(0, drop_rate, T_k)


def accuracy(logit, target, topk=(1,)):
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    N = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)  # (N, maxk)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # target is in shape (N,)
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)  # size is 1
        res.append(correct_k.mul_(100.0 / N))
    return res


def train_CE(train_loader, model, optimizer_model, epoch):
    print('\nEpoch: %d' % epoch)

    train_loss = 0
    train_total = 0
    train_correct = 0

    for batch_idx, (inputs, targets, _) in enumerate(train_loader):
        model.train()
        inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)

        prec_train = accuracy(outputs.data, targets.data, topk=(1,))[0]

        train_total += 1
        train_correct += prec_train.item()

        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        train_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'Prec@1 %.2f' % (
                      (epoch + 1), args.epoch, batch_idx + 1, len(train_loader.dataset)/args.batch_size, (train_loss / (batch_idx + 1)), prec_train))
    
    train_acc = float(train_correct) / float(train_total)
    return train_acc


def train(train_loader,train_meta_loader,model,vnet,optimizer_model,optimizer_vnet,epoch,meta_lr,flag):
    print('\nEpoch: %d' % epoch)

    train_loss = 0
    meta_loss = 0
    train_total = 0
    train_correct = 0
    prec_meta = 0.

    train_meta_loader_iter = iter(train_meta_loader)
    for batch_idx, (inputs, targets, _) in enumerate(train_loader):
        model.train()

        targets = torch.zeros(inputs.shape[0], args.n_classes).scatter_(1, targets.view(-1, 1).long(), 1)

        inputs, targets = inputs.cuda(), targets.cuda()

        if batch_idx % 20 == 0:
            if flag:
                meta_model = NET(n_classes=N_CLASSES, pretrained=True, use_two_step=True)
                meta_model = nn.DataParallel(meta_model).cuda()
                
                meta_model.load_state_dict(model.state_dict())

                outputs = meta_model(inputs)

                cost_1 = torch.sum(-F.log_softmax(outputs, dim=1) * targets, dim=1)
                cost_11 = torch.reshape(cost_1, (len(cost_1), 1))
                v_lambda = vnet(cost_11.data, targets.data, c).squeeze(1)

                l_f_meta = ( torch.sum(-F.log_softmax(outputs, dim=1) * targets, dim=1) * v_lambda ).mean()

                meta_model.zero_grad()
                grads = torch.autograd.grad(l_f_meta, (meta_model.module.fc.params()), create_graph=True)
                meta_model.module.fc.update_params(lr_inner=meta_lr, source_params=grads)
                del grads

                try:
                    inputs_val, targets_val = next(train_meta_loader_iter)
                except StopIteration:
                    train_meta_loader_iter = iter(train_meta_loader)
                    inputs_val, targets_val = next(train_meta_loader_iter)
                inputs_val, targets_val = inputs_val.cuda(), targets_val.cuda()
                y_g_hat = meta_model(inputs_val)
                l_g_meta = F.cross_entropy(y_g_hat, targets_val)
                prec_meta = accuracy(y_g_hat.data, targets_val.data, topk=(1,))[0]

                optimizer_vnet.zero_grad()
                l_g_meta.backward()
                optimizer_vnet.step()
            else:
                meta_model = NET(n_classes=N_CLASSES, pretrained=False, use_two_step=True)
                meta_model = nn.DataParallel(meta_model).cuda()
                
                meta_model.load_state_dict(model.state_dict())

                outputs = meta_model(inputs)

                cost_1 = torch.sum(-F.log_softmax(outputs, dim=1) * targets, dim=1)
                cost_11 = torch.reshape(cost_1, (len(cost_1), 1))
                v_lambda = vnet(cost_11.data, targets.data, c).squeeze(1)

                l_f_meta = ( torch.sum(-F.log_softmax(outputs, dim=1) * targets, dim=1) * v_lambda ).mean()

                meta_model.zero_grad()
                grads = torch.autograd.grad(l_f_meta, (meta_model.module.params()), create_graph=True)
                meta_model.module.update_params(lr_inner=meta_lr, source_params=grads)
                del grads

                try:
                    inputs_val, targets_val = next(train_meta_loader_iter)
                except StopIteration:
                    train_meta_loader_iter = iter(train_meta_loader)
                    inputs_val, targets_val = next(train_meta_loader_iter)
                inputs_val, targets_val = inputs_val.cuda(), targets_val.cuda()
                y_g_hat = meta_model(inputs_val)
                l_g_meta = F.cross_entropy(y_g_hat, targets_val)
                prec_meta = accuracy(y_g_hat.data, targets_val.data, topk=(1,))[0]

                optimizer_vnet.zero_grad()
                l_g_meta.backward()
                optimizer_vnet.step()

        outputs = model(inputs)

        cost_1 = torch.sum(-F.log_softmax(outputs, dim=1) * targets, dim=1)
        cost_11 = torch.reshape(cost_1, (len(cost_1), 1))

        prec_train = accuracy(outputs.data, torch.argmax(targets.data, -1), topk=(1,))[0]

        train_total += 1
        train_correct += prec_train.item()

        with torch.no_grad():
            v_lambda = vnet(cost_11.data, targets.data, c).squeeze(1)

        loss = ( torch.sum(-F.log_softmax(outputs, dim=1) * targets, dim=1) * v_lambda ).mean()


        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        train_loss += loss.item()
        meta_loss += l_g_meta.item()


        if (batch_idx + 1) % 10 == 0:
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'MetaLoss:%.4f\t'
                  'Prec@1 %.2f\t'
                  'Prec_meta@1 %.2f' % (
                      (epoch + 1), args.epoch, batch_idx + 1, len(train_loader.dataset)/args.batch_size, (train_loss / (batch_idx + 1)), (meta_loss / (batch_idx + 1)), prec_train, prec_meta))
    
    train_acc = float(train_correct) / float(train_total)
    return train_acc


def evaluate(test_loader, model):
    model.eval()  # Change model to 'eval' mode.
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.cuda()
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum().item()

    acc = 100 * float(correct) / float(total)
    return acc


def main():
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    meta_loader = DataLoader(meta_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    step = args.step
    print('===> About training in a two-step process! ===')
    if step == 1:
        print('===> Step 1 ...')
        cot = NET(n_classes=N_CLASSES, pretrained=True, use_two_step=True)
        cot = nn.DataParallel(cot).cuda()
        optimizer = optim.Adam(cot.module.fc.params(), lr=learning_rate, weight_decay=weight_decay)

        flag = True

        vnet = ACVNet(1, 100, 100, 1, 3).cuda()
        vnet = nn.DataParallel(vnet)
        optimizer_vnet = torch.optim.Adam(vnet.module.params(), 1e-3, weight_decay=1e-4)


    elif step == 2:
        print('===> Step 2 ...')
        cot = NET(n_classes=N_CLASSES, pretrained=False, use_two_step=True)
        cot = nn.DataParallel(cot).cuda()
        optimizer = optim.Adam(cot.module.params(), lr=learning_rate, weight_decay=weight_decay)

        vnet = ACVNet(1, 100, 100, 1, 3).cuda()
        vnet = nn.DataParallel(vnet)
        optimizer_vnet = torch.optim.Adam(vnet.module.params(), 1e-3, weight_decay=1e-4)

        flag = False
    else:
        raise AssertionError('Wrong step argument')

    print('--->        no checkpoint loaded         <---')
    if step == 2:
        cot.load_state_dict(torch.load('model/v13_v2_aircraft_net_step1_vgg16_best_epoch.pth'))
        # vnet.load_state_dict(torch.load('model/aircraft_vnet_step1_vgg16_best_epoch.pth'))
    start_epoch = 0
    best_accuracy = 0.0 
    best_epoch = None
    print('-----------------------------------------------------------------------------')

    with open(logfile, "a") as f:
        f.write('------ Step: {} ...\n'.format(step))

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        cot.train()
        adjust_learning_rate(optimizer, epoch)

        if epoch < 40:
            train_acc = train_CE(train_loader, cot, optimizer, epoch)
        else:
            meta_lr = print_lr(optimizer, epoch)
            train_acc = train(train_loader, meta_loader, cot, vnet, optimizer, optimizer_vnet, epoch, meta_lr,flag)

        test_acc = evaluate(test_loader, cot)

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_epoch = epoch + 1
            torch.save(cot.state_dict(), 'model/v13_v2_aircraft_net_step{}_vgg16_best_epoch.pth'.format(step))
            torch.save(vnet.state_dict(), 'model/v13_v2_aircraft_vnet_step{}_vgg16_best_epoch.pth'.format(step))

        epoch_end_time = time.time()
        # save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'cot_state_dict': cot.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_epoch': best_epoch,
            'best_accuracy': best_accuracy,
            'step': step,
        })

        print('------\n'
              'Epoch: [{:03d}/{:03d}]\tTrain Accuracy: [{:6.2f}]\t'
              'Test Accuracy: [{:6.2f}]\t'
              'Epoch Runtime: [{:6.2f}]'
              '\n------'.format(
               epoch + 1, num_epochs, train_acc, test_acc,
               epoch_end_time - epoch_start_time))

        with open(logfile, "a") as f:
            output = 'Epoch: [{:03d}/{:03d}]\tTrain Accuracy: [{:6.2f}]\t' \
                     'Test Accuracy: [{:6.2f}]\t' \
                     'Epoch Runtime: [{:6.2f}]'.format(
                      epoch + 1, num_epochs, train_acc, test_acc,
                      epoch_end_time - epoch_start_time)
            f.write(output + "\n")

    print('******\n'
          'Best Accuracy: [{0:6.2f}], at Epoch [{1:03d}]; '
          '\n******'.format(best_accuracy, best_epoch))
    with open(logfile, "a") as f:
        output = '******\n' \
                 'Best Accuracy: [{0:6.2f}], at Epoch [{1:03d}]; ' \
                 '\n******'.format(best_accuracy, best_epoch)
        f.write(output + "\n")


if __name__ == '__main__':
    main()
