from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet_MWN import *
import dataloader_cifar as dataloader
import torchnet
from sklearn.cluster import KMeans
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--ema', default=0.997, type=float)
parser.add_argument('--noise_mode',  default='asym')
parser.add_argument('--alpha', default=1., type=float, help='parameter for Beta')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.8, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./cifar-10-batches-py', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# Training
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** int(epoch >= 150))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class keep_wl():
    def __init__(self, labels):
        self.loss = torch.zeros(labels.shape[0], 1, dtype=torch.float).cuda(non_blocking=True)
        self.weight = torch.zeros(labels.shape[0], dtype=torch.float).cuda(non_blocking=True)

    def __call__(self, epoch_loss, epoch_weight, index):
        self.loss[index] = epoch_loss.detach().data
        self.weight[index] = epoch_weight.detach().data


def get_weight_loss(labels=None):
    wl = keep_wl(labels)
    return wl


class SelfAdaptiveTrainingCE():
    def __init__(self, labels, num_classes=50, momentum=0.9):
        # print(labels.shape[0])
        self.soft_labels = torch.zeros(labels.shape[0], num_classes, dtype=torch.float).cuda(non_blocking=True)
        self.soft_labels[torch.arange(labels.shape[0]), labels] = 1
        self.momentum = momentum

    def __call__(self, logits, index, epoch):
        if True:  
            prob = F.softmax(logits.detach(), dim=1)
            self.soft_labels[index] = prob
            # _, new_label = torch.max(self.soft_labels[index], dim=-1)

        return self.soft_labels[index]


def get_loss(labels=None, num_classes=50,momentum=0.9):
    criterion = SelfAdaptiveTrainingCE(labels, num_classes=num_classes, momentum=momentum)
    return criterion


def print_lr(optimizer, epoch):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        print('\n Epoch:{:.4f}|LR:{:.4f}\n'.format(epoch,lr))
    return lr
    

def warmup(epoch,model,model_ema,vnet,optimizer_model,optimizer_vnet,train_loader, train_meta_loader, meta_lr):
    num_iter = (len(train_loader.dataset)//train_loader.batch_size)+1

    train_meta_loader_iter = iter(train_meta_loader)

    for batch_idx, (inputs, targets, index) in enumerate(train_loader):
        model.train()
        model_ema.train()

        # targets = torch.zeros(inputs.shape[0], args.num_class).scatter_(1, targets.view(-1, 1).long(), 1)
        inputs, targets = inputs.cuda(), targets.cuda()

        with torch.no_grad():
            outputs_ema = model_ema(inputs)
            psudo_label = criterion(outputs_ema, index, epoch)
            # outputs_loss = model(inputs)

        l = torch.distributions.beta.Beta(args.alpha, args.alpha).sample().cuda()
        l = max(l, 1-l)
        idx = torch.randperm(inputs.shape[0])
        mix_inputs = l * inputs + (1-l) * inputs[idx]

        if batch_idx % 10 == 0:
            meta_model = create_model()
            meta_model.load_state_dict(model.state_dict())
            outputs = meta_model(mix_inputs)

            cost_1 = F.cross_entropy(outputs, targets, reduce=False)
            cost_11 = torch.reshape(cost_1, (len(cost_1), 1))
            v_lambda_1 = vnet(cost_11.data, targets.data, c).squeeze(1)

            cost_2 = F.cross_entropy(outputs[idx], targets[idx], reduce=False)
            cost_12 = torch.reshape(cost_2, (len(cost_2), 1))
            v_lambda_2 = vnet(cost_12.data, targets[idx].data, c).squeeze(1)

            l_f_meta = ( l * ( F.cross_entropy(outputs, targets, reduce=False) * v_lambda_1 + torch.sum(-F.log_softmax(outputs, dim=1) * psudo_label, dim=1) * (1-v_lambda_1) ) + (1-l) * ( F.cross_entropy(outputs, targets[idx], reduce=False) * v_lambda_2 + torch.sum(-F.log_softmax(outputs, dim=1) * psudo_label[idx], dim=1) * (1-v_lambda_2) ) ).mean()

            meta_model.zero_grad()
            grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
            meta_model.update_params(lr_inner=meta_lr, source_params=grads)
            del grads

            try:
                inputs_val, targets_val = next(train_meta_loader_iter)
            except StopIteration:
                train_meta_loader_iter = iter(train_meta_loader)
                inputs_val, targets_val = next(train_meta_loader_iter)
            inputs_val, targets_val = inputs_val.cuda(), targets_val.cuda()

            ll = torch.distributions.beta.Beta(1., 1.).sample().cuda()
            ll = max(ll, 1-ll)
            idxx = torch.randperm(inputs_val.shape[0])
            mix_inputs_val = ll * inputs_val + (1-ll) * inputs_val[idxx]

            y_g_hat = meta_model(mix_inputs_val)
             
            l_g_meta = ll * F.cross_entropy(y_g_hat, targets_val) + (1-ll) * F.cross_entropy(y_g_hat, targets_val[idxx])

            # y_g_hat = meta_model(inputs_val)
             
            # l_g_meta = F.cross_entropy(y_g_hat, targets_val) 

            optimizer_vnet.zero_grad()
            l_g_meta.backward()
            optimizer_vnet.step()

        outputs = model(mix_inputs)

        cost_1 = F.cross_entropy(outputs, targets, reduce=False)
        cost_11 = torch.reshape(cost_1, (len(cost_1), 1))

        cost_2 = F.cross_entropy(outputs[idx], targets[idx], reduce=False)
        cost_12 = torch.reshape(cost_2, (len(cost_2), 1))

        with torch.no_grad():
            v_lambda_1 = vnet(cost_11, targets, c).squeeze(1)
            v_lambda_2 = vnet(cost_12, targets[idx], c).squeeze(1)

        loss = ( l * ( F.cross_entropy(outputs, targets, reduce=False) * v_lambda_1 + torch.sum(-F.log_softmax(outputs, dim=1) * psudo_label, dim=1) * (1-v_lambda_1) ) + (1-l) * ( F.cross_entropy(outputs, targets[idx], reduce=False) * v_lambda_2 + torch.sum(-F.log_softmax(outputs, dim=1) * psudo_label[idx], dim=1) * (1-v_lambda_2) ) ).mean()
    
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        my_wl(cost_11, v_lambda_1, index)

        for (param, param_ema) in zip(model.params(), model_ema.params()):
            param_ema.data.mul_(args.ema).add_(1-args.ema, param.data)

        sys.stdout.write('\r')
        sys.stdout.write('%s | Epoch [%3d/%3d] Iter[%4d/%4d]\t CE-loss: %.4f'
                %(args.id, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()


def train_CE(train_loader, model, model_ema, optimizer,epoch):
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets, _) in enumerate(train_loader):
        model.train()
        model_ema.train()

        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        # print('loss:', loss)

        for (param, param_ema) in zip(model.params(), model_ema.params()):
            param_ema.data.mul_(args.ema).add_(1-args.ema, param.data)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}| Acc: {:.3f} ({}/{})'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), (train_loss / (batch_idx + 1)), 100. * correct / total,
                correct, total))

        
def test(net,test_loader):
    acc_meter.reset()
    net.eval()

    with torch.no_grad():
        for _, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            
            acc_meter.add(outputs, targets)
    accs = acc_meter.value()
    # print('acc:', 100. * correct / total)
    return accs


def eval_train(model):
    model.eval()
    num_iter = (len(eval_loader.dataset) // eval_loader.batch_size) + 1
    losses = torch.zeros(len(eval_loader.dataset))
    paths = []
    n = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, path) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets, reduce=False)
            for b in range(inputs.size(0)):
                losses[n] = loss[b]
                paths.append(path[b])
                n += 1
            sys.stdout.write('\r')
            sys.stdout.write('| Evaluating loss Iter[%3d/%3d]\t' % (batch_idx, num_iter))
            sys.stdout.flush()

    # losses = (losses - losses.min()) / (losses.max() - losses.min())
    # all_loss.append(losses)

    return losses, paths


def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model


stats_log=open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_stats.txt','w') 
test_log=open('./checkpoint/%s_%.1f_%s'%(args.dataset,args.r,args.noise_mode)+'_acc.txt','w')     

if args.dataset=='cifar10':
    warm_up = 10
    args.num_class = 10
elif args.dataset=='cifar100':
    warm_up = 30
    args.num_class = 100

loader = dataloader.cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=4,\
    root_dir=args.data_path,log=stats_log,noise_file='%s/%.1f_%s.json'%(args.data_path,args.r,args.noise_mode))

acc_meter = torchnet.meter.ClassErrorMeter(topk=[1,5], accuracy=True)

print('| Building net')
net = create_model()
net_ema = create_model()
net_ema.load_state_dict(net.state_dict())

optimizer = optim.SGD(net.params(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

test_loader = loader.run('test')
warmup_trainloader = loader.run('warmup')
eval_loader = loader.run('eval_train')


a = []
web_num = warmup_trainloader.dataset.noise_label
for i in range(args.num_class):
    a.append([web_num.count(i)])
    #print(i,' number is ', li.count(i))
print(len(web_num))

print(a)
es = KMeans(3)
es.fit(a)

c = es.labels_

print('c:', c.tolist())

w = [[],[],[]]
for i in range(3):
    for k, j in enumerate(c):
        if i == j:
            w[i].append(a[k][0])

print(w)

criterion = get_loss(labels=np.asarray(web_num), num_classes=args.num_class, momentum=0.)

my_wl = get_weight_loss(labels=np.asarray(web_num))


vnet = ACVNet(1, 100, 100, 1, 3)

vnet = vnet.cuda()
optimizer_vnet = torch.optim.Adam(vnet.params(), 1e-4, weight_decay=1e-4)

if (args.noise_mode == 'sym' and args.dataset == 'cifar100' and args.r == 0.8):
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-4)
    optimizer_vnet = torch.optim.Adam(vnet.params(), 1e-3, weight_decay=1e-4)
    args.num_epochs = 345

for epoch in range(args.num_epochs):
    if not (args.noise_mode == 'sym' and args.dataset == 'cifar100' and args.r == 0.8):
        adjust_learning_rate(optimizer, epoch)
    if epoch < warm_up:       
        print('Warmup Net')
        train_CE(warmup_trainloader, net, net_ema, optimizer, epoch)

    else: 
        if (args.noise_mode == 'sym' and args.dataset == 'cifar100' and args.r == 0.8):
            scheduler.step(epoch-45)
        if args.noise_mode == 'asym' and epoch > 149:
            args.ema = 1.

        train_imagenet_loader = loader.run('meta', losses)

        meta_lr = print_lr(optimizer, epoch)
 
        warmup(epoch,net,net_ema,vnet,optimizer,optimizer_vnet,warmup_trainloader,train_imagenet_loader, meta_lr)
        
    test_acc = test(net, test_loader)  
    test_acc_ema = test(net_ema, test_loader)  
        
    print("\n| Test Epoch #%d\t Test Acc: %.2f%% (%.2f%%) \n"%(epoch,test_acc[0],test_acc[1]))  
    test_log.write('Epoch:%d \t Test Acc: %.2f%% (%.2f%%) \n'%(epoch,test_acc[0],test_acc[1]))
    test_log.flush()  

    print("\n| Test Epoch #%d\t Test ema Acc: %.2f%% (%.2f%%) \n"%(epoch,test_acc_ema[0],test_acc_ema[1]))  
    test_log.write('Epoch:%d \t Test ema Acc: %.2f%% (%.2f%%) \n'%(epoch,test_acc_ema[0],test_acc_ema[1]))
    test_log.flush()  

    losses, _ = eval_train(net_ema)
    print('losses:', type(losses), losses.shape)


