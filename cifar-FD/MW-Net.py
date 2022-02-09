# -*- coding: utf-8 -*-

import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from copy import deepcopy
# from pandas import Series
from resnet import ResNet34, VNet, CVNet, ACVNet
from cifar_train_val_test import CIFAR10, CIFAR100
# from cifar import CIFAR10, CIFAR100
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from noise import noisify_with_P, noisify_cifar10_asymmetric, noisify_cifar100_asymmetric
from sklearn.cluster import KMeans
import sys
import copy


parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--corruption_prob', type=float, default=0.,
                    help='label noise')
parser.add_argument('--corruption_type', '-ctype', type=str, default='fip',
                    help='Type of corruption ("unif" or "flip" or "flip2").')
parser.add_argument('--noise_label_file', type=str, default='cifar10-1-0.35.npy')
parser.add_argument('--num_meta', type=int, default=1000)
parser.add_argument('--epochs', default=120, type=int,
                    help='number of total epochs to run')
parser.add_argument('--iters', default=60000, type=int,
                    help='number of total iters to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 100)')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--layers', default=28, type=int,
                    help='total number of layers (default: 28)')
parser.add_argument('--widen-factor', default=10, type=int,
                    help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='WideResNet-28-10', type=str,
                    help='name of experiment')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.set_defaults(augment=True)

args = parser.parse_args()
use_cuda = True
torch.manual_seed(args.seed)

print()
print(args)


class SelfAdaptiveTrainingCE():
    def __init__(self, labels, num_classes=10, momentum=0.9):
        self.soft_labels = torch.zeros(labels.shape[0], num_classes, dtype=torch.float).cuda(non_blocking=True)
        # self.soft_labels[torch.arange(labels.shape[0]), labels] = 1
        self.momentum = momentum

    def __call__(self, index, logits_ema, epoch):
        # obtain prob, then update running avg
        if epoch == 45:
            print('go epoch')
            self.soft_labels[index] = F.softmax(logits_ema.detach(), dim=1)
        else:
            prob = F.softmax(logits_ema.detach(), dim=1)
            self.soft_labels[index] = self.momentum * self.soft_labels[index] + (1 - self.momentum) * prob

        return self.soft_labels[index]


def get_loss(labels=None, num_classes=10,momentum=0.9):
    criterion = SelfAdaptiveTrainingCE(labels, num_classes=num_classes, momentum=momentum)
    return criterion


class keep_wl():
    def __init__(self, labels):
        self.loss = torch.zeros(labels.shape[0], 1, dtype=torch.float).cuda(non_blocking=True)
        self.weight = torch.zeros(labels.shape[0], dtype=torch.float).cuda(non_blocking=True)

    def __call__(self, epoch_loss=None, epoch_weight=None, index=None):
        self.loss[index] = epoch_loss.detach().data
        if epoch_weight is not None:
            self.weight[index] = epoch_weight.detach().data


def get_weight_loss(labels=None):
    wl = keep_wl(labels)
    return wl


def build_dataset():
    noise_label_path = os.path.join('noisy_labels', args.noise_label_file)
    noise_y = np.load(noise_label_path)
    print('Load noisy label from {}'.format(noise_label_path))

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.dataset == 'cifar10':
        trainset = CIFAR10(root='data', split='train', train_ratio=0.9, trust_ratio=0, download=True, transform=transform_train)

        valset = CIFAR10(root='data', split='val', train_ratio=0.9, trust_ratio=0, download=True, transform=transform_test)

        testset = CIFAR10(root='data', split='test', download=True, transform=transform_test)

        num_class = 10
    elif args.dataset == 'cifar100':
        trainset = CIFAR100(root='data', split='train', train_ratio=0.9, trust_ratio=0, download=True, transform=transform_train)

        valset = CIFAR100(root='data', split='val', train_ratio=0.9, trust_ratio=0, download=True, transform=transform_test)

        testset = CIFAR100(root='data', split='test', download=True, transform=transform_test)

        num_class = 100
    else:
        raise ValueError('Dataset should be cifar10 or cifar100.')

    print('train data size:', len(trainset))
    print('validation data size:', len(valset))
    print('test data size:', len(testset))

    # -- Sanity Check --
    num_noise_class = len(np.unique(noise_y))
    assert num_noise_class == num_class
    assert len(noise_y) == len(trainset)
    # -- generate noise --
    gt_clean_y = deepcopy(trainset.get_data_labels())
    y_train = noise_y.copy()

    noise_y_train = None
    p = None

    if args.corruption_type == "unif":
        noise_y_train, p, _ = noisify_with_P(y_train, nb_classes=num_class, noise=args.corruption_prob,
                                                        random_state=args.seed)
        trainset.update_corrupted_label(noise_y_train)
        print("apply uniform noise")
    else:
        if args.dataset == 'cifar10':
            noise_y_train, p, _ = noisify_cifar10_asymmetric(y_train, noise=args.corruption_prob, random_state=args.seed)
        elif args.dataset == 'cifar100':
            noise_y_train, p, _ = noisify_cifar100_asymmetric(y_train, noise=args.corruption_prob, random_state=args.seed)
        else:
            raise ValueError('Dataset should be cifar10 or cifar100.') 

        trainset.update_corrupted_label(noise_y_train)
        print("apply asymmetric noise")
    print("probability transition matrix:\n{}\n".format(p))

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    train_meta_loader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)

    return train_loader, train_meta_loader, test_loader, gt_clean_y, np.asarray(trainset.targets)


def build_model():
    model = ResNet34(args.dataset == 'cifar10' and 10 or 100)

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True

    return model


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


def adjust_learning_rate(optimizer, epochs):
    lr = args.lr * ((0.5 ** int(epochs >= 40)) * (0.5 ** int(epochs >= 80)))  # For WRN-28-10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def print_lr(optimizer, epoch):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        print('\n Epoch:{:.4f}|LR:{:.4f}\n'.format(epoch,lr))
    return lr


def test(model, test_loader):
    model.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for _, (inputs, targets, _, _) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            test_loss += F.cross_entropy(outputs, targets).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    return accuracy


def train_CE(train_loader, model, model_ema, optimizer,epoch):
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets, _, index) in enumerate(train_loader):
        model.train()
        model_ema.train()
        
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        cost = F.cross_entropy(outputs, targets, reduce=False)
        cost = torch.reshape(cost, (len(cost), 1))
        loss = torch.mean(cost)

        loss.backward()
        optimizer.step()

        my_wl(epoch_loss=cost, index=index)

        for (param, param_ema) in zip(model.params(), model_ema.params()):
            param_ema.data.mul_(0.999).add_(0.001, param.data)


        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}| Acc: {:.3f} ({}/{})'.format(
                epoch, batch_idx * len(inputs), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), (train_loss / (batch_idx + 1)), 100. * correct / total,
                correct, total))


# def train(train_loader, train_meta_loader, model, vnet, optimizer_model, optimizer_vnet, epoch, meta_lr):
def train(epoch,model,model_ema,vnet,optimizer_model,optimizer_vnet,train_loader, train_meta_loader, meta_lr):
    print('\nEpoch: %d' % epoch)

    train_loss = 0
    meta_loss = 0

    train_meta_loader_iter = iter(train_meta_loader)
    for batch_idx, (inputs, targets, _, index) in enumerate(train_loader):
        model.train()
        model_ema.train()

        targets = torch.zeros(inputs.shape[0], args.num_classes).scatter_(1, targets.view(-1, 1).long(), 1)
        inputs, targets = inputs.cuda(), targets.cuda()

        with torch.no_grad():
            outputs_ema = model_ema(inputs)
            psudo_label = get_label(index, outputs_ema, epoch)

        l = torch.distributions.beta.Beta(args.alpha, args.alpha).sample().cuda()
        # l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1-l)
        idx = torch.randperm(inputs.shape[0])
        mix_inputs = l * inputs + (1-l) * inputs[idx]
        
        if batch_idx % 10 == 0:
            meta_model = build_model().cuda()
            meta_model.load_state_dict(model.state_dict())

            outputs = meta_model(mix_inputs)

            cost_1 = torch.sum(-F.log_softmax(outputs, dim=1) * targets, dim=1)
            cost_11 = torch.reshape(cost_1, (len(cost_1), 1))
            v_lambda_1 = vnet(cost_11.data, targets.data, c).squeeze(1)

            cost_2 = torch.sum(-F.log_softmax(outputs, dim=1) * targets[idx], dim=1)
            cost_12 = torch.reshape(cost_2, (len(cost_2), 1))
            v_lambda_2 = vnet(cost_12.data, targets[idx].data, c).squeeze(1)

            # mix_v_lambda = l * v_lambda_1 + (1-l) * v_lambda_2

            l_f_meta = ( l * ( torch.sum(-F.log_softmax(outputs, dim=1) * targets, dim=1) * v_lambda_1 + torch.sum(-F.log_softmax(outputs, dim=1) * psudo_label, dim=1) * (1-v_lambda_1) )
                        +(1-l) * ( torch.sum(-F.log_softmax(outputs, dim=1) * targets[idx], dim=1) * v_lambda_2 + torch.sum(-F.log_softmax(outputs, dim=1) * psudo_label[idx], dim=1) * (1-v_lambda_2) )
                       ).mean()
                     
            meta_model.zero_grad()
            grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
            meta_model.update_params(lr_inner=meta_lr, source_params=grads)
            del grads

            try:
                inputs_val, targets_val, _, _ = next(train_meta_loader_iter)
            except StopIteration:
                train_meta_loader_iter = iter(train_meta_loader)
                inputs_val, targets_val, _, _ = next(train_meta_loader_iter)
            inputs_val, targets_val = inputs_val.cuda(), targets_val.cuda()
            y_g_hat = meta_model(inputs_val)
            l_g_meta = F.cross_entropy(y_g_hat, targets_val)
            # prec_meta = accuracy(y_g_hat.data, targets_val.data, topk=(1,))[0]

            optimizer_vnet.zero_grad()
            l_g_meta.backward()
            optimizer_vnet.step()

        outputs = model(mix_inputs)

        cost_1 = torch.sum(-F.log_softmax(outputs, dim=1) * targets, dim=1)
        cost_11 = torch.reshape(cost_1, (len(cost_1), 1))

        cost_2 = torch.sum(-F.log_softmax(outputs, dim=1) * targets[idx], dim=1)
        cost_12 = torch.reshape(cost_2, (len(cost_2), 1))

        with torch.no_grad():
            v_lambda_1 = vnet(cost_11, targets, c).squeeze(1)
            v_lambda_2 = vnet(cost_12, targets[idx], c).squeeze(1)

        loss = ( l * ( torch.sum(-F.log_softmax(outputs, dim=1) * targets, dim=1) * v_lambda_1 + torch.sum(-F.log_softmax(outputs, dim=1) * psudo_label, dim=1) * (1-v_lambda_1) )
                +(1-l) * ( torch.sum(-F.log_softmax(outputs, dim=1) * targets[idx], dim=1) * v_lambda_2 + torch.sum(-F.log_softmax(outputs, dim=1) * psudo_label[idx], dim=1) * (1-v_lambda_2) )
                ).mean()

        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        my_wl(cost_11, v_lambda_1, index)

        for (param, param_ema) in zip(model.params(), model_ema.params()):
            param_ema.data.mul_(0.999).add_(0.001, param.data)

        train_loss += loss.item()
        meta_loss += l_g_meta.item()

        if (batch_idx + 1) % 50 == 0:
            print('Epoch: [%d/%d]\t'
                  'Iters: [%d/%d]\t'
                  'Loss: %.4f\t'
                  'MetaLoss:%.4f\t' % (
                      (epoch + 1), args.epochs, batch_idx + 1, len(train_loader.dataset)/args.batch_size, (train_loss / (batch_idx + 1)), (meta_loss / (batch_idx + 1))))


def eval_train(model):
    model.eval()
    num_iter = (len(train_loader.dataset) // train_loader.batch_size) + 1
    losses = torch.zeros(len(train_loader.dataset)).cuda()
    all_targets = []
    n = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, _, index) in enumerate(train_loader):
            targets = torch.zeros(inputs.shape[0], args.num_classes).scatter_(1, targets.view(-1, 1).long(), 1)
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)

            loss = torch.sum(-F.log_softmax(outputs, dim=1) * targets, dim=1)
            # new_weight = torch.reshape(((my_wl.weight)[index]), (len( (my_wl.weight)[index]), 1))
            # new_label = targets * new_weight + ((criterion.soft_labels)[index]) * (1 - new_weight)

            # print('out:', (torch.sum(-F.log_softmax(outputs, dim=1) * targets, dim=1)).shape, (new_label).shape, ((criterion.soft_labels)[index]).shape, )
            # print('new_label:', new_label[0])

            losses[index] = loss
            sys.stdout.write('\r')
            sys.stdout.write('| Evaluating loss Iter[%3d/%3d]\t' % (batch_idx, num_iter))
            sys.stdout.flush()

    return losses


train_loader, _, test_loader, real, targets = build_dataset()
# create model
model = build_model()
model_ema = build_model()
model_ema.load_state_dict(model.state_dict())

if args.dataset == 'cifar10':
    args.num_classes = 10
if args.dataset == 'cifar100':
    args.num_classes = 100

# get the ac
a = []
for i in range(args.num_classes):
    a.append([train_loader.dataset.targets.count(i)])

print(len(train_loader.dataset.targets))

print('a:', a)
es = KMeans(3)
es.fit(a)
c = es.labels_

print('c:', c)

w = [[], [], []]

for i in range(3):
    for k, j in enumerate(c):
        if i == j:
            w[i].append(a[k][0])

print('w:', w)

vnet = ACVNet(1, 100, 100, 1, 3).cuda()

my_wl = get_weight_loss(labels=targets)
get_label = get_loss(labels=targets, num_classes=args.num_classes)

# folder = '%s_%s_%s_%s' % (args.dataset, args.corruption_type, args.corruption_prob, args.noise_label_file)
# if not os.path.exists('./ac_v220/' + folder):
#     os.makedirs('./ac_v220/' + folder)

# torch.save(np.asarray(real), './ac_v220/%s/real_label.pth' % folder)

# torch.save(targets, './ac_v220/%s/fake_label.pth' % folder)

optimizer_model = torch.optim.SGD(model.params(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
optimizer_vnet = torch.optim.Adam(vnet.params(), 1e-3, weight_decay=1e-4)

scheduler = CosineAnnealingWarmRestarts(optimizer_model, T_0=5, T_mult=2, eta_min=1e-4)


def main():
    best_acc = 0
    best_meta_acc =0
    for epoch in range(args.epochs):
        # adjust_learning_rate(optimizer_model, epoch)
        if epoch < 45:
            train_CE(train_loader, model, model_ema, optimizer_model, epoch)
        else:
            scheduler.step(epoch-45)

            losses = eval_train(model_ema)

            data_list = {}
            for j in range(args.num_classes):
                data_list[j] = [i for i, label in enumerate(targets) if label == j]

            idx_to_meta = []
            for _, img_id_list in data_list.items():
                _, indexs = torch.topk(losses[img_id_list], 10, largest=False)
                idx_to_meta.extend( list( set(img_id_list) - set( ((torch.tensor(img_id_list))[indexs]).tolist() ) ) )

            train_data_meta = copy.deepcopy(train_loader.dataset)
            train_data_meta.targets = np.delete(train_loader.dataset.targets, idx_to_meta, axis=0)
            train_data_meta.data = np.delete(train_loader.dataset.data, idx_to_meta, axis=0)

            print('validation data size:', len(train_data_meta))

            train_meta_loader = torch.utils.data.DataLoader(train_data_meta, batch_size=args.batch_size, shuffle=True, num_workers=args.prefetch, pin_memory=True)
 
            meta_lr = print_lr(optimizer_model, epoch)
            # train(train_loader, train_meta_loader, model, vnet, optimizer_model, optimizer_vnet, epoch, meta_lr)
            train(epoch,model,model_ema,vnet,optimizer_model,optimizer_vnet,train_loader, train_meta_loader, meta_lr)

            meta_acc = test(model=model, test_loader=train_meta_loader)
            if meta_acc >= best_meta_acc:
                best_meta_acc = meta_acc
                print('epoch best:', epoch)

        test_acc = test(model=model, test_loader=test_loader)

        if test_acc >= best_acc:
            best_acc = test_acc


        print('best accuracy:', best_acc, best_meta_acc)


if __name__ == '__main__':
    main()
