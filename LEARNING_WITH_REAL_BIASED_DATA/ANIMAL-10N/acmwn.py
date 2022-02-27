import sys
sys.path.append('..')
import os
import torch.nn as nn
import torch.nn.parallel
import random
import argparse
import numpy as np
from vgg_mwn import vgg19_bn, VNet
from dataset import Animal10
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
from utils import lrt_correction, check_folder
import logging
from termcolor import cprint
from sklearn.cluster import KMeans
import copy
import torch


class keep_wl():
    def __init__(self, labels):
        self.loss = torch.zeros(labels.shape[0], dtype=torch.float).cuda(non_blocking=True)
        self.weight = torch.zeros(labels.shape[0], dtype=torch.float).cuda(non_blocking=True)
        # self.loss1 = torch.zeros(labels.shape[0], 1, dtype=torch.float).cuda(non_blocking=True)
        # self.weight1 = torch.zeros(labels.shape[0], 1, dtype=torch.float).cuda(non_blocking=True)

    def __call__(self, epoch_loss=None, epoch_weight=None, index=None):
        self.loss[index] = epoch_loss.detach().data
        if epoch_weight is not None:
            self.weight[index] = epoch_weight.detach().data
        # self.loss1[index] = epoch_loss1.detach().data
        # self.weight1[index] = epoch_weight1.detach().data


def get_weight_loss(labels=None):
    wl = keep_wl(labels)
    return wl


class SelfAdaptiveTrainingCE():
    def __init__(self, labels, num_classes=50, momentum=0.9):
        # print(labels.shape[0])
        self.soft_labels = torch.zeros(labels.shape[0], num_classes, dtype=torch.float).cuda(non_blocking=True)
        self.soft_labels[torch.arange(labels.shape[0]), labels] = 1
        self.momentum = momentum

    def __call__(self, logits, index):
        prob = F.softmax(logits.detach(), dim=1)
        self.soft_labels[index] = self.momentum * self.soft_labels[index] + (1 - self.momentum) * prob

        return self.soft_labels[index]


def get_loss(labels=None, num_classes=50,momentum=0.9):
    criterion = SelfAdaptiveTrainingCE(labels, num_classes=num_classes, momentum=momentum)
    return criterion


def print_lr(optimizer, epoch):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        print('\n Epoch:{:.4f}|LR:{:.4f}\n'.format(epoch,lr))
    return lr
    

def get_dataset(model, eval_loader):
    model.eval()
    num_iter = (len(eval_loader.dataset) // eval_loader.batch_size) + 1
    losses = torch.zeros(len(eval_loader.dataset)).cuda()
    n = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets, reduce=False)
            losses[index] = loss

            sys.stdout.write('\r')
            sys.stdout.write('| Evaluating loss Iter[%3d/%3d]\t' % (batch_idx, num_iter))
            sys.stdout.flush()

    idx_to_meta = []

    data_list = {}
    for j in range(10):
        data_list[j] = [i for i, label in enumerate(eval_loader.dataset.targets) if label == j]

    for _, img_id_list in data_list.items():
        _, indexs = torch.topk(losses[img_id_list], 10, largest=False)
        idx_to_meta.extend(((torch.tensor(img_id_list))[indexs]).tolist())

    metadataset = copy.deepcopy(eval_loader.dataset)
 
    # print(type(metadataset.image_files), torch.tensor(metadataset.image_files))

    # metadataset.image_files = (torch.tensor(metadataset.image_files))[idx_to_meta]

    metadataset.image_files = [metadataset.image_files[i] for i in idx_to_meta]

    metadataset.targets = [metadataset.targets[i] for i in idx_to_meta]

    return metadataset


# random seed related
def _init_fn(worker_id):
    np.random.seed(77 + worker_id)


def main(arg_seed, arg_timestamp):
    random_seed = arg_seed
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True  # need to set to True as well

    print('Random Seed {}\n'.format(arg_seed))

    # -- training parameters
    num_epoch = args.epoch
    milestone = [50, 75]
    batch_size = args.batch
    num_workers = 2

    weight_decay = 1e-3
    gamma = 0.2
    current_delta = args.delta

    lr = args.lr
    start_epoch = 0

    # -- specify dataset
    # data augmentation
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    trainset = Animal10(split='train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                              worker_init_fn=_init_fn, drop_last=False)

    testset = Animal10(split='test', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size * 4, shuffle=False, num_workers=num_workers)

    num_class = 10

    print('train data size:', len(trainset))
    print('test data size:', len(testset))


    a = []
    for i in range(num_class):
        a.append([trainset.targets.count(i)])
        #print(i,' number is ', li.count(i))
    print(len(trainset.targets))

    # es = KMeans(3)
    # es.fit(a)
    # c = es.labels_

    # print('c:', c)

    qq = [1, 2, 0, 2, 0, 0, 1, 0, 0, 2]
    print(qq)

    mm = [2, 0, 1]
    c = []
    for i in qq:
        c.append(mm[i])

    w = [[],[],[]]
    for i in range(3):
        for k, j in enumerate(c):
            if i == j:
                w[i].append(a[k][0])

    print('w:', w)

    vnet = VNet(1, 100, 100, 1, 3)
    vnet = nn.DataParallel(vnet)
    vnet = vnet.cuda()

    # vnet.load_state_dict(torch.load('vnet_119.pth'))

    # -- set network, optimizer, scheduler, etc
    net = vgg19_bn(num_classes=num_class, pretrained=False)
    net = nn.DataParallel(net)

    net_ema = vgg19_bn(num_classes=num_class, pretrained=False)
    net_ema = nn.DataParallel(net_ema)
    net_ema.load_state_dict(net.state_dict())

    get_label = get_loss(labels=np.asarray(trainset.targets), num_classes=num_class, momentum=0.9)

    my_wl = get_weight_loss(labels=np.asarray(trainset.targets))

    optimizer = optim.SGD(net.module.params(), lr=lr, weight_decay=weight_decay)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestone, gamma=gamma)
    criterion = torch.nn.CrossEntropyLoss()

    optimizer_vnet = optim.Adam([{'params':vnet.module.feature.params(),'lr':1e-3}, {'params':vnet.module.classfier.params(),'lr':1e-3}], weight_decay=1e-4)

    # -- misc
    iterations = 0

    for epoch in range(20):
        train_correct = 0
        train_loss = 0
        train_total = 0

        net.train()
        net_ema.train()

        for i, (images, labels, indices) in enumerate(trainloader):
            if images.size(0) == 1:  # when batch size equals 1, skip, due to batch normalization
                continue

            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            cost = F.cross_entropy(outputs, labels, reduce=False)
            loss = torch.mean(cost)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            my_wl(epoch_loss=cost, index=indices)

            with torch.no_grad():
                for (param, param_ema) in zip(net.module.params(), net_ema.module.params()):
                    param_ema.data.mul_(0.997).add_(0.003, param.data)

            train_loss += loss.item()
            train_total += images.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()

            iterations += 1
            if iterations % 100 == 0:
                cur_train_acc = train_correct / train_total * 100.
                cur_train_loss = train_loss / train_total
                cprint('epoch: {}\titerations: {}\tcurrent train accuracy: {:.4f}\ttrain loss:{:.4f}'.format(
                    epoch, iterations, cur_train_acc, cur_train_loss), 'yellow')

        torch.save(net.state_dict(), '/data/animal/acmwn0/net_%s.pth' % epoch)
        torch.save(vnet.state_dict(), '/data/animal/acmwn0/vnet_%s.pth' % epoch)
        torch.save(my_wl.loss, '/data/animal/acmwn0/epoch_loss_%s.pth' % epoch)
        torch.save(my_wl.weight, '/data/animal/acmwn0/epoch_weight_%s.pth' % epoch)

        train_acc = train_correct / train_total * 100.

        cprint('epoch: {}'.format(epoch), 'yellow')
        cprint('train accuracy: {:.4f}\ntrain loss: {:.4f}'.format(train_acc, train_loss), 'yellow')

        exp_lr_scheduler.step()

        # testing
        net.eval()
        test_total = 0
        test_correct = 0
        with torch.no_grad():
            for i, (images, labels, _) in enumerate(testloader):
                images, labels = images.to(device), labels.to(device)

                outputs = net(images)

                test_total += images.size(0)
                _, predicted = outputs.max(1)
                test_correct += predicted.eq(labels).sum().item()

            test_acc = test_correct / test_total * 100.

        net_ema.eval()
        test_total = 0
        test_correct = 0
        with torch.no_grad():
            for i, (images, labels, _) in enumerate(testloader):
                images, labels = images.to(device), labels.to(device)

                outputs = net_ema(images)

                test_total += images.size(0)
                _, predicted = outputs.max(1)
                test_correct += predicted.eq(labels).sum().item()

            test_acc_ema = test_correct / test_total * 100.

        cprint('current test accuracy: {:.4f} >> current test ema accuracy: {:.4f}'.format(test_acc, test_acc_ema), 'cyan')


    for epoch in range(20, num_epoch):
        train_correct = 0
        train_loss = 0
        train_total = 0

        metadataset = get_dataset(net_ema, trainloader)


        train_meta_loader = torch.utils.data.DataLoader(metadataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                                worker_init_fn=_init_fn, drop_last=False)

        train_meta_loader_iter = iter(train_meta_loader)

        meta_lr = print_lr(optimizer, epoch)

        print('len:', len(train_meta_loader.dataset), len(trainloader.dataset))

        net.train()
        net_ema.train()

        train_meta_loader_iter = iter(train_meta_loader)
        for i, (inputs, targets, index) in enumerate(trainloader):
            if inputs.size(0) == 1:  # when batch size equals 1, skip, due to batch normalization
                continue

            inputs, targets = inputs.to(device), targets.to(device)

            with torch.no_grad():
                outputs_ema = net_ema(inputs)
                psudo_label = get_label(outputs_ema, index)

            l = torch.distributions.beta.Beta(torch.tensor([args.alpha]).cuda(),
                                              torch.tensor([args.alpha]).cuda() ).sample().cuda()
            l = max(l, 1-l)
            idx = torch.randperm(inputs.shape[0])
            mix_inputs = l * inputs + (1-l) * inputs[idx]

            if i % 10 == 0:
                meta_model = vgg19_bn(num_classes=num_class, pretrained=False)
                meta_model = nn.DataParallel(meta_model).cuda()
                meta_model.load_state_dict(net.state_dict())

                outputs = meta_model(mix_inputs)

                with torch.no_grad():
                    cost_1 = F.cross_entropy(outputs, targets, reduce=False)
                    cost_11 = torch.reshape(cost_1, (len(cost_1), 1))

                    cost_2 = F.cross_entropy(outputs, targets[idx], reduce=False)
                    cost_12 = torch.reshape(cost_2, (len(cost_2), 1))

                v_lambda_1 = vnet(cost_11.detach().data, targets.detach().data, c).squeeze(1)
                v_lambda_2 = vnet(cost_12.detach().data, targets[idx].detach().data, c).squeeze(1)

                l_f_meta = (l * ( F.cross_entropy(outputs, targets, reduce=False) * v_lambda_1
                            + torch.sum(-F.log_softmax(outputs, dim=1) * psudo_label, dim=1) * (1-v_lambda_1) 
                            ) 
                        + (1-l) * ( F.cross_entropy(outputs, targets[idx], reduce=False) * v_lambda_2
                                + torch.sum(-F.log_softmax(outputs, dim=1) * psudo_label[idx], dim=1) * (1-v_lambda_2) 
                                )

                        ).mean()

                meta_model.zero_grad()
                grads = torch.autograd.grad(l_f_meta, (meta_model.module.params()), create_graph=True)
                meta_model.module.update_params(lr_inner=meta_lr, source_params=grads)
                del grads

                try:
                    inputs_val, targets_val, _ = next(train_meta_loader_iter)
                except StopIteration:
                    train_meta_loader_iter = iter(train_meta_loader)
                    inputs_val, targets_val, _ = next(train_meta_loader_iter)
                inputs_val, targets_val = inputs_val.cuda(), targets_val.cuda()
                y_g_hat = meta_model(inputs_val)
                l_g_meta = F.cross_entropy(y_g_hat, targets_val)
                # prec_meta = accuracy(y_g_hat.data, targets_val.data, topk=(1,))[0]

                optimizer_vnet.zero_grad()
                l_g_meta.backward()
                optimizer_vnet.step()

            outputs = net(mix_inputs)

            with torch.no_grad():
                cost_1 = F.cross_entropy(outputs, targets, reduce=False)
                cost_11 = torch.reshape(cost_1, (len(cost_1), 1))

                cost_2 = F.cross_entropy(outputs, targets[idx], reduce=False)
                cost_12 = torch.reshape(cost_2, (len(cost_2), 1))

                v_lambda_1 = vnet(cost_11, targets, c).squeeze(1)
                v_lambda_2 = vnet(cost_12, targets[idx], c).squeeze(1)

            loss = (l * ( F.cross_entropy(outputs, targets, reduce=False) * v_lambda_1
                        + torch.sum(-F.log_softmax(outputs, dim=1) * psudo_label, dim=1) * (1-v_lambda_1) 
                        ) 
                    + (1-l) * ( F.cross_entropy(outputs, targets[idx], reduce=False) * v_lambda_2
                              + torch.sum(-F.log_softmax(outputs, dim=1) * psudo_label[idx], dim=1) * (1-v_lambda_2) 
                              )

                    ).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            my_wl(epoch_loss=cost_1, epoch_weight=v_lambda_1, index=index)

            with torch.no_grad():
                for (param, param_ema) in zip(net.module.params(), net_ema.module.params()):
                    param_ema.data.mul_(0.997).add_(0.003, param.data)

            train_loss += loss.item()
            train_total += inputs.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(targets).sum().item()

            iterations += 1
            if iterations % 100 == 0:
                cur_train_acc = train_correct / train_total * 100.
                cur_train_loss = train_loss / train_total
                cprint('epoch: {}\titerations: {}\tcurrent train accuracy: {:.4f}\ttrain loss:{:.4f}'.format(
                    epoch, iterations, cur_train_acc, cur_train_loss), 'yellow')

        train_acc = train_correct / train_total * 100.

        cprint('epoch: {}'.format(epoch), 'yellow')
        cprint('train accuracy: {:.4f}\ntrain loss: {:.4f}'.format(train_acc, train_loss), 'yellow')

        exp_lr_scheduler.step()

        torch.save(my_wl.loss, '/data/animal/acmwn0/epoch_loss_%s.pth' % epoch)
        torch.save(my_wl.weight, '/data/animal/acmwn0/epoch_weight_%s.pth' % epoch)
        torch.save(net.state_dict(), '/data/animal/acmwn0/net_%s.pth' % epoch)
        torch.save(vnet.state_dict(), '/data/animal/acmwn0/vnet_%s.pth' % epoch)

        # testing
        net.eval()
        test_total = 0
        test_correct = 0
        with torch.no_grad():
            for i, (images, labels, _) in enumerate(testloader):
                images, labels = images.to(device), labels.to(device)

                outputs = net(images)

                test_total += images.size(0)
                _, predicted = outputs.max(1)
                test_correct += predicted.eq(labels).sum().item()

            test_acc = test_correct / test_total * 100.

        cprint('>> current test accuracy: {:.4f}'.format(test_acc), 'cyan')

        net_ema.eval()
        test_total = 0
        test_correct = 0
        with torch.no_grad():
            for i, (images, labels, _) in enumerate(testloader):
                images, labels = images.to(device), labels.to(device)

                outputs = net_ema(images)

                test_total += images.size(0)
                _, predicted = outputs.max(1)
                test_correct += predicted.eq(labels).sum().item()

            test_acc_ema = test_correct / test_total * 100.

        cprint('current test accuracy: {:.4f} >> current test ema accuracy: {:.4f}'.format(test_acc, test_acc_ema), 'cyan')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default='1', help='delimited list input of GPUs', type=str)
    parser.add_argument('--timelog', action='store_false', help='whether to add time stamp to log file name')
    parser.add_argument("--warm_up", default=40, help="warm-up period", type=int)
    parser.add_argument("--rollWindow", default=10, help="rolling window to calculate the confidence", type=int)
    parser.add_argument("--eval_freq", default=2000, help="evaluation frequency (every a few iterations)", type=int)
    parser.add_argument("--batch", default=128, help="batch size", type=int)
    parser.add_argument("--epoch", default=100, help="total number of epochs", type=int)
    parser.add_argument("--seed", default=77, help="random seed", type=int)
    parser.add_argument("--lr", default=0.1, help="learning rate", type=float)
    parser.add_argument("--delta", default=0.3, help="delta", type=float)
    parser.add_argument('--alpha', default='1.', type=float)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    seed = args.seed

    main(seed, args.timelog)
