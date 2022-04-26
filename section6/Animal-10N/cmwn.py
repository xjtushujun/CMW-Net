import sys
sys.path.append('..')
import os
import torch.nn as nn
import torch.nn.parallel
import random
import argparse
import numpy as np
from vgg import vgg19_bn
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


class share(nn.Module):
    def __init__(self, input, hidden1, hidden2):
        super(share, self).__init__()
        self.layer = nn.Sequential( nn.Linear(input, hidden1), nn.ReLU(inplace=True) )

    def forward(self, x):
        output = self.layer(x)
        return output


class task(nn.Module):
    def __init__(self, hidden2, output, num_classes):
        super(task, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_classes):
            self.layers.append(nn.Sequential( nn.Linear(hidden2, output), nn.Sigmoid() ))

    def forward(self, x, num, c):
        si = x.shape[0]
        output = torch.tensor([]).cuda()
        for i in range(si):
            output = torch.cat(( output, self.layers[c[num[i]]]( x[i].unsqueeze(0) ) ),0)
        
        return output


class VNet(nn.Module):
    def __init__(self, input, hidden1, hidden2, output, num_classes):
        super(VNet, self).__init__()
        self.feature = share(input, hidden1, hidden2)
        self.classfier = task(hidden2, output, num_classes)

    def forward(self, x, num, c):
        # num = torch.argmax(num, -1)
        output = self.classfier( self.feature(x), num, c )
        #print('output:', output.shape)
        return output


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
                                              worker_init_fn=_init_fn, drop_last=True)

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
    # vnet = nn.DataParallel(vnet)
    vnet = vnet.cuda()

    vnet.load_state_dict(torch.load('vnet_119.pth'))

    # -- set network, optimizer, scheduler, etc
    net = vgg19_bn(num_classes=num_class, pretrained=False)
    net = nn.DataParallel(net)

    net_ema = vgg19_bn(num_classes=num_class, pretrained=False)
    net_ema = nn.DataParallel(net_ema)
    net_ema.load_state_dict(net.state_dict())

    get_label = get_loss(labels=np.asarray(trainset.targets), num_classes=num_class, momentum=0.9)

    optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestone, gamma=gamma)
    criterion = torch.nn.CrossEntropyLoss()

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
            loss = F.cross_entropy(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for (param, param_ema) in zip(net.parameters(), net_ema.parameters()):
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

        net.train()
        net_ema.train()

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

            with torch.no_grad():
                for (param, param_ema) in zip(net.parameters(), net_ema.parameters()):
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
    parser.add_argument('--gpus', default='0', help='delimited list input of GPUs', type=str)
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
