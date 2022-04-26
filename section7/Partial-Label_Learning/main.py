import os
import os.path
import torch 
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms 
import argparse
import numpy as np
import time

from utils.utils_loss import partial_loss
from utils.models_v1 import mlp
from cifar_models import *
from datasets.mnist_v1 import mnist
from datasets.fashion_v1 import fashion
from datasets.kmnist_v1 import kmnist
from datasets.cifar10_v1 import cifar10
from datasets.cifar100_v1 import cifar100
from sklearn.cluster import KMeans

torch.manual_seed(0); torch.cuda.manual_seed_all(0)

parser = argparse.ArgumentParser(
	prog='PRODEN demo file.',
	usage='Demo with partial labels.',
	description='A simple demo file with MNIST dataset.',
	epilog='end',
	add_help=True)

parser = argparse.ArgumentParser()
parser.add_argument('-lr', help='optimizer\'s learning rate', type=float, default=1e-1)
parser.add_argument('-wd', help='weight decay', type=float, default=5e-4)
parser.add_argument('-alpha', type=float, default=1.)
parser.add_argument('-bs', help='batch size', type=int, default=256)
parser.add_argument('-ep', help='number of epochs', type=int, default=500)
parser.add_argument('-ds', help='specify a dataset', type=str, default='mnist', choices=['mnist', 'fashion', 'kmnist', 'cifar10', 'cifar100'], required=False)
parser.add_argument('-model', help='model name', type=str, default='linear', choices=['linear', 'mlp', 'convnet', 'resnet'], required=False)
parser.add_argument('-decaystep', help='learning rate\'s decay step', type=int, default=500)
parser.add_argument('-decayrate', help='learning rate\'s decay rate', type=float, default=1)

parser.add_argument('-partial_type', help='flipping strategy', type=str, default='binomial', choices=['binomial', 'pair'])
parser.add_argument('-partial_rate', help='flipping probability', type=float, default=0.1)

parser.add_argument('-nw', help='multi-process data loading', type=int, default=4, required=False)
parser.add_argument('-dir', help='result save path', type=str, default='results/', required=False)

args = parser.parse_args()

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# load dataset
if args.ds == 'mnist':
    num_features = 28 * 28
    num_classes = 10
    num_training = 60000
    train_dataset = mnist(root='./mnist/',
                                download=True,  
                                train_or_not=True, 
                                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]), 
                                partial_type=args.partial_type,
                                partial_rate=args.partial_rate
    )
    test_dataset = mnist(root='./mnist/',
                               download=True,  
                               train_or_not=False, 
                               transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]),
                               partial_type=args.partial_type,
                               partial_rate=args.partial_rate
    )

if args.ds == 'fashion':
    num_features = 28 * 28
    num_classes = 10
    num_training = 60000
    train_dataset = fashion(root='./fashion/',
                                download=True,  
                                train_or_not=True, 
                                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]), 
                                partial_type=args.partial_type,
                                partial_rate=args.partial_rate
    ) 
    test_dataset = fashion(root='./fashionmnist/',
                               download=True,  
                               train_or_not=False, 
                               transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]),
                               partial_type=args.partial_type,
                               partial_rate=args.partial_rate
    )

if args.ds == 'kmnist':
    num_features = 28 * 28
    num_classes = 10
    num_training = 60000
    train_dataset = kmnist(root='./kmnist/',
                                download=True,  
                                train_or_not=True, 
                                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))]), 
                                partial_type=args.partial_type,
                                partial_rate=args.partial_rate
    ) 
    test_dataset = kmnist(root='./kmnist/',
                               download=True,  
                               train_or_not=False, 
                               transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))]), 
                               partial_type=args.partial_type,
                               partial_rate=args.partial_rate
    )

if args.ds == 'cifar10':
    input_channels = 3
    num_classes = 10
    dropout_rate = 0.25
    num_training = 50000
    train_dataset = cifar10(root='./cifar10/',
                                download=True,  
                                train_or_not=True, 
                                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]),
                                partial_type=args.partial_type,
                                partial_rate=args.partial_rate
    ) 
    test_dataset = cifar10(root='./cifar10/',
                                download=True,  
                                train_or_not=False, 
                                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]),
                                partial_type=args.partial_type,
                                partial_rate=args.partial_rate
    )

if args.ds == 'cifar100':
    input_channels = 3
    num_classes = 100
    dropout_rate = 0.25
    num_training = 50000
    train_dataset = cifar100(root='./cifar100/',
                                download=True,  
                                train_or_not=True, 
                                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]),
                                partial_type=args.partial_type,
                                partial_rate=args.partial_rate
    ) 
    test_dataset = cifar100(root='./cifar100/',
                                download=True,  
                                train_or_not=False, 
                                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]),
                                partial_type=args.partial_type,
                                partial_rate=args.partial_rate
    )


# result dir  
save_dir = './' + args.dir + '/mwn_v24/' + args.ds
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_file = os.path.join(save_dir, (args.partial_type + '_' + str(args.partial_rate) + '.txt'))

# calculate accuracy
def evaluate(loader, model):
    model.eval()     
    correct = 0
    total = 0
    for images, _, labels, _ in loader:
        images = images.cuda()
        labels = labels.cuda()
        output1 = model(images)
        # output = F.softmax(output1, dim=1)
        _, pred = torch.max(output1.data, 1) 
        total += images.size(0)
        # print('pred:', pred, '\n', labels)
        correct += (pred == labels).sum().item()
    acc = 100 * float(correct) / float(total)
    return acc


def print_lr(optimizer, epoch):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        print('\n Epoch:{:.4f}|LR:{:.4f}\n'.format(epoch,lr))
    return lr


def adjust_learning_rate(optimizer, epochs):
    lr = args.lr * ((0.1 ** int(epochs >= 80)) * (0.1 ** int(epochs >= 100)))  # For WRN-28-10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def eval_train(model, train_loader):
    model.eval()
    
    losses = torch.zeros(len(train_loader.dataset))
    indexes = []
    all_labels = []
    n = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, _, index) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets, reduce=False)
            for b in range(inputs.size(0)):
                losses[n] = loss[b]
                indexes.append(index[b])
                all_labels.append((targets.tolist())[b])
                n += 1

    idx_to_meta = []

    data_list = {}
    for j in range(num_classes):
        data_list[j] = [i for i, label in enumerate(all_labels) if label == j]

    for _, img_id_list in data_list.items():
        _, temp_index = torch.topk(losses[img_id_list], 10, largest=False)
        idx_to_meta.extend((((torch.tensor(indexes))[img_id_list])[temp_index]).tolist())
        
    return idx_to_meta


class SelfAdaptiveTrainingCE():
    def __init__(self, labels, num_classes=10, momentum=0.9):
        self.soft_labels = torch.zeros(labels.shape[0], num_classes, dtype=torch.float).cuda(non_blocking=True)
        # self.soft_labels[torch.arange(labels.shape[0]), labels] = 1
        self.momentum = momentum

    def __call__(self, logits_ema, index, epoch):
        # obtain prob, then update running avg
        if epoch == 45:
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

    def __call__(self, epoch_loss, epoch_weight, index):
        self.loss[index] = epoch_loss.detach().data
        self.weight[index] = epoch_weight.detach().data


def get_weight_loss(labels=None):
    wl = keep_wl(labels)
    return wl


# learning rate 
# lr_plan = [args.lr] * args.ep 
# for i in range(0, args.ep):
#     lr_plan[i] = args.lr * args.decayrate ** (i // args.decaystep)
# def adjust_learning_rate(optimizer, epoch):
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr_plan[epoch]


def main():
    print ('loading dataset...')
    # _, train_dataset.train_final_labels = torch.max( torch.load('./save/%s_%f/labels_%d.pth' % (args.ds, args.partial_rate, 40)), dim=-1)

    print(torch.sum( (train_dataset.train_final_labels == train_dataset.train_labels).float() ))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.bs, 
                                               num_workers=args.nw,
                                               drop_last=False, 
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.bs, 
                                              num_workers=args.nw,
                                              drop_last=False,
                                              shuffle=False)

    # get the ac
    a = []
    for i in range(num_classes):
        a.append([train_loader.dataset.train_final_labels.tolist().count(i)])

    print(len(train_loader.dataset.train_final_labels))

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

    get_label = get_loss(labels=np.array(train_loader.dataset.train_final_labels), num_classes=num_classes, momentum=0.9)

    my_wl = get_weight_loss(labels=np.array(train_loader.dataset.train_final_labels))


    # print ('building model...')
    if args.model == 'linear':
        net = linear(n_inputs=num_features, n_outputs=num_classes)
    elif args.model == 'mlp':
        net = mlp(n_inputs=num_features, n_outputs=num_classes)
        net_ema = mlp(n_inputs=num_features, n_outputs=num_classes)
    elif args.model == 'convnet':
        net = convnet(input_channels=input_channels, n_outputs=num_classes, dropout_rate=dropout_rate)
    elif args.model == 'resnet':
        net = resnet(depth=32, n_outputs=num_classes)
        net_ema = resnet(depth=32, n_outputs=num_classes)
    net = net.cuda()
    print (net.params)

    # net_ema = mlp(n_inputs=num_features, n_outputs=num_classes)
    net_ema = net_ema.cuda()
    net_ema.load_state_dict(net.state_dict())

    optimizer = torch.optim.SGD(net.params(), lr=args.lr, weight_decay=args.wd, momentum=0.9)

    # if not os.path.exists('./save/mwn_v24/%s_%f' % (args.ds, args.partial_rate)):
    #     os.makedirs('./save/mwn_v24/%s_%f' % (args.ds, args.partial_rate))

    # torch.save(train_loader.dataset.train_labels, './save/mwn_v24/%s_%f/real_labels.pth' % (args.ds, args.partial_rate))   

    vnet = ACVNet(1, 100, 100, 1, 3).cuda()
    optimizer_vnet = torch.optim.Adam(vnet.params(), 1e-3, weight_decay=1e-4)


    for epoch in range(0, 45):
        # torch.save(net.state_dict(), './save/mwn_v24/%s_%f/net_%d.pth' % (args.ds, args.partial_rate, epoch))
        # torch.save(net_ema.state_dict(), './save/mwn_v24/%s_%f/ema_net_%d.pth' % (args.ds, args.partial_rate, epoch))
        
        print ('training...')
        net.train()
        net_ema.train()
        adjust_learning_rate(optimizer, epoch)

        for i, (inputs, targets, _, _) in enumerate(train_loader): 
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for (param, param_ema) in zip(net.params(), net_ema.params()):
                param_ema.data.mul_(0.999).add_(0.001, param.data)

        print ('evaluating model...')  
        net.eval()     
        train_acc = evaluate(train_loader, net)
        test_acc = evaluate(test_loader, net)

        with open(save_file, 'a') as file:
            file.write(str(int(epoch)) + ': Training Acc.: ' + str(round(train_acc, 4)) + ' , Test Acc.: ' + str(round(test_acc, 4)) + '\n')


    for epoch in range(45, args.ep):
        # torch.save(net.state_dict(), './save/mwn_v24/%s_%f/net_%d.pth' % (args.ds, args.partial_rate, epoch))
        # torch.save(net_ema.state_dict(), './save/mwn_v24/%s_%f/ema_net_%d.pth' % (args.ds, args.partial_rate, epoch))
        
        # torch.save(vnet.state_dict(), './save/mwn_v24/%s_%f/vnet_%d.pth' % (args.ds, args.partial_rate, epoch))
        # torch.save(my_wl.weight, './save/mwn_v24/%s_%f/epoch_weight_%d.pth' % (args.ds, args.partial_rate, epoch))
        # torch.save(my_wl.loss, './save/mwn_v24/%s_%f/epoch_loss_%d.pth' % (args.ds, args.partial_rate, epoch))
        # torch.save(get_label.soft_labels, './save/mwn_v24/%s_%f/soft_labels_%d.pth' % (args.ds, args.partial_rate, epoch))
        
        print ('training...')
        # net.train()
        adjust_learning_rate(optimizer, epoch)

        index_meta = eval_train(net, train_loader)
        # print('index_meta', len(index_meta), index_meta)

        if args.ds == 'cifar100':
            meta_dataset = cifar100(root='./cifar100/', train_or_not=True, 
                                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]), 
                                index=index_meta, partial_type=args.partial_type, partial_rate=args.partial_rate )

            meta_loader = torch.utils.data.DataLoader(dataset=meta_dataset, batch_size=args.bs, num_workers=args.nw, shuffle=True, drop_last=False)

        if args.ds == 'cifar10':
            meta_dataset = cifar10(root='./cifar10/', train_or_not=True, 
                                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]), 
                                index=index_meta, partial_type=args.partial_type, partial_rate=args.partial_rate )

            meta_loader = torch.utils.data.DataLoader(dataset=meta_dataset, batch_size=args.bs, num_workers=args.nw, shuffle=True, drop_last=False)

        if args.ds == 'kmnist':
            meta_dataset = kmnist(root='./kmnist/',
                                        download=True,  
                                        train_or_not=True, 
                                        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))]), 
                                        partial_type=args.partial_type,
                                        partial_rate=args.partial_rate,  
                                        index=index_meta
            ) 
            meta_loader = torch.utils.data.DataLoader(dataset=meta_dataset, batch_size=args.bs, num_workers=args.nw, shuffle=True, drop_last=False)


        if args.ds == 'mnist':
            meta_dataset = mnist(root='./mnist/',
                                        download=True,  
                                        train_or_not=True, 
                                        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]), 
                                        partial_type=args.partial_type,
                                        partial_rate=args.partial_rate,  
                                        index=index_meta
            )
            meta_loader = torch.utils.data.DataLoader(dataset=meta_dataset, batch_size=args.bs, num_workers=args.nw, shuffle=True, drop_last=False)

        if args.ds == 'fashion':
            meta_dataset = fashion(root='./fashion/',
                                        download=True,  
                                        train_or_not=True, 
                                        transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]), 
                                        partial_type=args.partial_type,
                                        partial_rate=args.partial_rate,  
                                        index=index_meta
            ) 
            meta_loader = torch.utils.data.DataLoader(dataset=meta_dataset, batch_size=args.bs, num_workers=args.nw, shuffle=True, drop_last=False)

        print('meta_loader', torch.sum( (meta_dataset.train_final_labels == meta_dataset.train_labels).float() ))

        meta_loader_iter = iter(meta_loader)

        meta_lr = print_lr(optimizer, epoch)

        for i, (inputs, targets, _, index) in enumerate(train_loader): 
            print(i)
            net.train()
            net_ema.train()

            targets = torch.zeros(inputs.shape[0], num_classes).scatter_(1, targets.view(-1, 1).long(), 1)
            inputs, targets = inputs.cuda(), targets.cuda()

            # with torch.no_grad():
            #     outputs_ema = net_ema(inputs)
            #     psudo_label = get_label(outputs_ema, index, epoch)

            # l = torch.distributions.beta.Beta(args.alpha, args.alpha).sample().cuda()
            # # l = np.random.beta(args.alpha, args.alpha)
            # l = max(l, 1-l)
            # idx = torch.randperm(inputs.shape[0])
            # mix_inputs = l * inputs + (1-l) * inputs[idx]
            # mix_targets = l * targets + (1-l) * targets[idx]
            # mix_psudo_label = l * psudo_label + (1-l) * psudo_label[idx]
            

            if i % 20 == 0:
                # meta_model = mlp(n_inputs=num_features, n_outputs=num_classes)

                if args.model == 'linear':
                    meta_model = linear(n_inputs=num_features, n_outputs=num_classes)
                elif args.model == 'mlp':
                    meta_model = mlp(n_inputs=num_features, n_outputs=num_classes)
                elif args.model == 'convnet':
                    meta_model = convnet(input_channels=input_channels, n_outputs=num_classes, dropout_rate=dropout_rate)
                elif args.model == 'resnet':
                    meta_model = resnet(depth=32, n_outputs=num_classes)

                meta_model.cuda()
                meta_model.load_state_dict(net.state_dict())
                outputs = meta_model(inputs)

                cost_1 = torch.sum(-F.log_softmax(outputs, dim=1) * targets, dim=1)
                cost_11 = torch.reshape(cost_1, (len(cost_1), 1))
                v_lambda = vnet(cost_11.data, targets.data, c).squeeze(1)

                l_f_meta = torch.sum( torch.sum(-F.log_softmax(outputs, dim=1) * targets, dim=1) * v_lambda )/(torch.sum(v_lambda).detach().data)
                        
                meta_model.zero_grad()
                grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
                meta_model.update_params(lr_inner=meta_lr, source_params=grads)
                del grads

                try:
                    inputs_val, targets_val, _, _ = next(meta_loader_iter)
                except StopIteration:
                    meta_loader_iter = iter(meta_loader)
                    inputs_val, targets_val, _, _ = next(meta_loader_iter)
                inputs_val, targets_val = inputs_val.cuda(), targets_val.cuda()
                y_g_hat = meta_model(inputs_val)
                l_g_meta = F.cross_entropy(y_g_hat, targets_val)
                # prec_meta = accuracy(y_g_hat.data, targets_val.data, topk=(1,))[0]

                optimizer_vnet.zero_grad()
                l_g_meta.backward()
                optimizer_vnet.step()

            outputs = net(inputs)

            cost_1 = torch.sum(-F.log_softmax(outputs, dim=1) * targets, dim=1)
            cost_11 = torch.reshape(cost_1, (len(cost_1), 1))

            with torch.no_grad():
                v_lambda = vnet(cost_11, targets, c).squeeze(1)

            loss = torch.sum( torch.sum(-F.log_softmax(outputs, dim=1) * targets, dim=1) * v_lambda )/(torch.sum(v_lambda).detach().data)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            my_wl(cost_11, v_lambda, index)

            # for (param, param_ema) in zip(net.params(), net_ema.params()):
            #     param_ema.data.mul_(0.999).add_(0.001, param.data)

        print ('evaluating model...') 
        net.eval()      
        train_acc = evaluate(train_loader, net)
        test_acc = evaluate(test_loader, net)
        test_acc_ema = evaluate(test_loader, net_ema)

        with open(save_file, 'a') as file:
            file.write(str(int(epoch)) + ': Training Acc.: ' + str(round(train_acc, 4)) + ' , Test Acc.: ' + str(round(test_acc, 4)) + ' , Test Ema Acc.: ' + str(round(test_acc_ema, 4)) + '\n')
    

if __name__=='__main__':
    main()
