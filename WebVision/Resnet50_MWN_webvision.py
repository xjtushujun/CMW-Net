import os
import sys
import torch
import random
import torchnet
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
# import dataloader_webvision as dataloader
from dataloaders import dataloader_webvision_v7 as dataloader

# from Inc_ACMWN_v1 import *
from models.resnet_mwn_v2 import SupCEResNet, VNet
from copy import deepcopy
from sklearn.cluster import KMeans


# os.environ['CUDA_VISIBLE_DEVICES']='0,1,2'

parser = argparse.ArgumentParser(description='PyTorch WebVision Training')
parser.add_argument('--batch_size', default=32, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--id', default='',type=str)
parser.add_argument('--seed', default=123)
parser.add_argument('--num_class', default=50, type=int)
parser.add_argument('--data_path', default='./data/', type=str, help='path to dataset')
parser.add_argument('--alpha', default='0.6', type=float)
parser.add_argument('--imagenet_data_path', default='./dataset/', type=str, help='path to ImageNet validation')
parser.add_argument('--num_batches', default=None, type=int)

args = parser.parse_args()

#torch.cuda.set_device([0, 1])
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def adjust_learning_rate(optimizer, epochs):
    lr = args.lr * ((0.1 ** int(epochs >= 50)))  # For WRN-28-10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


class keep_wl():
    def __init__(self, labels):
        self.loss = torch.zeros(labels.shape[0], 1, dtype=torch.float).cuda(non_blocking=True)
        self.weight = torch.zeros(labels.shape[0], dtype=torch.float).cuda(non_blocking=True)
        # self.loss1 = torch.zeros(labels.shape[0], 1, dtype=torch.float).cuda(non_blocking=True)
        # self.weight1 = torch.zeros(labels.shape[0], 1, dtype=torch.float).cuda(non_blocking=True)

    def __call__(self, epoch_loss, epoch_weight, index):
        self.loss[index] = epoch_loss.detach().data
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
    

def warmup(epoch,model,model_ema,vnet,optimizer_model,optimizer_vnet,train_loader,train_meta_loader,meta_lr):
    num_iter = (len(train_loader.dataset)//train_loader.batch_size)+1
    train_meta_loader_iter = iter(train_meta_loader)

    for batch_idx, (inputs, targets, index) in enumerate(train_loader):
        model.train()
        model_ema.train()

        targets = torch.zeros(inputs.shape[0], args.num_class).scatter_(1, targets.view(-1, 1).long(), 1)
        inputs, targets = inputs.cuda(), targets.cuda()

        l = torch.distributions.beta.Beta(torch.tensor([args.alpha]).cuda(),
                                          torch.tensor([args.alpha]).cuda() ).sample().cuda()
        l = max(l, 1-l)
        idx = torch.randperm(inputs.shape[0])
        mix_inputs = l * inputs + (1-l) * inputs[idx]

        with torch.no_grad():
            outputs_ema = model_ema(inputs)
            psudo_label = criterion(outputs_ema, index)

        if batch_idx % 10 == 0:
            meta_model = create_model()
            meta_model.load_state_dict(model.state_dict())
            outputs = meta_model(mix_inputs)

            with torch.no_grad():
                cost_1 = torch.sum(-F.log_softmax(outputs, dim=1) * targets, dim=1)
                cost_11 = torch.reshape(cost_1, (len(cost_1), 1))
                
                cost_2 = torch.sum(-F.log_softmax(outputs, dim=1) * targets[idx], dim=1)
                cost_12 = torch.reshape(cost_2, (len(cost_2), 1))
                
            v_lambda_1 = vnet(cost_11.detach().data, targets.detach().data, c).squeeze(1)
            v_lambda_2 = vnet(cost_12.detach().data, targets[idx].detach().data, c).squeeze(1)

            l_f_meta = ( l     * ( torch.sum(-F.log_softmax(outputs, dim=1) * targets, dim=1) * v_lambda_1
                            + torch.sum(-F.log_softmax(outputs, dim=1) * psudo_label, dim=1) * (1-v_lambda_1)
                            ) 
                    +(1-l) * (torch.sum(-F.log_softmax(outputs, dim=1) * targets[idx], dim=1) * v_lambda_2
                            +torch.sum(-F.log_softmax(outputs, dim=1) * psudo_label[idx], dim=1) * (1-v_lambda_2)
                            ) 
                    ).mean()
                     
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

            ll = torch.distributions.beta.Beta(torch.tensor([1.]).cuda(),
                                              torch.tensor([1.]).cuda() ).sample().cuda()
            # l = max(l, 1-l)
            idxx = torch.randperm(inputs_val.shape[0])
            mix_inputs_val = ll * inputs_val + (1-ll) * inputs_val[idxx]

            y_g_hat = meta_model(mix_inputs_val)
            l_g_meta = ll * F.cross_entropy(y_g_hat, targets_val) + (1-ll) * F.cross_entropy(y_g_hat, targets_val[idxx])
            # prec_meta = accuracy(y_g_hat.data, targets_val.data, topk=(1,))[0]

            optimizer_vnet.zero_grad()
            l_g_meta.backward()
            optimizer_vnet.step()
      
        outputs = model(mix_inputs)

        with torch.no_grad():
            cost_1 = torch.sum(-F.log_softmax(outputs, dim=1) * targets, dim=1)
            cost_11 = torch.reshape(cost_1, (len(cost_1), 1))
            v_lambda_1 = vnet(cost_11, targets, c).squeeze(1)

            cost_2 = torch.sum(-F.log_softmax(outputs, dim=1) * targets[idx], dim=1)
            cost_12 = torch.reshape(cost_2, (len(cost_2), 1))
            v_lambda_2 = vnet(cost_12, targets[idx], c).squeeze(1)

        loss = ( l     * ( torch.sum(-F.log_softmax(outputs, dim=1) * targets, dim=1) * v_lambda_1
                         + torch.sum(-F.log_softmax(outputs, dim=1) * psudo_label, dim=1) * (1-v_lambda_1)
                         ) 
                +(1-l) * (torch.sum(-F.log_softmax(outputs, dim=1) * targets[idx], dim=1) * v_lambda_2
                         +torch.sum(-F.log_softmax(outputs, dim=1) * psudo_label[idx], dim=1) * (1-v_lambda_2)
                         ) 
                ).mean()
    
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        my_wl(cost_11, v_lambda_1, index)

        for (param, param_ema) in zip(model.params(), model_ema.params()):
            param_ema.data.mul_(0.997).add_(0.003, param.data)

        sys.stdout.write('\r')
        sys.stdout.write('%s | Epoch [%3d/%3d] Iter[%4d/%4d]\t CE-loss: %.4f'
                %(args.id, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()


def train_CE(train_loader, model, model_ema, optimizer,epoch):
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    correct = 0
    total = 0
    # best_acc = 0
    for batch_idx, (inputs, targets, _) in enumerate(train_loader):
        model.train()
        model_ema.train()
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        for (param, param_ema) in zip(model.params(), model_ema.params()):
            param_ema.data.mul_(0.997).add_(0.003, param.data)

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
    correct = 0
    total = 0
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            # loss = F.cross_entropy(outputs, targets)
            # _, predicted = torch.max(outputs, 1) 

            # _, predicted = outputs.max(1)
            # total += targets.size(0)
            # correct += predicted.eq(targets).sum().item()
            
            acc_meter.add(outputs, targets)
    accs = acc_meter.value()
    # print('acc:', 100. * correct / total)
    return accs


def create_model():
    chekpoint = torch.load('pretrained/ckpt_webvision_{}.pth'.format('resnet50'))
    sd = {}
    for ke in chekpoint['model']:
        nk = ke.replace('module.', '')
        sd[nk] = chekpoint['model'][ke]
    model = SupCEResNet('resnet50', num_classes=args.num_class, pool=True)
    model.load_state_dict(sd, strict=False)
    # model = ResNet32(num_classes=args.num_class)
    # model = nn.DataParallel(model)
    model = model.cuda()
    return model


def eval_train(model):
    model.eval()
    num_iter = (len(eval_loader.dataset) // eval_loader.batch_size) + 1
    losses = torch.zeros(len(eval_loader.dataset))
    n = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets, path) in enumerate(eval_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets, reduce=False)
            for b in range(inputs.size(0)):
                losses[n] = loss[b]
                n += 1
            sys.stdout.write('\r')
            sys.stdout.write('| Evaluating loss Iter[%3d/%3d]\t' % (batch_idx, num_iter))
            sys.stdout.flush()

    # losses = (losses - losses.min()) / (losses.max() - losses.min())
    # all_loss.append(losses)

    return losses


stats_log=open('./checkpoint/%s'%(args.id)+'v99_stats.txt','w') 
test_log=open('./checkpoint/%s'%(args.id)+'v99_acc.txt','w')

loader = dataloader.webvision_dataloader(batch_size=args.batch_size, num_workers=5, root_dir=args.data_path,
                                         root_imagenet_dir=args.imagenet_data_path, log=stats_log,
                                         num_class=args.num_class, num_batches=args.num_batches)

print('| Building net')
net = create_model()
net_ema = create_model()
net_ema.load_state_dict(net.state_dict())

cudnn.benchmark = True

acc_meter = torchnet.meter.ClassErrorMeter(topk=[1,5], accuracy=True)

web_valloader = loader.run('test')
imagenet_valloader = loader.run('imagenet')
# train_imagenet_loader = loader.run('train_imagenet')   
warmup_trainloader = loader.run('warmup')
eval_loader = loader.run('eval_train')
print('Warmup Net')

a = []
web_num = [value for _, value in warmup_trainloader.dataset.train_labels.items()]
for i in range(50):
    a.append([web_num.count(i)])
    #print(i,' number is ', li.count(i))
print(len(web_num))

es = KMeans(3)
es.fit(a)
c = es.labels_

# qq = [2,2,2,2,0,1,0,2,0,2,2,2,1,2,1,1,0,0,0,0,2,0,2,0,1,2,2,0,2,2,2,0,1,2,1,0,0,2,0,1,1,2,0,2,0,2,2,2,1,1]
# print(qq)

# mm = [1, 0, 2]
# c = []
# for i in qq:
#     c.append(mm[i])

w = [[],[],[]]
for i in range(3):
    for k, j in enumerate(c):
        if i == j:
            w[i].append(a[k][0])

print(w)

criterion = get_loss(labels=np.asarray(web_num), num_classes=args.num_class, momentum=0.9)

my_wl = get_weight_loss(labels=np.asarray(web_num))

folder = 'v99'
if not os.path.exists('/data/save/' + folder):
    os.makedirs('/data/save/' + folder)

torch.save(np.asarray(web_num), '/data/save/%s/fake_label.pth' % folder)

vnet = VNet(1, 100, 100, 1, 3)
# vnet = nn.DataParallel(vnet)
vnet = vnet.cuda()

# vnet.load_state_dict(torch.load('./vnet_119.pth'))

optimizer = optim.SGD(net.params(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer_vnet = torch.optim.Adam([{'params':vnet.feature.params(),'lr':1e-4}, {'params':vnet.classfier.params(),'lr':1e-4}], weight_decay=1e-4)

for epoch in range(args.num_epochs):
    adjust_learning_rate(optimizer, epoch)
    if (epoch) < 5:
        train_CE(warmup_trainloader, net, net_ema, optimizer,epoch)
    else:
        losses = eval_train(net_ema)
        train_imagenet_loader = loader.run('meta', losses)
        
        meta_lr = print_lr(optimizer, epoch)
        warmup(epoch,net,net_ema,vnet,optimizer,optimizer_vnet,warmup_trainloader, train_imagenet_loader,meta_lr)
                 
    web_acc = test(net,web_valloader)  
    imagenet_acc = test(net,imagenet_valloader)

    web_acc_ema = test(net_ema,web_valloader)  
    imagenet_acc_ema = test(net_ema,imagenet_valloader)
    
    torch.save(criterion.soft_labels, '/data/save/%s/soft_label_%d.pth' % (folder, epoch))
    torch.save(my_wl.loss, '/data/save/%s/epoch_loss_%d.pth' % (folder, epoch))
    torch.save(my_wl.weight, '/data/save/%s/epoch_weight_%d.pth' % (folder, epoch))

    torch.save(net.state_dict(), '/data/%s/net_%d.pth' % (folder, epoch))
    torch.save(net_ema.state_dict(), '/data/%s/net_ema_%d.pth' % (folder, epoch))
    torch.save(vnet.state_dict(), '/data/%s/vnet_%d.pth' % (folder, epoch))
        
    print("\n| Test Epoch #%d\t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n"%(epoch,web_acc[0],web_acc[1],imagenet_acc[0],imagenet_acc[1]))  
    test_log.write('Epoch:%d \t WebVision Acc: %.2f%% (%.2f%%) \t ImageNet Acc: %.2f%% (%.2f%%)\n'%(epoch,web_acc[0],web_acc[1],imagenet_acc[0],imagenet_acc[1]))
    test_log.flush()  

    print("\n| Test Epoch #%d\t WebVision ema Acc: %.2f%% (%.2f%%) \t ImageNet ema Acc: %.2f%% (%.2f%%)\n"%(epoch,web_acc_ema[0],web_acc_ema[1],imagenet_acc_ema[0],imagenet_acc_ema[1]))  
    test_log.write('Epoch:%d \t WebVision ema Acc: %.2f%% (%.2f%%) \t ImageNet ema Acc: %.2f%% (%.2f%%)\n'%(epoch,web_acc_ema[0],web_acc_ema[1],imagenet_acc_ema[0],imagenet_acc_ema[1]))
    test_log.flush()  

