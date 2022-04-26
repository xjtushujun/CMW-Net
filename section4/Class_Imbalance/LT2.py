import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import time
import argparse
import random
import copy
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

from data_utils import *
from resnet import *
import shutil
from sklearn.cluster import KMeans


# parse arguments
parser = argparse.ArgumentParser(description='Imbalanced Example')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--num_meta', type=int, default=10,
                    help='The number of meta data for each class.')
parser.add_argument('--imb_factor', type=float, default=0.1)
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--split', type=int, default=1000)
parser.add_argument('--alpha', type=float, default=2.)
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
args = parser.parse_args()
print(args)


class keep_wl():
    def __init__(self, labels):
        self.loss = torch.zeros(labels.shape[0], dtype=torch.float).cuda(non_blocking=True)
        self.weight = torch.zeros(labels.shape[0], dtype=torch.float).cuda(non_blocking=True)

    def __call__(self, epoch_loss, epoch_weight, index):
        self.loss[index] = epoch_loss.detach().data
        self.weight[index] = epoch_weight.detach().data


def get_weight_loss(labels=None):
    wl = keep_wl(labels)
    return wl


kwargs = {'num_workers': 1, 'pin_memory': True}
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

train_data, test_dataset = build_dataset(args.dataset,args.num_meta)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)

np.random.seed(42)
random.seed(42)
# make imbalanced data
torch.manual_seed(args.seed)
classe_labels = range(args.num_classes)

data_list = {}


for j in range(args.num_classes):
    data_list[j] = [i for i, label in enumerate(train_loader.dataset.train_labels) if label == j]


img_num_list = get_img_num_per_cls(args.dataset,args.imb_factor, 0)
print(img_num_list)
print(sum(img_num_list))
print(min(img_num_list))

idx_to_del = []
# idx_to_meta = []

for cls_idx, img_id_list in data_list.items():
    random.shuffle(img_id_list)
    img_num = img_num_list[int(cls_idx)]
    idx_to_del.extend(img_id_list[img_num:])
    # idx_to_meta.extend(img_id_list[min(img_num_list):])

print(len(idx_to_del))

imbalanced_train_dataset = copy.deepcopy(train_data)
imbalanced_train_dataset.train_labels = np.delete(train_loader.dataset.train_labels, idx_to_del, axis=0)
imbalanced_train_dataset.train_data = np.delete(train_loader.dataset.train_data, idx_to_del, axis=0)
imbalanced_train_loader = torch.utils.data.DataLoader(imbalanced_train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)


# print(imbalanced_train_dataset.train_labels)
a = []
for i in range(args.num_classes):
    a.append([imbalanced_train_dataset.train_labels.tolist().count(i)])
    #print(i,' number is ', li.count(i))
# print(len(web_num))

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

torch.save(imbalanced_train_dataset.train_labels, './acmwn/label_%f.pth' % args.imb_factor)

my_wl = get_weight_loss(labels=imbalanced_train_dataset.train_labels)

# validation_loader = torch.utils.data.DataLoader(train_data_meta, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    # create model
    model = build_model()
    vnet = VNet(1, 100, 100, 1, 3).cuda()

    optimizer_a = torch.optim.SGD(model.params(), args.lr,
                                  momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)

    optimizer_c = torch.optim.Adam(vnet.params(), 1e-4, weight_decay=1e-5, amsgrad=True)

    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().cuda()
    # model.load_state_dict(torch.load('./acmwn21/model_%d_%f_%s.pth' % (159, args.imb_factor, args.dataset)))

    for epoch in range(0, args.epochs):
        adjust_learning_rate(optimizer_a, epoch)

        if epoch < 160:
            train(imbalanced_train_loader, validation_loader,model, optimizer_a,epoch)
            # if epoch > 150:
            #     torch.save(model.state_dict(), './acmwn21/model_%d_%f_%s.pth' % (epoch, args.imb_factor, args.dataset))

        else:
            data_list = {}
            for j in range(args.num_classes):
                data_list[j] = [i for i, label in enumerate(imbalanced_train_loader.dataset.train_labels) if label == j]

            idx_to_meta = []

            for cls_idx, img_id_list in data_list.items():
                random.shuffle(img_id_list)
                idx_to_meta.extend(img_id_list[min(img_num_list):])

            # print('idx_to_meta:', len(idx_to_meta))

            train_data_meta = copy.deepcopy(imbalanced_train_loader.dataset)
            train_data_meta.train_labels = np.delete(imbalanced_train_loader.dataset.train_labels, idx_to_meta, axis=0)
            train_data_meta.train_data = np.delete(imbalanced_train_loader.dataset.train_data, idx_to_meta, axis=0)

            validation_loader = torch.utils.data.DataLoader(train_data_meta, batch_size=args.batch_size, shuffle=True, **kwargs)
            # print(validation_loader.dataset.__dict__)

            train_meta(imbalanced_train_loader, validation_loader, model, vnet, optimizer_a, optimizer_c, epoch)
            # train(imbalanced_train_loader, validation_loader,model, optimizer_a,epoch)
            
            # torch.save(vnet.state_dict(), './acmwn21/vnet_%d_%f_%s.pth' % (epoch, args.imb_factor, args.dataset))
            # torch.save(model.state_dict(), './acmwn21/model_%d_%f_%s.pth' % (epoch, args.imb_factor, args.dataset))

            # torch.save(my_wl.loss, './acmwn21/epoch_loss_%d_%f_%s.pth' % (epoch, args.imb_factor, args.dataset))
            # torch.save(my_wl.weight, './acmwn21/epoch_weight_%d_%f_%s.pth' % (epoch, args.imb_factor, args.dataset))

        prec1, preds, gt_labels = validate(test_loader, model, criterion, epoch)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

    print('Best accuracy: ', best_prec1)


def train(train_loader, validation_loader,model,optimizer_a,epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    meta_losses = AverageMeter()
    top1 = AverageMeter()
    meta_top1 = AverageMeter()
    model.train()

    weight_eps_class = [0 for _ in range(int(args.num_classes))]
    total_seen_class = [0 for _ in range(int(args.num_classes))]
    for i, (input, target, _) in enumerate(train_loader):
        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)

        y_f = model(input_var)
        cost_w = F.cross_entropy(y_f, target_var, reduce=False)
        l_f = torch.mean(cost_w) # * w)
        prec_train = accuracy(y_f.data, target_var.data, topk=(1,))[0]


        losses.update(l_f.item(), input.size(0))
        #meta_losses.update(l_g_meta.item(), input.size(0))
        top1.update(prec_train.item(), input.size(0))
        #meta_top1.update(prec_meta.item(), input.size(0))

        optimizer_a.zero_grad()
        l_f.backward()
        optimizer_a.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  #'Meta_Loss {meta_loss.val:.4f} ({meta_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                  #'meta_Prec@1 {meta_top1.val:.3f} ({meta_top1.avg:.3f})'.format(
                epoch, i, len(train_loader),
                loss=losses,top1=top1))


def train_meta(train_loader, validation_loader,model,vnet,optimizer_a,optimizer_c,epoch):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    meta_losses = AverageMeter()
    top1 = AverageMeter()
    meta_top1 = AverageMeter()
    model.train()

    iter_validation_loader = iter(validation_loader)

    for i, (inputs, targets, index) in enumerate(train_loader):

        # targets = torch.zeros(inputs.shape[0], args.num_classes).scatter_(1, targets.view(-1, 1).long(), 1)
        input_var = to_var(inputs, requires_grad=False)
        target_var = to_var(targets, requires_grad=False)

        meta_model = ResNet32(args.dataset == 'cifar10' and 10 or 100).cuda()
        meta_model.load_state_dict(model.state_dict())

        y_f_hat = meta_model(input_var)

        cost_1 = LDAMLoss(img_num_list)(y_f_hat, target_var)
        cost_11 = torch.reshape(cost_1, (len(cost_1), 1))
        v_lambda = vnet(cost_11.data, target_var.data, c).squeeze(1)

        l_f_meta = (torch.sum( LDAMLoss(img_num_list)(y_f_hat, target_var) * v_lambda ))/(torch.sum(v_lambda).detach().data)

        meta_model.zero_grad()
        grads = torch.autograd.grad(l_f_meta, (meta_model.params()), create_graph=True)
        meta_lr = args.lr * ((0.01 ** int(epoch >= 160)) * (0.01 ** int(epoch >= 180)))
        meta_model.update_params(lr_inner=meta_lr, source_params=grads)
        del grads

        try:
            input_validation, target_validation, _ = next(iter_validation_loader)
        except StopIteration:
            iter_validation_loader = iter(validation_loader)
            input_validation, target_validation, _ = next(iter_validation_loader)
        input_validation_var = to_var(input_validation, requires_grad=False)
        target_validation_var = to_var(target_validation, requires_grad=False)
        y_g_hat = meta_model(input_validation_var)
        l_g_meta = F.cross_entropy(y_g_hat, target_validation_var)
        prec_meta = accuracy(y_g_hat.data, target_validation_var.data, topk=(1,))[0]

        optimizer_c.zero_grad()
        l_g_meta.backward()
        optimizer_c.step()


        y_f_hat = model(input_var)
        with torch.no_grad():
            cost_1 = LDAMLoss(img_num_list)(y_f_hat, target_var)
            cost_11 = torch.reshape(cost_1, (len(cost_1), 1))
            v_lambda = vnet(cost_11, target_var, c).squeeze(1)

        l_f = (torch.sum( LDAMLoss(img_num_list)(y_f_hat, target_var) * v_lambda ))/(torch.sum(v_lambda).detach().data)

        losses.update(l_f.item(), inputs.size(0))
        meta_losses.update(l_g_meta.item(), inputs.size(0))
        # top1.update(prec_train.item(), input.size(0))
        meta_top1.update(prec_meta.item(), inputs.size(0))

        optimizer_a.zero_grad()
        l_f.backward()
        optimizer_a.step()

        my_wl(cost_1, v_lambda, index)

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  #'Meta_Loss {meta_loss.val:.4f} ({meta_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                  #'meta_Prec@1 {meta_top1.val:.3f} ({meta_top1.avg:.3f})'.format(
                epoch, i, len(train_loader),
                loss=losses,top1=top1))


def validate(val_loader, model, criterion, epoch):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    true_labels = []
    preds = []

    end = time.time()
    for i, (input, target, index) in enumerate(val_loader):
        # print(target, index)
        target = target.cuda()#async=True)
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        with torch.no_grad():
            output = model(input_var)
        loss = criterion(output, target_var)

        output_numpy = output.data.cpu().numpy()
        preds_output = list(output_numpy.argmax(axis=1))

        true_labels += list(target_var.data.cpu().numpy())
        preds += preds_output


        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    # log to TensorBoard
    # import pdb; pdb.set_trace()

    return top1.avg, preds, true_labels


def build_model():
    model = ResNet32(args.dataset == 'cifar10' and 10 or 100)
    # print('Number of model parameters: {}'.format(
    #     sum([p.data.nelement() for p in model.parameters()])))

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True


    return model

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR divided by 10 at 160th, and 180th epochs"""
    # lr = args.lr * ((0.2 ** int(epoch >= 60)) * (0.2 ** int(epoch >= 120))* (0.2 ** int(epoch >= 160)))
    #lr = args.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 90)))
    lr = args.lr * ((0.01 ** int(epoch >= 160)) * (0.01 ** int(epoch >= 180)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# def adjust_learning_rate_v1(oargs, optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     epoch = epoch + 1
#     if epoch <= 5:
#         lr = args.lr * epoch / 5
#     elif epoch > 180:
#         lr = args.lr * 0.0001
#     elif epoch > 160:
#         lr = args.lr * 0.01
#     else:
#         lr = args.lr
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


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


def save_checkpoint(args, state, is_best):
    
    filename = '%s/%s/ckpt.pth.tar' % ('checkpoint', 'ours')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))

if __name__ == '__main__':
    main()


