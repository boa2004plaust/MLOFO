from __future__ import print_function
import os
import sys
import math
import time
import datetime
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler
from datasets.DataLoader import ImageDataset
from models.Resnet import resnet50
from models.MSmodel import MS_stage3
from utils import mkdir_if_missing, save_checkpoint, Logger, jigsaw_generator

parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')

# Fixed
parser.add_argument('--height', type=int, default=448, help="height of an image (default: 448)")
parser.add_argument('--width', type=int, default=448, help="width of an image (default: 448)")

# Option
parser.add_argument('--total-epochs', default=100, type=int, help="total epochs to run")
parser.add_argument('--batch-size', default=64, type=int, help="train batch size")
parser.add_argument('-lr', '--learning-rate', default=0.0035, type=float, help="initial learning rate")
parser.add_argument('--warmup-epochs', default=5, type=int, help="warmup-epochs")

# Setting
parser.add_argument('-d', '--dataset', type=str, default='CUB')
parser.add_argument('-a', '--arch', type=str, default='resnet50')
parser.add_argument('--scheduler', type=str, default='warmup', help="optimization algorithm")
parser.add_argument('--exp-dir', type=str, default='log')
parser.add_argument('--measure', type=str, default='cos')
parser.add_argument('-s', '--nsample', type=int, default=4, help='the number of identities sampled')
parser.add_argument('-w', '--weight', type=float, default=0.001, help='the weights for IM and CE')

args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = '3,2,1,0'


def GenIdx(train_label):
    color_pos = []
    unique_label = np.unique(train_label)
    for i in range(len(unique_label)):
        tmp_pos = [k for k, v in enumerate(train_label) if v == unique_label[i]]
        color_pos.append(tmp_pos)

    return color_pos


class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label: labels of two modalities
            color_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_label, color_pos, batchSize, per_img):
        uni_label = np.unique(train_label)
        self.n_classes = len(uni_label)

        sample_color = np.arange(batchSize)
        N = len(train_label)

        # per_img = 4
        per_id = batchSize / per_img
        for j in range(N // batchSize + 1):
            batch_idx = np.random.choice(uni_label, int(per_id), replace=False)

            for s, i in enumerate(range(0, batchSize, per_img)):
                sample_color[i:i + per_img] = np.random.choice(color_pos[batch_idx[s]],
                                                               per_img, replace=False)

            if j == 0:
                index = sample_color
            else:
                index = np.hstack((index, sample_color))

        self.index = index
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index)))

    def __len__(self):
        return self.N


def max_v(x, y):
    if x > y:
        return x
    else:
        return y


class PrototypeMetricLoss(nn.Module):
    def __init__(self, margin=0.1, measure='l2', nsample=4, weight=0.0001):
        super(PrototypeMetricLoss, self).__init__()
        self.margin = margin
        self.nsample = nsample
        self.weight = weight
        self.measure = measure
        if measure == 'l2':
            self.dist = nn.MSELoss(reduction='sum')
        if measure == 'cos':
            self.dist = nn.CosineSimilarity(dim=0)
        if measure == 'l1':
            self.dist = nn.L1Loss()

    def forward(self, feat1, label1):
        label_num = len(label1.unique())
        feat1 = feat1.chunk(label_num, 0)
        # print(label1)
        for i in range(label_num):
            center = torch.mean(feat1[i], dim=0)
            center_mat = center.repeat((self.nsample, 1))
            if self.measure == 'l2' or self.measure == 'l1':
                if i == 0:
                    dist = self.weight * self.dist(feat1[i], center_mat) / feat1[i].shape[0]
                else:
                    dist += self.weight * self.dist(feat1[i], center_mat) / feat1[i].shape[0]
            elif self.measure == 'cos':
                cos_dist = 1 - self.dist(feat1[i], center_mat)
                if i == 0:
                    dist = self.weight * (cos_dist.sum() / feat1[i].shape[0])
                else:
                    dist += self.weight * (cos_dist.sum() / feat1[i].shape[0])

        return dist


def test_epoch(dataloader, net, criterion, use_cuda, device):
    net.eval()
    epoch_correct = [0, 0, 0, 0]
    epoch_acc = [0, 0, 0, 0]
    epoch_com_correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)

            output_list, _ = net(inputs)

            output_com = output_list[0] + output_list[1]
            for i in range(2, len(output_list)):
                output_com += output_list[i]

            total += targets.size(0)
            for i in range(len(output_list)):
                _, predicted = torch.max(output_list[i].data, 1)
                epoch_correct[i] += predicted.eq(targets.data).cpu().sum()

            _, predicted_com = torch.max(output_com.data, 1)
            epoch_com_correct += predicted_com.eq(targets.data).cpu().sum()

    for i in range(len(epoch_correct)):
        epoch_acc[i] = 100. * float(epoch_correct[i]) / total
    epoch_com_acc = 100. * float(epoch_com_correct) / total

    return epoch_acc, epoch_com_acc


def train_epoch(dataloader, net, criterion, use_cuda, device,
                optimizer, scheduler):
    net.train()
    epoch_ce_loss = [0, 0, 0, 0]
    epoch_pm_loss = [0, 0, 0, 0]
    epoch_correct = [0, 0, 0, 0]
    epoch_ce_loss_sum = 0
    epoch_pm_loss_sum = 0
    epoch_total_loss = 0
    epoch_com_correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)

        optimizer.zero_grad()
        v = random.random()
        if v > 0.75:
            inputs = jigsaw_generator(inputs, 8)
        elif v > 0.50:
            inputs = jigsaw_generator(inputs, 4)
        elif v > 0.25:
            inputs = jigsaw_generator(inputs, 2)
        else:
            inputs = inputs

        output_list, feature_list = net(inputs)

        iter_ce_loss_list = [0, 0, 0, 0]
        iter_pm_loss_list = [0, 0, 0, 0]
        iter_ce_loss_sum = 0
        iter_pm_loss_sum = 0
        for i in range(len(output_list)):
            w = 1
            if i == 3:
                w = 2
            iter_ce_loss_list[i] = criterion[0](output_list[i], targets) * w
            iter_pm_loss_list[i] = criterion[1](feature_list[i], targets) * w
            iter_ce_loss_sum += iter_ce_loss_list[i]
            iter_pm_loss_sum += iter_pm_loss_list[i]
        iter_total_loss = iter_ce_loss_sum + iter_pm_loss_sum
        iter_total_loss.backward()
        optimizer.step()

        total += targets.size(0)
        for i in range(len(output_list)):
            _, predicted = torch.max(output_list[i].data, 1)
            epoch_correct[i] += predicted.eq(targets.data).cpu().sum()
            epoch_ce_loss[i] += iter_ce_loss_list[i].item()
            epoch_pm_loss[i] += iter_pm_loss_list[i].item()
        epoch_ce_loss_sum += iter_ce_loss_sum.item()
        epoch_pm_loss_sum += iter_pm_loss_sum.item()
        epoch_total_loss += iter_total_loss.item()

        output_com = output_list[0] + output_list[1]
        for i in range(2, len(output_list)):
            output_com += output_list[i]
        _, predicted_com = torch.max(output_com.data, 1)
        epoch_com_correct += predicted_com.eq(targets.data).cpu().sum()

        if batch_idx % 50 == 0:
            print('Step: %d, ' % (batch_idx))
            for i in range(len(output_list)):
                print('Cls%d_Acc: %6.2f%% (%5d/%5d), L%d_ce: %.6f, L%d_im: %.6f' % (
                    i+1, 100. * float(epoch_correct[i]) / total, epoch_correct[i], total,
                    i+1, epoch_ce_loss[i] / total, i+1, epoch_pm_loss[i] / total))
    scheduler.step()

    epoch_acc = [0, 0, 0, 0]
    for i in range(len(epoch_correct)):
        epoch_acc[i] = 100. * float(epoch_correct[i]) / total
        epoch_ce_loss[i] = epoch_ce_loss[i] / total
        epoch_pm_loss[i] = epoch_pm_loss[i] / total

    epoch_com_acc = 100. * float(epoch_com_correct) / total
    epoch_ce_loss_sum = epoch_ce_loss_sum / total
    epoch_pm_loss_sum = epoch_pm_loss_sum / total
    epoch_total_loss = epoch_total_loss / total

    return epoch_acc, epoch_ce_loss, epoch_pm_loss, epoch_com_acc, epoch_ce_loss_sum, epoch_pm_loss_sum, epoch_total_loss


def train_mlofo(dataset_name, dataset, arch, total_epoch, batch_size=64, learn_rate=0.0035,
               lr_scheduler='warmup', warmup_epochs=10, store_name=None,
               measure='l2', weight=0.0001, nsample=4):

    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = ImageDataset(dataset_name+'_train', dataset.train, transform=transform_train)
    testset = ImageDataset(dataset_name+'_test', dataset.test, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=16)

    print('==> Preparing model..')
    net = resnet50(pretrained=True)
    # for mo in net.layer4[0].modules():
    #     if isinstance(mo, nn.Conv2d):
    #         mo.stride = (1, 1)
    for param in net.parameters():
        param.requires_grad = True
    net = MS_stage3(net, 512, classes_num=dataset.num_train_pids)
    netp = torch.nn.DataParallel(net, device_ids=[0,1,2,3])

    # GPU
    device = torch.device("cuda:0")
    net.cuda()
    cudnn.benchmark = True

    CELoss = nn.CrossEntropyLoss()
    IMLoss = PrototypeMetricLoss(measure=measure, nsample=nsample, weight=weight)

    param_setting = [
        {'params': net.features.parameters(), 'lr': 0.1 * learn_rate},
        {'params': net.conv_block1.parameters(), 'lr': learn_rate},
        {'params': net.feat1.parameters(), 'lr': learn_rate},
        {'params': net.classifier1.parameters(), 'lr': learn_rate},
        {'params': net.conv_block2.parameters(), 'lr': learn_rate},
        {'params': net.feat2.parameters(), 'lr': learn_rate},
        {'params': net.classifier2.parameters(), 'lr': learn_rate},
        {'params': net.conv_block3.parameters(), 'lr': learn_rate},
        {'params': net.feat3.parameters(), 'lr': learn_rate},
        {'params': net.classifier3.parameters(), 'lr': learn_rate},
        {'params': net.feat_concat.parameters(), 'lr': learn_rate},
        {'params': net.classifier_concat.parameters(), 'lr': learn_rate}
    ]
    optimizer = optim.SGD(param_setting, momentum=0.9, weight_decay=5e-4)

    if lr_scheduler == 'warmup':
        warm_up_epochs = warmup_epochs
        warm_up_with_cosine_lr = lambda epoch: epoch / warm_up_epochs if epoch <= warm_up_epochs else 0.5 * (
                math.cos((epoch - warm_up_epochs) / (total_epoch - warm_up_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    elif lr_scheduler == 'no-warmup':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch)

    print('==> Training...')
    elapsed_time_train = 0
    elapsed_time_test = 0
    max_val_acc = 0
    color_pos = GenIdx(dataset.train_label)
    for epoch in range(0, total_epoch):
        print('\nEpoch: %d' % epoch)
        epoch_start_time = time.time()
        sampler = IdentitySampler(dataset.train_label, color_pos, batch_size, nsample)
        trainset.cIndex = sampler.index
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                  sampler=sampler, num_workers=16,
                                                  drop_last=True)
        train_acc, train_ce_loss, train_pm_loss, train_com_acc, train_ce_loss_sum, train_pm_loss_sum, train_total_loss = train_epoch(trainloader, netp, [CELoss, IMLoss], use_cuda,
                                       device, optimizer, scheduler)
        elapsed_time_train += round(time.time() - epoch_start_time)

        print('Iteration %d, ' % (epoch))
        for i in range(len(train_acc)):
            print('\tTrain Cls%d_Acc: %6.2f%%, L%d_ce: %.6f, L%d_pm: %.6f' % (
                i + 1, train_acc[i], i + 1, train_ce_loss[i], i + 1, train_pm_loss[i]))
        print('Iteration %d, Train_Acc = %6.2f%%, Train_CE_Loss = %.6f, Train_IM_Loss = %.6f, Train_Loss = %.6f' % (epoch,
            train_com_acc, train_ce_loss_sum, train_pm_loss_sum, train_total_loss))

        epoch_start_time = time.time()
        val_acc, val_com_acc = test_epoch(testloader, netp, CELoss, use_cuda, device)
        elapsed_time_test += round(time.time() - epoch_start_time)

        print('Iteration %d, ' % (epoch))
        for i in range(len(train_acc)):
            print('\tTest Cls%d_Acc: %6.2f%%' % (
                i + 1, val_acc[i]))
        print('Iteration %d, Test_Cls_Com_Acc = %5.2f%%' % (
            epoch, val_com_acc))

        epoch_elapsed = str(datetime.timedelta(seconds=elapsed_time_train))
        print("Train Elapsed Time (h:m:s): {}.".format(epoch_elapsed))

        epoch_elapsed = str(datetime.timedelta(seconds=elapsed_time_test))
        print("Test Elapsed Time (h:m:s): {}.".format(epoch_elapsed))

        if val_com_acc > max_val_acc:
            max_val_acc = val_com_acc
            save_checkpoint(netp, store_name, device)


if __name__ == '__main__':
    args.exp_dir = args.dataset + '_' + args.arch + '_mlofo' + \
                   '_epochs' + str(args.total_epochs) + \
                   '_lr' + str(args.learning_rate) + \
                   '_' + args.scheduler + str(args.warmup_epochs) + \
                   '_dist' + args.measure + \
                   '_sample' + str(args.nsample) + \
                   '_weight' + str(args.weight) + args.exp_dir

    mkdir_if_missing(args.exp_dir)
    sys.stdout = Logger(os.path.join(args.exp_dir, 'log_train.txt'))

    print(args)
    # setup output
    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    if args.dataset == 'CUB':
        from datasets.CUB_200_2011 import CUB_200_2011
        dataset = CUB_200_2011()
    elif args.dataset == 'Aircraft':
        from datasets.FGVC_Aircraft import FGVC_Aircraft
        dataset = FGVC_Aircraft()
    elif args.dataset == 'Cars':
        from datasets.Stanford_Cars import Stanford_Cars
        dataset = Stanford_Cars()
    elif args.dataset == 'Carrier':
        from datasets.Aircraft_Carrier import Aircraft_Carrier
        dataset = Aircraft_Carrier()
    else:
        print("Dataset is error.")
        exit()

    train_mlofo(dataset_name=args.dataset,
                dataset=dataset,
                arch=args.arch,
                total_epoch=args.total_epochs,
                batch_size=args.batch_size,
                learn_rate=args.learning_rate,
                lr_scheduler=args.scheduler,
                warmup_epochs=args.warmup_epochs,
                store_name=args.exp_dir,
                measure=args.measure,
                weight=args.weight,
                nsample=args.nsample
                )
