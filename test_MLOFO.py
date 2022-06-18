from __future__ import print_function
import os
import sys
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler
from datasets.DataLoader import ImageDataset
from models.MSmodel import MS_stage3
from utils import mkdir_if_missing, Logger

parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')

# Fixed
parser.add_argument('--height', type=int, default=448, help="height of an image (default: 448)")
parser.add_argument('--width', type=int, default=448, help="width of an image (default: 448)")

# Option
parser.add_argument('--batch-size', default=24, type=int, help="train batch size")

# Setting
parser.add_argument('-d', '--dataset', type=str, default='CUB')
parser.add_argument('--model-dir', type=str, default='cub')
parser.add_argument('--exp-dir', type=str, default='log')

args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = '3,2,1,0'


def test_epoch(dataloader, net, criterion, use_cuda, device):
    net.eval()
    test_correct = [0, 0, 0, 0]
    test_acc = [0, 0, 0, 0]
    test_com_correct = 0
    total = 0
    from utils import jigsaw_generator
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)

            output_list, _ = net(inputs)

            output_com = output_list[0] + output_list[1]
            for i in range(2, len(output_list)):
                output_com += output_list[i]

            total += targets.size(0)
            for i in range(len(output_list)):
                _, predicted = torch.max(output_list[i].data, 1)
                test_correct[i] += predicted.eq(targets.data).cpu().sum()

            _, predicted_com = torch.max(output_com.data, 1)
            test_com_correct += predicted_com.eq(targets.data).cpu().sum()

    for i in range(len(test_correct)):
        test_acc[i] = 100. * float(test_correct[i]) / total
    test_com_acc = 100. * float(test_com_correct) / total
    print('Test Results:')
    for i in range(len(test_acc)):
        print('\tTest Cls%d_Acc: %6.2f%%' % (i + 1, test_acc[i]))
    print('Test_Cls_Com_Acc = %6.2f%%\n' % (test_com_acc))


def test_main(dataset_name, dataset, batch_size=64, model_dir=''):
    print('==> Preparing data..')
    transform_test = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    testset = ImageDataset(dataset_name, dataset.test, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=16)

    print('==> Preparing model..')
    net = torch.load(model_dir+"/model.pth")
    # GPU
    device = torch.device("cuda:0")
    net.to(device)
    cudnn.benchmark = True

    CELoss = nn.CrossEntropyLoss()

    test_epoch(testloader, net, CELoss, use_cuda, device)


if __name__ == '__main__':
    args.exp_dir = args.model_dir

    mkdir_if_missing(args.exp_dir)
    sys.stdout = Logger(os.path.join(args.exp_dir, 'log_evaluation.txt'))

    print(args)
    # setup output
    use_cuda = torch.cuda.is_available()
    # use_cuda = False
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

    test_main(dataset_name=args.dataset,
              dataset=dataset,
              batch_size=args.batch_size,
              model_dir=args.model_dir,
               )
