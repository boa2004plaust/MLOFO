from __future__ import absolute_import
import os
import sys
import errno
import random
import os.path as osp
import torch


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def save_checkpoint(net, store_name, device):
    net.cpu()
    torch.save(net, './' + store_name + '/model.pth')
    net.to(device)


def jigsaw_generator(images, n):
    l = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])
    block_size = 112 // n
    rounds = n ** 2
    random.shuffle(l)
    jigsaws = images.clone()
    for i in range(rounds):
        x, y = l[i]
        temp = jigsaws[..., 0:block_size, 0:block_size].clone()
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
                                                y * block_size:(y + 1) * block_size].clone()
        jigsaws[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp

    return jigsaws


from PIL import Image
import torch
import torchvision.transforms as transforms
from datasets.DataLoader import read_image


def PIL2Tensor(image):
    loader = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
    ])
    image = loader(image).unsqueeze(0)
    return image


def Tensor2PIL(tensor):
    unloader = transforms.ToPILImage()
    image = tensor.clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


if __name__ == '__main__':

    print("--------------------------------")
    images = read_image('/home/deep/data/FineGrained/CUB_200_2011/CUB_200_2011/images/012.Yellow_headed_Blackbird/Yellow_Headed_Blackbird_0056_8455.jpg')
    # images.show()

    inputs = PIL2Tensor(images)
    inputs1 = jigsaw_generator(inputs, 8)
    inputs2 = jigsaw_generator(inputs, 4)
    inputs3 = jigsaw_generator(inputs, 2)
    inputs4 = jigsaw_generator(inputs, 1)
    images1 = Tensor2PIL(inputs1)
    images1.show()
    images1.save('vis/step1.jpg')
    images2 = Tensor2PIL(inputs2)
    images2.show()
    images2.save('vis/step2.jpg')
    images3 = Tensor2PIL(inputs3)
    images3.show()
    images3.save('vis/step3.jpg')
    images4 = Tensor2PIL(inputs4)
    images4.show()
    images4.save('vis/step4.jpg')

