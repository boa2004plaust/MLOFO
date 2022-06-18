# encoding: utf-8

import os
import os.path as osp
from datasets.Databases import FGVCDatabase


class Aircraft_Carrier(FGVCDatabase):
    def __init__(self, root=r'/home/deep/JiabaoWang/data/FGVC/Aircraft_Carrier/', verbose=True, **kwargs):
        super(Aircraft_Carrier, self).__init__()
        train, test, self.train_list, self.test_list = self._process_dir(root)
        if verbose:
            print("=> Aircraft_Carrier loaded")
            self.print_dataset_statistics(train, test)

        self.train = train
        self.test = test
        self.train_label = []
        for _, label in self.train:
            self.train_label.append(label)

        self.num_train_pids, self.num_train_imgs = self.get_imagedata_info(self.train)
        self.num_test_pids, self.num_test_imgs = self.get_imagedata_info(self.test)

    def _process_dir(self, root):
        images_train = osp.join(root, 'trainval_label.txt')
        images_test = osp.join(root, 'test_label.txt')

        train_dataset = []
        train_image_list = []
        with open(images_train, 'r', encoding='UTF-8') as f:
            lines_images = f.readlines()
            for line in lines_images:
                strs = line.split(' ')
                image_path = strs[0]
                label = strs[1].strip()
                # print(label)
                image_info = [image_path, int(label)]
                train_dataset.append(image_info)
                _, filename = os.path.split(image_path)
                train_image_list.append(filename)

        test_dataset = []
        test_image_list = []
        with open(images_test, 'r', encoding='UTF-8') as f:
            lines_train_test = f.readlines()
            for line in lines_train_test:
                strs = line.split(' ')
                image_path = strs[0]
                label = strs[1].strip()
                image_info = [image_path, int(label)]
                test_dataset.append(image_info)
                _, filename = os.path.split(image_path)
                test_image_list.append(filename)

        return train_dataset, test_dataset, train_image_list, test_image_list


if __name__ == '__main__':
    dataset = Aircraft_Carrier()
    for idx, x in enumerate(dataset.train):
        if idx % 500 == 0:
            print(x)
    print("--------------------------------")
    for idx, x in enumerate(dataset.test):
        if idx % 500 == 0:
            print(x)
