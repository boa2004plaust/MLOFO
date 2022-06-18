# encoding: utf-8

import os
import os.path as osp
from datasets.Databases import FGVCDatabase


class FGVC_Aircraft(FGVCDatabase):
    def __init__(self, root=r'/home/deep/JiabaoWang/data/FGVC/FGVC_Aircraft/data/', verbose=True, **kwargs):
        super(FGVC_Aircraft, self).__init__()
        train, test, self.train_list, self.test_list = self._process_dir(root)
        if verbose:
            print("=> FGVC_Aircraft loaded")
            self.print_dataset_statistics(train, test)

        self.train = train
        self.test = test
        self.train_label = []
        for _, label in self.train:
            self.train_label.append(label)

        self.num_train_pids, self.num_train_imgs = self.get_imagedata_info(self.train)
        self.num_test_pids, self.num_test_imgs = self.get_imagedata_info(self.test)

    def _process_dir(self, root):
        classes = osp.join(root, 'variants.txt')
        images_train = osp.join(root, 'images_variant_trainval.txt')
        images_test = osp.join(root, 'images_variant_test.txt')

        class_list = {}
        with open(classes, 'r', encoding='UTF-8') as f:
            i = 0
            lines_classes = f.readlines()
            for line in lines_classes:
                class_list[str(line).replace('\n', '')] = i
                i = i+1

        train_dataset = []
        train_image_list = []
        with open(images_train, 'r', encoding='UTF-8') as f:
            lines_images = f.readlines()
            for line in lines_images:
                image_path = str(line[:7]).strip()+'.jpg'
                label = class_list[str(line[7:-1]).strip()]
                image_info = [os.path.join(root, 'images', image_path), int(label)]
                train_dataset.append(image_info)
                train_image_list.append(image_path)

        test_dataset = []
        test_image_list = []
        with open(images_test, 'r', encoding='UTF-8') as f:
            lines_train_test = f.readlines()
            for line in lines_train_test:
                image_path = str(line[:7]).strip() + '.jpg'
                label = class_list[str(line[7:-1]).strip()]
                image_info = [os.path.join(root, 'images', image_path), int(label)]
                test_dataset.append(image_info)
                test_image_list.append(image_path)

        return train_dataset, test_dataset, train_image_list, test_image_list


if __name__ == '__main__':
    dataset = FGVC_Aircraft()
    for idx, x in enumerate(dataset.train):
        if idx % 500 == 0:
            print(x)
    print("--------------------------------")
    for idx, x in enumerate(dataset.test):
        if idx % 500 == 0:
            print(x)