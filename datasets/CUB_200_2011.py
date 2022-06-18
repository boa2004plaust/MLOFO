# encoding: utf-8

import os
import os.path as osp
from datasets.Databases import FGVCDatabase
from PIL import Image

class CUB_200_2011(FGVCDatabase):
    def __init__(self, root=r'/home/deep/JiabaoWang/data/FGVC/CUB_200_2011/', verbose=True, **kwargs):
        super(CUB_200_2011, self).__init__()
        train, test, self.train_list, self.test_list = self._process_dir(root)
        if verbose:
            print("=> CUB_200_2011 loaded")
            self.print_dataset_statistics(train, test)

        self.train = train
        self.test = test
        self.train_label = []
        for _, label in self.train:
            self.train_label.append(label)

        self.num_train_pids, self.num_train_imgs = self.get_imagedata_info(self.train)
        self.num_test_pids, self.num_test_imgs = self.get_imagedata_info(self.test)

    def _process_dir(self, root):
        classes = osp.join(root, 'classes.txt')
        bounding_boxes = osp.join(root, 'bounding_boxes.txt')
        images = osp.join(root, 'images.txt')
        image_class_label = osp.join(root, 'image_class_labels.txt')
        train_test_split = osp.join(root, 'train_test_split.txt')

        image_list = []
        with open(images, 'r', encoding='UTF-8') as f:
            lines_images = f.readlines()
            for line in lines_images:
                strs = line.split(' ')
                image_list.append(str(strs[1]).replace('\n',''))

        bounding_boxes_list = []
        with open(bounding_boxes, 'r', encoding='UTF-8') as f:
            lines_images = f.readlines()
            for line in lines_images:
                strs = line.split(' ')
                bbox = [float(strs[1]), float(strs[2]), float(strs[3]), float(strs[4])]
                bounding_boxes_list.append(bbox)

        class_list = []
        with open(image_class_label, 'r', encoding='UTF-8') as f:
            lines_classes = f.readlines()
            for line in lines_classes:
                strs = line.split(' ')
                class_list.append(int(strs[1]))
        # print(set(class_list))

        train_test_list = []
        with open(train_test_split, 'r', encoding='UTF-8') as f:
            lines_train_test = f.readlines()
            for line in lines_train_test:
                strs = line.split(' ')
                train_test_list.append(int(strs[1]))

        train_dataset = []
        test_dataset = []
        train_image_list = []
        test_image_list = []
        for image_path, label, bbox, train_test in zip(image_list, class_list, bounding_boxes_list, train_test_list):
            if train_test > 0:
                image_info = [os.path.join(root, 'images', image_path), int(label)-1]
                # print(image_info)
                train_dataset.append(image_info)
                train_image_list.append([image_path, bbox])
            else:
                image_info = [os.path.join(root, 'images', image_path), int(label)-1]
                # print(image_info)
                test_dataset.append(image_info)
                test_image_list.append([image_path, bbox])

        return train_dataset, test_dataset, train_image_list, test_image_list


if __name__ == '__main__':
    dataset = CUB_200_2011()
    for idx, x in enumerate(dataset.train):
        if idx % 500 == 0:
            print(x)
    print("--------------------------------")
    for idx, x in enumerate(dataset.test):
        if idx % 500 == 0:
            print(x)
