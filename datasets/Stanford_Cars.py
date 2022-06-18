# encoding: utf-8

import os
import os.path as osp
from datasets.Databases import FGVCDatabase


class Stanford_Cars(FGVCDatabase):
    def __init__(self, root=r'/home/deep/JiabaoWang/data/FGVC/Stanford_Cars/', verbose=True, **kwargs):
        super(Stanford_Cars, self).__init__()
        train, test, self.train_list, self.test_list = self._process_dir(root)
        if verbose:
            print("=> Stanford_Cars loaded")
            self.print_dataset_statistics(train, test)

        self.train = train
        self.test = test
        self.train_label = []
        for _, label in self.train:
            self.train_label.append(label)

        self.num_train_pids, self.num_train_imgs = self.get_imagedata_info(self.train)
        self.num_test_pids, self.num_test_imgs = self.get_imagedata_info(self.test)

    def _process_dir(self, root):
        data_file = osp.join(root, 'cars_annos.mat')

        import scipy.io as sio
        data = sio.loadmat(data_file)
        test_flags = data['annotations']['test'][0]
        labels = data['annotations']['class'][0]
        images = data['annotations']['relative_im_path'][0]

        train_dataset = []
        test_dataset = []
        train_image_list = []
        test_image_list = []
        for test_flag, image_path, label in zip(test_flags, images, labels):
            image_info = [os.path.join(root, str(image_path[0])), int(label[0][0]) - 1]
            if test_flag[0][0]:
                train_dataset.append(image_info)
                train_image_list.append(str(image_path[0]))
            else:
                test_dataset.append(image_info)
                test_image_list.append(str(image_path[0]))

        return train_dataset, test_dataset, train_image_list, test_image_list


if __name__ == '__main__':
    dataset = Stanford_Cars()
    for idx, x in enumerate(dataset.train):
        if idx % 500 == 0:
            print(x)
    print("--------------------------------")
    for idx, x in enumerate(dataset.test):
        if idx % 500 == 0:
            print(x)
