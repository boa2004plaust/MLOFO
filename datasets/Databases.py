# encoding: utf-8


class FGVCDatabase(object):
    """
    Base class of Fine Grained Image Classification dataset
    """

    def get_imagedata_info(self, data):
        classes = []
        for _, label in data:
            classes.append(label)
        class_set = set(classes)
        num_classes = len(class_set)
        num_imgs = len(data)
        return num_classes, num_imgs

    def print_dataset_statistics(self, train, test):
        num_train_pids, num_train_imgs = self.get_imagedata_info(train)
        num_test_pids, num_test_imgs = self.get_imagedata_info(test)

        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images   ")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}      ".format(num_train_pids, num_train_imgs))
        print("  test     | {:5d} | {:8d}      ".format(num_test_pids, num_test_imgs))
        print("  ------------------------------")
