import os
import random
import matplotlib.pyplot as plt
import shutil

"""
Utilities for managing eurosat dataset

"""

class data_manipulator:
    """
    data manipulator: takes dir with class name/imgs format and splits into
    train/test sets
    """
    def __init__(self, base_dir):
        self._base_dir = base_dir

    def train_test_split(self, validation=False, data_split=0.80):
        classes = os.listdir(self._base_dir)
        train_dir = self._base_dir + "/train_data"
        test_dir = self._base_dir + "/test_data"
        os.mkdir(train_dir)
        os.mkdir(test_dir)
        if validation:
            os.mkdir(self._base_dir + "/validation_data")
        for c in classes:
            train_class_dir = train_dir+"/"+c
            test_class_dir = test_dir + "/" + c
            os.mkdir(train_class_dir)
            os.mkdir(test_class_dir)
            if not validation:
                c_path = self._base_dir + "/" + c
                c_dir = os.listdir(c_path)
                #shuffle images so we take random ones each time for split
                random.shuffle(c_dir)
                print(c_dir)
                train = c_dir[0:int(data_split*len(c_dir))]
                test = c_dir[int(data_split*len(c_dir)):]
                #move all train data to train folder
                for train_data in train:
                    shutil.copy(c_path+ "/" + train_data, train_class_dir)
                # move all test data to test folder
                for test_data in test:
                    shutil.copy(c_path+ "/" + test_data, test_class_dir)
            """TODO: add validation"""
        print(classes)

def split_data(basedir, data_split=0.80):
    """
    quicker for calls in py console
    """
    manip = data_manipulator(basedir)
    manip.train_test_split(data_split=data_split)

def show_batch(image_batch):
    """
    visualization of batch
    :param image_batch: from ds_train typically
    :param label_batch: from ds_train typically
    :return: None`
    """
    fig, ax = plt.subplots(4, 3, figsize=(12,8))
    for n, b in enumerate(image_batch[0]):

        ax[n%4, n%3].imshow(b.numpy()
                            )
        plt.axis('off')

    plt.show()