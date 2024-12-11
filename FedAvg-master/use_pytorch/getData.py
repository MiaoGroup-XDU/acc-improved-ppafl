import numpy as np
import gzip
import os
import platform
import pickle
import torchvision
import torch
from torchvision import transforms as transforms


class GetDataSet(object):
    def __init__(self, dataSetName, isIID):
        self.name = dataSetName
        self.train_data = None
        self.train_label = None
        self.train_data_size = None
        self.test_data = None
        self.test_label = None
        self.test_data_size = None

        self._index_in_train_epoch = 0

        if self.name == 'mnist':
            self.mnistDataSetConstruct(isIID)
        elif self.name == 'cifar10':
            self.load_data(isIID)
        else:
            pass


    def mnistDataSetConstruct(self, isIID):
        data_dir = r'E:\代码\FedAvg-master(1)\FedAvg-master\data\MNIST'
        # data_dir = r'./data/MNIST'
        train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
        train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
        test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
        test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
        train_images = extract_images(train_images_path)
        train_labels = extract_labels(train_labels_path)
        test_images = extract_images(test_images_path)
        test_labels = extract_labels(test_labels_path)
        # print(test_labels)

        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        assert train_images.shape[3] == 1
        assert test_images.shape[3] == 1
        train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])
        test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])
        # print(train_images.shape)

        train_images = train_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        if isIID:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
        else:
            # print(train_labels)
            labels = np.argmax(train_labels, axis=1)
            # print(labels)
            order = np.argsort(labels)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]


        self.test_data = test_images
        self.test_label = test_labels

    # 加载cifar10 的数据
    def load_data(self, isIID):
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True,
                                                 transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True,
                                                transform=test_transform)
        #print('cifar10')
        train_data = train_set.data.transpose((0,3,1,2))  # (50000, 32, 32, 3)
        print(train_data.shape)
        train_labels = train_set.targets
        train_labels = np.array(train_labels)  # 将标签转化为
        # train_labels = dense_to_one_hot(train_labels)
        # print(type(train_labels))  # <class 'numpy.ndarray'>
        # print(train_labels.shape)  # (50000,)

        test_data = test_set.data.transpose((0,3,1,2))  # 测试数据
        test_labels = test_set.targets
        test_labels = np.array(test_labels)
        # test_labels = dense_to_one_hot(test_labels)

        # print(test_labels)
        # test_labels = tensor.index_select(0, test_labels)
        # y_one_hot = torch.zeros(len(test_labels), 10).scatter_(1, test_labels, 1)
        # print(y_one_hot)
        # print(test_labels)
        # print()

        # train_data = train_data.transpose((0, 2, 3, 1))  # convert to HWC
        # test_data = test_data.transpose((0, 2, 3, 1))  # convert to HWC

        self.train_data_size = train_data.shape[0]
        self.test_data_size = test_data.shape[0]

        # print(train_data.shape)
        # 将训练集转化为（50000，32*32*3）矩阵
        # print(train_data.shape[1])
        # print(train_data.shape[3])
        # train_images = train_data.reshape(train_data.shape[0],
        #                                   train_data.shape[1], train_data.shape[2], train_data.shape[3])
        # # train_images = train_data.reshape(train_data.shape[0],
        # #                                   train_data.shape[1] * train_data.shape[2] * train_data.shape[3])
        # # print(train_images.shape)
        # # 将测试集转化为（10000，32*32*3）矩阵
        # test_images = test_data.reshape(test_data.shape[0],
        #                                 test_data.shape[1], test_data.shape[2],  test_data.shape[3])
        # # test_images = test_data.reshape(test_data.shape[0],
        # #                                 test_data.shape[1] * test_data.shape[2] * test_data.shape[3])

        # ---------------------------归一化处理------------------------------#
        train_images = train_data.astype(np.float32)
        # 数组对应元素位置相乘
        train_images = np.multiply(train_images, 1.0 / 255.0)
        # print(train_images[0:10,5:10])
        test_images = test_data.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)
        # ----------------------------------------------------------------#
        # print(train_images)
        '''
            一工有60000个样本
            100个客户端
            IID：
                我们首先将数据集打乱，然后为每个Client分配600个样本。
            Non-IID：
                我们首先根据数据标签将数据集排序(即MNIST中的数字大小)，
                然后将其划分为200组大小为300的数据切片，然后分给每个Client两个切片。
        '''
        if isIID:
            # 这里将50000 个训练集随机打乱
            order = np.arange(self.train_data_size)
            print(self.train_data_size)
            print(order)
            np.random.shuffle(order)
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
            # print(self.train_data)
        else:
            # 按照标签的
            # labels = np.argmax(train_labels, axis=1)
            # 对数据标签进行排序
            order = np.argsort(train_labels)
            print(self.train_data_size)
            print(order)
            # print("标签下标排序")
            # print(train_labels[order[20000:25000]])
            self.train_data = train_images[order]
            self.train_label = train_labels[order]
            # print(self.train_data)
        # print(self.train_label)


        self.test_data = test_images
        self.test_label = test_labels


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                    'Invalid magic number %d in MNIST label file: %s' %
                    (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        # print(labels)
        return dense_to_one_hot(labels)


if __name__=="__main__":
    'test data set'
    mnistDataSet = GetDataSet('cifar10', True) # test NON-IID
    if type(mnistDataSet.train_data) is np.ndarray and type(mnistDataSet.test_data) is np.ndarray and \
            type(mnistDataSet.train_label) is np.ndarray and type(mnistDataSet.test_label) is np.ndarray:
        print('the type of data is numpy ndarray')
    else:
        print('the type of data is not numpy ndarray')
    print('the shape of the train data set is {}'.format(mnistDataSet.train_data.shape))
    print('the shape of the test data set is {}'.format(mnistDataSet.test_data.shape))
    print(mnistDataSet.train_label[0:100], mnistDataSet.train_label[11000:11100])

