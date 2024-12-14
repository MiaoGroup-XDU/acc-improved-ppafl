import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getData import GetDataSet
import cv2


class client(object):
    def __init__(self, trainDataSet, dev):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None
        self.dataset_size = 5000
        # self.isToPreprocess = preprocess
        # print('init')
        # if self.isToPreprocess == 1:
        #     print("ispre")
        #     self.preprocess()

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters,mu):
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                loss = lossFun(preds, label)
                loss.backward()
                opti.step()
                opti.zero_grad()

        return Net.state_dict()

    def preprocess(self):
        print("preprocess")
        new_images = []
        shape = (24, 24, 3)
        for i in range(self.dataset_size):
            old_image = self.train_ds[i, :, :, :]
            old_image = np.pad(old_image, [[4, 4], [4, 4], [0, 0]], 'constant')
            left = np.random.randint(old_image.shape[0] - shape[0] + 1)
            top = np.random.randint(old_image.shape[1] - shape[1] + 1)
            new_image = old_image[left: left + shape[0], top: top + shape[1], :]

            if np.random.random() < 0.5:
                new_image = cv2.flip(new_image, 1)

            mean = np.mean(new_image)
            std = np.max([np.std(new_image),
                          1.0 / np.sqrt(self.train_ds.shape[1] * self.train_ds.shape[2] * self.train_ds.shape[3])])
            new_image = (new_image - mean) / std

            new_images.append(new_image)

        self.train_ds = new_images

    def local_val(self):
        pass


class ClientsGroup(object):
    def __init__(self, dataSetName, isIID, numOfClients, dev, mu):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}

        self.test_data_loader = None

        self.dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):
        # print(self.data_set_name)
        mnistDataSet = GetDataSet(self.data_set_name, self.is_iid)

        test_data = torch.tensor(mnistDataSet.test_data)
        # print(torch.tensor(mnistDataSet.test_label))
        if self.data_set_name == 'mnist':
            test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)
        else:
            test_label = torch.tensor(mnistDataSet.test_label)
        # print(test_label)
        self.test_data_loader = DataLoader(TensorDataset(test_data, test_label), batch_size=100, shuffle=False)

        train_data = mnistDataSet.train_data
        train_label = mnistDataSet.train_label

        shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2
        shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)
        for i in range(self.num_of_clients):
            shards_id1 = shards_id[i * 2]
            shards_id2 = shards_id[i * 2 + 1]
            data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            # if i == 0:
            #     print(label_shards1)
            #     print(label_shards2)
            #     print(len(np.vstack((label_shards1, label_shards2))))
            #     print(np.vstack((label_shards1, label_shards2)))
            local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            # if i == 0:
            #     print(local_label)
            if self.data_set_name == 'mnist':
                local_label = np.argmax(local_label, axis=1)
            elif self.data_set_name == 'cifar10':
                local_label = np.ndarray.flatten(local_label)
            # if i == 0:
            #     print(local_label)
            # print(torch.tensor(local_data))
            # print(len(torch.tensor(local_data)))
            preprocess = 1 if self.data_set_name == 'cifar10' else 0
            someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
            self.clients_set['client{}'.format(i)] = someone

if __name__=="__main__":
    MyClients = ClientsGroup('mnist', True, 100, 1)
    print(MyClients.clients_set['client10'].train_ds[0:100])
    print(MyClients.clients_set['client11'].train_ds[400:500])


