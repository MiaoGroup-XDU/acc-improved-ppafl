import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getData import GetDataSet


class client(object):
    def __init__(self, trainDataSet, dev, mu):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None
        self.mu = mu

    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters, mu):
        Net.load_state_dict(global_parameters, strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        for epoch in range(localEpoch):
            i = 0
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                # data = torch.unsqueeze(data, dim=0)  # cifar10时才用这个
                # print(data.shape)
                preds = Net(data)
                # label = torch.tensor(label)
                losss = lossFun(preds, label.long())
                # print(losss)
                proximal_term = 0.0
                # iterate through the current and global model parameters
                # *******************************异步方法记得接触注释
                # for key, value in Net.state_dict().items():
                #     proximal_term += (global_parameters[key] - Net.state_dict()[key]).norm(2)
                # ****************************************
                    # print(proximal_term)
                # for w, w_t in zip(Net.state_dict(), global_parameters):
                #     # update the proximal term
                #     print(w)
                #     print(w_t)
                #     proximal_term += (w - w_t).norm(2)
                # print(proximal_term)
                # print("---------------")
                loss = losss + (mu/2) * proximal_term
                # print(loss)
                loss.backward()
                opti.step()
                opti.zero_grad()
                i+=1

        return Net.state_dict()

    def local_val(self):
        pass


class ClientsGroup(object):
    def __init__(self, dataSetName, isIID, numOfClients, dev, mu):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}
        self.mu = mu

        self.test_data_loader = None

        self.dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):
        mnistDataSet = GetDataSet(self.data_set_name, self.is_iid)

        test_data = torch.tensor(mnistDataSet.test_data)
        # print(torch.tensor(mnistDataSet.test_label))
        if self.data_set_name=='mnist':
            test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)
        else:
            test_label= torch.tensor(mnistDataSet.test_label)

        self.test_data_loader = DataLoader(TensorDataset( test_data, test_label), batch_size=100, shuffle=False)

        train_data = mnistDataSet.train_data
        train_label = mnistDataSet.train_label

        print(train_data.shape)

        shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2
        shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)
        for i in range(self.num_of_clients):
            shards_id1 = shards_id[i * 2]
            shards_id2 = shards_id[i * 2 + 1]
            data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))
            # print(local_label.flatten().squeeze())
            if self.data_set_name=='mnist':
                local_label = np.argmax(local_label, axis=1)
            someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label.flatten().squeeze())), self.dev, self.mu)
            self.clients_set['client{}'.format(i)] = someone
        # print(self.data_set_name)

if __name__=="__main__":
    MyClients = ClientsGroup('mnist', True, 100, 1)
    print(MyClients.clients_set['client10'].train_ds[0:100])
    print(MyClients.clients_set['client11'].train_ds[400:500])


