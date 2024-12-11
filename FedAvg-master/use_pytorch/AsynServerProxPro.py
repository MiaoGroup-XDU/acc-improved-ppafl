# 这个是AsyFL方案，加权更新，使用服务器保存的陈旧模型参与聚合，添加了mu项，还没有计算欧几里得距离
import os
import argparse

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import Mnist_2NN, Mnist_CNN
from ProxClients import ClientsGroup, client
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=20, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=1,
                    help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_2nn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=1, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=1, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=100, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')  # 0是noniid,1是iid
parser.add_argument('--mu', type=float, default=0.01, help='proximal term constant')


def tes_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


# def StaleClientDecide(num_in_comm, PercentageOfStale):
#     # 这个函数用来在num_in_comm个客户端中选择百分之PercentageOfStale个客户端作为掉线的客户端
#     clients_in_comm = ['client{}'.format(i) for i in range(num_in_comm)]  # 生成num_in_comm个客户端
#     index = [i for i in order[0:num_in_comm][num_in_comm*PercentageOfStale]]  # 随机生成指定数目延迟客户端的索引
#     return clients_in_comm, index

def flagUpdate(flag, time_stamp, epoch, percentageOfStale):
    num = flag.count(0)  # 统计当前轮次中有几个延迟客户端
    index = np.random.permutation(num_in_comm)[0:int(num_in_comm * percentageOfStale - num)]  # 随机生成指定数目延迟客户端的索引
    # print(index)
    for j in range(len(index)):  # 对于上一轮没有延迟的客户，将延迟标志设为0，时间戳设为上一轮次，上一轮延迟则不做更改
        if flag[index[j]] != 0:
            flag[index[j]] = 0
            time_stamp[index[j]] = epoch
    return flag, time_stamp


def stale_timeUpdate(flag, stale_time, stale_threshold):
    index = [i for i, x in enumerate(flag) if x == 0]  # 延迟客户端的索引
    for j in range(len(index)):
        # print(j)
        if stale_time[index[j]] == 0:  # 如果该客户端上一轮没有延迟则设置延迟轮数，如果上一轮延迟则不做更改
            # print(index[j])
            stale_time[index[j]] = np.random.permutation(stale_threshold)[0] + 1  # 随机确定客户端延迟的轮数
            # print(stale_time[index[j]])
    return stale_time


if __name__ == "__main__":
    args = parser.parse_args()
    args = args.__dict__

    tes_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net = None
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN()
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    net = net.to(dev)

    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=args['learning_rate'])

    myClients = ClientsGroup('mnist', args['IID'], args['num_of_clients'], dev, args["mu"])
    testDataLoader = myClients.test_data_loader

    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    AsyAccuracy = []  # 存放模型准确率

    global_parameters = {}  # 最新的全局模型
    stale_global_parameters = {}  # 陈旧的全局模型
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()
        stale_global_parameters[key] = var.clone()

    percentageOfStale = 0.2  # 表示延迟客户端所占总客户端数目的百分比
    stale_threshold = 8  # 每个客户端延迟的阈值
    flag = [1] * 20  # 客户端是否参与训练的标志，0代表此轮不参与训练，1代表此轮参与训练。
    weight = [1] * 20  # 客户端在聚合时的权重
    stale_time = [0] * 20  # 客户端延时的轮次，0表示不延迟
    time_stamp = [0] * 20  # 时间戳，表示客户端参与的最近一次训练的全局轮次
    threshold = 3  # 使用未参与本轮训练的客户端的陈旧模型参与聚合，这是陈旧阈值。


    for i in range(args['num_comm']):
        print("communicate round {}".format(i + 1))

        order = np.random.permutation(args['num_of_clients'])
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

        sum_parameters = None
        u = []  # 没有参与训练的客户端。

        if i != 0:
            flag, time_stamp = flagUpdate(flag, time_stamp, i, percentageOfStale)
            stale_time = stale_timeUpdate(flag, stale_time, stale_threshold)
            print("客户端的延迟轮次为：{}".format(stale_time))

            # 对于正常客户端：
            id = [key for key, value in enumerate(flag) if value == 1]  # 获取这些客户端的索引
            for m in range(len(id)):
                weight[id[m]] = 1  # 设置权重
                # print('客户{}的权重分数为{}'.format(id[m], weight[id[m]]))
            # 对于延迟客户端：
            iid = [key for key, value in enumerate(flag) if value == 0]  # 获取这些客户端的索引
            s = []  # 存放参与此轮训练的客户端中延迟客户端的索引
            for m in range(len(iid)):
                stale_time[iid[m]] -= 1
                # print(stale_time[iid[m]])
                if stale_time[iid[m]] == 0:
                    s.append(iid[m])
                    flag[iid[m]] = 1
                    # weight[iid[m]] = (i + 1) / (((i+1) - time_stamp[iid[m]])*args['num_comm'])  # 设置权重
                    # weight[iid[m]] = ((i + 1)/args['num_comm']) * (1 - ((i+1) - time_stamp[iid[m]])/(i + 1)) # 设置权重
                    # weight[iid[m]] = 1  # 设置权重
                    weight[iid[m]] = 2**(-((i+1)-time_stamp[iid[m]]))  # 设置权重
                    print('客户{}的权重分数为{}'.format(iid[m], weight[iid[m]]))
            print("参与此轮训练的延迟客户端：{}".format(s))
            print("未参与此轮训练的延迟客户端：{}".format(list(set(iid) - set(s))))
            # 对于未参与训练的客户端：
            if i < 20:
                uu = list(set(iid) - set(s))  # 存放未参与此轮训练的客户端
                for m in range(len(uu)):
                    # flag[u[m]] = 1
                    if (i + 1) - time_stamp[uu[m]] <= threshold:
                        # weight[u[m]] = ((i + 1)/args['num_comm']) * (1 - ((i+1) - time_stamp[iid[m]])/(i + 1)) # 设置权重
                        # weight[iid[m]] = 1  # 设置权重
                        weight[uu[m]] = 2 ** (-((i + 1) - time_stamp[uu[m]]))  # 设置权重
                        u.append(uu[m])
                        print("{}的权重分数为{}(此轮未到达但是用该客户端的陈旧模型)".format(uu[m], weight[uu[m]]))
                    else:
                        print("此轮未使用客户端{}的陈旧模型参与聚合".format(uu[m]))
            else:
                u = []
        # 获取参与此轮训练的客户端的索引（包括正常的和延迟的：）并参与训练
        iiiid = [key for key, value in enumerate(flag) if value == 1]
        iiid = iiiid + u
        print("参与此轮训练的所有客户端：{}".format(iiiid))
        time = []
        for b in range(20):
            time.append(time_stamp[b])
        print("客户端的时间戳：{}".format(time))
        # 计算所有客户端权重分数的和
        sum_weight = 0
        for m in range(len(iiid)):
            sum_weight = sum_weight + weight[iiid[m]]
        for m in range(len(iiid)):
            print('******{}******'.format(iiid[m]))
            # 对于上一轮参与了训练这一轮也参与训练的客户端，说明客户端的模型不是延迟模型
            if i - time_stamp[iiid[m]] == 1 or i == 0:
                local_parameters = myClients.clients_set[clients_in_comm[iiid[m]]].localUpdate(args['epoch'],
                                                                                               args['batchsize'], net,
                                                                                               loss_func, opti,
                                                                                               global_parameters,
                                                                                               args["mu"])
            # 对于上一轮没参加训练这一轮参加训练的客户端，说明该客户端上传的模型是延迟的模型
            else:
                # 获取客户端最后一次参与训练所对应的全局模型，并用这个全局模型获得客户端的延迟模型
                stale_global_parameters = torch.load(os.path.join(args['save_path'],
                                                                  '{}_num_comm{}'.format(args['model_name'],
                                                                                         time_stamp[iiid[m]] - 1)))
                local_parameters = myClients.clients_set[clients_in_comm[iiid[m]]].localUpdate(args['epoch'],
                                                                                               args['batchsize'], net,
                                                                                               loss_func, opti,
                                                                                               stale_global_parameters,
                                                                                               args["mu"])
            # print(type(local_parameters['fc1weight.txt']))
            # np.savetxt('fc1weight.txt', np.array(local_parameters['fc1weight.txt']))
            # np.savetxt('fc2weight.txt', np.array(local_parameters['fc2weight.txt']))
            # np.savetxt('fc3weight.txt', np.array(local_parameters['fc3weight.txt']))
            # 加权聚合
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone() * (weight[iiid[m]] / sum_weight)
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var] * (weight[iiid[m]] / sum_weight)
        for m in range(len(iiiid)):
            time_stamp[iiiid[m]] = i + 1  # 设置本轮到达的客户端的时间戳为当前训练轮次

        # cl = 0
        # for client in tqdm(clients_in_comm):
        #     local_parameters = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], net,
        #                                                                  loss_func, opti, global_parameters)
        #     if sum_parameters is None:
        #         sum_parameters = {}
        #         for key, var in local_parameters.items():
        #             sum_parameters[key] = var.clone()
        #     else:
        #         for var in sum_parameters:
        #             sum_parameters[var] = sum_parameters[var] + local_parameters[var]
        #     cl += 1

        for var in global_parameters:
            # global_parameters[var] = (sum_parameters[var] / num_in_comm)
            global_parameters[var] = sum_parameters[var]

        with torch.no_grad():
            if (i + 1) % args['val_freq'] == 0:
                net.load_state_dict(global_parameters, strict=True)
                sum_accu = 0
                num = 0
                for data, label in testDataLoader:
                    data, label = data.to(dev), label.to(dev)
                    preds = net(data)
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1
                print('accuracy: {}'.format(sum_accu / num))
                AsyAccuracy.append(sum_accu / num)

        if (i + 1) % args['save_freq'] == 0:
            torch.save(net.state_dict(), os.path.join(args['save_path'],
                                                      '{}_num_comm{}'.format(args['model_name'], i)))
            # torch.save(net, os.path.join(args['save_path'],
            #                              '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
            #                                                                                     i, args['epoch'],
            #                                                                                     args['batchsize'],
            #                                                                                     args['learning_rate'],
            #                                                                                     args['num_of_clients'],
            #                                                                                     args['cfraction'])))
        r = os.getcwd() + "\\nonIID\FedProx_AsyAccuracy_Pro"
        np.savetxt(r + '\FedProx_AsyAccuracy_Pro(staleThreshold={},percent={},mu={},threshold={},20stale,newnewWeight).txt'.format(
            stale_threshold, percentageOfStale, args["mu"], threshold), AsyAccuracy)
