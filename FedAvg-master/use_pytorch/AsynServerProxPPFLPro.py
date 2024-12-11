# 这个是改进版的AsyFL方案，没有使用加权更新，增加了落后客户端的本地迭代轮次(即将改为增加学习率)，也添加了mu项，
# 使用服务器保存的陈旧模型更新，也进行了量化，并且延迟客户端的选择上，不是每轮都有四个八个客户端延迟而改为
# 最多有四个或八个等等个延迟，不聚合延迟轮次大于threshold的延迟模型，但还没有添加掩码
import os
import argparse
import random
import time
from Crypto.Util import number
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import Mnist_2NN, Mnist_CNN
from ProxClients import ClientsGroup, client
import os
from encryption import encryption

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
parser.add_argument('--mu', type=float, default=0.1, help='proximal term constant')
parser.add_argument('--q_width', type=int, default=16)


def tes_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


# def StaleClientDecide(num_in_comm, PercentageOfStale):
#     # 这个函数用来在num_in_comm个客户端中选择百分之PercentageOfStale个客户端作为掉线的客户端
#     clients_in_comm = ['client{}'.format(i) for i in range(num_in_comm)]  # 生成num_in_comm个客户端
#     index = [i for i in order[0:num_in_comm][num_in_comm*PercentageOfStale]]  # 随机生成指定数目延迟客户端的索引
#     return clients_in_comm, index

k0=2048
k1=20
k2=160
p=number.getPrime(k0)
q=number.getPrime(k0)
N=p*q
#sk=(p,L)
#pk=(k0,k1,k2,N)
L=number.getPrime(k2)

def she_enc(p,L,m):#m明文值
    r = random.getrandbits(k2)
    r1 = random.getrandbits(k0)
    return ((r*L+m)*(1+r1*p))%N
def she_dec(p,L,c):#c密文值值
    m=(c%p)%L
    if m<L/2:
        return  m
    else:
        return  m-L

def flagUpdate(flag, time_stamp, epoch, percentageOfStale):
    num = flag.count(0)  # 统计当前轮次中有几个延迟客户端
    if num == 0:
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


def quantize_per_layer(party, r_maxs, bit_width=16):
    result = []
    for component, r_max in zip(party, r_maxs):
        x, _ = encryption.quantize_matrix_stochastic(component, bit_width=bit_width, r_max=r_max)
        result.append(x)
    return np.array(result)


def unquantize_per_layer(party, r_maxs, bit_width=16):
    result = []
    for component, r_max in zip(party, r_maxs):
        result.append(encryption.unquantize_matrix(component, bit_width=bit_width, r_max=r_max).astype(np.float32))
    return np.array(result)


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
    # opti = optim.SGD(net.parameters(), lr=args['learning_rate'])

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
    stale_threshold = 4  # 每个客户端延迟的阈值
    flag = [1] * 20  # 客户端是否参与训练的标志，0代表此轮不参与训练，1代表此轮参与训练。
    weight = [1] * 20  # 客户端在聚合时的权重
    stale_time = [0] * 20  # 客户端延时的轮次，0表示不延迟
    time_stamp = [0] * 20  # 时间戳，表示客户端参与的最近一次训练的全局轮次
    threshold = 6  # 使用未参与本轮训练的客户端的陈旧模型参与聚合，这是陈旧阈值。


    for i in range(args['num_comm']):
        print("communicate round {}".format(i + 1))
        number = 0
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
                    if (i+1) - time_stamp[iid[m]] > threshold:
                        weight[iid[m]] = 0
                        number+=1
                    else:
                        weight[iid[m]] = 1  # 设置权重
                    # weight[iid[m]] = 2**(-((i+1)-time_stamp[iid[m]]))  # 设置权重
                    print('客户{}的权重分数为{}'.format(iid[m], weight[iid[m]]))
            print("参与此轮训练的延迟客户端：{}".format(s))
            print("未参与此轮训练的延迟客户端：{}".format(list(set(iid) - set(s))))
            # 对于未参与训练的客户端：
            if i < 100:
                uu = list(set(iid) - set(s))  # 存放未参与此轮训练的客户端
                for m in range(len(uu)):
                    # flag[u[m]] = 1
                    if (i + 1) - time_stamp[uu[m]] <= threshold:
                        # weight[u[m]] = ((i + 1)/args['num_comm']) * (1 - ((i+1) - time_stamp[iid[m]])/(i + 1)) # 设置权重
                        # weight[uu[m]] = 1  # 设置权重
                        # weight[uu[m]] = 2 ** (-((i + 1) - time_stamp[uu[m]]))  # 设置权重
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
        # sum_weight = 0
        # for m in range(len(iiid)):
        #     sum_weight = sum_weight + weight[iiid[m]]
        rmax = []
        k = []

        for m in range(len(iiid)):
            print('******{}******'.format(iiid[m]))
            local_gradients = []
            # 对于上一轮参与了训练这一轮也参与训练的客户端，说明客户端的模型不是延迟模型
            if i - time_stamp[iiid[m]] == 1 or i == 0:
                opti = optim.SGD(net.parameters(), lr=args['learning_rate'])
                local_parameters = myClients.clients_set[clients_in_comm[iiid[m]]].localUpdate(args['epoch'],
                                                                                               args['batchsize'], net,
                                                                                               loss_func, opti,
                                                                                               global_parameters,
                                                                                               args["mu"])
            # 对于上一轮没参加训练这一轮参加训练的客户端，说明该客户端上传的模型是延迟的模型
            else:
                # 获取客户端最后一次参与训练所对应的全局模型，并用这个全局模型获得客户端的延迟模型
                opti = optim.SGD(net.parameters(), lr=0.02)
                stale_global_parameters = torch.load(os.path.join(args['save_path'],
                                                                  '{}_num_comm{}'.format(args['model_name'],
                                                                                         time_stamp[iiid[m]] - 1)))
                local_parameters = myClients.clients_set[clients_in_comm[iiid[m]]].localUpdate(args['epoch'],
                                                                                               args['batchsize'], net,
                                                                                               loss_func, opti,
                                                                                               stale_global_parameters,
                                                                                               args["mu"])
                # # 获取本轮的模型更新
                # for key, var in local_parameters.items():
                #     local_gradients[key] = local_parameters[key] - stale_global_parameters[key]
                #     # 获取本轮的模型更新
            keys = []
            for key, var in local_parameters.items():
                keys.append(key)
                local_gradients.append(local_parameters[key].numpy() - global_parameters[key].numpy())
            k = keys
            # 裁剪
            sizes = [item.size * args['num_of_clients'] for item in local_gradients]
            max_values = []
            min_values = []
            # print(local_gradients)
            # print(sizes)
            # for key, var in local_parameters.items():
            #     max_values.append([np.max(var.numpy())])
            #     min_values.append([np.min(var.numpy())])
            for layer_idx in range(len(local_gradients)):
                max_values.append([np.max([local_gradients[layer_idx]])])
                min_values.append([np.min([local_gradients[layer_idx]])])
            grads_max_min = np.concatenate([np.array(max_values), np.array(min_values)], axis=1)
            clipping_thresholds = encryption.calculate_clip_threshold_aciq_g(grads_max_min, sizes,
                                                                             bit_width=args['q_width'])
            # print(clipping_thresholds)

            r_maxs = [x * args['num_of_clients'] for x in clipping_thresholds]
            rmax = r_maxs
            # print(r_maxs)
            grads_batch_clients = [encryption.clip_with_threshold(local_gradients, clipping_thresholds)]
            # grads_batch_clients = [encryption.clip_with_threshold(item, clipping_thresholds)
            #  for item in local_parameters]
            # 量化
            local_gradients = [quantize_per_layer(local_gradients, r_maxs, bit_width=args['q_width'])]
            # print(local_gradients)

            #SHE加密

            print(local_gradients)

            # 聚合
            if sum_parameters is None:
                sum_parameters = {}
                for ith in range(len(local_gradients[0])):
                    # print(len(local_gradients[0]))
                    # print(ith)
                    # print(type(local_gradients[0][ith]))
                    # print(local_gradients[0][ith])
                    # local_gradients[ith] = local_gradients[ith].astype()
                    sum_parameters[keys[ith]] = torch.from_numpy(local_gradients[0][ith])
                # for key, var in local_parameters.items():
                #     sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    index = keys.index(var)
                    sum_parameters[var] = sum_parameters[var] + torch.from_numpy(local_gradients[0][index])*weight[iiid[m]]
        for m in range(len(iiiid)):
            time_stamp[iiiid[m]] = i + 1  # 设置本轮到达的客户端的时间戳为当前训练轮次
        # 反量化
        sum_gradients = []
        for key, var in sum_parameters.items():
            sum_gradients.append(var.numpy())
        sum_gradients = unquantize_per_layer(sum_gradients, rmax, bit_width=args['q_width'])
        # print(len(sum_gradients))

        for ith in range(len(sum_gradients)):
            sum_parameters[k[ith]] = torch.from_numpy(sum_gradients[ith])

        for var in global_parameters:
            # global_parameters[var] = (sum_parameters[var] / num_in_comm)
            global_parameters[var] = sum_parameters[var]/(len(iiid)+number) + global_parameters[var]

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
        r = os.getcwd() + "\\nonIID\FedProx_AsyAccuracy_PPFLPro\MNIST"
        np.savetxt(r + '\FedProx_AsyAccuracy_Pro(staleThreshold={},percent={},mu={},threshold={},100stale,notAggUnderThre,5local,0.02lr,newflag).txt'.format(
            stale_threshold, percentageOfStale, args["mu"], threshold), AsyAccuracy)
