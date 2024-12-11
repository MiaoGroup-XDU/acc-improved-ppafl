# 这是使用了量化的fedavg方案
import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import Mnist_2NN, Mnist_CNN
from clients import ClientsGroup, client
from encryption import encryption


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=20, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=1, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_2nn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=1, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=100, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')


def tes_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

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


if __name__=="__main__":
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

    myClients = ClientsGroup('mnist', args['IID'], args['num_of_clients'], dev)
    testDataLoader = myClients.test_data_loader

    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()
    Accuracy = []

    rmax = []
    k = []

    for i in range(args['num_comm']):
        print("communicate round {}".format(i+1))
        local_gradients = []
        order = np.random.permutation(args['num_of_clients'])
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

        sum_parameters = None
        for client in tqdm(clients_in_comm):
            local_parameters = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], net,
                                                                         loss_func, opti, global_parameters)

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
            print(sizes)
            # for key, var in local_parameters.items():
            #     max_values.append([np.max(var.numpy())])
            #     min_values.append([np.min(var.numpy())])
            for layer_idx in range(len(local_gradients)):
                max_values.append([np.max([local_gradients[layer_idx]])])
                min_values.append([np.min([local_gradients[layer_idx]])])
            grads_max_min = np.concatenate([np.array(max_values), np.array(min_values)], axis=1)
            clipping_thresholds = encryption.calculate_clip_threshold_aciq_g(grads_max_min, sizes,
                                                                             bit_width=args['q_width'])
            print(clipping_thresholds)

            r_maxs = [x * args['num_of_clients'] for x in clipping_thresholds]
            rmax = r_maxs
            print(r_maxs)
            grads_batch_clients = [encryption.clip_with_threshold(local_gradients, clipping_thresholds)]
            # grads_batch_clients = [encryption.clip_with_threshold(item, clipping_thresholds)
            #  for item in local_parameters]
            # 量化
            local_gradients = [quantize_per_layer(local_gradients, r_maxs, bit_width=args['q_width'])]
            # print(local_gradients)
            # 聚合
            if sum_parameters is None:
                sum_parameters = {}
                for ith in range(len(local_gradients[0])):
                    sum_parameters[keys[ith]] = torch.from_numpy(local_gradients[0][ith])
            else:
                for var in sum_parameters:
                    index = keys.index(var)
                    sum_parameters[var] = sum_parameters[var] + torch.from_numpy(local_gradients[0][index])

            # if sum_parameters is None:
            #     sum_parameters = {}
            #     for key, var in local_parameters.items():
            #         sum_parameters[key] = var.clone()
            # else:
            #     for var in sum_parameters:
            #         sum_parameters[var] = sum_parameters[var] + local_parameters[var]
        # 反量化
        sum_gradients = []
        for key, var in sum_parameters.items():
            sum_gradients.append(var.numpy())
        sum_gradients = unquantize_per_layer(sum_gradients, rmax, bit_width=args['q_width'])
        print(len(sum_gradients))

        for ith in range(len(sum_gradients)):
            sum_parameters[k[ith]] = torch.from_numpy(sum_gradients[ith])
        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / num_in_comm)

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
                Accuracy.append(sum_accu / num)

        # if (i + 1) % args['save_freq'] == 0:
        #     torch.save(net, os.path.join(args['save_path'],
        #                                  '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
        #                                                                                         i, args['epoch'],
        #                                                                                         args['batchsize'],
        #                                                                                         args['learning_rate'],
        #                                                                                         args['num_of_clients'],
        #                                                                                         args['cfraction'])))
        np.savetxt('Accuracy,nonIID,fmnist,quan.txt', Accuracy)
