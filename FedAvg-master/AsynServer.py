import os
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from Models import Models
from clients import clients, user


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0,1,2,3,4,5,6,7', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=10, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=1, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--modelname', type=str, default='cifar10_cnn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=100, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')


def tes_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

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

if __name__=='__main__':
    args = parser.parse_args()

    # GPU preparation
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    tes_mkdir(args.save_path)

    if args.modelname == 'mnist_2nn' or args.modelname == 'mnist_cnn':
        datasetname = 'mnist'
        with tf.variable_scope('inputs') as scope:
            inputsx = tf.placeholder(tf.float32, [None, 784])
            inputsy = tf.placeholder(tf.float32, [None, 10])
    elif args.modelname == 'cifar10_cnn':
        datasetname = 'cifar10'
        with tf.variable_scope('inputs') as scope:
            inputsx = tf.placeholder(tf.float32, [None, 24, 24, 3])
            inputsy = tf.placeholder(tf.float32, [None, 10])

    myModel = Models(args.modelname, inputsx)

    predict_label = tf.nn.softmax(myModel.outputs)
    with tf.variable_scope('loss') as scope:
        Cross_entropy = -tf.reduce_mean(inputsy * tf.log(predict_label), axis=1)

    with tf.variable_scope('train') as scope:
        optimizer = tf.train.GradientDescentOptimizer(args.learning_rate)
        train = optimizer.minimize(Cross_entropy)

    with tf.variable_scope('validation') as scope:
        correct_prediction = tf.equal(tf.argmax(predict_label, axis=1), tf.argmax(inputsy, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    saver = tf.train.Saver(max_to_keep=3)

    # ---------------------------------------- train --------------------------------------------- #
    with tf.Session(config=tf.ConfigProto(
            log_device_placement=False, \
            allow_soft_placement=True, \
            gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        sess.run(tf.initialize_all_variables())

        myClients = clients(args.num_of_clients, datasetname,
                            args.batchsize, args.epoch, sess, train, inputsx, inputsy, is_IID=args.IID)

        vars = tf.trainable_variables()
        global_vars = sess.run(vars)
        num_in_comm = int(max(args.num_of_clients * args.cfraction, 1))

        AsyAccuracy = []  # 存放模型准确率

        global_parameters = {}  # 最新的全局模型
        stale_global_parameters = {}  # 陈旧的全局模型
        for key, var in net.state_dict().items():
            global_parameters[key] = var.clone()
            stale_global_parameters[key] = var.clone()

        percentageOfStale = 0.8  # 表示延迟客户端所占总客户端数目的百分比
        stale_threshold = 8  # 每个客户端延迟的阈值
        flag = [1] * 20  # 客户端是否参与训练的标志，0代表此轮不参与训练，1代表此轮参与训练。
        weight = [1] * 20  # 客户端在聚合时的权重
        stale_time = [0] * 20  # 客户端延时的轮次，0表示不延迟
        time_stamp = [0] * 20  # 时间戳，表示客户端参与的最近一次训练的全局轮次

        for i in range(args.num_comm):
            print("communicate round {}".format(i))
            order = np.arange(args.num_of_clients)
            np.random.shuffle(order)
            clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

            sum_vars = None
            for client in tqdm(clients_in_comm):
                local_vars = myClients.ClientUpdate(client, global_vars)
                if sum_vars is None:
                    sum_vars = local_vars
                else:
                    for sum_var, local_var in zip(sum_vars, local_vars):
                        sum_var += local_var

            global_vars = []
            for var in sum_vars:
                global_vars.append(var / num_in_comm)

            if i % args.val_freq == 0:
                for variable, value in zip(vars, global_vars):
                    variable.load(value, sess)
                test_data = myClients.test_data
                test_label = myClients.test_label
                print(sess.run(accuracy, feed_dict={inputsx: test_data, inputsy: test_label}))

            if i % args.save_freq == 0:
                checkpoint_name = os.path.join(args.save_path, '{}_comm'.format(args.modelname) +
                                               'IID{}_communication{}'.format(args.IID, i+1)+ '.ckpt')
                save_path = saver.save(sess, checkpoint_name)
