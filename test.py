import numpy
import numpy as np
import matplotlib.pyplot as plt
import os

import torch

if __name__ == '__main__':
    accuracy = [1,2]
    r = os.getcwd()
    np.savetxt(r + '\\accu', accuracy)

    label = np.random.randint(0, 10, size=(8, 1))
    label = torch.LongTensor(label)
    # label = [2,3,4]
    print(torch.nn.functional.one_hot(label, num_classes=10))
    # ones = torch.sparse.torch.eye(10)
    # print(ones.index_select(0,label,None))

    nIID42 = np.loadtxt(
        os.getcwd() + "\Cifar10-Accuracy-cnn2,nonIID.txt")
    plt.plot(nIID42, marker='o', markersize=5)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.show()
    # c = [1,1,1,1,0,0,0,0,0]
    # b = [1,1]
    # print(list(set(c)-set(b)))
    # print(c.count(0))
    # print([i for i, x in enumerate(c) if x == 0])
    # index = [numpy.random.permutation(4)[0:3]]
    # print(index)
    # print([numpy.random.permutation(4)[0:3]])
    # for i in range(8):
    #     print(numpy.random.permutation(4))
    # r = os.getcwd() + "\IID\FedProx_AsyAccuracy"
    # print("./IID/FedProx_AsyAccuracy")

    # nIID42 = np.loadtxt(
    #     os.getcwd() + "\\nonIID\FedProx_AsyAccuracy\FedProx_AsyAccuracy(staleThreshold=4,percent=0.2,mu=0.01,newWeight).txt")
    # nIID44 = np.loadtxt(
    #     os.getcwd() + "\\nonIID\FedProx_AsyAccuracy\FedProx_AsyAccuracy(staleThreshold=4,percent=0.4,mu=0.001,newWeight).txt")
    # nIID46 = np.loadtxt(
    #     os.getcwd() + "\\nonIID\AsyAccuracy\AsyAccuracy(staleThreshold=4,percent=0.6,newWeight1).txt")
    # nIID48 = np.loadtxt(
    #     os.getcwd() + "\\nonIID\AsyAccuracy\AsyAccuracy(staleThreshold=4,percent=0.8,newWeight1).txt")
    # proxnIID = np.loadtxt(os.getcwd() + "\\nonIID\AccuracyProx,nonIID,mu=0.001.txt")
    # avgnIID = np.loadtxt(os.getcwd() + "\\nonIID\Accuracy,nonIID.txt")
    #
    # plt.plot(nIID42, marker='o', markersize=5)  # 绘制折线图，添加数据点，设置点的大小
    # plt.plot(nIID44, marker='o', markersize=5)
    # plt.plot(nIID46, marker='o', markersize=5)
    # plt.plot(nIID48, marker='o', markersize=5)
    # plt.plot(proxnIID, marker='o', markersize=5)
    # plt.plot(avgnIID, marker='o', markersize=5)
    # plt.legend(['20%', '40%', '60%',
    #             '80%', 'FedProx', 'FedAvg', ])  # 'AsyFed_4_0.2', 'AsyFed_4_0.4', 'AsyFed_8_0.2',
    #
    # plt.title('non-IID,staleThreshold=4')
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.grid()
    # plt.show()

    # IID42 = np.loadtxt(os.getcwd() + "\IID\FedProx_AsyAccuracy\FedProx_AsyAccuracy(staleThreshold=4,percent=0.2,mu=0.1,newWeight).txt")
    # IID44 = np.loadtxt(os.getcwd() + "\IID\FedProx_AsyAccuracy\FedProx_AsyAccuracy(staleThreshold=4,percent=0.4,mu=0.1,newWeight).txt")
    # IID46 = np.loadtxt(os.getcwd() + "\IID\FedProx_AsyAccuracy\FedProx_AsyAccuracy(staleThreshold=4,percent=0.6,mu=0.1,newWeight).txt")
    # IID48 = np.loadtxt(os.getcwd() + "\IID\FedProx_AsyAccuracy\FedProx_AsyAccuracy(staleThreshold=4,percent=0.8,mu=0.1,newWeight).txt")
    # proxIID = np.loadtxt(os.getcwd() + "\IID\AccuracyProx,IID,mu=0.01.txt")
    # avgIID = np.loadtxt(os.getcwd() + "\IID\Accuracy,IID.txt")
    #
    # plt.plot(IID42, marker='o', markersize=5)  # 绘制折线图，添加数据点，设置点的大小
    # plt.plot(IID44, marker='o', markersize=5)
    # plt.plot(IID46, marker='o', markersize=5)
    # plt.plot(IID48, marker='o', markersize=5)
    # plt.plot(proxIID, marker='o', markersize=5)
    # plt.plot(avgIID, marker='o', markersize=5)
    # plt.legend(['20%', '40%', '60%',
    #             '80%','FedProx','FedAvg',])  # 'AsyFed_4_0.2', 'AsyFed_4_0.4', 'AsyFed_8_0.2',
    #
    # plt.title('IID,staleThreshold=4')
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.grid()
    # plt.show()

    fedavgIID = np.loadtxt(os.getcwd() + "\IID\Accuracy,IID.txt")
    fedproxIID = np.loadtxt(os.getcwd() + "\IID\AccuracyProx,IID,mu=0.01.txt")
    fedavgNIID = np.loadtxt(os.getcwd() + "\\nonIID\Accuracy,nonIID.txt")
    fedproxNIID = np.loadtxt(os.getcwd() + "\\nonIID\AccuracyProx,nonIID,mu=0.001.txt")
    plt.plot(fedavgIID, marker='o', markersize=5)  # 绘制折线图，添加数据点，设置点的大小
    plt.plot(fedproxIID, marker='o', markersize=5)
    plt.plot(fedavgNIID[0:50], marker='o', markersize=5)
    plt.plot(fedproxNIID[0:50], marker='o', markersize=5)
    plt.legend(['FedAvg,iid',  'FedProx,iid', 'FedAvg,non-iid', 'FedProx,non-iid'])  # 'AsyFed_4_0.2', 'AsyFed_4_0.4', 'AsyFed_8_0.2',
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.show()


    # FedAvg = np.loadtxt("Accuracy,nonIID.txt")
    # #Asy4_2 = np.loadtxt("AsyAccuracy(staleThreshold=4,percent=0.2).txt")
    # Asy4_2n = np.loadtxt("AsyAccuracy(staleThreshold=4,percent=0.2,newWeight).txt")
    # #Asy4_4 = np.loadtxt("AsyAccuracy(staleThreshold=4,percent=0.4).txt")
    # Asy4_4n = np.loadtxt("AsyAccuracy(staleThreshold=4,percent=0.4,newWeight).txt")
    # #Asy8_2 = np.loadtxt("AsyAccuracy(staleThreshold=8,percent=0.2).txt")
    # Asy8_2n = np.loadtxt("AsyAccuracy(staleThreshold=8,percent=0.2,newWeight1).txt")
    # # Asy8_4 = np.loadtxt("AsyAccuracy(staleThreshold=8,percent=0.4).txt")
    # # syn8_4 = np.loadtxt("AsyAccuracy(staleThreshold=8,percent=0.4,withSameWeight).txt")
    # Asy8_4n = np.loadtxt("AsyAccuracy(staleThreshold=8,percent=0.4,newWeight).txt")
    # Asy4_6n = np.loadtxt("AsyAccuracy(staleThreshold=4,percent=0.6,newWeight1).txt")
    # Asy8_6n = np.loadtxt("AsyAccuracy(staleThreshold=8,percent=0.6,newWeight1).txt")
    # Asy4_8n = np.loadtxt("AsyAccuracy(staleThreshold=4,percent=0.8,newWeight1).txt")
    #
    #
    # plt.plot(FedAvg, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
    # plt.plot(Asy4_2n, marker='o', markersize=3)
    # #plt.plot(Asy4_4, marker='o', markersize=3)
    # plt.plot(Asy4_4n, marker='o', markersize=3)
    # plt.plot(Asy8_2n, marker='o', markersize=3)
    # # plt.plot(Asy8_4, marker='o', markersize=3)
    # # plt.plot(syn8_4, marker='o', markersize=3)
    # plt.plot(Asy8_4n, marker='o', markersize=3)
    # plt.plot(Asy4_6n, marker='o', markersize=3)
    # plt.plot(Asy8_6n, marker='o', markersize=3)
    # plt.plot(Asy4_8n, marker='o', markersize=3)
    #
    # plt.legend(['FedAvg',  'AsyFed_4_0.2n', 'synFed_4_0.4n', 'AsyFed_8_0.2n', 'AsyFed_8_0.4n', 'AsyFed_4_0.6n', 'AsyFed_8_0.6n', 'AsyFed_4_0.8n'])  # 'AsyFed_4_0.2', 'AsyFed_4_0.4', 'AsyFed_8_0.2',
    #
    # plt.show()
