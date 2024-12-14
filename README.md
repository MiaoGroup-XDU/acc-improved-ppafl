# Accuracy-Improved Privacy-Preserving Asynchronous Federated Learning in IoT

This repository contains official PyTorch implementation of the following paper
>**Accuracy-Improved Privacy-Preserving Asynchronous Federated Learning in IoT**
>

**Abstract:** *To address the asynchronous challenges stemming from user resource constraints and intermittent network connections in distributed Internet of Things (IoT) systems, Asynchronous Federated Learning (AFL) has been extensively explored in both academic and industrial domains. However, existing AFL approaches still struggle to effectively tackle the issue of model aggregation information loss due to delayed users, which leads to degraded model accuracy. To alleviate the impact of delayed or unavailable model updates on model aggregation, we propose a novel model update enhancement method to compensate for the loss of model aggregation information. Specifically, we utilize delayed model updates as an update agent for this user and correct the weights of these updates within the delayed rounds threshold to ensure the delayed user’s contribution to the model aggregation. Additionally, we integrate Symmetric Homomorphic Encryption (SHE) into AFL to ensure user privacy while simultaneously minimizing computational overhead. Lastly, we conduct extensive experiments to demonstrate that our scheme improves model accuracy by 6.53% compared to state-of-the-art
solutions.*

## Installation
This repository is built in PyTorch 1.4.0 and Python 3.7.6. See the requirements.txt for the installation of dependencies required to run our scheme.


```
pip install -r requirements.txt
```


### prepare data sets

You are supposed to prepare the data set by yourself. MNIST can be downloaded on http://yann.lecun.com/exdb/mnist/, and CIFAR-10 can be downloaded on http://www.cs.toronto.edu/~kriz/cifar.html. These data sets should be put into /data/MNIST and /data/CIFAR-10 when the download is finished.

### usage

Run the code

```
python main.py -nc 100 -cf 0.1 -E 5 -B 10 -mn mnist_cnn  -ncomm 1000 -iid 0 -lr 0.01 -vf 20 -g 0
```

which means there are 100 clients,  we randomly select 10 in each communicating round.  The data set are allocated in Non-IID way.  The epoch and batch size are set to 5 and 10. The learning rate is 0.01, we validate the codes every 20 rounds during the training, training stops after 1000 rounds. There are three models to do experiments: mnist_2nn mnist_cnn and cifar_cnn, and we choose mnist_cnn in this command. Notice the data set path when run the code of pytorch-version(you can take the source code out of the 'use_pytorch' folder). 



