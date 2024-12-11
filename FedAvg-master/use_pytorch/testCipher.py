import math

import tenseal as ts
import numpy as np
import time
import random
import sys
from Crypto.Util import number


k0 = 2048
k1 = 20
k2 = 160
p = number.getPrime(k0)
q = number.getPrime(k0)
N = p * q
# sk=(p,L)
# pk=(k0,k1,k2,N)
L = number.getPrime(k2)


def she_enc(p, L, m):  # m明文值
    r = random.getrandbits(k2)
    r1 = random.getrandbits(k0)
    return ((r * L + m) * (1 + r1 * p)) % N


def she_dec(p, L, c):  # c密文值值
    m = (c % p) % L
    if m < L / 2:
        return m
    else:
        return m - L


# Setup TenSEAL context
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
    # poly_modulus_degree=32768,  # 决定coeff里参数的个数，32768大概20多次乘法，65536大概40多次乘法。
    # coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 40, 40, 40,
    #                      40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60]  # 两个60之间的素数的位数
    # 要跟context.global_scale对应，例如context.global_scale是2的40次方，这里就是40位。(这里40的数量就是乘法的次数)
    # coeff_mod_bit_sizes=[60, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
    #                      30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
    #                      30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
    #                      30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
    #                      30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
    #                      60]
)
context.generate_galois_keys()
context.global_scale = 2 ** 40

file = open(r'E:\代码\FedAvg-master(1)+SHE\FedAvg-master\use_pytorch\\fc3weight.txt')
weight1 = []
for line in file.readlines():
    curLine = line.strip().split(" ")
    weight1.append(curLine[:])
# print(len(weight1))
w1 = np.array(weight1).flatten().tolist()
print(len(w1))

# w = w1[0:10000]
# print(w)
w2 = np.array(w1)
w2.astype(np.float)
print(type(w2.astype(np.float)[2]))
print(w2.astype(np.float).nbytes) # 统计明文包含的字节数
print(sys.getsizeof(w2))
she = []

w3 = w2.astype(np.float)
w4= []
for i in range(len(w1)):
    # print(w3[i])
    w4.append(math.ceil(w3[i]*10000000))
    she.append(she_enc(p, L, w4[i]))
print(np.array(she).nbytes) # 统计密文包含的字节数
print(sys.getsizeof(she))
# np.savetxt('w', w2.astype(np.float))
e = ts.ckks_vector(context, w1)
a = e.serialize() # 序列化为字节流
# print(a)
print(len(a)) # 统计CKKS密文包含的字节数
print(sys.getsizeof(a))

r = []
# flashe加密时间
# for i in range(len(w1)):
#     res2 = random.randrange(1, 11, len(w1))
#     r.append(res2)
#     # b = float(w1[i]) + r[i]
# starttime = time.clock()
# for i in range(len(w1)):
#     b = float(w1[i])+r[i]
# endtime = time.clock()
# print(endtime-starttime)

# flashe解密时间
# starttime = time.clock()
# for i in range(len(w1)):
#     b = float(w1[i])-r[i]
# endtime = time.clock()
# print(endtime-starttime)

# print(len(np.array(weight1).flatten().tolist()))
# encrypted vectors
# CKKS加解密时间
# starttime = time.time()
# enc_v1 = ts.ckks_vector(context, w1)
# endtime = time.time()
# print(endtime-starttime)
# stime = time.time()
# enc_v1.decrypt()
# etime = time.time()
# print(etime-stime)
# a = enc_v1.serialize()
# print(len(a))

# v1 = [0, 1, 2, 3, 4]
# v2 = [4, 3, 2, 1, 0]
# v3=[1,1,1,1,1]
#
# # encrypted vectors
# enc_v1 = ts.ckks_vector(context, v1)
# enc_v2 = ts.ckks_vector(context, v2)
# enc_v3 = ts.ckks_vector(context, v3)
#
# result = enc_v1 + enc_v2
# print(type(result.ciphertext()[0]))
# result.decrypt() # ~ [4, 4, 4, 4, 4]
# a = result.serialize()
# # print(int.from_bytes(a, byteorder='big', signed=False))
# # b = enc_v3.serialize()
# # print(len(b))
# print(a)
# print(len(a))
# print(type(a))
# # print(int(a))
# print(result.decrypt())
#
#
# result = enc_v1.dot(enc_v2)
# result.decrypt() # ~ [10]
#
# matrix = [
#   [73, 0.5, 8],
#   [81, -5, 66],
#   [-100, -78, -2],
#   [0, 9, 17],
#   [69, 11 , 10],
# ]
# result = enc_v1.matmul(matrix)
# result.decrypt() # ~ [157, -90, 153]
