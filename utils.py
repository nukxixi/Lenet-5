import struct
import numpy as np
import math
import random

def readDataset(dataset):
    (image, label) = dataset
    with open(label, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(image, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    return (img, lbl)

# 图像的zero padding
def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, ), (pad, ), (pad, ), (0, )), 'constant', constant_values=(0, 0))    
    return X_pad

# 图像标准化
def normalize(image, mode='lenet5'):
    image -= image.min()
    image = image / image.max()
    # range = [0,1]
    if mode == '0p1':
        return image
    # range = [-1,1]
    elif mode == 'n1p1':
        image = image * 2 - 1
    # range = [-0.1,1.175]   
    elif mode == 'lenet5':
        image = image * 1.275 - 0.1
    return image

# 初始化权重和偏置
def initialize(kernel_shape, mode='Fan-in'):
    b_shape = (1,1,1,kernel_shape[-1]) if len(kernel_shape)==4 else (kernel_shape[-1],)
    if mode == 'Gaussian_dist':
        mu, sigma = 0, 0.1
        weight = np.random.normal(mu, sigma,  kernel_shape) 
        bias   = np.ones(b_shape)*0.01
        
    elif mode == 'Fan-in':
        Fi = np.prod(kernel_shape)/kernel_shape[-1]
        weight = np.random.uniform(-2.4/Fi, 2.4/Fi, kernel_shape)    
        bias   = np.ones(b_shape)*0.01     
    return weight, bias

# 更新权重
def update(weight, bias, dW, db, vw, vb, lr, momentum=0, weight_decay=0):
    vw_u = momentum*vw - weight_decay*lr*weight - lr*dW
    vb_u = momentum*vb - weight_decay*lr*bias   - lr*db
    weight_u = weight + vw_u
    bias_u   = bias   + vb_u
    return weight_u, bias_u, vw_u, vb_u 

#  mini-batches
def random_mini_batches(image, label, mini_batch_size = 256, one_batch=False):
    m = image.shape[0]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_image = image[permutation,:,:,:]
    shuffled_label = label[permutation]

    if one_batch:
        mini_batch_image = shuffled_image[0: mini_batch_size,:,:,:]
        mini_batch_label = shuffled_label[0: mini_batch_size]
        return (mini_batch_image, mini_batch_label)

    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_image = shuffled_image[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_label = shuffled_label[k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_image, mini_batch_label)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:
        mini_batch_image = shuffled_image[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_label = shuffled_label[num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_image, mini_batch_label)
        mini_batches.append(mini_batch)
    return mini_batches

import numpy as np

bitmap = np.zeros((10,84))
bitmap[0]=np.array([
    [-1, +1, +1, +1, +1, +1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] + \
    [-1, -1, +1, +1, +1, -1, -1] + \
    [-1, +1, +1, -1, +1, +1, -1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [-1, +1, +1, -1, +1, +1, -1] + \
    [-1, -1, +1, +1, +1, -1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] \
])
bitmap[1]=np.array([
    [-1, -1, -1, +1, +1, -1, -1] + \
    [-1, -1, +1, +1, +1, -1, -1] + \
    [-1, +1, +1, +1, +1, -1, -1] + \
    [-1, -1, -1, +1, +1, -1, -1] + \
    [-1, -1, -1, +1, +1, -1, -1] + \
    [-1, -1, -1, +1, +1, -1, -1] + \
    [-1, -1, -1, +1, +1, -1, -1] + \
    [-1, -1, -1, +1, +1, -1, -1] + \
    [-1, -1, -1, +1, +1, -1, -1] + \
    [-1, +1, +1, +1, +1, +1, +1] + \
    [-1, -1, -1, -1, -1, -1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] \
])
bitmap[2]=np.array([
    [-1, +1, +1, +1, +1, +1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] + \
    [-1, +1, +1, +1, +1, +1, -1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [+1, -1, -1, -1, -1, +1, +1] + \
    [-1, -1, -1, -1, +1, +1, -1] + \
    [-1, -1, +1, +1, +1, -1, -1] + \
    [-1, +1, +1, -1, -1, -1, -1] + \
    [+1, +1, -1, -1, -1, -1, -1] + \
    [+1, +1, +1, +1, +1, +1, +1] + \
    [-1, -1, -1, -1, -1, -1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] \
])
bitmap[3]=np.array([
    [+1, +1, +1, +1, +1, +1, +1] + \
    [-1, -1, -1, -1, -1, +1, +1] + \
    [-1, -1, -1, -1, +1, +1, -1] + \
    [-1, -1, -1, +1, +1, -1, -1] + \
    [-1, -1, +1, +1, +1, +1, -1] + \
    [-1, -1, -1, -1, -1, +1, +1] + \
    [-1, -1, -1, -1, -1, +1, +1] + \
    [-1, -1, -1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [-1, +1, +1, +1, +1, +1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] \
])
bitmap[4]=np.array([
    [-1, +1, +1, +1, +1, +1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] + \
    [-1, +1, +1, -1, -1, +1, +1] + \
    [-1, +1, +1, -1, -1, +1, +1] + \
    [+1, +1, +1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, +1, +1, +1] + \
    [-1, +1, +1, +1, +1, +1, +1] + \
    [-1, -1, -1, -1, -1, +1, +1] + \
    [-1, -1, -1, -1, -1, +1, +1] \
])
bitmap[5]=np.array([
    [-1, +1, +1, +1, +1, +1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] + \
    [+1, +1, +1, +1, +1, +1, +1] + \
    [+1, +1, -1, -1, -1, -1, -1] + \
    [+1, +1, -1, -1, -1, -1, -1] + \
    [-1, +1, +1, +1, +1, -1, -1] + \
    [-1, -1, +1, +1, +1, +1, -1] + \
    [-1, -1, -1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [-1, +1, +1, +1, +1, +1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] \
])
bitmap[6]=np.array([
    [-1, -1, +1, +1, +1, +1, -1] + \
    [-1, +1, +1, -1, -1, -1, -1] + \
    [+1, +1, -1, -1, -1, -1, -1] + \
    [+1, +1, -1, -1, -1, -1, -1] + \
    [+1, +1, +1, +1, +1, +1, -1] + \
    [+1, +1, +1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [+1, +1, +1, -1, -1, +1, +1] + \
    [-1, +1, +1, +1, +1, +1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] \
])
bitmap[7]=np.array([
    [+1, +1, +1, +1, +1, +1, +1] + \
    [-1, -1, -1, -1, -1, +1, +1] + \
    [-1, -1, -1, -1, -1, +1, +1] + \
    [-1, -1, -1, -1, +1, +1, -1] + \
    [-1, -1, -1, +1, +1, -1, -1] + \
    [-1, -1, -1, +1, +1, -1, -1] + \
    [-1, -1, +1, +1, -1, -1, -1] + \
    [-1, -1, +1, +1, -1, -1, -1] + \
    [-1, -1, +1, +1, -1, -1, -1] + \
    [-1, -1, +1, +1, -1, -1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] \
])
bitmap[8]=np.array([
    [-1, +1, +1, +1, +1, +1, -1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [-1, +1, +1, +1, +1, +1, -1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [-1, +1, +1, +1, +1, +1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] \
])
bitmap[9]=np.array([
    [-1, +1, +1, +1, +1, +1, -1] + \
    [+1, +1, -1, -1, +1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, -1, +1, +1] + \
    [+1, +1, -1, -1, +1, +1, +1] + \
    [-1, +1, +1, +1, +1, +1, +1] + \
    [-1, -1, -1, -1, -1, +1, +1] + \
    [-1, -1, -1, -1, -1, +1, +1] + \
    [-1, -1, -1, -1, +1, +1, -1] + \
    [-1, +1, +1, +1, +1, -1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] + \
    [-1, -1, -1, -1, -1, -1, -1] \
])

def rbf_init_weight():
    return bitmap

#池化层
def pool_forward(A_prev, hparameters, mode):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))      
    for h in range(n_H):
        for w in range(n_W):
            A_prev_slice = A_prev[:, h*stride:h*stride+f, w*stride:w*stride+f, :]
            if mode == "max":
                A[:, h, w, :] = np.max(A_prev_slice, axis=(1,2))
            elif mode == "average":
                A[:, h, w, :] = np.average(A_prev_slice, axis=(1,2))

    cache = (A_prev, hparameters)
    assert(A.shape == (m, n_H, n_W, n_C))
    return A, cache

#反向池化
def pool_backward(dA, cache, mode):
    (A_prev, hparameters) = cache
    
    stride = hparameters["stride"]
    f = hparameters["f"]

    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape #256,28,28,6
    m, n_H, n_W, n_C = dA.shape                    #256,14,14,6
    
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev)) #256,28,28,6
        
    for h in range(n_H):
        for w in range(n_W):
            vert_start, horiz_start  = h*stride, w*stride
            vert_end,   horiz_end    = vert_start+f, horiz_start+f
            
            if mode == "max":
                A_prev_slice = A_prev[:, vert_start: vert_end, horiz_start: horiz_end, :] 
                A_prev_slice = np.transpose(A_prev_slice, (1,2,3,0))
                mask = A_prev_slice==A_prev_slice.max((0,1))           
                mask = np.transpose(mask, (3,2,0,1))                   
                dA_prev[:, vert_start: vert_end, horiz_start: horiz_end, :] \
                      += np.transpose(np.multiply(dA[:, h, w, :][:,:,np.newaxis,np.newaxis],mask), (0,2,3,1))

            elif mode == "average":
                da = dA[:, h, w, :][:,np.newaxis,np.newaxis,:]  #256*1*1*6
                dA_prev[:, vert_start: vert_end, horiz_start: horiz_end, :] += np.repeat(np.repeat(da, 2, axis=1), 2, axis=2)/f/f
    
    assert(dA_prev.shape == A_prev.shape)
    return dA_prev

def subsampling_forward(A_prev, weight, b, hparameters):
    A_, cache = pool_forward(A_prev, hparameters, 'average') 
    A = A_ * weight + b
    cache_A = (cache, A_)
    return A, cache_A

def subsampling_backward(dA, weight, b, cache_A_):
    (cache, A_) = cache_A_
    db = dA
    dW = np.sum(np.multiply(dA, A_))
    dA_ = dA * weight
    dA_prev = pool_backward(dA_, cache, 'average') 
    return dA_prev, dW, db

#最大池化反向传播
def create_mask_from_window(x):
    mask = x==np.max(x) 
    return mask

#平均池化反向传播
def distribute_value(dz, shape):
    (n_H, n_W) = shape
    a = np.ones(shape) * dz/n_H/n_W
    return a

# activations
import numpy as np

#Squashing function used in LeNet-5
def LeNet5_squash(x):
    return 1.7159*np.tanh(2*x/3)
def d_LeNet5_squash(x):
    return 1.14393*(1-np.power(tanh(2*x/3),2))
def d2_LeNet5_squash(x):
    return -1.52524*((tanh(2/3*x)))*(1-np.power(tanh(2/3*x),2))

#sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def d_sigmoid(x):
    return np.exp(-x) / np.power((1+np.exp(-x)),2)
def d2_sigmoid(x):
    return 2*np.exp(-2*x)/np.power(np.exp(-x)+1,3)  - np.exp(-x) / np.power((1+np.exp(-x)),2)

#tanh
def tanh(x):
    return np.tanh(x)
def d_tanh(x):
    return 1/np.power(np.cosh(x),2)
def d2_tanh(x):
    return -2*(tanh(x))/np.power(np.cosh(x),2)

#ReLU
def ReLU(x):
    return np.where(x>0, x, 0)
def d_ReLU(x):
    return np.where(x>0, 1, 0)
def d2_ReLU(x):
    return np.zeros(d_ReLU(x).shape)

alpha = {"prelu":0.1, "elu":0.5}
def PReLU(x, a=alpha["prelu"]):
    return np.where(x>0, x, a*x)
def d_PReLU(x, a=alpha["prelu"]):
    return np.where(x>0, 1, a)
def d2_PReLU(x, a=alpha["prelu"]):
    return np.zeros(d_PReLU(x, a).shape)

#ELU
def ELU(x, a=alpha["elu"]):
    return np.where(x > 0, x, a*(np.exp(x) - 1))
def d_ELU(x, a=alpha["elu"]):
    return np.where(x > 0, 1, ELU(x, a)+a)
def d2_ELU(x, a=alpha["elu"]):
    return np.where(x > 0, 0, ELU(x, a)+a)

def activation_func():
    actf = [LeNet5_squash, sigmoid, tanh, ReLU, PReLU, ELU]
    actfName = [act.__name__ for act in actf]
    d_actf = [d_LeNet5_squash, d_sigmoid, d_tanh, d_ReLU, d_PReLU, d_ELU]
    d_actfName = [d_act.__name__ for d_act in d_actf]
    return (actf, d_actf), actfName

import numpy as np 
from scipy.signal import convolve2d

# 前向传播
def conv_forward(A_prev, W, b, hparameters):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    n_H = int((n_H_prev + 2*pad - f)/stride + 1)
    n_W = int((n_W_prev + 2*pad - f)/stride + 1)
    
    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev, pad)
    for h in range(n_H):
        for w in range(n_W):
            A_slice_prev = A_prev_pad[:, h*stride:h*stride+f, w*stride:w*stride+f, :]
            Z[:, h, w, :] = np.tensordot(A_slice_prev, W, axes=([1,2,3],[0,1,2])) + b
                            
    assert(Z.shape == (m, n_H, n_W, n_C))
    cache = (A_prev, W, b, hparameters)
    return Z, cache

# 卷积层反向传播
def conv_backward(dZ, cache):

    (A_prev, W, b, hparameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape

    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    (m, n_H, n_W, n_C) = dZ.shape
    
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))
    
    if pad != 0:
        A_prev_pad = zero_pad(A_prev, pad)
        dA_prev_pad = zero_pad(dA_prev, pad)
    else:
        A_prev_pad = A_prev
        dA_prev_pad = dA_prev
    
    for h in range(n_H):
        for w in range(n_W):
            vert_start, horiz_start  = h*stride, w*stride
            vert_end,   horiz_end    = vert_start+f, horiz_start+f
            
            A_slice = A_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end, :]
            
            dA_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end, :] += np.transpose(np.dot(W, dZ[:, h, w, :].T), (3,0,1,2))

            dW += np.dot(np.transpose(A_slice, (1,2,3,0)), dZ[:, h, w, :])
            db += np.sum(dZ[:, h, w, :], axis=0)
            
    dA_prev = dA_prev_pad if pad == 0 else dA_prev_pad[:,pad:-pad, pad:-pad, :]
        
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    return dA_prev, dW, db

def conv_SDLM(dZ, cache):
    (A_prev, W, b, hparameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    (m, n_H, n_W, n_C) = dZ.shape
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))
    
    if pad != 0:
        A_prev_pad = zero_pad(A_prev, pad)
        dA_prev_pad = zero_pad(dA_prev, pad)
    else:
        A_prev_pad = A_prev
        dA_prev_pad = dA_prev
    
    for h in range(n_H):
        for w in range(n_W):
            vert_start, horiz_start  = h*stride, w*stride
            vert_end,   horiz_end    = vert_start+f, horiz_start+f
            
            A_slice = A_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end, :]
            dA_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end, :] += np.transpose(np.dot(np.power(W,2), dZ[:, h, w, :].T), (3,0,1,2))
            dW += np.dot(np.transpose(np.power(A_slice,2), (1,2,3,0)), dZ[:, h, w, :])
    dA_prev = dA_prev_pad if pad == 0 else dA_prev_pad[:,pad:-pad, pad:-pad, :]
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    return dA_prev, dW

#单步卷积
def conv_single_step(a_slice_prev, W, b):
    s = a_slice_prev * W
    Z = np.sum(s)
    Z = Z + b
    return Z


def Conv3D(image3D, filter3D, b, stride):
    output = []
    n_C_prev, n_C = filter3D.shape[2], filter3D.shape[3]
    for c in range(n_C):
        output_c = 0
        for c_prev in range(n_C_prev):
            output_c += convolve2d(image3D[:,:,c_prev], np.rot90(filter3D[:,:,c_prev,c],2),'valid')[::stride,::stride] 
        output_c += b[0,0,0,c]
        output += [output_c]
    return np.transpose(np.array(output),(1,2,0))


def conv_forward_scipy(A_prev, W, b, hparameters):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    n_H = int((n_H_prev + 2*pad - f)/stride + 1)
    n_W = int((n_W_prev + 2*pad - f)/stride + 1)
    
    A_prev_pad = zero_pad(A_prev, pad)
    
    Z = np.empty((m, n_H, n_W, n_C))
    for i in range(m): 
        Z[i,:,:,:] = Conv3D(A_prev_pad[i,:,:,:], W, b, stride)
                                        
    assert(Z.shape == (m, n_H, n_W, n_C))
    cache = (A_prev, W, b, hparameters)
    return Z, cache


import numpy as np

class ConvLayer(object):
    def __init__(self, kernel_shape, hparameters, init_mode='Gaussian_dist'):
        self.hparameters = hparameters
        self.weight, self.bias = initialize(kernel_shape, init_mode)
        self.v_w, self.v_b = np.zeros(kernel_shape), np.zeros((1,1,1,kernel_shape[-1]))
        
    def foward_prop(self, input_map):
        output_map, self.cache = conv_forward(input_map, self.weight, self.bias, self.hparameters)
        return output_map
    
    def back_prop(self, dZ, momentum, weight_decay):
        dA_prev, dW, db = conv_backward(dZ, self.cache)
        self.weight, self.bias, self.v_w, self.v_b = \
            update(self.weight, self.bias, dW, db, self.v_w, self.v_b, self.lr, momentum, weight_decay)
        return dA_prev  
    
    def SDLM(self, d2Z, mu, lr_global):
        d2A_prev, d2W = conv_SDLM(d2Z, self.cache)
        h = np.sum(d2W)/d2Z.shape[0]
        self.lr = lr_global / (mu + h)
        return d2A_prev  
    
class ConvLayer_maps(object):
    def __init__(self, kernel_shape, hparameters, mapping, init_mode='Gaussian_dist'):
        self.hparameters = hparameters
        self.mapping     = mapping
        self.wb   = []      # [weight, bias]
        self.v_wb = []      # [v_w,    v_b]
        for i in range(len(self.mapping)):
            weight_shape = (kernel_shape[0], kernel_shape[1], len(self.mapping[i]), 1)
            w, b = initialize(weight_shape, init_mode)
            self.wb.append([w, b])
            self.v_wb.append([np.zeros(w.shape), np.zeros(b.shape)])
        
    def foward_prop(self, input_map):
        self.iputmap_shape = input_map.shape #(n_m,14,14,6)
        self.caches = []
        output_maps = []
        for i in range(len(self.mapping)):
            output_map, cache = conv_forward(input_map[:,:,:,self.mapping[i]], self.wb[i][0], self.wb[i][1], self.hparameters)
            output_maps.append(output_map)
            self.caches.append(cache)
        output_maps = np.swapaxes(np.array(output_maps),0,4)[0]
        return output_maps
    
    def back_prop(self, dZ, momentum, weight_decay):
        dA_prevs = np.zeros(self.iputmap_shape)
        for i in range(len(self.mapping)):
            dA_prev, dW, db = conv_backward(dZ[:,:,:,i:i+1], self.caches[i])
            self.wb[i][0], self.wb[i][1], self.v_wb[i][0], self.v_wb[i][1] =\
                update(self.wb[i][0], self.wb[i][1], dW, db, self.v_wb[i][0], self.v_wb[i][1], self.lr, momentum, weight_decay)
            dA_prevs[:,:,:,self.mapping[i]] += dA_prev
        return dA_prevs 
    
    def SDLM(self, d2Z, mu, lr_global):
        h = 0
        d2A_prevs = np.zeros(self.iputmap_shape)
        for i in range(len(self.mapping)):
            d2A_prev, d2W = conv_SDLM(d2Z[:,:,:,i:i+1], self.caches[i])
            d2A_prevs[:,:,:,self.mapping[i]] += d2A_prev
            h += np.sum(d2W)
        self.lr = lr_global / (mu + h/d2Z.shape[0])
        return d2A_prevs 

class PoolingLayer(object):
    def __init__(self, hparameters, mode):
        self.hparameters = hparameters
        self.mode = mode
        
    def foward_prop(self, input_map):   # n,28,28,6 / n,10,10,16
        A, self.cache = pool_forward(input_map, self.hparameters, self.mode)
        return A
    
    def back_prop(self, dA):
        dA_prev = pool_backward(dA, self.cache, self.mode)
        return dA_prev
    
    def SDLM(self, d2A):
        d2A_prev = pool_backward(d2A, self.cache, self.mode)
        return d2A_prev

class Subsampling(object):
    def __init__(self, n_kernel, hparameters):
        self.hparameters = hparameters
        self.weight = np.random.normal(0, 0.1, (1,1,1,n_kernel)) 
        self.bias   = np.random.normal(0, 0.1, (1,1,1,n_kernel)) 
        self.v_w = np.zeros(self.weight.shape)
        self.v_b = np.zeros(self.bias.shape)
        
    def foward_prop(self, input_map):   # n,28,28,6 / n,10,10,16
        A, self.cache = subsampling_forward(input_map, self.weight, self.bias, self.hparameters)
        return A
    
    def back_prop(self, dA, momentum, weight_decay):
        dA_prev, dW, db = subsampling_backward(dA, A_, weight, b, self.cache)
        self.weight, self.bias, self.v_w, self.v_b = \
            update(self.weight, self.bias, dW, db, self.v_w, self.v_b, self.lr, momentum, weight_decay)
        return dA_prev
    
    def SDLM(self, d2A, mu, lr_global):
        d2A_prev, d2W, _ = subsampling_backward(dA, A_, weight, b, self.cache)
        h = np.sum(d2W)/d2A.shape[0]
        self.lr = lr_global / (mu + h)
        return d2A_prev

class Activation(object):
    def __init__(self, mode):    
        (act, d_act), actfName = activation_func()
        act_index  = actfName.index(mode)
        self.act   = act[act_index]
        self.d_act = d_act[act_index]
        
    def foward_prop(self, input_image): 
        self.input_image = input_image
        return self.act(input_image)
    
    def back_prop(self, dZ):
        dA = np.multiply(dZ, self.d_act(self.input_image)) 
        return dA
    
    def SDLM(self, d2Z):  #d2_LeNet5_squash
        dA = np.multiply(d2Z, np.power(self.d_act(self.input_image),2)) 
        return dA

class FCLayer(object):
    def __init__(self, weight_shape, init_mode='Gaussian_dist'): 
        
        # Initialization
        self.v_w, self.v_b = np.zeros(weight_shape), np.zeros((weight_shape[-1],))
        self.weight, self.bias = initialize(weight_shape, init_mode)
        
    def foward_prop(self, input_array):
        self.input_array = input_array  #(n_m, 120)
        return np.matmul(self.input_array, self.weight) # (n_m, 84)
        
    def back_prop(self, dZ, momentum, weight_decay):
        dA = np.matmul(dZ, self.weight.T)               # (n_m, 84) * (84, 120) = (n_m, 120)
        dW = np.matmul(self.input_array.T, dZ)          # (n_m, 120).T * (n_m, 84) = (120, 84)
        db = np.sum(dZ.T, axis=1)                       # (84,)
        
        self.weight, self.bias, self.v_w, self.v_b = \
            update(self.weight, self.bias, dW, db, self.v_w, self.v_b, self.lr, momentum, weight_decay)
        return dA
    
    def SDLM(self, d2Z, mu, lr_global):
        d2A = np.matmul(d2Z, np.power(self.weight.T,2))
        d2W = np.matmul(np.power(self.input_array.T,2), d2Z)
        h = np.sum(d2W)/d2Z.shape[0]
        self.lr = lr_global / (mu + h)
        return d2A
    
class RBFLayer_trainable_weight(object):
    def __init__(self, weight_shape, init_weight=None, init_mode='Gaussian_dist'): 
        self.weight_shape = weight_shape # =(10, 84)
        self.v_w = np.zeros(weight_shape)

        if init_weight.shape == (10,84):
            self.weight = init_weight
        else:
            self.weight, _ = initialize(weight_shape, init_mode)
        
    def foward_prop(self, input_array, label, mode):
        
        if mode == 'train':
            self.input_array = input_array
            self.weight_label = self.weight[label,:]  #(n_m, 84)
            loss = 0.5 * np.sum(np.power(input_array - self.weight_label, 2), axis=1, keepdims=True)  #(n_m, )
            return np.sum(np.squeeze(loss))
        
        if mode == 'test':
            subtract_weight = (input_array[:,np.newaxis,:] - np.array([self.weight]*input_array.shape[0])) # (n_m,10,84)
            rbf_class = np.sum(np.power(subtract_weight,2), axis=2) # (n_m, 10)
            class_pred = np.argmin(rbf_class, axis=1) # (n_m,)
            error01 = np.sum(label != class_pred)
            return error01, class_pred

    def back_prop(self, label, lr, momentum, weight_decay):
        dy_predict = -self.weight_label + self.input_array
        dW_target  = -dy_predict
        dW = np.zeros(self.weight_shape) # (10,84)
        
        for i in range(len(label)):  
            dW[label[i],:] += dW_target[i,:]
            
        self.v_w = momentum*self.v_w - weight_decay*lr*self.weight - lr*dW
        self.weight += self.v_w

        return dy_predict

bitmap = rbf_init_weight()

class RBFLayer(object):
    def __init__(self, weight):        
        self.weight = weight  # (10, 84)
        
    def foward_prop(self, input_array, label, mode):
        if mode == 'train':
            self.input_array = input_array
            self.weight_label = self.weight[label,:]  #(n_m, 84)
            loss = 0.5 * np.sum(np.power(input_array - self.weight_label, 2), axis=1, keepdims=True)  #(n_m, )
            return np.sum(np.squeeze(loss))
        if mode == 'test':
            subtract_weight = (input_array[:,np.newaxis,:] - np.array([self.weight]*input_array.shape[0])) # (n_m,10,84)
            rbf_class = np.sum(np.power(subtract_weight,2), axis=2) # (n_m, 10)
            class_pred = np.argmin(rbf_class, axis=1) # (n_m,)
            error01 = np.sum(label != class_pred)
            return error01, class_pred
        
    def back_prop(self):
        dy_predict = -self.weight_label + self.input_array    #(n_m, 84)
        return dy_predict
    
    def SDLM(self):
        # d2y_predict
        return np.ones(self.input_array.shape)
    
    
class LeNet5(object):
    def __init__(self):
        kernel_shape = {"C1": (5,5,1,6),
                        "C3": (5,5,6,16),
                        "C5": (5,5,16,120),
                        "F6": (120,84),
                        "OUTPUT": (84,10)}
        
        hparameters_convlayer = {"stride": 1, "pad": 0}
        hparameters_pooling   = {"stride": 2, "f": 2}        
        
        self.C1 = ConvLayer(kernel_shape["C1"], hparameters_convlayer)
        self.a1 = Activation("LeNet5_squash")
        self.S2 = PoolingLayer(hparameters_pooling, "average")
        
        self.C3 = ConvLayer_maps(kernel_shape["C3"], hparameters_convlayer, C3_mapping)
        self.a2 = Activation("LeNet5_squash")
        self.S4 = PoolingLayer(hparameters_pooling, "average")
        
        self.C5 = ConvLayer(kernel_shape["C5"], hparameters_convlayer)
        self.a3 = Activation("LeNet5_squash")

        self.F6 = FCLayer(kernel_shape["F6"])
        self.a4 = Activation("LeNet5_squash")
        
        self.Output = RBFLayer(bitmap)
        
    def Forward_Propagation(self, input_image, input_label, mode): 
        self.label = input_label
        self.C1_FP = self.C1.foward_prop(input_image)
        self.a1_FP = self.a1.foward_prop(self.C1_FP)
        self.S2_FP = self.S2.foward_prop(self.a1_FP)

        self.C3_FP = self.C3.foward_prop(self.S2_FP)
        self.a2_FP = self.a2.foward_prop(self.C3_FP)
        self.S4_FP = self.S4.foward_prop(self.a2_FP)

        self.C5_FP = self.C5.foward_prop(self.S4_FP)
        self.a3_FP = self.a3.foward_prop(self.C5_FP)

        self.flatten = self.a3_FP[:,0,0,:]
        self.F6_FP = self.F6.foward_prop(self.flatten)
        self.a4_FP = self.a4.foward_prop(self.F6_FP)  
        
        out  = self.Output.foward_prop(self.a4_FP, input_label, mode)

        return out 
        
    def Back_Propagation(self, momentum, weight_decay):
        dy_pred = self.Output.back_prop()
        
        dy_pred = self.a4.back_prop(dy_pred)
        F6_BP = self.F6.back_prop(dy_pred, momentum, weight_decay)
        reverse_flatten = F6_BP[:,np.newaxis,np.newaxis,:]
        
        reverse_flatten = self.a3.back_prop(reverse_flatten) 
        C5_BP = self.C5.back_prop(reverse_flatten, momentum, weight_decay)
        
        S4_BP = self.S4.back_prop(C5_BP)
        S4_BP = self.a2.back_prop(S4_BP)
        C3_BP = self.C3.back_prop(S4_BP, momentum, weight_decay) 
        
        S2_BP = self.S2.back_prop(C3_BP)
        S2_BP = self.a1.back_prop(S2_BP)  
        C1_BP = self.C1.back_prop(S2_BP, momentum, weight_decay)
        
    def SDLM(self, mu, lr_global):
        d2y_pred = self.Output.SDLM()
        d2y_pred = self.a4.SDLM(d2y_pred)
        
        F6_SDLM = self.F6.SDLM(d2y_pred, mu, lr_global)
        reverse_flatten = F6_SDLM[:,np.newaxis,np.newaxis,:]
        
        reverse_flatten = self.a3.SDLM(reverse_flatten) 
        C5_SDLM = self.C5.SDLM(reverse_flatten, mu, lr_global)
        
        S4_SDLM = self.S4.SDLM(C5_SDLM)
        S4_SDLM = self.a2.SDLM(S4_SDLM)
        C3_SDLM = self.C3.SDLM(S4_SDLM, mu, lr_global)
        
        S2_SDLM = self.S2.SDLM(C3_SDLM)
        S2_SDLM = self.a1.SDLM(S2_SDLM)  
        C1_SDLM = self.C1.SDLM(S2_SDLM, mu, lr_global)