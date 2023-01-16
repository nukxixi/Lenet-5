#import package
import numpy as np 
import sys
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib as mpl
import time
from tqdm import *
from utils import *

test_image_path = './mnist/t10k-images-idx3-ubyte'
test_label_path = './mnist/t10k-labels-idx1-ubyte'
train_image_path = './mnist/train-images-idx3-ubyte'
train_label_path = './mnist/train-labels-idx1-ubyte'
trainset = (train_image_path, train_label_path)
testset = (test_image_path, test_label_path)
(train_image, train_label) = readDataset(trainset)
(test_image, test_label) = readDataset(testset)
n_m, n_m_test = len(train_label), len(test_label)
print("Training image shape:", train_image.shape)
print("Testing image shape: ", test_image.shape)
print("Size of the training set: ", n_m)
print("Size of the test set: ", n_m_test)
print("Image shape: ", train_image[0].shape)

train_image_normalized_pad = normalize(zero_pad(train_image[:,:,:,np.newaxis], 2),'lenet5')
test_image_normalized_pad  = normalize(zero_pad(test_image[:,:,:,np.newaxis],  2),'lenet5')

C3_mapping = [[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,0],[5,0,1],\
              [0,1,2,3],[1,2,3,4],[2,3,4,5],[3,4,5,0],[4,5,0,1],[5,0,1,2],\
              [0,1,3,4],[1,2,4,5],[0,2,3,5],\
              [0,1,2,3,4,5]]
bitmap = rbf_init_weight()


class LeNet5(object):
    def __init__(self):
        kernel_shape = {"C1": (5,5,1,6),
                        "C3": (5,5,6,16),    ### C3 has designated combinations
                        "C5": (5,5,16,120),  ### It's actually a FC layer
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

ConvNet = LeNet5()

epoch_orig, lr_global_orig = 12, np.array([5e-4]*2 + [2e-4]*3 + [1e-4]*3 + [5e-5]*4 + [1e-5]*8)

epoches, lr_global_list = epoch_orig, lr_global_orig*100

momentum = 0.9
weight_decay = 0
batch_size = 256

st = time.time()
cost_last, count = np.Inf, 0
err_rate_list = []
for epoch in range(0,epoches):
    print("---------- epoch", epoch+1, "begin ----------")
    (batch_image, batch_label) = random_mini_batches(train_image_normalized_pad, train_label, mini_batch_size = 500, one_batch=True)
    ConvNet.Forward_Propagation(batch_image, batch_label, 'train')
    lr_global = lr_global_list[epoch]
    ConvNet.SDLM(0.02, lr_global)

    
    ste = time.time()
    cost = 0
    mini_batches = random_mini_batches(train_image_normalized_pad, train_label, batch_size)
    pbar = trange(len(mini_batches))
    for i in pbar:
        batch_image, batch_label = mini_batches[i]
        
        loss = ConvNet.Forward_Propagation(batch_image, batch_label, 'train')     
        cost += loss
        
        ConvNet.Back_Propagation(momentum, weight_decay) 

        # print progress
        if i%(int(len(mini_batches)/100))==0:
            tqdm.write("progress:"+str(int(100*(i+1)/len(mini_batches)))+"%, "+"loss ="+str(cost/(i+1)))
    
    print ("Done, loss", epoch+1, ":", cost/len(mini_batches),"                                             ")
    print("Please wait for some moment...")
    error01_train, _ = ConvNet.Forward_Propagation(train_image_normalized_pad, train_label, 'test')  
    error01_test, _  = ConvNet.Forward_Propagation(test_image_normalized_pad,  test_label,  'test')     
    err_rate_list.append([error01_train/60000, error01_test/10000])
    print("0/1 error of training set:",  error01_train, "/", len(train_label))
    print("0/1 error of testing set: ",  error01_test,  "/", len(test_label))
    print("Time used: ",time.time() - ste, "sec")
    print("---------- epoch", epoch+1, "end ------------")
    with open('model_data_'+str(epoch)+'.pkl', 'wb') as output:
        pickle.dump(ConvNet, output, pickle.HIGHEST_PROTOCOL)

    
err_rate_list = np.array(err_rate_list).T
print("Total time used: ", time.time() - st, "sec")

# This shows the error rate of training and testing data after each epoch
x = np.arange(epoches)
plt.xlabel('epoch')
plt.ylabel('ErrorRate')
plt.plot(x, err_rate_list[0])
plt.plot(x, err_rate_list[1])
plt.legend(['train data', 'test data'], loc='upper right')
plt.show()
         
with open('model_final.pkl', 'rb') as input_:
    ConvNet = pickle.load(input_)

test_image_normalized_pad = normalize(zero_pad(test_image[:,:,:,np.newaxis], 2), 'lenet5')
error01, class_pred = ConvNet.Forward_Propagation(test_image_normalized_pad, test_label, 'test')
print("error rate:", error01/len(class_pred))
print("correct rare:",(len(class_pred)-error01)/len(class_pred))