import pandas as pd
import numpy as np
import random

def partial_frames(data,interval):
    train_data=data[:,-interval:,:]
    print(train_data.shape)
    return train_data

def case_list(data_num,lane_data,step):
    data_list=[]
    for i in range(data_num):
      f=False
      for j in range(step-1):
        if(lane_data[i][step-1-j]!=lane_data[i][step-1-j-1]):
          f=True
          break
      if(f==False):#B
        data_list.append(0)
      if(f==True):#A
        data_list.append(1)
    return data_list

def pro_list(data_num,train_list,pred):
    data_list=[]
    for i in range(data_num):
        data_list.append(train_list[i*pred])
    return data_list

def shuffle(X,Y):
  np.random.seed(10)
  randomList = np.arange(X.shape[0])
  np.random.shuffle(randomList)
  return X[randomList], Y[randomList]

def splitData(X,Y,rate):
  X_train = X[int(X.shape[0]*rate):]
  Y_train = Y[int(Y.shape[0]*rate):]
  X_val = X[:int(X.shape[0]*rate)]
  Y_val = Y[:int(Y.shape[0]*rate)]
  return X_train, Y_train, X_val, Y_val

def sampling_from_pool(data,label,num):
    if(num<len(data)):
      seq=np.random.permutation(len(data))
      X=[]
      for i in range(num):
        fetch_id=seq[i]
        X.append(data[fetch_id])
    elif(num==len(data)):
      X=data
    elif(num>len(data)):
      X=data
      for i in range(num-len(data)):
        rand_id=random.randint(0,len(data)-1)#random generate
        X.append(data[rand_id])

    y=label*num
    return np.array(X),np.array(y)

def combineData(d0,l0,d1,l1,d2,l2,Note):
    d=np.concatenate((d0,d1,d2),axis=0)
    l=np.concatenate((l0,l1,l2),axis=0)
    print(Note+"Data Shape")
    print("0,1,2: ",d0.shape,d1.shape,d2.shape)
    print(d.shape)
    print(l.shape)  
    return d,l
def combineData_4(d0,l0,d1,l1,d2,l2,d3,l3,Note):
    d=np.concatenate((d0,d1,d2,d3),axis=0)
    l=np.concatenate((l0,l1,l2,l3),axis=0)
    print(Note+"Data Shape")
    print("0,1: ",d0.shape,d1.shape,d2.shape,d3.shape)
    print(d.shape)
    print(l.shape)  
    return d,l
def combineData_2(d0,l0,d1,l1,Note):
    d=np.concatenate((d0,d1),axis=0)
    l=np.concatenate((l0,l1),axis=0)
    print(Note+"Data Shape")
    print("0,1: ",d0.shape,d1.shape)
    print(d.shape)
    print(l.shape)  
    return d,l