import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.models import load_model
import random
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from utilize import *
import time 

# model params
train_frames=10 
data_frames=20 
# model_dir="./model_Bi/"
# model_dir="./model_single/"
# model_dir="./model_ext/"
model_dir="./model/"
dataset_root="./data/"
cate="" #which type of data set to predict, '' indicates all data
single_test=False
attr_list=[2,3,5,6,7,9] #feature inputs for model
attr_num=len(attr_list)

# load model
single_model=False
if(single_model):
    print("Use single stage model.")
    # model_cate="_re"
    # model_cate="_npnew"
    # model_cate="_ALL"
    model_cate = ""
    model_name="best_weights"+model_cate+".h5"
    model=load_model(model_dir+model_name)
else:
    print("Use two stage model.")
    model_cate0="_B"
    model_cate1="_An"
    model_name0="best_weights"+model_cate0+".h5"
    model_name1="best_weights"+model_cate1+".h5"
    model0=load_model(model_dir+model_name0)
    model1=load_model(model_dir+model_name1)

# load data
df = pd.read_csv(dataset_root+"/test"+cate+".csv")
label=df.iloc[:,-1].values
data_num=int(len(label)/data_frames)
test_label=pro_list(data_num=data_num,train_list=label,pred=data_frames)
lane_data=df.iloc[:,7].values
lane_data=np.reshape(lane_data,[data_num,data_frames])


#for 2-stage model, case_label stores corresponding model's ID for each test case
case_label=case_list(data_num=data_num,lane_data=lane_data,step=data_frames)
case_label=np.array(case_label)
#for check
case0=(case_label==0).astype(int)
case1=(case_label==1).astype(int)
tl0=((case0*test_label)==1).astype(int)
tl1=((case1*test_label)==1).astype(int)
print(np.sum(tl0),np.sum(tl1))


data=df.iloc[:,attr_list].values
data=np.reshape(data,[data_num,data_frames,attr_num])
data=partial_frames(data,10)#extract 10 frames from 20 frames for testing 
turn_left_test=[]
turn_right_test=[]
keep_lane_test=[]
cnt0=cnt1=cnt2=0
test_order0=[]
test_order1=[]
test_order2=[]
case_order0=[]
case_order1=[]
case_order2=[]
for i in range(len(test_label)):
    if test_label[i]==0:
        cnt0+=1
        keep_lane_test.append(data[i])
        test_order0.append(i)
        case_order0.append(case_label[i])
    elif test_label[i]==1:
        cnt1+=1
        turn_left_test.append(data[i])
        test_order1.append(i)
        case_order1.append(case_label[i])
    elif test_label[i]==2:
        cnt2+=1
        turn_right_test.append(data[i])
        test_order2.append(i)
        case_order2.append(case_label[i])
print("Test-record Proportion:(keep lane, left, right)",cnt0,cnt1,cnt2)

print("Testing data description:")
if(single_test==False):
    d0_test,l0_test=sampling_from_pool(keep_lane_test,[[1,0,0]],len(keep_lane_test))
    d1_test,l1_test=sampling_from_pool(turn_left_test,[[0,1,0]],len(turn_left_test))
    d2_test,l2_test=sampling_from_pool(turn_right_test,[[0,0,1]],len(turn_right_test))
    test_order=test_order0+test_order1+test_order2
    case_order=case_order0+case_order1+case_order2
    case_order=np.array(case_order)
    print(test_order[-20:])
    test_data,test_label=combineData(d0_test,l0_test,d1_test,l1_test,d2_test, l2_test,"Test")
    # test_data,test_label=combineData_2(d1_test,l1_test,d2_test,l2_test,"Test")
else:
    d_test,l_test=sampling_from_pool(keep_lane_test,[[1,0,0]],len(keep_lane_test))
    test_data,test_label=d_test,l_test


#testing process
true_category=np.argmax(test_label,axis=1)
if(single_model==True):
    result=model.predict(test_data)
    loss, accuracy = model.evaluate(test_data,test_label)
    print("Loss, Acc:",loss,accuracy)
    pred_category=np.argmax(result,axis=1)
else:
    result0=model0.predict(test_data)
    result1=model1.predict(test_data)

    result0=np.argmax(result0,axis=1)
    result1=np.argmax(result1,axis=1)
    # result0+=1
    # result1+=1
    pred_category=(case_order==0).astype(int)*result0+(case_order==1).astype(int)*result1
    # pred_category=(case_order==1).astype(int)*result1
    # pred_category=result0

m=confusion_matrix(true_category,pred_category)
print(m)
acc_list=[]
for i in range(len(m)):
    tot=0
    for j in range(len(m[i])):
        tot+=m[i][j]
    if(tot==0):
        acc=0
    else:
        acc=float(m[i][i])/float(tot)
    acc_list.append(acc)
print(acc_list)

# for i in range(len(pred_category)):
#     if(pred_category[i]==2 and true_category[i]==1):
#         case1=df[test_order[i]*data_frames:(test_order[i]+1)*data_frames]
#         print(str(case1['Vehicle_ID'].values[-1])+' '+str(case1['Frame_ID'].values[-1]))


